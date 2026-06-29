"""v4 runtime: one capture+inference loop per camera, one shared cap manager.

Each camera runs in its own thread (``CameraWorker``): read frame -> infer ->
update that camera's tracker -> hand finished tracks to the shared
``CapEventManager``. A lightweight coordinator loop on the calling thread drives
the composite preview, periodic performance snapshots, and the cap-event merge
flush, and tears everything down cleanly when ``stop_event`` is set.

There is deliberately no frame pairing, anchor geometry, prediction or snapshot
machinery here - the only cross-camera logic is the de-dup inside the manager.
"""

from __future__ import annotations

import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta
from typing import Callable

from gpio_output import GPIOOutputPin

from .actuation import NullGPIOOutputPin, RejectScheduler
from .config import RuntimeConfig, class_name, validate_config
from .decision import CapEventManager
from .model import create_onnx_session, postprocess, preprocess, resolve_imgsz, resolve_model_path
from .tracking import CameraTracker
from .types import Box, CapturedFrame, PerfSnapshot, RuntimeCallbacks


CAP_PROP_FRAME_WIDTH = 3
CAP_PROP_FRAME_HEIGHT = 4
CAP_PROP_FPS = 5
CAP_PROP_BUFFERSIZE = 38
PERF_EMIT_INTERVAL_S = 0.5


class Clock:
    """Monotonic clock with wall-clock formatting for human-readable logs."""

    def __init__(self, time_fn: Callable[[], float] = time.monotonic):
        self.time_fn = time_fn
        self.origin_monotonic = float(time_fn())
        self.origin_wall = datetime.now().astimezone()

    def monotonic(self) -> float:
        return float(self.time_fn())

    def format(self, timestamp: float | None = None) -> str:
        moment = self.monotonic() if timestamp is None else float(timestamp)
        wall = self.origin_wall + timedelta(seconds=moment - self.origin_monotonic)
        return wall.isoformat(timespec="milliseconds")


# --------------------------------------------------------------------------- #
# Camera helpers (copied/stripped from v3; Linux/V4L2 oriented).
# --------------------------------------------------------------------------- #

def parse_cameras(cameras) -> tuple[list[str | int], list[str]]:
    sources: list[str | int] = []
    device_paths: list[str] = []
    for camera in cameras:
        text = str(camera).strip()
        if text.isdigit():
            sources.append(int(text))
            device_paths.append(f"/dev/video{text}")
        else:
            sources.append(text)
            device_paths.append(text)
    return sources, device_paths


def _v4l2_unavailable(device_path: str) -> bool:
    return not sys.platform.startswith("linux") or not str(device_path).startswith("/dev/")


def set_camera_format(
    device_path: str,
    width: int,
    height: int,
    fps: int,
    *,
    pixel_format: str,
    log_fn: Callable[..., None] = print,
) -> None:
    if _v4l2_unavailable(device_path):
        return
    command = [
        "v4l2-ctl",
        "-d",
        str(device_path),
        f"--set-fmt-video=width={int(width)},height={int(height)},pixelformat={pixel_format}",
        f"--set-parm={int(fps)}",
    ]
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True)
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "").strip()
            log_fn(f"[CAMERA][WARN] format command failed for {device_path} rc={result.returncode}: {detail}")
    except OSError as exc:
        log_fn(f"[CAMERA][WARN] unable to set format for {device_path}: {exc}")


def set_camera_controls(device_path: str, exposure: int, *, log_fn: Callable[..., None] = print) -> None:
    if _v4l2_unavailable(device_path):
        return
    command = ["v4l2-ctl", "-d", str(device_path), f"--set-ctrl=exposure_time_absolute={int(exposure)}"]
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True)
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "").strip()
            log_fn(f"[CAMERA][WARN] exposure command failed for {device_path} rc={result.returncode}: {detail}")
    except OSError as exc:
        log_fn(f"[CAMERA][WARN] unable to set exposure for {device_path}: {exc}")


def open_cam(source, width: int, height: int, fps: int, pixel_format: str):
    import cv2

    camera = cv2.VideoCapture(source, cv2.CAP_V4L2 if sys.platform.startswith("linux") else 0)
    camera.set(CAP_PROP_FRAME_WIDTH, width)
    camera.set(CAP_PROP_FRAME_HEIGHT, height)
    camera.set(CAP_PROP_FPS, fps)
    try:
        camera.set(CAP_PROP_BUFFERSIZE, 1)  # keep latency low: grab the freshest frame
    except Exception:
        pass
    if hasattr(cv2, "VideoWriter_fourcc"):
        camera.set(6, cv2.VideoWriter_fourcc(*pixel_format))
    return camera


def _camera_is_open(camera) -> bool:
    is_opened = getattr(camera, "isOpened", None)
    if is_opened is None:
        return True
    try:
        return bool(is_opened())
    except Exception:
        return False


def validate_opened_cameras(cameras, camera_sources, device_paths) -> None:
    failed = []
    for index, camera in enumerate(cameras):
        if _camera_is_open(camera):
            continue
        source = camera_sources[index] if index < len(camera_sources) else "unknown"
        device_path = device_paths[index] if index < len(device_paths) else "unknown"
        failed.append(f"camera {index} source={source!r} device={device_path!r}")
    if failed:
        raise RuntimeError(
            "Unable to open v4 camera(s): "
            + "; ".join(failed)
            + ". Check --cams, cabling/power, permissions, and `v4l2-ctl --list-devices`."
        )


# --------------------------------------------------------------------------- #
# Frame / overlay helpers.
# --------------------------------------------------------------------------- #

def _to_box(box) -> Box:
    return (float(box[0]), float(box[1]), float(box[2]), float(box[3]), float(box[4]), int(box[5]))


def mirror_frame_horizontal(frame):
    try:
        mirrored = frame[:, ::-1]
        return mirrored.copy() if hasattr(mirrored, "copy") else mirrored
    except Exception:
        pass
    try:
        import cv2

        return cv2.flip(frame, 1)
    except Exception:
        return frame


def draw_boxes(frame, boxes: tuple[Box, ...]):
    """Green for undefected, red for dirt_defect, with a class+confidence label."""

    import cv2

    for box in boxes:
        x1, y1, x2, y2, conf, cls = box
        color = (0, 0, 255) if int(cls) == 1 else (0, 200, 0)
        label = class_name(int(cls)) or str(int(cls))
        try:
            cv2.rectangle(frame, (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2))), color, 2)
            cv2.putText(
                frame,
                f"{label}:{conf:.2f}",
                (int(round(x1)), max(0, int(round(y1)) - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )
        except Exception:
            return frame
    return frame


def compose_preview(frames: list[object]):
    """Side-by-side composite of the (already annotated) per-camera frames."""

    import cv2
    import numpy as np

    usable = [frame for frame in frames if frame is not None]
    if not usable:
        return None
    if len(usable) == 1:
        return usable[0]
    height = min(frame.shape[0] for frame in usable)
    resized = [cv2.resize(frame, (int(frame.shape[1] * height / frame.shape[0]), height)) for frame in usable]
    spacer = np.zeros((height, 6, 3), dtype=np.uint8)
    parts: list[object] = []
    for index, frame in enumerate(resized):
        if index:
            parts.append(spacer)
        parts.append(frame)
    return np.hstack(parts)


# --------------------------------------------------------------------------- #
# Shared state + camera worker.
# --------------------------------------------------------------------------- #

class SharedRuntimeState:
    """Thread-safe slots holding each camera's latest frame, boxes and stats."""

    def __init__(self, camera_count: int):
        self.camera_count = int(camera_count)
        self._lock = threading.Lock()
        self._frames: list[object | None] = [None] * self.camera_count
        self._boxes: list[tuple[Box, ...]] = [tuple() for _ in range(self.camera_count)]
        self._captured = [0] * self.camera_count
        self._processed = [0] * self.camera_count
        self._inference_ms: list[float | None] = [None] * self.camera_count

    def record_capture(self, index: int) -> None:
        with self._lock:
            self._captured[index] += 1

    def publish(self, index: int, frame, boxes: tuple[Box, ...], inference_ms: float) -> None:
        with self._lock:
            self._frames[index] = frame
            self._boxes[index] = boxes
            self._processed[index] += 1
            self._inference_ms[index] = inference_ms

    def latest_frames(self) -> tuple[list[object | None], list[tuple[Box, ...]]]:
        with self._lock:
            return list(self._frames), list(self._boxes)

    def perf_counts(self) -> tuple[list[int], list[int], list[float | None]]:
        with self._lock:
            return list(self._captured), list(self._processed), list(self._inference_ms)


class CameraWorker:
    """Read+infer loop for a single camera, feeding its tracker and the manager."""

    def __init__(
        self,
        *,
        camera_index: int,
        camera,
        session,
        input_name: str,
        model_imgsz: int,
        reject_threshold: float,
        mirror_horizontal: bool,
        tracker: CameraTracker,
        manager: CapEventManager,
        shared: SharedRuntimeState,
        preprocess_fn,
        postprocess_fn,
        stop_event: threading.Event,
        time_fn: Callable[[], float],
        sleep_fn: Callable[[float], None],
        log_fn: Callable[..., None],
    ):
        self.camera_index = int(camera_index)
        self.camera = camera
        self.session = session
        self.input_name = input_name
        self.model_imgsz = int(model_imgsz)
        self.reject_threshold = float(reject_threshold)
        self.mirror_horizontal = bool(mirror_horizontal)
        self.tracker = tracker
        self.manager = manager
        self.shared = shared
        self.preprocess_fn = preprocess_fn
        self.postprocess_fn = postprocess_fn
        self.stop_event = stop_event
        self.time_fn = time_fn
        self.sleep_fn = sleep_fn
        self.log_fn = log_fn
        self._sequence = 0
        self._thread = threading.Thread(target=self._run, name=f"cap-line-v4-camera-{self.camera_index}", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def join(self, timeout: float | None = None) -> None:
        self._thread.join(timeout)

    def _read(self) -> CapturedFrame | None:
        try:
            ok, frame = self.camera.read()
        except Exception:
            ok, frame = False, None
        if not ok or frame is None:
            return None
        if self.mirror_horizontal:
            frame = mirror_frame_horizontal(frame)
        self._sequence += 1
        return CapturedFrame(self.camera_index, frame, float(self.time_fn()), self._sequence)

    def _finish_due_tracks(self, now: float) -> None:
        for track in self.tracker.collect_finished(now):
            self.manager.handle_finished_track(track)

    def _run(self) -> None:
        while not self.stop_event.is_set():
            captured = self._read()
            if captured is None:
                # No frame this iteration: still advance timeouts so a cap that
                # just left view finishes even during a quiet stretch.
                self._finish_due_tracks(float(self.time_fn()))
                self.sleep_fn(0.005)
                continue
            self.shared.record_capture(self.camera_index)
            inference_start = float(self.time_fn())
            try:
                tensor, meta = self.preprocess_fn(captured.frame, self.model_imgsz)
                output = self.session.run(None, {self.input_name: tensor})[0]
                raw_boxes = self.postprocess_fn(output, meta, conf_threshold=self.reject_threshold)
            except Exception as exc:
                self.log_fn(f"[CAMERA {self.camera_index}][WARN] inference failed: {exc}")
                self.sleep_fn(0.005)
                continue
            boxes = tuple(_to_box(box) for box in raw_boxes)
            inference_ms = (float(self.time_fn()) - inference_start) * 1000.0
            self.shared.publish(self.camera_index, captured.frame, boxes, inference_ms)
            self.tracker.update(boxes, captured.timestamp)
            self._finish_due_tracks(float(self.time_fn()))


# --------------------------------------------------------------------------- #
# Entry point.
# --------------------------------------------------------------------------- #

def run_detection(
    config: RuntimeConfig,
    callbacks: RuntimeCallbacks | None = None,
    stop_event: threading.Event | None = None,
    *,
    pin_factory=GPIOOutputPin,
    camera_factory: Callable[[int, str | int, RuntimeConfig], object] | None = None,
    session_factory: Callable[[str, int], object] | None = None,
    preprocess_fn=None,
    postprocess_fn=None,
    time_fn: Callable[[], float] = time.monotonic,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> None:
    validate_config(config)
    callbacks = callbacks or RuntimeCallbacks()
    log_fn = callbacks.log_fn
    stop_event = stop_event or threading.Event()
    clock = Clock(time_fn)

    model_path, preset_imgsz = resolve_model_path(config.model)
    camera_sources, device_paths = parse_cameras(config.cameras)
    width, height = config.resolution
    for device_path in device_paths:
        set_camera_format(device_path, width, height, config.target_fps, pixel_format=config.pixel_format, log_fn=log_fn)
        set_camera_controls(device_path, config.exposure, log_fn=log_fn)

    active_camera_factory = camera_factory or (
        lambda _index, source, cfg: open_cam(source, cfg.resolution[0], cfg.resolution[1], cfg.target_fps, cfg.pixel_format)
    )
    cameras = [active_camera_factory(index, source, config) for index, source in enumerate(camera_sources)]
    try:
        validate_opened_cameras(cameras, camera_sources, device_paths)
    except Exception:
        for camera in cameras:
            if hasattr(camera, "release"):
                camera.release()
        raise

    active_preprocess = preprocess_fn or preprocess
    active_postprocess = postprocess_fn or postprocess
    active_session_factory = session_factory or create_onnx_session
    sessions = [active_session_factory(model_path, config.onnx_intra_op_threads) for _ in camera_sources]
    input_metas = [session.get_inputs()[0] for session in sessions]
    input_names = [meta.name for meta in input_metas]
    model_imgsz = resolve_imgsz(input_metas[0], config.imgsz, preset_imgsz)

    scheduler = RejectScheduler(
        trigger_pin=config.trigger_pin,
        trigger_duration=config.trigger_duration,
        trigger_min_gap=config.trigger_min_gap,
        pin_factory=NullGPIOOutputPin if config.simulate_gpio else pin_factory,
        log_fn=log_fn,
        time_fn=time_fn,
        sleep_fn=sleep_fn,
    )
    manager = CapEventManager(
        config,
        scheduler=scheduler,
        time_fn=time_fn,
        clock=clock,
        history_callback=callbacks.history_callback,
        log_fn=log_fn,
    )
    track_timeout_s = float(config.track_timeout_ms) / 1000.0
    trackers = [
        CameraTracker(index, track_iou=config.track_iou, track_timeout_s=track_timeout_s)
        for index in range(len(camera_sources))
    ]
    shared = SharedRuntimeState(len(camera_sources))
    workers = [
        CameraWorker(
            camera_index=index,
            camera=cameras[index],
            session=sessions[index],
            input_name=input_names[index],
            model_imgsz=model_imgsz,
            reject_threshold=config.reject_threshold,
            mirror_horizontal=config.mirror_cameras[index],
            tracker=trackers[index],
            manager=manager,
            shared=shared,
            preprocess_fn=active_preprocess,
            postprocess_fn=active_postprocess,
            stop_event=stop_event,
            time_fn=time_fn,
            sleep_fn=sleep_fn,
            log_fn=log_fn,
        )
        for index in range(len(camera_sources))
    ]

    preview_enabled = (
        callbacks.preview_callback is not None
        and not config.no_display
        and float(config.live_preview_fps) > 0.0
    )
    preview_interval_s = 1.0 / float(config.live_preview_fps) if preview_enabled else 0.0
    preview_broken = False
    start_time = clock.monotonic()
    last_preview = 0.0
    last_perf = 0.0

    log_fn(f"Using v4 model: {model_path} imgsz={model_imgsz} target_fps={config.target_fps} gpio={scheduler.backend_name}")
    try:
        for worker in workers:
            worker.start()
        while not stop_event.is_set():
            now = clock.monotonic()
            manager.flush_expired(now)

            if preview_enabled and not preview_broken and (now - last_preview) >= preview_interval_s:
                last_preview = now
                try:
                    frames, boxes = shared.latest_frames()
                    annotated = []
                    for frame, frame_boxes in zip(frames, boxes):
                        if frame is None:
                            continue
                        drawable = frame.copy() if hasattr(frame, "copy") else frame
                        annotated.append(draw_boxes(drawable, frame_boxes))
                    composite = compose_preview(annotated)
                    if composite is not None:
                        callbacks.preview_callback(composite)
                except Exception as exc:  # cv2/numpy missing or a draw error: stop trying
                    preview_broken = True
                    log_fn(f"[PREVIEW][WARN] disabled: {exc}")

            if callbacks.performance_callback is not None and (now - last_perf) >= PERF_EMIT_INTERVAL_S:
                last_perf = now
                callbacks.performance_callback(_perf_snapshot(shared, manager, scheduler, start_time, clock))

            sleep_fn(0.005)
    finally:
        stop_event.set()
        for worker in workers:
            worker.join(timeout=2.0)
        # Finalize any caps still mid-track so the last cap before stop is logged.
        for tracker in trackers:
            for track in tracker.flush():
                manager.handle_finished_track(track)
        manager.finalize_all()
        for camera in cameras:
            if hasattr(camera, "release"):
                camera.release()
        scheduler.close()


def _perf_snapshot(
    shared: SharedRuntimeState,
    manager: CapEventManager,
    scheduler: RejectScheduler,
    start_time: float,
    clock: Clock,
) -> PerfSnapshot:
    elapsed = max(1e-6, clock.monotonic() - start_time)
    captured, processed, inference_ms = shared.perf_counts()
    return PerfSnapshot(
        elapsed_s=elapsed,
        capture_fps_by_camera=tuple(count / elapsed for count in captured),
        processed_fps_by_camera=tuple(count / elapsed for count in processed),
        inference_ms_by_camera=tuple(inference_ms),
        gpio_backend=scheduler.backend_name,
        caps_seen=manager.caps_seen,
        rejects=manager.rejects,
    )
