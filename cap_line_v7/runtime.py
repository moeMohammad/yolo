"""v7 runtime: one capture+two-stage-inference loop per camera, one shared cap manager.

Each camera has a continuously-draining capture thread and a one-slot latest-
frame handoff to its inference worker. The worker runs the two-stage pipeline
per frame: cap detector -> dedup -> crop each cap box from the ORIGINAL frame
-> dirt classifier -> P(dirt) per box. Tracking accumulates those
probabilities and hands only finished, physically-qualified tracks to the
shared ``CapEventManager``. A lightweight coordinator loop drives preview,
metrics, event flushing, and fail-safe shutdown.

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

from .actuation import NullGPIOOutputPin, RejectScheduler
from .config import RuntimeConfig, class_name, validate_config
from .decision import CapEventManager
from .model import (
    classify_dirt_probability,
    create_onnx_session,
    deduplicate_boxes,
    postprocess,
    preprocess,
    resolve_imgsz,
    resolve_model_path,
)
from .tracking import CameraTracker
from .types import Box, CapturedFrame, PerfSnapshot, RuntimeCallbacks


def resolve_pin_factory(config: RuntimeConfig):
    """Pick the GPIO driver for the configured backend.

    v7 targets the Jetson Nano rig: the default is the untouched Jetson driver
    from ``gpio_output.py``. The Raspberry Pi driver (gpiozero) stays selectable
    via ``gpio_backend: "rpi"``. ``simulate_gpio`` overrides both.
    """

    if config.simulate_gpio:
        return NullGPIOOutputPin
    if config.gpio_backend == "jetson":
        from gpio_output import GPIOOutputPin

        return GPIOOutputPin
    from rpi_gpio_output import RPiGPIOOutputPin

    return RPiGPIOOutputPin


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
            "Unable to open v7 camera(s): "
            + "; ".join(failed)
            + ". Check --cams, cabling/power, permissions, and `v4l2-ctl --list-devices`."
        )


# --------------------------------------------------------------------------- #
# Frame / overlay helpers.
# --------------------------------------------------------------------------- #

def _to_box(box) -> Box:
    return (float(box[0]), float(box[1]), float(box[2]), float(box[3]), float(box[4]), int(box[5]))


def _display_box(box, p_dirt: float | None, frame_dirt_threshold: float) -> Box:
    """Fold the classifier verdict into the box tuple for preview/tracking.

    Geometry comes from the detector; class and confidence reflect the crop
    classifier (class 1 + P(dirt) for a dirty frame, class 0 + P(clean) for a
    clean one). Without a classifier score the box stays class 0 with the
    detector confidence — presence only, no dirt claim.
    """

    x1, y1, x2, y2 = (float(value) for value in box[:4])
    if p_dirt is None:
        return (x1, y1, x2, y2, float(box[4]), 0)
    probability = min(1.0, max(0.0, float(p_dirt)))
    if probability >= float(frame_dirt_threshold):
        return (x1, y1, x2, y2, probability, 1)
    return (x1, y1, x2, y2, 1.0 - probability, 0)


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


def draw_boxes(
    frame,
    boxes: tuple[Box, ...],
    *,
    presence_line_axis: str | None = None,
    presence_line_ratio: float | None = None,
    presence_direction: str = "positive",
):
    """Draw detections plus the physical-presence gate used for actuation."""

    import cv2

    if presence_line_axis in {"x", "y"} and presence_line_ratio is not None:
        try:
            height, width = frame.shape[:2]
            if presence_line_axis == "x":
                coordinate = int(round(width * float(presence_line_ratio)))
                start, end = (coordinate, 0), (coordinate, height - 1)
            else:
                coordinate = int(round(height * float(presence_line_ratio)))
                start, end = (0, coordinate), (width - 1, coordinate)
            cv2.line(frame, start, end, (255, 255, 0), 2)
            direction_marker = {"positive": "+", "negative": "-"}.get(presence_direction, "+/-")
            cv2.putText(
                frame,
                f"PRESENCE GATE {presence_line_axis}{direction_marker}",
                (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                1,
            )
        except Exception:
            pass
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

    def clear_boxes(self, index: int) -> None:
        """Clear stale overlays after a read/inference failure."""

        with self._lock:
            self._boxes[index] = tuple()

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
        classifier_session,
        classifier_input_name: str,
        classifier_imgsz: int,
        crop_margin: float,
        frame_dirt_threshold: float,
        max_classified_boxes: int,
        detect_threshold: float,
        duplicate_iou_threshold: float,
        max_frame_age_s: float,
        mirror_horizontal: bool,
        tracker: CameraTracker,
        manager: CapEventManager,
        shared: SharedRuntimeState,
        preprocess_fn,
        postprocess_fn,
        classify_fn,
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
        self.classifier_session = classifier_session
        self.classifier_input_name = classifier_input_name
        self.classifier_imgsz = int(classifier_imgsz)
        self.crop_margin = float(crop_margin)
        self.frame_dirt_threshold = float(frame_dirt_threshold)
        self.max_classified_boxes = max(1, int(max_classified_boxes))
        self.classify_fn = classify_fn
        self.detect_threshold = float(detect_threshold)
        self.duplicate_iou_threshold = float(duplicate_iou_threshold)
        self.max_frame_age_s = max(0.001, float(max_frame_age_s))
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
        self._capture_condition = threading.Condition()
        self._fatal_lock = threading.Lock()
        self.fatal_error: str | None = None
        self._latest_capture: CapturedFrame | None = None
        self._last_processed_sequence = 0
        self._capture_thread = threading.Thread(
            target=self._capture_entry,
            name=f"cap-line-v7-capture-{self.camera_index}",
            daemon=True,
        )
        self._thread = threading.Thread(
            target=self._worker_entry,
            name=f"cap-line-v7-camera-{self.camera_index}",
            daemon=True,
        )

    def start(self) -> None:
        self._capture_thread.start()
        self._thread.start()

    def join(self, timeout: float | None = None) -> bool:
        """Join both worker threads and report whether shutdown completed."""

        if timeout is None:
            self._thread.join()
            self._capture_thread.join()
        else:
            deadline = time.monotonic() + max(0.0, float(timeout))
            self._thread.join(max(0.0, deadline - time.monotonic()))
            self._capture_thread.join(max(0.0, deadline - time.monotonic()))
        return not self._thread.is_alive() and not self._capture_thread.is_alive()

    def _mark_fatal(self, role: str, exc: BaseException) -> None:
        with self._fatal_lock:
            if self.fatal_error is None:
                self.fatal_error = (
                    f"camera {self.camera_index} {role} crashed with {type(exc).__name__}"
                )
        self.stop_event.set()

    def _capture_entry(self) -> None:
        try:
            self._capture_loop()
        except BaseException as exc:
            self._mark_fatal("capture thread", exc)

    def _worker_entry(self) -> None:
        try:
            self._run()
        except BaseException as exc:
            self._mark_fatal("inference thread", exc)

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

    def _capture_loop(self) -> None:
        """Continuously drain V4L2 and retain only the freshest frame.

        Capture and inference used to be serial. On a slow Jetson inference,
        OpenCV could then deliver an old cap sequence long after the cap had
        left. A one-slot handoff deliberately drops intermediate frames.
        """

        while not self.stop_event.is_set():
            captured = self._read()
            if captured is None:
                self.shared.clear_boxes(self.camera_index)
                self.sleep_fn(0.005)
                continue
            self.shared.record_capture(self.camera_index)
            with self._capture_condition:
                self._latest_capture = captured
                self._capture_condition.notify_all()
            # Real V4L2 reads block at the camera frame rate. This tiny yield
            # also keeps injected/non-blocking test cameras from starving the
            # inference consumer.
            self.sleep_fn(0.001)

    def _next_capture(self) -> CapturedFrame | None:
        with self._capture_condition:
            while not self.stop_event.is_set():
                captured = self._latest_capture
                if captured is not None and captured.sequence > self._last_processed_sequence:
                    self._last_processed_sequence = captured.sequence
                    return captured
                self._capture_condition.wait(timeout=0.05)
        return None

    def _finish_due_tracks(self, now: float) -> None:
        for track in self.tracker.collect_finished(now):
            self.manager.handle_finished_track(track)

    def _run(self) -> None:
        while not self.stop_event.is_set():
            captured = self._next_capture()
            if captured is None:
                continue
            inference_start = float(self.time_fn())
            try:
                tensor, meta = self.preprocess_fn(captured.frame, self.model_imgsz)
                output = self.session.run(None, {self.input_name: tensor})[0]
                raw_boxes = self.postprocess_fn(output, meta, conf_threshold=self.detect_threshold)
            except Exception as exc:
                self.log_fn(f"[CAMERA {self.camera_index}][WARN] inference failed: {exc}")
                self.shared.clear_boxes(self.camera_index)
                # Even a failed inference advances the capture timeline. Expire
                # the previous cap before a later detection can revive it.
                self._finish_due_tracks(captured.timestamp)
                self.sleep_fn(0.005)
                continue
            if self.stop_event.is_set():
                break  # never turn an inference that completed during stop into a pulse
            unique_boxes = deduplicate_boxes(raw_boxes, iou_threshold=self.duplicate_iou_threshold)
            # Stage 2: score each cap crop with the dirt classifier. A failed
            # classification yields None (no dirt evidence for that frame) —
            # the conservative direction for actuation.
            dirt_probs: list[float | None] = []
            for box_index, unique_box in enumerate(unique_boxes):
                if box_index >= self.max_classified_boxes:
                    dirt_probs.append(None)
                    continue
                try:
                    dirt_probs.append(
                        self.classify_fn(
                            self.classifier_session,
                            self.classifier_input_name,
                            captured.frame,
                            unique_box,
                            classifier_imgsz=self.classifier_imgsz,
                            crop_margin=self.crop_margin,
                        )
                    )
                except Exception as exc:
                    self.log_fn(f"[CAMERA {self.camera_index}][WARN] classification failed: {exc}")
                    dirt_probs.append(None)
            if self.stop_event.is_set():
                break
            inference_completed = float(self.time_fn())
            frame_age = inference_completed - captured.timestamp
            if frame_age > self.max_frame_age_s:
                self.shared.clear_boxes(self.camera_index)
                self._finish_due_tracks(captured.timestamp)
                self.log_fn(
                    f"[CAMERA {self.camera_index}][STALE] dropped inference result for a "
                    f"{frame_age * 1000.0:.0f} ms-old frame "
                    f"(limit={self.max_frame_age_s * 1000.0:.0f} ms)"
                )
                continue
            boxes = tuple(
                _display_box(box, p_dirt, self.frame_dirt_threshold)
                for box, p_dirt in zip(unique_boxes, dirt_probs)
            )
            inference_ms = (inference_completed - inference_start) * 1000.0
            self.shared.publish(self.camera_index, captured.frame, boxes, inference_ms)
            try:
                frame_size = (int(captured.frame.shape[1]), int(captured.frame.shape[0]))
            except Exception:
                frame_size = None
            # Expire old tracks *before* association. Updating first allowed a
            # late frame (or the next cap) to revive a track whose timeout had
            # already elapsed, combining separate physical presences.
            self._finish_due_tracks(captured.timestamp)
            self.tracker.update(boxes, captured.timestamp, frame_size, dirt_probs)


# --------------------------------------------------------------------------- #
# Entry point.
# --------------------------------------------------------------------------- #

def run_detection(
    config: RuntimeConfig,
    callbacks: RuntimeCallbacks | None = None,
    stop_event: threading.Event | None = None,
    *,
    pin_factory=None,
    camera_factory: Callable[[int, str | int, RuntimeConfig], object] | None = None,
    session_factory: Callable[[str, int], object] | None = None,
    preprocess_fn=None,
    postprocess_fn=None,
    classify_fn=None,
    time_fn: Callable[[], float] = time.monotonic,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> None:
    validate_config(config)
    callbacks = callbacks or RuntimeCallbacks()
    log_fn = callbacks.log_fn
    stop_event = stop_event or threading.Event()
    clock = Clock(time_fn)

    model_path, preset_imgsz = resolve_model_path(config.model)
    classifier_path, classifier_preset_imgsz = resolve_model_path(config.classifier_model)
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
    active_classify = classify_fn or classify_dirt_probability
    active_session_factory = session_factory or create_onnx_session
    sessions = [active_session_factory(model_path, config.onnx_intra_op_threads) for _ in camera_sources]
    input_metas = [session.get_inputs()[0] for session in sessions]
    input_names = [meta.name for meta in input_metas]
    model_imgsz = resolve_imgsz(input_metas[0], config.imgsz, preset_imgsz)
    classifier_sessions = [
        active_session_factory(classifier_path, config.onnx_intra_op_threads) for _ in camera_sources
    ]
    classifier_input_metas = [session.get_inputs()[0] for session in classifier_sessions]
    classifier_input_names = [meta.name for meta in classifier_input_metas]
    classifier_imgsz = resolve_imgsz(
        classifier_input_metas[0], config.classifier_imgsz, classifier_preset_imgsz
    )

    if pin_factory is None or config.simulate_gpio:
        pin_factory = resolve_pin_factory(config)
    scheduler = RejectScheduler(
        trigger_pin=config.trigger_pin,
        trigger_duration=config.trigger_duration,
        trigger_min_gap=config.trigger_min_gap,
        max_queue_age=float(config.trigger_max_queue_age_ms) / 1000.0,
        max_lateness=float(config.trigger_max_lateness_ms) / 1000.0,
        pin_factory=pin_factory,
        log_fn=log_fn,
        time_fn=time_fn,
        sleep_fn=sleep_fn,
        cancel_event=stop_event,
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
        CameraTracker(
            index,
            track_iou=config.track_iou,
            track_timeout_s=track_timeout_s,
            min_defect_frames=config.min_defect_frames,
            frame_dirt_threshold=config.frame_dirt_threshold,
            track_dirt_threshold=config.track_dirt_threshold,
            presence_clear_s=float(config.presence_clear_ms) / 1000.0,
            min_track_frames=config.min_track_frames,
            min_track_travel_ratio=config.min_track_travel_ratio,
            min_track_directionality=config.min_track_directionality,
            presence_line_axis=config.presence_line_axis,
            presence_line_ratio=config.presence_line_ratio,
            presence_direction=config.presence_direction,
            max_track_gap_s=float(config.max_track_gap_ms) / 1000.0,
        )
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
            classifier_session=classifier_sessions[index],
            classifier_input_name=classifier_input_names[index],
            classifier_imgsz=classifier_imgsz,
            crop_margin=config.crop_margin,
            frame_dirt_threshold=config.frame_dirt_threshold,
            max_classified_boxes=config.max_classified_boxes,
            detect_threshold=config.detect_threshold,
            duplicate_iou_threshold=config.duplicate_iou_threshold,
            max_frame_age_s=float(config.max_frame_age_ms) / 1000.0,
            mirror_horizontal=config.mirror_cameras[index],
            tracker=trackers[index],
            manager=manager,
            shared=shared,
            preprocess_fn=active_preprocess,
            postprocess_fn=active_postprocess,
            classify_fn=active_classify,
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

    log_fn(
        f"Using v7 two-stage models: detector={model_path} imgsz={model_imgsz} "
        f"classifier={classifier_path} imgsz={classifier_imgsz} crop_margin={config.crop_margin:.2f} "
        f"target_fps={config.target_fps} gpio={scheduler.backend_name} "
        f"presence_gate={config.presence_line_axis}"
        f"@{config.presence_line_ratio:.3f}/{config.presence_direction} "
        f"track={config.min_track_frames}frames "
        f"dirt={config.min_defect_frames}frames@{config.frame_dirt_threshold:.2f} "
        f"track_dirt_threshold={config.track_dirt_threshold:.2f}"
    )
    try:
        for worker in workers:
            worker.start()
        while not stop_event.is_set():
            if scheduler.fatal_error is not None:
                raise RuntimeError(f"[REJECT][FATAL] {scheduler.fatal_error}")
            camera_fatal = next(
                (worker.fatal_error for worker in workers if worker.fatal_error is not None),
                None,
            )
            if camera_fatal is not None:
                raise RuntimeError(f"[CAMERA][FATAL] {camera_fatal}")
            now = clock.monotonic()
            manager.flush_expired(now)

            if callbacks.test_fire_poll is not None and callbacks.test_fire_poll():
                # Manual operator pulse through the runtime's own pin (event 0).
                if stop_event.is_set():
                    break
                log_fn(f"[TEST FIRE] manual pulse via {scheduler.backend_name}")
                try:
                    scheduler.enqueue(0, now)
                except RuntimeError:
                    break  # stop/close won the race; never resurrect a pulse

            if preview_enabled and not preview_broken and (now - last_preview) >= preview_interval_s:
                last_preview = now
                try:
                    frames, boxes = shared.latest_frames()
                    annotated = []
                    for frame, frame_boxes in zip(frames, boxes):
                        if frame is None:
                            continue
                        drawable = frame.copy() if hasattr(frame, "copy") else frame
                        annotated.append(
                            draw_boxes(
                                drawable,
                                frame_boxes,
                                presence_line_axis=config.presence_line_axis,
                                presence_line_ratio=config.presence_line_ratio,
                                presence_direction=config.presence_direction,
                            )
                        )
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
        # Cancel the actuator first. Worker/capture teardown can block on
        # inference or V4L2; no queued pulse may fire during that wait.
        scheduler_closed_cleanly = scheduler.close()
        for camera in cameras:
            if hasattr(camera, "release"):
                camera.release()
        stuck_camera_workers = [
            worker.camera_index for worker in workers if not worker.join(timeout=2.0)
        ]
        camera_fatals = [
            worker.fatal_error for worker in workers if worker.fatal_error is not None
        ]
        # Stopping is a safety boundary: never convert a partial/unconfirmed
        # track into a new fire request. Pending scheduler jobs are cancelled
        # below rather than drained through the valve.
        for tracker in trackers:
            tracker.flush()
        manager.finalize_all()
        if not scheduler_closed_cleanly:
            raise RuntimeError(
                f"[REJECT][FATAL] {scheduler.fatal_error or 'reject scheduler shutdown failed'}"
            )
        if camera_fatals:
            raise RuntimeError(f"[CAMERA][FATAL] {'; '.join(camera_fatals)}")
        if stuck_camera_workers:
            raise RuntimeError(
                f"[CAMERA][FATAL] worker shutdown timed out for camera(s) {stuck_camera_workers}"
            )


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
