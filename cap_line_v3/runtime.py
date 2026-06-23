from __future__ import annotations

import csv
import concurrent.futures
import json
import os
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

from gpio_output import GPIOOutputPin

from .actuation import NullGPIOOutputPin, RejectScheduler
from .config import DEFAULT_MODEL, RuntimeConfig, validate_config
from .decision import TrackedCapManager, decide_decision_ready
from .geometry import box_spans_line_coordinate, class_name, frame_line_coordinate
from .pairing import select_synchronized_frame_pair
from .preview import CameraPreviewView, resolve_preview_views
from .types import (
    Box,
    CapturedFrame,
    DetectionHistoryRecord,
    DetectionPacket,
    RuntimeCallbacks,
    RuntimePerformanceSnapshot,
    TimingLogRecord,
    TrackObservation,
)


MODEL_SEARCH_DIRS = (Path(__file__).resolve().parent.parent, Path(__file__).resolve().parent.parent / "model")
CAP_PROP_FRAME_WIDTH = 3
CAP_PROP_FRAME_HEIGHT = 4
CAP_PROP_FPS = 5
MIN_CAP_FINALIZE_QUIET_S = 0.50


def format_timestamp(clock_origin_wall: datetime, origin_monotonic: float, timestamp: float) -> str:
    from datetime import timedelta

    return (clock_origin_wall + timedelta(seconds=float(timestamp) - float(origin_monotonic))).isoformat(
        timespec="milliseconds"
    )


class Clock:
    def __init__(self, time_fn: Callable[[], float] = time.monotonic):
        self.time_fn = time_fn
        self.origin_monotonic = float(time_fn())
        self.origin_wall = datetime.now().astimezone()

    def monotonic(self) -> float:
        return float(self.time_fn())

    def format(self, timestamp: float | None = None) -> str:
        return format_timestamp(self.origin_wall, self.origin_monotonic, self.monotonic() if timestamp is None else timestamp)


def resolve_model_path(model: str) -> tuple[str, int | None]:
    requested = str(model or DEFAULT_MODEL)
    candidates = []
    path = Path(os.path.expanduser(requested))
    if path.is_absolute() or path.parent != Path("."):
        candidates.extend([path, Path(__file__).resolve().parent.parent / path])
    else:
        candidates.extend([directory / path for directory in MODEL_SEARCH_DIRS])
    for candidate in candidates:
        if candidate.exists():
            return str(candidate), infer_model_imgsz_from_name(str(candidate))
    return str(candidates[0]), infer_model_imgsz_from_name(str(candidates[0]))


def infer_model_imgsz_from_name(path: str) -> int | None:
    name = Path(path).name
    for token in name.replace("-", "_").split("_"):
        if token.isdigit():
            value = int(token)
            if 128 <= value <= 4096:
                return value
    return None


def parse_cameras(cameras: tuple[str, str]) -> tuple[list[str | int], list[str]]:
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


def set_camera_format(
    device_path: str,
    width: int,
    height: int,
    fps: int,
    *,
    pixel_format: str,
    log_fn: Callable[..., None] = print,
) -> None:
    if os.name == "nt" or not str(device_path).startswith("/dev/"):
        return
    command = [
        "v4l2-ctl",
        "-d",
        str(device_path),
        f"--set-fmt-video=width={int(width)},height={int(height)},pixelformat={pixel_format}",
        f"--set-parm={int(fps)}",
    ]
    try:
        subprocess.run(command, check=False, capture_output=True, text=True)
    except OSError as exc:
        log_fn(f"[CAMERA][WARN] unable to set format for {device_path}: {exc}")


def set_camera_controls(device_path: str, exposure: int, *, log_fn: Callable[..., None] = print) -> None:
    if os.name == "nt" or not str(device_path).startswith("/dev/"):
        return
    command = ["v4l2-ctl", "-d", str(device_path), f"--set-ctrl=exposure_time_absolute={int(exposure)}"]
    try:
        subprocess.run(command, check=False, capture_output=True, text=True)
    except OSError as exc:
        log_fn(f"[CAMERA][WARN] unable to set exposure for {device_path}: {exc}")


def open_cam(source: str | int, width: int, height: int, fps: int, pixel_format: str):
    import cv2

    camera = cv2.VideoCapture(source, cv2.CAP_V4L2 if os.name != "nt" else 0)
    camera.set(CAP_PROP_FRAME_WIDTH, width)
    camera.set(CAP_PROP_FRAME_HEIGHT, height)
    camera.set(CAP_PROP_FPS, fps)
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


def validate_opened_cameras(
    cameras: list[object],
    camera_sources: list[str | int],
    device_paths: list[str],
) -> None:
    failed = []
    for index, camera in enumerate(cameras):
        if _camera_is_open(camera):
            continue
        source = camera_sources[index] if index < len(camera_sources) else "unknown"
        device_path = device_paths[index] if index < len(device_paths) else "unknown"
        failed.append(f"camera {index} source={source!r} device={device_path!r}")
    if failed:
        raise RuntimeError(
            "Unable to open V3 camera(s): "
            + "; ".join(failed)
            + ". Check --cams/UI camera settings, cable/power, permissions, and v4l2-ctl --list-devices."
        )


def create_onnx_session(model_path: str, intra_op_threads: int):
    import onnxruntime as ort

    options = ort.SessionOptions() if hasattr(ort, "SessionOptions") else None
    if options is not None:
        options.intra_op_num_threads = max(1, int(intra_op_threads))
        options.inter_op_num_threads = 1
    providers = [
        provider
        for provider in ("TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider")
        if provider in list(getattr(ort, "get_available_providers", lambda: [])())
    ] or ["CPUExecutionProvider"]
    if options is None:
        return ort.InferenceSession(model_path, providers=providers)
    return ort.InferenceSession(model_path, sess_options=options, providers=providers)


def resolve_imgsz(input_meta, override: int | None, preset: int | None) -> int:
    if override:
        return int(override)
    shape = list(getattr(input_meta, "shape", []) or [])
    for value in reversed(shape):
        if isinstance(value, int) and value > 0:
            return int(value)
    return int(preset or 640)


def letterbox_resize(image_bgr, new_shape: tuple[int, int] = (640, 640), color=(114, 114, 114)):
    import cv2

    original_height, original_width = image_bgr.shape[:2]
    scale = min(new_shape[0] / original_height, new_shape[1] / original_width)
    resized_width = int(round(original_width * scale))
    resized_height = int(round(original_height * scale))
    resized = cv2.resize(image_bgr, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    pad_width = new_shape[1] - resized_width
    pad_height = new_shape[0] - resized_height
    pad_left = int(round(pad_width / 2 - 0.1))
    pad_right = int(round(pad_width / 2 + 0.1))
    pad_top = int(round(pad_height / 2 - 0.1))
    pad_bottom = int(round(pad_height / 2 + 0.1))
    padded = cv2.copyMakeBorder(
        resized,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=color,
    )
    return padded, float(scale), (pad_left, pad_top)


def preprocess(frame, model_imgsz: int):
    import cv2
    import numpy as np

    image, resize_scale, padding = letterbox_resize(
        frame,
        new_shape=(int(model_imgsz), int(model_imgsz)),
    )
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = image.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    return tensor, {
        "scale": float(resize_scale),
        "pad_left": int(padding[0]),
        "pad_top": int(padding[1]),
        "frame_shape": frame.shape,
        "img_size": int(model_imgsz),
    }


def postprocess(output, preprocess_meta, conf_threshold: float):
    import numpy as np

    detections = np.asarray(output, dtype=np.float32)
    if detections.ndim == 3 and detections.shape[0] == 1:
        detections = detections[0]
    if detections.ndim != 2:
        return []
    if detections.shape[1] != 6 and detections.shape[0] == 6:
        detections = detections.T
    if detections.shape[1] != 6:
        return []

    scale = float(preprocess_meta["scale"])
    pad_left = float(preprocess_meta["pad_left"])
    pad_top = float(preprocess_meta["pad_top"])
    frame_h, frame_w = preprocess_meta["frame_shape"][:2]
    img_size = int(preprocess_meta["img_size"])

    boxes = []
    for detection in detections:
        x1, y1, x2, y2, score, class_id_value = detection[:6]
        score = float(score)
        if score < float(conf_threshold):
            continue
        coords = np.asarray([x1, y1, x2, y2], dtype=np.float32)
        if float(np.max(np.abs(coords))) <= 1.5:
            coords[[0, 2]] *= img_size
            coords[[1, 3]] *= img_size
        x1, y1, x2, y2 = coords.tolist()
        x1 = (x1 - pad_left) / scale
        y1 = (y1 - pad_top) / scale
        x2 = (x2 - pad_left) / scale
        y2 = (y2 - pad_top) / scale
        x1 = max(0.0, min(float(frame_w) - 1.0, x1))
        y1 = max(0.0, min(float(frame_h) - 1.0, y1))
        x2 = max(0.0, min(float(frame_w) - 1.0, x2))
        y2 = max(0.0, min(float(frame_h) - 1.0, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        class_id = int(round(float(class_id_value)))
        if class_id not in (0, 1):
            continue
        boxes.append([x1, y1, x2, y2, score, class_id])
    boxes.sort(key=lambda box: float(box[4]), reverse=True)
    return boxes


def _to_box(box: list[float] | tuple[float, ...]) -> Box:
    return (float(box[0]), float(box[1]), float(box[2]), float(box[3]), float(box[4]), int(box[5]))


def draw_boxes(frame, boxes: tuple[Box, ...]):
    import cv2

    for box in boxes:
        x1, y1, x2, y2, conf, cls = box
        color = (0, 0, 255) if int(cls) == 1 else (0, 200, 0)
        label = class_name(int(cls)) or str(int(cls))
        try:
            cv2.rectangle(frame, (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2))), color, 2)
            cv2.putText(frame, f"{label}:{conf:.2f}", (int(round(x1)), max(0, int(round(y1)) - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        except Exception:
            return frame
    return frame


def draw_anchor_line(frame, axis: str, ratio: float):
    import cv2

    height, width = frame.shape[:2]
    try:
        if axis == "x":
            x = max(0, min(width - 1, int(round(width * ratio))))
            cv2.line(frame, (x, 0), (x, height - 1), (255, 215, 0), 2)
        else:
            y = max(0, min(height - 1, int(round(height * ratio))))
            cv2.line(frame, (0, y), (width - 1, y), (255, 215, 0), 2)
    except Exception:
        return frame
    return frame


def compose_preview(frames: list[object]):
    import cv2
    import numpy as np

    if not frames:
        return None
    if len(frames) == 1:
        return frames[0]
    height = min(frame.shape[0] for frame in frames)
    resized = [cv2.resize(frame, (int(frame.shape[1] * height / frame.shape[0]), height)) for frame in frames]
    spacer = np.zeros((height, 6, 3), dtype=np.uint8)
    parts = []
    for index, frame in enumerate(resized):
        if index:
            parts.append(spacer)
        parts.append(frame)
    return np.hstack(parts)


class DirectCameraReader:
    def __init__(self, camera, camera_index: int, time_fn: Callable[[], float]):
        self.camera = camera
        self.camera_index = int(camera_index)
        self.time_fn = time_fn
        self.sequence = 0
        self.captured = 0

    def start(self) -> None:
        return None

    def latest(self) -> CapturedFrame | None:
        started = float(self.time_fn())
        ok, frame = self.camera.read()
        captured_at = float(self.time_fn())
        if not ok or frame is None:
            return None
        self.sequence += 1
        self.captured += 1
        return CapturedFrame(self.camera_index, frame, captured_at, self.sequence, (captured_at - started) * 1000.0)

    def stop(self) -> None:
        return None


class LatestFrameCameraReader:
    """Continuously drain one camera and expose only its freshest frame."""

    def __init__(
        self,
        camera,
        camera_index: int,
        *,
        target_fps: int | float | None,
        time_fn: Callable[[], float] = time.monotonic,
        sleep_fn: Callable[[float], None] = time.sleep,
    ):
        self.camera = camera
        self.camera_index = int(camera_index)
        self.target_fps = None if target_fps is None else float(target_fps)
        self.time_fn = time_fn
        self.sleep_fn = sleep_fn
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._latest: CapturedFrame | None = None
        self._captured = 0
        self._started = False
        self._thread = threading.Thread(
            target=self._run,
            name=f"cap-line-v3-camera-{self.camera_index}",
            daemon=True,
        )

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._thread.start()

    def latest(self) -> CapturedFrame | None:
        with self._lock:
            return self._latest

    @property
    def captured(self) -> int:
        with self._lock:
            return int(self._captured)

    @property
    def sequence(self) -> int:
        return self.captured

    def stop(self) -> None:
        self._stop_event.set()
        if self._started:
            self._thread.join()

    def _run(self) -> None:
        min_interval_s = 0.0 if not self.target_fps or self.target_fps <= 0.0 else 1.0 / self.target_fps
        while not self._stop_event.is_set():
            started_at = float(self.time_fn())
            try:
                ok, frame = self.camera.read()
            except Exception:
                ok, frame = False, None
            captured_at = float(self.time_fn())
            if ok and frame is not None:
                with self._lock:
                    self._captured += 1
                    self._latest = CapturedFrame(
                        self.camera_index,
                        frame,
                        captured_at,
                        self._captured,
                        (captured_at - started_at) * 1000.0,
                    )
            elif not self._stop_event.is_set():
                self.sleep_fn(0.01)

            if min_interval_s > 0.0 and not self._stop_event.is_set():
                remaining_s = min_interval_s - (float(self.time_fn()) - started_at)
                if remaining_s > 0.0:
                    self.sleep_fn(remaining_s)


class LivePreviewPublisher:
    """Publish smooth live previews while inference updates boxes more slowly."""

    def __init__(
        self,
        camera_readers,
        preview_callback: Callable[[object], None],
        *,
        anchor_axis: str,
        anchor_line_ratio: float,
        preview_fps: int | float,
        overlay_target_fps: int | float,
        stop_event=None,
        compose_preview_fn: Callable[[list[object]], object] | None = None,
        draw_boxes_fn: Callable[[object, tuple[Box, ...]], object] | None = None,
        draw_anchor_line_fn: Callable[[object, str, float], object] | None = None,
        preview_latency_compensation_ms: int | float = 0.0,
        actuation_snapshot_hold_ms: int | float = 0.0,
        time_fn: Callable[[], float] = time.monotonic,
        sleep_fn: Callable[[float], None] = time.sleep,
    ):
        self.camera_readers = list(camera_readers)
        self.preview_callback = preview_callback
        self.anchor_axis = anchor_axis
        self.anchor_line_ratio = float(anchor_line_ratio)
        self.preview_fps = float(preview_fps)
        self.overlay_target_fps = float(overlay_target_fps)
        self.external_stop_event = stop_event
        self.compose_preview_fn = compose_preview_fn or compose_preview
        self.draw_boxes_fn = draw_boxes_fn or draw_boxes
        self.draw_anchor_line_fn = draw_anchor_line_fn or draw_anchor_line
        self.preview_latency_compensation_ms = float(preview_latency_compensation_ms)
        self.actuation_snapshot_hold_s = max(0.0, float(actuation_snapshot_hold_ms) / 1000.0)
        self.time_fn = time_fn
        self.sleep_fn = sleep_fn
        self._stop_event = threading.Event()
        self._overlay_lock = threading.Lock()
        self._previous_packet: DetectionPacket | None = None
        self._current_packet: DetectionPacket | None = None
        self._snapshot_lock = threading.Lock()
        self._actuation_snapshots: dict[int, tuple[float, CameraPreviewView]] = {}
        self._stats_lock = threading.Lock()
        self._published_count = 0
        self._latest_overlay_age_ms: float | None = None
        self._started = False
        self._thread = threading.Thread(
            target=self._run,
            name="cap-line-v3-live-preview",
            daemon=True,
        )

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._started:
            self._thread.join()

    @property
    def published_count(self) -> int:
        with self._stats_lock:
            return int(self._published_count)

    @property
    def latest_overlay_age_ms(self) -> float | None:
        with self._stats_lock:
            return self._latest_overlay_age_ms

    def update_packet(self, packet: DetectionPacket) -> None:
        with self._overlay_lock:
            self._previous_packet = self._current_packet
            self._current_packet = packet

    def update_overlay(self, frame_pair, boxes_by_camera) -> None:
        packet = DetectionPacket(
            frame_pair,
            tuple(tuple(_to_box(box) for box in camera_boxes) for camera_boxes in boxes_by_camera),
            tuple(),
        )
        self.update_packet(packet)

    def _should_stop(self) -> bool:
        return self._stop_event.is_set() or (
            self.external_stop_event is not None and self.external_stop_event.is_set()
        )

    def _run(self) -> None:
        min_interval_s = 0.0 if self.preview_fps <= 0.0 else 1.0 / self.preview_fps
        while not self._should_stop():
            loop_started_at = float(self.time_fn())
            latest_frames = tuple(reader.latest() for reader in self.camera_readers)
            if latest_frames and all(frame is not None for frame in latest_frames):
                live_frames = tuple(frame for frame in latest_frames if frame is not None)
                with self._overlay_lock:
                    previous_packet = self._previous_packet
                    current_packet = self._current_packet
                preview_views = resolve_preview_views(
                    previous_packet,
                    current_packet,
                    live_frames,
                    target_fps=self.overlay_target_fps,
                    anchor_axis=self.anchor_axis,
                    anchor_line_ratio=self.anchor_line_ratio,
                    preview_latency_compensation_ms=self.preview_latency_compensation_ms,
                )
                preview_views = self._hold_actuation_snapshots(preview_views)
                annotated = []
                for view in preview_views:
                    captured = view.captured
                    frame = captured.frame.copy() if hasattr(captured.frame, "copy") else captured.frame
                    frame = self.draw_boxes_fn(frame, view.boxes)
                    frame = self.draw_anchor_line_fn(frame, self.anchor_axis, self.anchor_line_ratio)
                    annotated.append(frame)
                preview = self.compose_preview_fn(annotated)
                if preview is not None:
                    self.preview_callback(preview)
                    overlay_age_ms = None
                    if current_packet is not None:
                        overlay_age_ms = (
                            max(float(frame.timestamp) for frame in live_frames)
                            - float(current_packet.frame_pair.pair_timestamp)
                        ) * 1000.0
                    with self._stats_lock:
                        self._published_count += 1
                        self._latest_overlay_age_ms = overlay_age_ms

            if min_interval_s > 0.0 and not self._should_stop():
                remaining_s = min_interval_s - (float(self.time_fn()) - loop_started_at)
                if remaining_s > 0.0:
                    self.sleep_fn(remaining_s)

    def _hold_actuation_snapshots(
        self,
        preview_views: tuple[CameraPreviewView, ...],
    ) -> tuple[CameraPreviewView, ...]:
        if self.actuation_snapshot_hold_s <= 0.0:
            return preview_views
        now = float(self.time_fn())
        held_views = list(preview_views)
        with self._snapshot_lock:
            expired = [
                camera_index
                for camera_index, (expires_at, _view) in self._actuation_snapshots.items()
                if expires_at <= now
            ]
            for camera_index in expired:
                self._actuation_snapshots.pop(camera_index, None)

            for camera_index, view in enumerate(preview_views):
                if view.boxes:
                    self._actuation_snapshots[camera_index] = (
                        now + self.actuation_snapshot_hold_s,
                        view,
                    )
                    continue
                snapshot = self._actuation_snapshots.get(camera_index)
                if snapshot is None:
                    continue
                _expires_at, snapshot_view = snapshot
                held_views[camera_index] = snapshot_view
        return tuple(held_views)


def _frame_size(frame) -> tuple[int, int]:
    shape = getattr(frame, "shape", (0, 0, 0))
    return int(shape[1]), int(shape[0])


def _record_history(event_id: int, decision, clock: Clock, config: RuntimeConfig) -> DetectionHistoryRecord:
    return DetectionHistoryRecord(
        recorded_at=clock.format(decision.decision_ready_time),
        runtime_event_id=int(event_id),
        result=decision.result,
        final_class_name=decision.final_class_name,
        final_score=decision.final_score,
        decision_source=decision.decision_source,
        camera_labels=list(config.cameras),
        camera_votes={
            index: {
                "class_id": vote.class_id,
                "score": vote.score,
                "observation_count": vote.observation_count,
            }
            for index, vote in decision.camera_votes.items()
        },
        anchor_time=clock.format(decision.anchor_time),
        trigger_delay_s=decision.trigger_delay_s,
    )


def _timing_record(event_id: int, decision, clock: Clock) -> TimingLogRecord:
    return TimingLogRecord(
        recorded_at=clock.format(decision.decision_ready_time),
        runtime_event_id=int(event_id),
        result=decision.result,
        final_class_name=decision.final_class_name,
        anchor_time=clock.format(decision.anchor_time),
        decision_time=clock.format(decision.decision_ready_time),
        requested_fire_time=clock.format(decision.requested_fire_time),
    )


def _write_timing_record(config: RuntimeConfig, record: TimingLogRecord) -> None:
    os.makedirs(config.timing_log_dir, exist_ok=True)
    path = Path(config.timing_log_dir) / f"timing_{record.recorded_at[:10]}.csv"
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(record.__dataclass_fields__.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow({key: getattr(record, key) for key in record.__dataclass_fields__})


def _write_debug_artifact(
    config: RuntimeConfig,
    *,
    event_id: int,
    decision,
    packet: DetectionPacket,
    clock: Clock,
) -> None:
    os.makedirs(config.debug_dir, exist_ok=True)
    os.makedirs(config.pictures_dir, exist_ok=True)
    recorded_label = clock.format(decision.decision_ready_time).replace(":", "").replace("-", "")
    prefix = f"event_{int(event_id)}_{decision.result}_{recorded_label}"
    payload = {
        "event_id": int(event_id),
        "recorded_at": clock.format(decision.decision_ready_time),
        "result": decision.result,
        "final_class_name": decision.final_class_name,
        "final_score": decision.final_score,
        "decision_source": decision.decision_source,
        "anchor_time": clock.format(decision.anchor_time),
        "requested_fire_time": clock.format(decision.requested_fire_time),
        "pair_sequences": list(packet.frame_pair.sequences),
        "pair_timestamps": list(packet.frame_pair.timestamps),
        "pair_skew_ms": packet.frame_pair.skew_ms,
        "boxes_by_camera": [
            [[float(value) for value in box] for box in camera_boxes]
            for camera_boxes in packet.boxes_by_camera
        ],
    }
    json_path = Path(config.debug_dir) / f"{prefix}.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    try:
        import cv2
    except Exception:
        return
    for captured in packet.frame_pair.frames:
        raw_path = Path(config.pictures_dir) / f"{prefix}_cam{captured.camera_index}.jpg"
        try:
            cv2.imwrite(str(raw_path), captured.frame)
        except Exception:
            continue


def _run_camera_inference(
    *,
    camera_index: int,
    captured: CapturedFrame,
    session,
    input_name: str,
    model_imgsz: int,
    preprocess_fn: Callable[[object, int], tuple[object, dict]],
    postprocess_fn: Callable[..., list[list[float]]],
    tracking_threshold: float,
    anchor_axis: str,
    anchor_line_ratio: float,
    clock: Clock,
) -> tuple[int, tuple[Box, ...], float, list[TrackObservation]]:
    inference_start = clock.monotonic()
    input_tensor, meta = preprocess_fn(captured.frame, model_imgsz)
    output = session.run(None, {input_name: input_tensor})[0]
    boxes = tuple(_to_box(box) for box in postprocess_fn(output, meta, conf_threshold=tracking_threshold))
    inference_ms = (clock.monotonic() - inference_start) * 1000.0
    frame_size = _frame_size(captured.frame)
    line_coordinate = frame_line_coordinate(
        frame_size,
        axis=anchor_axis,
        ratio=anchor_line_ratio,
    )
    observations = [
        TrackObservation(
            camera_index=camera_index,
            box=box,
            timestamp=captured.timestamp,
            frame_size=frame_size,
            at_actuation_line=box_spans_line_coordinate(
                box,
                axis=anchor_axis,
                line_coordinate=line_coordinate,
            ),
        )
        for box in boxes
    ]
    return camera_index, boxes, inference_ms, observations


def run_detection(
    config: RuntimeConfig,
    callbacks: RuntimeCallbacks | None = None,
    *,
    stop_event=None,
    pin_factory=GPIOOutputPin,
    camera_factory: Callable[[int, str | int, RuntimeConfig], object] | None = None,
    session_factory: Callable[[str, int], object] | None = None,
    preprocess_fn: Callable[[object, int], tuple[object, dict]] | None = None,
    postprocess_fn: Callable[..., list[list[float]]] | None = None,
    compose_preview_fn: Callable[[list[object]], object] | None = None,
    draw_boxes_fn: Callable[[object, tuple[Box, ...]], object] | None = None,
    draw_anchor_line_fn: Callable[[object, str, float], object] | None = None,
    time_fn: Callable[[], float] = time.monotonic,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> None:
    validate_config(config)
    callbacks = callbacks or RuntimeCallbacks()
    log_fn = callbacks.log_fn
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
    readers = [
        LatestFrameCameraReader(
            camera,
            index,
            target_fps=config.target_fps,
            time_fn=time_fn,
            sleep_fn=sleep_fn,
        )
        for index, camera in enumerate(cameras)
    ]
    for reader in readers:
        reader.start()

    active_preprocess = preprocess_fn or preprocess
    active_postprocess = postprocess_fn or postprocess
    active_compose_preview = compose_preview_fn or compose_preview
    live_preview: LivePreviewPublisher | None = None
    if callbacks.preview_callback is not None and float(config.live_preview_fps) > 0.0:
        live_preview = LivePreviewPublisher(
            readers,
            callbacks.preview_callback,
            anchor_axis=config.anchor_axis,
            anchor_line_ratio=config.anchor_line_ratio,
            preview_fps=config.live_preview_fps,
            overlay_target_fps=config.target_fps,
            stop_event=stop_event,
            compose_preview_fn=active_compose_preview,
            draw_boxes_fn=draw_boxes_fn,
            draw_anchor_line_fn=draw_anchor_line_fn,
            preview_latency_compensation_ms=config.preview_latency_compensation_ms,
            actuation_snapshot_hold_ms=config.actuation_snapshot_hold_ms,
            time_fn=time_fn,
            sleep_fn=sleep_fn,
        )

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
    start_time = clock.monotonic()
    frame_count = 0
    dropped_pairs = 0
    last_sequences: tuple[int, ...] | None = None
    current_packet: DetectionPacket | None = None
    cap_manager = TrackedCapManager(
        camera_count=len(camera_sources),
        merge_window_seconds=max(
            float(config.merge_window_ms) / 1000.0,
            float(config.pair_max_skew_ms) / 1000.0,
        ),
        finalize_quiet_seconds=max(
            float(config.finalize_quiet_ms) / 1000.0,
            MIN_CAP_FINALIZE_QUIET_S,
        ),
        anchor_axis=config.anchor_axis,
        anchor_line_ratio=config.anchor_line_ratio,
    )
    queued_trigger_event_ids: set[int] = set()
    inference_executor: concurrent.futures.ThreadPoolExecutor | None = None
    if not config.serial_inference:
        inference_executor = concurrent.futures.ThreadPoolExecutor(max_workers=len(camera_sources))

    try:
        if live_preview is not None:
            live_preview.start()
        log_fn(f"Using V3 model: {model_path} target_fps={config.target_fps}")
        if live_preview is not None:
            log_fn(
                "Live preview: "
                f"{config.live_preview_fps:g} fps; camera display is decoupled from inference"
            )
        while stop_event is None or not stop_event.is_set():
            latest_frames = tuple(reader.latest() for reader in readers)
            frame_pair = select_synchronized_frame_pair(
                latest_frames,
                last_sequences,
                max_skew_ms=config.pair_max_skew_ms,
            )
            if frame_pair is None:
                dropped_pairs += 1
                sleep_fn(0.001)
                if stop_event is not None and stop_event.is_set():
                    break
                if time_fn() - start_time > 1.0 and camera_factory is not None:
                    break
                continue
            last_sequences = frame_pair.sequences
            inference_jobs = [
                dict(
                    camera_index=camera_index,
                    captured=captured,
                    session=sessions[camera_index],
                    input_name=input_names[camera_index],
                    model_imgsz=model_imgsz,
                    preprocess_fn=active_preprocess,
                    postprocess_fn=active_postprocess,
                    tracking_threshold=config.tracking_threshold,
                    anchor_axis=config.anchor_axis,
                    anchor_line_ratio=config.anchor_line_ratio,
                    clock=clock,
                )
                for camera_index, captured in enumerate(frame_pair.frames)
            ]
            if inference_executor is None:
                inference_results = [_run_camera_inference(**job) for job in inference_jobs]
            else:
                futures = [inference_executor.submit(_run_camera_inference, **job) for job in inference_jobs]
                inference_results = [future.result() for future in futures]

            boxes_by_camera: list[tuple[Box, ...]] = [tuple() for _ in frame_pair.frames]
            inference_ms: list[float] = [0.0 for _ in frame_pair.frames]
            observations = []
            for camera_index, boxes, camera_inference_ms, camera_observations in sorted(
                inference_results,
                key=lambda result: result[0],
            ):
                boxes_by_camera[camera_index] = boxes
                inference_ms[camera_index] = camera_inference_ms
                observations.extend(camera_observations)
            packet = DetectionPacket(frame_pair, tuple(boxes_by_camera), tuple(inference_ms))
            current_packet = packet
            if live_preview is not None:
                live_preview.update_packet(packet)
            if observations:
                cap_manager.update(observations)

            decision_ready_time = clock.monotonic()
            for tracked_cap in cap_manager.open_caps():
                if tracked_cap.event_id in queued_trigger_event_ids:
                    continue
                decision = decide_decision_ready(
                    tracked_cap,
                    config=config,
                    decision_ready_time=decision_ready_time,
                    camera_count=len(camera_sources),
                )
                if decision is not None:
                    queued_trigger_event_ids.add(tracked_cap.event_id)
                    scheduler.enqueue(tracked_cap.event_id, decision.requested_fire_time)
                    history = _record_history(tracked_cap.event_id, decision, clock, config)
                    timing = _timing_record(tracked_cap.event_id, decision, clock)
                    _write_timing_record(config, timing)
                    _write_debug_artifact(
                        config,
                        event_id=tracked_cap.event_id,
                        decision=decision,
                        packet=packet,
                        clock=clock,
                    )
                    if callbacks.history_callback:
                        callbacks.history_callback(history)
                    if callbacks.timing_log_callback:
                        callbacks.timing_log_callback(timing)
            cap_manager.pop_finalized(decision_ready_time)

            frame_count += 1
            elapsed = max(0.000001, clock.monotonic() - start_time)
            preview_count = live_preview.published_count if live_preview is not None else 0
            snapshot = RuntimePerformanceSnapshot(
                frame_count=frame_count,
                target_fps=int(config.target_fps),
                elapsed_s=elapsed,
                capture_fps_by_camera=tuple(reader.captured / elapsed for reader in readers),
                processed_fps=frame_count / elapsed,
                preview_fps=preview_count / elapsed,
                latest_pair_skew_ms=frame_pair.skew_ms,
                dropped_pairs=dropped_pairs,
                overlay_age_ms=(
                    live_preview.latest_overlay_age_ms
                    if live_preview is not None
                    else ((clock.monotonic() - current_packet.frame_pair.pair_timestamp) * 1000.0 if current_packet else None)
                ),
            )
            if callbacks.performance_callback is not None:
                callbacks.performance_callback(snapshot)

            target_interval_s = 1.0 / max(1.0, float(config.target_fps))
            sleep_fn(max(0.0, target_interval_s - (clock.monotonic() - frame_pair.pair_timestamp)))
            if camera_factory is not None and frame_count >= 2:
                break
    finally:
        if inference_executor is not None:
            inference_executor.shutdown(wait=True)
        if live_preview is not None:
            live_preview.stop()
        for reader in readers:
            reader.stop()
        for camera in cameras:
            if hasattr(camera, "release"):
                camera.release()
        scheduler.close()
