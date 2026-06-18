from __future__ import annotations

import csv
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
from .decision import TrackedCap, decide_decision_ready
from .geometry import box_spans_line_coordinate, frame_line_coordinate
from .pairing import select_synchronized_frame_pair
from .preview import predict_preview_overlay
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
        try:
            cv2.rectangle(frame, (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2))), color, 2)
            cv2.putText(frame, f"{int(cls)}:{conf:.2f}", (int(round(x1)), max(0, int(round(y1)) - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
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
    readers = [DirectCameraReader(camera, index, time_fn) for index, camera in enumerate(cameras)]
    for reader in readers:
        reader.start()

    active_session_factory = session_factory or create_onnx_session
    sessions = [active_session_factory(model_path, config.onnx_intra_op_threads) for _ in camera_sources]
    input_metas = [session.get_inputs()[0] for session in sessions]
    input_names = [meta.name for meta in input_metas]
    model_imgsz = resolve_imgsz(input_metas[0], config.imgsz, preset_imgsz)
    active_preprocess = preprocess_fn or preprocess
    active_postprocess = postprocess_fn or postprocess
    active_compose_preview = compose_preview_fn or compose_preview
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
    preview_count = 0
    dropped_pairs = 0
    last_sequences: tuple[int, ...] | None = None
    previous_packet: DetectionPacket | None = None
    current_packet: DetectionPacket | None = None
    tracked_cap: TrackedCap | None = None

    try:
        log_fn(f"Using V3 model: {model_path} target_fps={config.target_fps}")
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
                if time_fn() - start_time > 0.05 and camera_factory is not None:
                    break
                continue
            last_sequences = frame_pair.sequences
            boxes_by_camera: list[tuple[Box, ...]] = []
            inference_ms: list[float] = []
            observations = []
            for camera_index, captured in enumerate(frame_pair.frames):
                inference_start = clock.monotonic()
                input_tensor, meta = active_preprocess(captured.frame, model_imgsz)
                output = sessions[camera_index].run(None, {input_names[camera_index]: input_tensor})[0]
                boxes = tuple(_to_box(box) for box in active_postprocess(output, meta, conf_threshold=config.tracking_threshold))
                boxes_by_camera.append(boxes)
                inference_ms.append((clock.monotonic() - inference_start) * 1000.0)
                frame_size = _frame_size(captured.frame)
                line_coordinate = frame_line_coordinate(
                    frame_size,
                    axis=config.anchor_axis,
                    ratio=config.anchor_line_ratio,
                )
                for box in boxes:
                    observations.append(
                        TrackObservation(
                            camera_index=camera_index,
                            box=box,
                            timestamp=captured.timestamp,
                            frame_size=frame_size,
                            at_actuation_line=box_spans_line_coordinate(
                                box,
                                axis=config.anchor_axis,
                                line_coordinate=line_coordinate,
                            ),
                        )
                    )
            packet = DetectionPacket(frame_pair, tuple(boxes_by_camera), tuple(inference_ms))
            previous_packet, current_packet = current_packet, packet
            if observations:
                if tracked_cap is None:
                    tracked_cap = TrackedCap(event_id=1, created_at=frame_pair.pair_timestamp, last_seen_at=frame_pair.pair_timestamp)
                for observation in observations:
                    tracked_cap.add_observation(observation)
                decision = decide_decision_ready(tracked_cap, config=config, decision_ready_time=clock.monotonic(), camera_count=len(camera_sources))
                if decision is not None:
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

            overlay = predict_preview_overlay(previous_packet, current_packet, frame_pair.frames, target_fps=config.target_fps)
            annotated = []
            for captured, boxes in zip(frame_pair.frames, overlay):
                frame = captured.frame.copy()
                draw_boxes(frame, boxes)
                draw_anchor_line(frame, config.anchor_axis, config.anchor_line_ratio)
                annotated.append(frame)
            preview = active_compose_preview(annotated)
            if preview is not None and callbacks.preview_callback is not None:
                callbacks.preview_callback(preview)
                preview_count += 1

            frame_count += 1
            elapsed = max(0.000001, clock.monotonic() - start_time)
            snapshot = RuntimePerformanceSnapshot(
                frame_count=frame_count,
                target_fps=int(config.target_fps),
                elapsed_s=elapsed,
                capture_fps_by_camera=tuple(reader.captured / elapsed for reader in readers),
                processed_fps=frame_count / elapsed,
                preview_fps=preview_count / elapsed,
                latest_pair_skew_ms=frame_pair.skew_ms,
                dropped_pairs=dropped_pairs,
                overlay_age_ms=(clock.monotonic() - current_packet.frame_pair.pair_timestamp) * 1000.0 if current_packet else None,
            )
            if callbacks.performance_callback is not None:
                callbacks.performance_callback(snapshot)

            target_interval_s = 1.0 / max(1.0, float(config.target_fps))
            sleep_fn(max(0.0, target_interval_s - (clock.monotonic() - frame_pair.pair_timestamp)))
            if camera_factory is not None and frame_count >= 2:
                break
    finally:
        for reader in readers:
            reader.stop()
        for camera in cameras:
            if hasattr(camera, "release"):
                camera.release()
        scheduler.close()
