#!/usr/bin/env python3
"""
Greyscale two-class cap inspection runtime for conveyor deployment.

- Runs a two-class YOLO ONNX model (`undefected`, `dirt_defect`).
- Converts frames to greyscale for inference while leaving capture, preview, and
  saved frame behavior aligned with the RGB V2 runtime.
- Tracks one physical cap across frames and both cameras so repeated detections
  do not trigger GPIO multiple times.
- Names tracking and reject thresholds separately so sensitivity is explicit.
- Captures timestamped debug frame bursts around actuation for miss analysis.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import math
import os
import subprocess
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from queue import Queue
from typing import Any, Callable

import cap_line_runtime as base
from gpio_output import GPIOOutputPin


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CLASS_NAMES = base.CLASS_NAMES
DEFECT_CLASS_ID = base.DEFECT_CLASS_ID
DEFAULT_MODEL = "dirtv3_grey.onnx"
MODEL_ALIASES = {
    "default": DEFAULT_MODEL,
    "best": DEFAULT_MODEL,
    "dirt": DEFAULT_MODEL,
    "latest": DEFAULT_MODEL,
}
MODEL_SEARCH_DIRS = (
    SCRIPT_DIR,
    os.path.join(SCRIPT_DIR, "model"),
)
DEFAULT_MODEL_PATH = os.path.join(SCRIPT_DIR, "model", DEFAULT_MODEL)
DEFAULT_TIMING_LOG_DIR = os.path.join(SCRIPT_DIR, "data", "timing_logs_v2")
RESOURCES_DIR = os.path.join(SCRIPT_DIR, "resources")
DEFAULT_DEBUG_DIR = os.path.join(RESOURCES_DIR, "debugging")
DEFAULT_PICTURES_DIR = os.path.join(RESOURCES_DIR, "pictures_grey")
DEFAULT_SESSION_LOG_DIR = os.path.join(DEFAULT_DEBUG_DIR, "sessions")
DEFAULT_REVIEW_DIR = DEFAULT_DEBUG_DIR
TRACKING_DETECTION_THRESHOLD = 0.45
DEFECT_REJECT_THRESHOLD = 0.45
GLOBAL_DETECTION_THRESHOLD = TRACKING_DETECTION_THRESHOLD
DUPLICATE_BOX_IOU_THRESHOLD = 0.65
DEFAULT_DEBUG_BURST_BEFORE_FRAMES = 3
DEFAULT_DEBUG_BURST_AFTER_FRAMES = 3
DEFAULT_CAMERA_RESOLUTION = base.DEFAULT_CAMERA_RESOLUTION
DEFAULT_CAMERA_FPS = base.DEFAULT_CAMERA_FPS
DEFAULT_CAMERA_PIXEL_FORMAT = base.DEFAULT_CAMERA_PIXEL_FORMAT
DEFAULT_ONNX_INTRA_OP_THREADS = max(1, (os.cpu_count() or 2) // 2)
DEFAULT_PERF_LOG_INTERVAL_S = 5.0
DEFAULT_PAIR_MAX_SKEW_MS = 40.0
DEFAULT_SAVE_QUEUE_WARNING_THRESHOLD = 25
DEFAULT_NOZZLE_DISTANCE_MM = base.DEFAULT_NOZZLE_DISTANCE_MM
DEFAULT_BELT_SPEED_MM_PER_S = base.DEFAULT_BELT_SPEED_MM_PER_S
DEFAULT_TRIGGER_PIN = base.DEFAULT_TRIGGER_PIN
DEFAULT_TRIGGER_OFFSET_S = base.DEFAULT_TRIGGER_OFFSET_S
DEFAULT_FINALIZE_QUIET_MS = base.DEFAULT_FINALIZE_QUIET_MS
DEFAULT_LATENCY_COMPENSATION_MS = base.DEFAULT_LATENCY_COMPENSATION_MS
TIMING_LOG_HEADERS = base.TIMING_LOG_HEADERS
REVIEW_MANIFEST_HEADERS = base.REVIEW_MANIFEST_HEADERS
DEBUG_MANIFEST_HEADERS = [
    "recorded_at",
    "event_id",
    "result",
    "review_reason",
    "decision_source",
    "final_class",
    "final_score",
    "score_summary",
    "cam0_vote",
    "cam1_vote",
    "model_path",
    "preview_path",
    "cam0_annotated_path",
    "cam1_annotated_path",
    "raw_cam0_path",
    "raw_cam1_path",
    "json_path",
]
PICTURE_MANIFEST_HEADERS = [
    "recorded_at",
    "event_id",
    "result",
    "final_class",
    "final_score",
    "decision_source",
    "cam0_vote",
    "cam1_vote",
    "anchor_axis",
    "anchor_line_ratio",
    "raw_cam0_path",
    "raw_cam1_path",
]
CAP_PROP_FRAME_WIDTH = 3
CAP_PROP_FRAME_HEIGHT = 4
CAP_PROP_FPS = 5

DetectionHistoryRecord = base.DetectionHistoryRecord
TimingLogRecord = base.TimingLogRecord
RuntimeClock = base.RuntimeClock
NullGPIOOutputPin = base.NullGPIOOutputPin
RejectEnqueueResult = base.RejectEnqueueResult
RejectExecution = base.RejectExecution
RejectScheduler = base.RejectScheduler
TimingCsvLogger = base.TimingCsvLogger
CameraTrack = base.CameraTrack
TrackObservation = base.TrackObservation
ClosedTrack = base.ClosedTrack
TrackUpdate = base.TrackUpdate
CameraLifecycleTracker = base.CameraLifecycleTracker
CameraVote = base.CameraVote
CapEvaluation = base.CapEvaluation
TrackedCapDecision = base.TrackedCapDecision
class_name = base.class_name
round_float = base.round_float
copy_frame = base.copy_frame
copy_frames = base.copy_frames
infer_model_imgsz_from_name = base.infer_model_imgsz_from_name
parse_cameras = base.parse_cameras
set_camera_controls = base.set_camera_controls
set_camera_format = base.set_camera_format
normalize_camera_pixel_format = base.normalize_camera_pixel_format
open_cam = base.open_cam
resolve_imgsz = base.resolve_imgsz
def preprocess(frame, img_size: int = 640):
    import cv2
    import numpy as np

    if frame is None:
        raise ValueError("frame must not be None")

    letterboxed_bgr, resize_scale, padding = base.letterbox_resize(
        frame,
        new_shape=(img_size, img_size),
        color=(114, 114, 114),
    )
    if letterboxed_bgr.ndim == 2:
        greyscale = letterboxed_bgr
    elif letterboxed_bgr.ndim == 3 and letterboxed_bgr.shape[2] == 1:
        greyscale = letterboxed_bgr[:, :, 0]
    elif letterboxed_bgr.ndim == 3 and letterboxed_bgr.shape[2] == 3:
        greyscale = cv2.cvtColor(letterboxed_bgr, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError(
            "frame must be a 2D greyscale image or a 1/3-channel image"
        )

    img = greyscale.astype(np.float32) / 255.0
    img = np.repeat(np.expand_dims(img, axis=0), 3, axis=0)
    img = np.expand_dims(img, axis=0)
    return img, {
        "scale": float(resize_scale),
        "pad_left": int(padding[0]),
        "pad_top": int(padding[1]),
        "frame_shape": frame.shape,
        "img_size": int(img_size),
    }


draw_anchor_line = base.draw_anchor_line
draw_boxes = base.draw_boxes
compose_preview = base.compose_preview
reference_coordinate = base.reference_coordinate
did_cross_reference_line = base.did_cross_reference_line
calculate_trigger_delay = base.calculate_trigger_delay
compute_requested_trigger_delay = base.compute_requested_trigger_delay
resolve_anchor_time = base.resolve_anchor_time
build_tracked_cap_decision = base.build_tracked_cap_decision
camera_vote_payload = base.camera_vote_payload
format_camera_vote = base.format_camera_vote
build_history_record = base.build_history_record
build_timing_log_record = base.build_timing_log_record
format_class_scores = base.format_class_scores


def _isoformat_timestamp_label(recorded_at: str) -> str:
    if recorded_at:
        try:
            return datetime.fromisoformat(recorded_at).strftime("%Y%m%d_%H%M%S_%f")[:-3]
        except ValueError:
            pass
    return datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")[:-3]


def _isoformat_record_day(recorded_at: str) -> str:
    if recorded_at:
        return recorded_at.split("T", 1)[0]
    return datetime.now().astimezone().strftime("%Y-%m-%d")


def debug_event_prefix(recorded_at: str, event_id: int, result: str) -> str:
    return f"{_isoformat_timestamp_label(recorded_at)}_event{event_id}_{result}"


def copy_boxes_by_camera(boxes_by_camera: list[list[list[float]]]) -> list[list[list[float]]]:
    return [
        [[float(value) for value in box] for box in camera_boxes]
        for camera_boxes in boxes_by_camera
    ]


def _json_default(value: Any) -> Any:
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, datetime):
        return value.isoformat(timespec="milliseconds")
    if hasattr(value, "__dict__"):
        return dict(value.__dict__)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


class SessionLogTee:
    """Wraps a log_fn so every call is also appended to a session log file."""

    def __init__(self, original_log_fn: Callable[..., None], log_path: str):
        self._original_log_fn = original_log_fn
        self._log_path = os.path.abspath(log_path)
        os.makedirs(os.path.dirname(self._log_path) or ".", exist_ok=True)
        self._lock = threading.Lock()

    @property
    def log_path(self) -> str:
        return self._log_path

    def __call__(self, *args, **kwargs):
        try:
            self._original_log_fn(*args, **kwargs)
        finally:
            message = " ".join(str(arg) for arg in args)
            timestamp = datetime.now().astimezone().isoformat(timespec="milliseconds")
            with self._lock:
                try:
                    with open(self._log_path, "a", encoding="utf-8") as handle:
                        handle.write(f"{timestamp} {message}\n")
                except OSError:
                    pass


@dataclass
class FrameSnapshot:
    frame_index: int
    timestamp: float
    raw_frames: list[object] = field(default_factory=list)
    annotated_frames: list[object] = field(default_factory=list)
    boxes_by_camera: list[list[list[float]]] = field(default_factory=list)
    read_duration_ms: float | None = None
    frame_interval_ms: float | None = None
    inference_ms_by_camera: list[float] = field(default_factory=list)
    processing_duration_ms: float | None = None
    pair_skew_ms: float | None = None
    pair_sequences: list[int] = field(default_factory=list)


@dataclass(frozen=True)
class CapturedFrame:
    camera_index: int
    frame: object
    timestamp: float
    sequence: int
    read_duration_ms: float | None = None


@dataclass(frozen=True)
class FramePair:
    frames: list[object]
    timestamps: list[float]
    sequences: list[int]
    read_duration_ms_by_camera: list[float | None]
    pair_timestamp: float
    skew_ms: float

    @property
    def read_duration_ms(self) -> float | None:
        return _mean(
            [
                duration
                for duration in self.read_duration_ms_by_camera
                if duration is not None
            ]
        )


@dataclass(frozen=True)
class PairedInferenceResult:
    frame_pair: FramePair
    camera_results: list[CameraInferenceResult]

    @property
    def boxes_by_camera(self) -> list[list[list[float]]]:
        return [result.boxes for result in self.camera_results]

    @property
    def inference_ms_by_camera(self) -> list[float]:
        return [result.inference_ms for result in self.camera_results]


@dataclass
class DebugFrameSnapshot:
    frame_index: int
    timestamp_monotonic: float
    timestamp_iso: str
    offset_from_actuation_ms: float | None
    raw_frames: list[object] = field(default_factory=list)
    annotated_frames: list[object] = field(default_factory=list)
    boxes_by_camera: list[list[list[float]]] = field(default_factory=list)
    read_duration_ms: float | None = None
    frame_interval_ms: float | None = None
    inference_ms_by_camera: list[float] = field(default_factory=list)
    processing_duration_ms: float | None = None
    pair_skew_ms: float | None = None
    pair_sequences: list[int] = field(default_factory=list)


@dataclass(frozen=True)
class CameraProperties:
    camera_index: int
    source: str
    requested_width: int | None
    requested_height: int | None
    requested_fps: float | None
    actual_width: int | None
    actual_height: int | None
    actual_fps: float | None


@dataclass(frozen=True)
class CameraInferenceResult:
    camera_index: int
    boxes: list[list[float]]
    inference_ms: float


@dataclass(frozen=True)
class RuntimePerformanceSnapshot:
    frame_count: int
    elapsed_s: float
    fps: float
    capture_fps_by_camera: list[float | None]
    frame_interval_ms: float | None
    read_duration_ms: float | None
    inference_ms_by_camera: list[float | None]
    processing_duration_ms: float | None
    latest_pair_skew_ms: float | None = None
    average_pair_skew_ms: float | None = None
    stale_pair_drops: int = 0
    accepted_pairs: int = 0


@dataclass
class PairingStats:
    accepted_pairs: int = 0
    stale_pair_drops: int = 0
    latest_skew_ms: float | None = None
    _total_skew_ms: float = 0.0
    _last_stale_sequences: list[int] = field(default_factory=list, repr=False)

    def record_accepted(self, pair: FramePair) -> None:
        self.accepted_pairs += 1
        self.latest_skew_ms = float(pair.skew_ms)
        self._total_skew_ms += float(pair.skew_ms)

    def record_stale_drop(self, sequences: list[int], skew_ms: float) -> bool:
        normalized_sequences = [int(sequence) for sequence in sequences]
        if normalized_sequences == self._last_stale_sequences:
            return False
        self._last_stale_sequences = normalized_sequences
        self.stale_pair_drops += 1
        self.latest_skew_ms = float(skew_ms)
        return True

    @property
    def average_pair_skew_ms(self) -> float | None:
        if self.accepted_pairs <= 0:
            return None
        return self._total_skew_ms / self.accepted_pairs


class RuntimePerformanceStats:
    def __init__(self, interval_s: float, *, camera_count: int, start_time: float):
        self.interval_s = max(0.0, float(interval_s))
        self.camera_count = max(0, int(camera_count))
        self.latest_snapshot: RuntimePerformanceSnapshot | None = None
        self._latest_capture_sequences_by_camera: list[int] = [
            0 for _ in range(self.camera_count)
        ]
        self._reset(float(start_time))

    def record(
        self,
        *,
        now: float,
        frame_interval_ms: float | None,
        read_duration_ms: float | None,
        inference_ms_by_camera: list[float],
        processing_duration_ms: float | None,
        capture_sequences_by_camera: list[int] | None = None,
        pair_skew_ms: float | None = None,
        pairing_stats: PairingStats | None = None,
    ) -> RuntimePerformanceSnapshot | None:
        self._frame_count += 1
        if capture_sequences_by_camera is not None:
            for camera_index in range(self.camera_count):
                if camera_index >= len(capture_sequences_by_camera):
                    continue
                self._latest_capture_sequences_by_camera[camera_index] = int(
                    capture_sequences_by_camera[camera_index]
                )
        if frame_interval_ms is not None:
            self._frame_interval_ms.append(float(frame_interval_ms))
        if read_duration_ms is not None:
            self._read_duration_ms.append(float(read_duration_ms))
        if processing_duration_ms is not None:
            self._processing_duration_ms.append(float(processing_duration_ms))
        if pair_skew_ms is not None:
            self._pair_skew_ms.append(float(pair_skew_ms))
        self._pairing_stats = pairing_stats
        for camera_index in range(self.camera_count):
            if camera_index >= len(inference_ms_by_camera):
                continue
            self._inference_ms_by_camera[camera_index].append(
                float(inference_ms_by_camera[camera_index])
            )

        snapshot = self._build_snapshot(float(now))
        self.latest_snapshot = snapshot
        if self.interval_s <= 0.0 or snapshot.elapsed_s < self.interval_s:
            return None

        self._reset(float(now))
        self.latest_snapshot = snapshot
        return snapshot

    def _reset(self, start_time: float) -> None:
        self._window_started_at = float(start_time)
        self._capture_window_start_sequences = list(
            self._latest_capture_sequences_by_camera
        )
        self._frame_count = 0
        self._frame_interval_ms: list[float] = []
        self._read_duration_ms: list[float] = []
        self._processing_duration_ms: list[float] = []
        self._pair_skew_ms: list[float] = []
        self._pairing_stats: PairingStats | None = None
        self._inference_ms_by_camera: list[list[float]] = [
            [] for _ in range(self.camera_count)
        ]

    def _build_snapshot(self, now: float) -> RuntimePerformanceSnapshot:
        elapsed_s = max(0.0, float(now) - self._window_started_at)
        fps = 0.0 if elapsed_s <= 0.0 else self._frame_count / elapsed_s
        capture_fps_by_camera = []
        for camera_index in range(self.camera_count):
            if elapsed_s <= 0.0:
                capture_fps_by_camera.append(0.0)
                continue
            captured_frames = (
                self._latest_capture_sequences_by_camera[camera_index]
                - self._capture_window_start_sequences[camera_index]
            )
            capture_fps_by_camera.append(max(0.0, captured_frames / elapsed_s))
        return RuntimePerformanceSnapshot(
            frame_count=int(self._frame_count),
            elapsed_s=elapsed_s,
            fps=fps,
            capture_fps_by_camera=capture_fps_by_camera,
            frame_interval_ms=_mean(self._frame_interval_ms),
            read_duration_ms=_mean(self._read_duration_ms),
            inference_ms_by_camera=[
                _mean(values) for values in self._inference_ms_by_camera
            ],
            processing_duration_ms=_mean(self._processing_duration_ms),
            latest_pair_skew_ms=(
                self._pair_skew_ms[-1] if self._pair_skew_ms else None
            ),
            average_pair_skew_ms=_mean(self._pair_skew_ms),
            stale_pair_drops=(
                int(self._pairing_stats.stale_pair_drops)
                if self._pairing_stats is not None
                else 0
            ),
            accepted_pairs=(
                int(self._pairing_stats.accepted_pairs)
                if self._pairing_stats is not None
                else 0
            ),
        )


class LatestFrameCameraReader:
    """Continuously drains one camera and exposes the freshest captured frame."""

    def __init__(
        self,
        camera: object,
        *,
        camera_index: int,
        target_fps: float | None,
        time_fn: Callable[[], float] = time.monotonic,
        sleep_fn: Callable[[float], None] = time.sleep,
    ):
        self.camera = camera
        self.camera_index = int(camera_index)
        self.target_fps = None if target_fps is None else float(target_fps)
        self._time_fn = time_fn
        self._sleep_fn = sleep_fn
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._latest: CapturedFrame | None = None
        self._sequence = 0
        self._thread = threading.Thread(
            target=self._run,
            name=f"cap-line-camera-{self.camera_index}",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join()

    def latest(self) -> CapturedFrame | None:
        with self._lock:
            return self._latest

    @property
    def sequence(self) -> int:
        with self._lock:
            return self._sequence

    def _run(self) -> None:
        min_interval_s = (
            0.0
            if self.target_fps is None or self.target_fps <= 0.0
            else 1.0 / self.target_fps
        )
        while not self._stop_event.is_set():
            read_started_at = self._time_fn()
            try:
                ok, frame = self.camera.read()  # type: ignore[attr-defined]
            except Exception:
                ok, frame = False, None
            captured_at = self._time_fn()
            if ok and frame is not None:
                with self._lock:
                    self._sequence += 1
                    self._latest = CapturedFrame(
                        camera_index=self.camera_index,
                        frame=frame,
                        timestamp=captured_at,
                        sequence=self._sequence,
                        read_duration_ms=(captured_at - read_started_at) * 1000.0,
                    )
            elif not self._stop_event.is_set():
                self._sleep_fn(0.01)

            if min_interval_s > 0.0 and not self._stop_event.is_set():
                elapsed_s = self._time_fn() - read_started_at
                remaining_s = min_interval_s - elapsed_s
                if remaining_s > 0.0:
                    self._sleep_fn(remaining_s)


def is_fresh_frame_pair(
    sequences: list[int],
    last_processed_sequences: list[int] | None,
) -> bool:
    if last_processed_sequences is None:
        return True
    if len(sequences) != len(last_processed_sequences):
        return True
    return all(
        int(sequence) > int(last_sequence)
        for sequence, last_sequence in zip(sequences, last_processed_sequences)
    )


def has_new_frame_for_pair(
    sequences: list[int],
    last_processed_sequences: list[int] | None,
) -> bool:
    if last_processed_sequences is None:
        return True
    if len(sequences) != len(last_processed_sequences):
        return True
    return any(
        int(sequence) > int(last_sequence)
        for sequence, last_sequence in zip(sequences, last_processed_sequences)
    )


def build_frame_pair(captured_frames: list[CapturedFrame]) -> FramePair:
    timestamps = [float(capture.timestamp) for capture in captured_frames]
    sequences = [int(capture.sequence) for capture in captured_frames]
    pair_timestamp = max(timestamps)
    skew_ms = (max(timestamps) - min(timestamps)) * 1000.0
    return FramePair(
        frames=[capture.frame for capture in captured_frames],
        timestamps=timestamps,
        sequences=sequences,
        read_duration_ms_by_camera=[
            capture.read_duration_ms for capture in captured_frames
        ],
        pair_timestamp=pair_timestamp,
        skew_ms=skew_ms,
    )


def select_synchronized_frame_pair(
    latest_frames: list[CapturedFrame | None],
    last_processed_sequences: list[int] | None,
    *,
    max_skew_ms: float,
    pairing_stats: PairingStats | None = None,
    log_fn: Callable[..., None] | None = None,
) -> FramePair | None:
    if not latest_frames or not all(frame is not None for frame in latest_frames):
        return None

    captured_frames = [frame for frame in latest_frames if frame is not None]
    sequences = [int(frame.sequence) for frame in captured_frames]
    if not has_new_frame_for_pair(sequences, last_processed_sequences):
        return None

    frame_pair = build_frame_pair(captured_frames)
    if frame_pair.skew_ms > float(max_skew_ms):
        recorded_drop = False
        if pairing_stats is not None:
            recorded_drop = pairing_stats.record_stale_drop(
                frame_pair.sequences,
                frame_pair.skew_ms,
            )
        if recorded_drop and log_fn is not None:
            log_fn(
                "[PAIR][WARN] dropped stale camera pair "
                f"skew_ms={frame_pair.skew_ms:.1f} "
                f"max_skew_ms={float(max_skew_ms):.1f} "
                f"sequences={frame_pair.sequences}"
            )
        return None

    if not is_fresh_frame_pair(frame_pair.sequences, last_processed_sequences):
        return None

    if pairing_stats is not None:
        pairing_stats.record_accepted(frame_pair)
    return frame_pair


def copy_frame_snapshot(snapshot: FrameSnapshot) -> FrameSnapshot:
    return FrameSnapshot(
        frame_index=int(snapshot.frame_index),
        timestamp=float(snapshot.timestamp),
        raw_frames=copy_frames(snapshot.raw_frames),
        annotated_frames=copy_frames(snapshot.annotated_frames),
        boxes_by_camera=copy_boxes_by_camera(snapshot.boxes_by_camera),
        read_duration_ms=snapshot.read_duration_ms,
        frame_interval_ms=snapshot.frame_interval_ms,
        inference_ms_by_camera=[float(value) for value in snapshot.inference_ms_by_camera],
        processing_duration_ms=snapshot.processing_duration_ms,
        pair_skew_ms=snapshot.pair_skew_ms,
        pair_sequences=[int(sequence) for sequence in snapshot.pair_sequences],
    )


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def format_optional_float(value: float | None, *, digits: int = 1) -> str:
    if value is None:
        return "unknown"
    return f"{float(value):.{digits}f}"


def read_camera_property(camera: object, property_id: int) -> float | None:
    try:
        value = camera.get(property_id)  # type: ignore[attr-defined]
    except Exception:
        return None
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric_value) or numeric_value <= 0.0:
        return None
    return numeric_value


def read_camera_properties(
    camera: object,
    *,
    camera_index: int,
    source: object,
    requested_width: int | None,
    requested_height: int | None,
    requested_fps: float | None,
) -> CameraProperties:
    actual_width = read_camera_property(camera, CAP_PROP_FRAME_WIDTH)
    actual_height = read_camera_property(camera, CAP_PROP_FRAME_HEIGHT)
    actual_fps = read_camera_property(camera, CAP_PROP_FPS)
    return CameraProperties(
        camera_index=int(camera_index),
        source=str(source),
        requested_width=None if requested_width is None else int(requested_width),
        requested_height=None if requested_height is None else int(requested_height),
        requested_fps=None if requested_fps is None else float(requested_fps),
        actual_width=None if actual_width is None else int(round(actual_width)),
        actual_height=None if actual_height is None else int(round(actual_height)),
        actual_fps=actual_fps,
    )


def camera_properties_mismatch(properties: CameraProperties) -> bool:
    if (
        properties.requested_width is not None
        and properties.actual_width is not None
        and abs(properties.requested_width - properties.actual_width) >= 1
    ):
        return True
    if (
        properties.requested_height is not None
        and properties.actual_height is not None
        and abs(properties.requested_height - properties.actual_height) >= 1
    ):
        return True
    if (
        properties.requested_fps is not None
        and properties.actual_fps is not None
        and abs(properties.requested_fps - properties.actual_fps) >= 1.0
    ):
        return True
    return False


def log_camera_properties(
    properties: CameraProperties,
    *,
    log_fn: Callable[..., None] = print,
) -> None:
    requested_size = (
        "unknown"
        if properties.requested_width is None or properties.requested_height is None
        else f"{properties.requested_width}x{properties.requested_height}"
    )
    actual_size = (
        "unknown"
        if properties.actual_width is None or properties.actual_height is None
        else f"{properties.actual_width}x{properties.actual_height}"
    )
    requested_fps = format_optional_float(properties.requested_fps)
    actual_fps = format_optional_float(properties.actual_fps)
    log_fn(
        f"[CAMERA] index={properties.camera_index} source={properties.source} "
        f"requested={requested_size}@{requested_fps} actual={actual_size}@{actual_fps}"
    )
    if camera_properties_mismatch(properties):
        log_fn(
            f"[CAMERA][WARN] index={properties.camera_index} source={properties.source} "
            "actual camera format differs from requested format"
        )


def camera_properties_payload(properties: CameraProperties) -> dict[str, Any]:
    return {
        "camera_index": int(properties.camera_index),
        "source": properties.source,
        "requested_width": properties.requested_width,
        "requested_height": properties.requested_height,
        "requested_fps": round_float(properties.requested_fps, 3),
        "actual_width": properties.actual_width,
        "actual_height": properties.actual_height,
        "actual_fps": round_float(properties.actual_fps, 3),
        "mismatch": camera_properties_mismatch(properties),
    }


def performance_snapshot_payload(
    snapshot: RuntimePerformanceSnapshot | None,
) -> dict[str, Any] | None:
    if snapshot is None:
        return None
    return {
        "frame_count": int(snapshot.frame_count),
        "elapsed_s": round_float(snapshot.elapsed_s, 3),
        "processed_fps": round_float(snapshot.fps, 3),
        "fps": round_float(snapshot.fps, 3),
        "capture_fps_by_camera": [
            round_float(value, 3) for value in snapshot.capture_fps_by_camera
        ],
        "frame_interval_ms": round_float(snapshot.frame_interval_ms, 3),
        "read_duration_ms": round_float(snapshot.read_duration_ms, 3),
        "inference_ms_by_camera": [
            round_float(value, 3) for value in snapshot.inference_ms_by_camera
        ],
        "processing_duration_ms": round_float(snapshot.processing_duration_ms, 3),
        "latest_pair_skew_ms": round_float(snapshot.latest_pair_skew_ms, 3),
        "average_pair_skew_ms": round_float(snapshot.average_pair_skew_ms, 3),
        "stale_pair_drops": int(snapshot.stale_pair_drops),
        "accepted_pairs": int(snapshot.accepted_pairs),
    }


def log_performance_snapshot(
    snapshot: RuntimePerformanceSnapshot,
    *,
    log_fn: Callable[..., None] = print,
) -> None:
    inference_text = ",".join(
        format_optional_float(value) for value in snapshot.inference_ms_by_camera
    )
    capture_fps_text = ",".join(
        format_optional_float(value) for value in snapshot.capture_fps_by_camera
    )
    log_fn(
        f"[PERF] processed_fps={snapshot.fps:.1f} "
        f"capture_fps=[{capture_fps_text}] "
        f"frame_interval_ms={format_optional_float(snapshot.frame_interval_ms)} "
        f"read_ms={format_optional_float(snapshot.read_duration_ms)} "
        f"infer_ms=[{inference_text}] "
        f"processing_ms={format_optional_float(snapshot.processing_duration_ms)} "
        f"pair_skew_ms={format_optional_float(snapshot.latest_pair_skew_ms)} "
        f"pair_drops={snapshot.stale_pair_drops}"
    )


@dataclass
class DebugCaptureTask:
    event_id: int
    recorded_at: str
    result: str
    review_reason: str
    decision_source: str
    final_class: str | None
    final_score: float | None
    score_summary: str
    cam0_vote: str
    cam1_vote: str
    model_path: str
    preview_frame: object | None = None
    annotated_frames: list[object] = field(default_factory=list)
    raw_frames: list[object] = field(default_factory=list)
    frame_burst: list[DebugFrameSnapshot] = field(default_factory=list)
    json_payload: dict = field(default_factory=dict)


@dataclass
class LinePictureTask:
    event_id: int
    recorded_at: str
    result: str
    final_class: str | None
    final_score: float | None
    decision_source: str
    cam0_vote: str
    cam1_vote: str
    raw_frames: list[object] = field(default_factory=list)
    anchor_axis: str = "x"
    anchor_line_ratio: float = 0.5


class DebugCaptureWriter:
    """Writes debug artifacts and line-frame training pictures on a background thread.

    Run on a background thread to keep the inference loop responsive.
    """

    def __init__(
        self,
        directory: str = DEFAULT_DEBUG_DIR,
        *,
        pictures_dir: str = DEFAULT_PICTURES_DIR,
        queue_warning_threshold: int = DEFAULT_SAVE_QUEUE_WARNING_THRESHOLD,
        write_image_fn: Callable[[object, str], bool] | None = None,
        log_fn: Callable[..., None] = print,
    ):
        self.directory = os.path.abspath(directory)
        self.debug_dir = self.directory
        self.pictures_dir = os.path.abspath(pictures_dir)
        self.queue_warning_threshold = max(0, int(queue_warning_threshold))
        os.makedirs(self.debug_dir, exist_ok=True)
        os.makedirs(self.pictures_dir, exist_ok=True)
        self._write_image_fn = write_image_fn or self._default_write_image
        self._log = log_fn
        self._queue: Queue = Queue()
        self._thread = threading.Thread(
            target=self._run,
            name="cap-line-debug-capture",
            daemon=True,
        )
        self._thread.start()

    @staticmethod
    def _default_write_image(frame, path: str) -> bool:
        import cv2

        return bool(
            cv2.imwrite(
                path,
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), 95],
            )
        )

    @property
    def queue_depth(self) -> int:
        return self._queue.qsize()

    def submit(self, task: DebugCaptureTask) -> int:
        self._queue.put(task)
        queue_depth = self._queue.qsize()
        self._log(
            f"[DEBUG] queued event={task.event_id} reason={task.review_reason} "
            f"backlog={queue_depth}"
        )
        self._warn_if_queue_backlogged(queue_depth)
        return queue_depth

    def submit_line_picture(self, task: LinePictureTask) -> int:
        self._queue.put(task)
        queue_depth = self._queue.qsize()
        self._log(
            f"[PICTURE] queued event={task.event_id} result={task.result} "
            f"backlog={queue_depth}"
        )
        self._warn_if_queue_backlogged(queue_depth)
        return queue_depth

    def _warn_if_queue_backlogged(self, queue_depth: int) -> None:
        if self.queue_warning_threshold <= 0:
            return
        if queue_depth < self.queue_warning_threshold:
            return
        self._log(
            "[SAVE][WARN] capture save queue backlog "
            f"depth={queue_depth} threshold={self.queue_warning_threshold}"
        )

    def write_trigger_completion(
        self,
        *,
        event_id: int,
        recorded_at: str,
        payload: dict,
    ) -> str | None:
        day = _isoformat_record_day(recorded_at)
        prefix = debug_event_prefix(recorded_at, event_id, "trigger")
        debug_day_dir = os.path.join(self.debug_dir, day)
        try:
            os.makedirs(debug_day_dir, exist_ok=True)
            sidecar_path = os.path.join(debug_day_dir, f"{prefix}_fire.json")
            with open(sidecar_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, default=_json_default)
            return sidecar_path
        except OSError as exc:
            self._log(f"[DEBUG] trigger sidecar error event={event_id} {exc}")
            return None

    def _manifest_path(self, day: str) -> str:
        return os.path.join(self.debug_dir, f"{day}.csv")

    def _picture_manifest_path(self, day: str) -> str:
        return os.path.join(self.pictures_dir, f"{day}.csv")

    def _run(self) -> None:
        while True:
            task = self._queue.get()
            if task is None:
                self._queue.task_done()
                break
            try:
                if isinstance(task, LinePictureTask):
                    self._handle_line_picture_task(task)
                else:
                    self._handle_task(task)
            except Exception as exc:
                event_id = getattr(task, "event_id", "unknown")
                self._log(f"[DEBUG] error event={event_id} {exc}")
            finally:
                self._queue.task_done()

    def _write_camera_frames(
        self,
        frames: list[object],
        *,
        path_for_camera: Callable[[int], str],
        log_prefix: str,
        event_id: int,
    ) -> list[str]:
        paths = ["" for _ in frames]
        write_jobs: list[tuple[int, str, concurrent.futures.Future]] = []
        valid_frames = [
            (camera_index, frame)
            for camera_index, frame in enumerate(frames)
            if frame is not None
        ]
        if not valid_frames:
            return paths

        max_workers = min(2, len(valid_frames))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            for camera_index, frame in valid_frames:
                path = path_for_camera(camera_index)
                write_jobs.append(
                    (
                        camera_index,
                        path,
                        executor.submit(self._write_image_fn, frame, path),
                    )
                )

            for camera_index, path, future in write_jobs:
                try:
                    wrote_image = bool(future.result())
                except Exception as exc:
                    wrote_image = False
                    self._log(
                        f"{log_prefix} cam{camera_index} write error "
                        f"event={event_id} {exc}"
                    )
                if wrote_image:
                    paths[camera_index] = path
                else:
                    self._log(
                        f"{log_prefix} cam{camera_index} write failed event={event_id}"
                    )
        return paths

    def _handle_task(self, task: DebugCaptureTask) -> None:
        day = _isoformat_record_day(task.recorded_at)
        prefix = debug_event_prefix(task.recorded_at, task.event_id, task.result)
        debug_day_dir = os.path.join(self.debug_dir, day)
        os.makedirs(debug_day_dir, exist_ok=True)

        preview_path = ""
        if task.preview_frame is not None:
            preview_path = os.path.join(debug_day_dir, f"{prefix}_preview.jpg")
            if not self._write_image_fn(task.preview_frame, preview_path):
                preview_path = ""
                self._log(f"[DEBUG] preview write failed event={task.event_id}")

        annotated_paths = self._write_camera_frames(
            task.annotated_frames,
            path_for_camera=lambda camera_index: os.path.join(
                debug_day_dir, f"{prefix}_cam{camera_index}_annot.jpg"
            ),
            log_prefix="[DEBUG] annotated",
            event_id=task.event_id,
        )

        raw_paths = self._write_camera_frames(
            task.raw_frames,
            path_for_camera=lambda camera_index: os.path.join(
                debug_day_dir, f"{prefix}_cam{camera_index}_raw.jpg"
            ),
            log_prefix="[DEBUG] raw",
            event_id=task.event_id,
        )

        frame_burst_artifacts: list[dict[str, Any]] = []
        for snapshot_index, snapshot in enumerate(task.frame_burst):
            snapshot_payload: dict[str, Any] = {
                "frame_index": int(snapshot.frame_index),
                "timestamp_monotonic": round_float(snapshot.timestamp_monotonic, 6),
                "timestamp_iso": snapshot.timestamp_iso,
                "offset_from_actuation_ms": round_float(
                    snapshot.offset_from_actuation_ms, 3
                )
                if snapshot.offset_from_actuation_ms is not None
                else None,
                "read_duration_ms": round_float(snapshot.read_duration_ms, 3),
                "frame_interval_ms": round_float(snapshot.frame_interval_ms, 3),
                "inference_ms_by_camera": [
                    round_float(value, 3) for value in snapshot.inference_ms_by_camera
                ],
                "processing_duration_ms": round_float(
                    snapshot.processing_duration_ms, 3
                ),
                "pair_skew_ms": round_float(snapshot.pair_skew_ms, 3),
                "pair_sequences": [
                    int(sequence) for sequence in snapshot.pair_sequences
                ],
                "boxes_by_camera": copy_boxes_by_camera(snapshot.boxes_by_camera),
                "raw_paths": {},
                "annotated_paths": {},
            }
            for camera_index, frame in enumerate(snapshot.raw_frames):
                if frame is None:
                    continue
                raw_path = os.path.join(
                    debug_day_dir,
                    f"{prefix}_burst{snapshot_index:02d}_cam{camera_index}_raw.jpg",
                )
                if self._write_image_fn(frame, raw_path):
                    snapshot_payload["raw_paths"][str(camera_index)] = raw_path
                else:
                    self._log(
                        f"[DEBUG] burst raw cam{camera_index} write failed "
                        f"event={task.event_id} frame={snapshot.frame_index}"
                    )
            for camera_index, frame in enumerate(snapshot.annotated_frames):
                if frame is None:
                    continue
                annotated_path = os.path.join(
                    debug_day_dir,
                    f"{prefix}_burst{snapshot_index:02d}_cam{camera_index}_annot.jpg",
                )
                if self._write_image_fn(frame, annotated_path):
                    snapshot_payload["annotated_paths"][str(camera_index)] = (
                        annotated_path
                    )
                else:
                    self._log(
                        f"[DEBUG] burst annot cam{camera_index} write failed "
                        f"event={task.event_id} frame={snapshot.frame_index}"
                    )
            frame_burst_artifacts.append(snapshot_payload)

        payload = dict(task.json_payload)
        artifacts = dict(payload.get("artifacts", {}))
        artifacts["preview_path"] = preview_path
        for camera_index, path in enumerate(annotated_paths):
            artifacts[f"cam{camera_index}_annotated_path"] = path
        for camera_index, path in enumerate(raw_paths):
            artifacts[f"raw_cam{camera_index}_path"] = path
        artifacts["frame_burst"] = frame_burst_artifacts
        payload["artifacts"] = artifacts

        json_path = os.path.join(debug_day_dir, f"{prefix}.json")
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=_json_default)

        self._append_manifest_row(
            task,
            day=day,
            preview_path=preview_path,
            annotated_paths=annotated_paths,
            raw_paths=raw_paths,
            json_path=json_path,
        )

        self._log(
            f"[DEBUG] saved event={task.event_id} reason={task.review_reason} "
            f"json={json_path}"
        )

    def _handle_line_picture_task(self, task: LinePictureTask) -> None:
        day = _isoformat_record_day(task.recorded_at)
        pictures_day_dir = os.path.join(self.pictures_dir, day)
        os.makedirs(pictures_day_dir, exist_ok=True)

        raw_prefix = (
            f"{_isoformat_timestamp_label(task.recorded_at)}_event{task.event_id}_line"
        )
        raw_paths = self._write_camera_frames(
            task.raw_frames,
            path_for_camera=lambda camera_index: os.path.join(
                pictures_day_dir, f"{raw_prefix}_cam{camera_index}.jpg"
            ),
            log_prefix="[PICTURE] raw",
            event_id=task.event_id,
        )
        if len(raw_paths) < 2 or not raw_paths[0] or not raw_paths[1]:
            self._log(
                f"[PICTURE][WARN] skipped manifest event={task.event_id}; "
                "incomplete cam0/cam1 image write"
            )
            return

        self._append_picture_manifest_row(task, day=day, raw_paths=raw_paths)
        self._log(
            f"[PICTURE] saved event={task.event_id} result={task.result} "
            f"frames={sum(1 for path in raw_paths if path)}"
        )

    def _append_manifest_row(
        self,
        task: DebugCaptureTask,
        *,
        day: str,
        preview_path: str,
        annotated_paths: list[str],
        raw_paths: list[str],
        json_path: str,
    ) -> None:
        manifest_path = self._manifest_path(day)
        needs_header = (
            not os.path.exists(manifest_path) or os.path.getsize(manifest_path) == 0
        )
        with open(manifest_path, "a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=DEBUG_MANIFEST_HEADERS)
            if needs_header:
                writer.writeheader()
            writer.writerow(
                {
                    "recorded_at": task.recorded_at,
                    "event_id": task.event_id,
                    "result": task.result,
                    "review_reason": task.review_reason,
                    "decision_source": task.decision_source,
                    "final_class": task.final_class or "",
                    "final_score": (
                        ""
                        if task.final_score is None
                        else round_float(task.final_score, 6)
                    ),
                    "score_summary": task.score_summary,
                    "cam0_vote": task.cam0_vote,
                    "cam1_vote": task.cam1_vote,
                    "model_path": task.model_path,
                    "preview_path": preview_path,
                    "cam0_annotated_path": (
                        annotated_paths[0] if len(annotated_paths) > 0 else ""
                    ),
                    "cam1_annotated_path": (
                        annotated_paths[1] if len(annotated_paths) > 1 else ""
                    ),
                    "raw_cam0_path": raw_paths[0] if len(raw_paths) > 0 else "",
                    "raw_cam1_path": raw_paths[1] if len(raw_paths) > 1 else "",
                    "json_path": json_path,
                }
            )

    def _append_picture_manifest_row(
        self,
        task: LinePictureTask,
        *,
        day: str,
        raw_paths: list[str],
    ) -> None:
        manifest_path = self._picture_manifest_path(day)
        needs_header = (
            not os.path.exists(manifest_path) or os.path.getsize(manifest_path) == 0
        )
        with open(manifest_path, "a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=PICTURE_MANIFEST_HEADERS)
            if needs_header:
                writer.writeheader()
            writer.writerow(
                {
                    "recorded_at": task.recorded_at,
                    "event_id": task.event_id,
                    "result": task.result,
                    "final_class": task.final_class or "",
                    "final_score": (
                        ""
                        if task.final_score is None
                        else round_float(task.final_score, 6)
                    ),
                    "decision_source": task.decision_source,
                    "cam0_vote": task.cam0_vote,
                    "cam1_vote": task.cam1_vote,
                    "anchor_axis": task.anchor_axis,
                    "anchor_line_ratio": round_float(task.anchor_line_ratio, 6),
                    "raw_cam0_path": raw_paths[0] if len(raw_paths) > 0 else "",
                    "raw_cam1_path": raw_paths[1] if len(raw_paths) > 1 else "",
                }
            )

    def close(self) -> None:
        self._queue.put(None)
        self._thread.join()


# Backward-compat aliases for code/tests still referencing the old names.
ReviewCaptureTask = DebugCaptureTask
ReviewCaptureWriter = DebugCaptureWriter


def candidate_model_paths(model_name: str) -> list[str]:
    expanded_name = os.path.expanduser(model_name)
    base_names = [expanded_name]
    if not expanded_name.lower().endswith(".onnx"):
        base_names.append(f"{expanded_name}.onnx")

    candidates: list[str] = []
    seen_candidates: set[str] = set()

    def add_candidate(path: str) -> None:
        normalized = os.path.abspath(path) if not os.path.isabs(path) else path
        if normalized in seen_candidates:
            return
        seen_candidates.add(normalized)
        candidates.append(normalized)

    if os.path.isabs(expanded_name):
        for name in base_names:
            add_candidate(name)
        return candidates

    if os.path.dirname(expanded_name):
        for name in base_names:
            add_candidate(name)
            add_candidate(os.path.join(SCRIPT_DIR, name))
        return candidates

    for name in base_names:
        add_candidate(name)
        for search_dir in MODEL_SEARCH_DIRS:
            add_candidate(os.path.join(search_dir, name))

    return candidates


def resolve_model_path(model_arg: str | None) -> tuple[str, int | None]:
    requested_model = str(model_arg).strip() if model_arg is not None else ""
    requested_model = requested_model or DEFAULT_MODEL
    requested_model = MODEL_ALIASES.get(requested_model.lower(), requested_model)

    for candidate in candidate_model_paths(requested_model):
        if os.path.exists(candidate):
            return candidate, infer_model_imgsz_from_name(candidate)

    raise FileNotFoundError(
        f"Model not found for '{requested_model}'. Default expected at {DEFAULT_MODEL_PATH}"
    )


def postprocess(output, preprocess_meta, conf_threshold: float = TRACKING_DETECTION_THRESHOLD):
    boxes = base.postprocess(
        output,
        preprocess_meta,
        conf_threshold=conf_threshold,
    )
    return deduplicate_yolo_boxes(boxes)


def create_onnx_session(ort, model_path: str, intra_op_threads: int):
    session_options = None
    if hasattr(ort, "SessionOptions"):
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = max(1, int(intra_op_threads))
        session_options.inter_op_num_threads = 1
    if session_options is None:
        return ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    return ort.InferenceSession(
        model_path,
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )


def infer_camera_frame(
    camera_index: int,
    frame: object,
    session: object,
    input_name: str,
    model_imgsz: int,
    clock: RuntimeClock,
    *,
    tracking_threshold: float = TRACKING_DETECTION_THRESHOLD,
) -> CameraInferenceResult:
    inference_started_at = clock.monotonic()
    input_tensor, preprocess_meta = preprocess(frame, model_imgsz)
    output = session.run(None, {input_name: input_tensor})[0]
    try:
        boxes = postprocess(output, preprocess_meta, conf_threshold=tracking_threshold)
    except TypeError as exc:
        if "conf_threshold" not in str(exc):
            raise
        boxes = postprocess(output, preprocess_meta)
    return CameraInferenceResult(
        camera_index=int(camera_index),
        boxes=boxes,
        inference_ms=(clock.monotonic() - inference_started_at) * 1000.0,
    )


def infer_frame_pair(
    frames: list[object],
    sessions: list[object],
    input_names: list[str],
    model_imgsz: int,
    clock: RuntimeClock,
    executor: concurrent.futures.Executor | None,
    *,
    serial: bool = False,
    tracking_threshold: float = TRACKING_DETECTION_THRESHOLD,
) -> list[CameraInferenceResult]:
    if serial or executor is None or len(frames) <= 1:
        return [
            infer_camera_frame(
                camera_index,
                frame,
                sessions[camera_index],
                input_names[camera_index],
                model_imgsz,
                clock,
                tracking_threshold=tracking_threshold,
            )
            for camera_index, frame in enumerate(frames)
        ]

    futures = [
        executor.submit(
            infer_camera_frame,
            camera_index,
            frame,
            sessions[camera_index],
            input_names[camera_index],
            model_imgsz,
            clock,
            tracking_threshold=tracking_threshold,
        )
        for camera_index, frame in enumerate(frames)
    ]
    return [future.result() for future in futures]


def infer_paired_frame(
    frame_pair: FramePair,
    sessions: list[object],
    input_names: list[str],
    model_imgsz: int,
    clock: RuntimeClock,
    executor: concurrent.futures.Executor | None,
    *,
    serial: bool = False,
    tracking_threshold: float = TRACKING_DETECTION_THRESHOLD,
) -> PairedInferenceResult:
    return PairedInferenceResult(
        frame_pair=frame_pair,
        camera_results=infer_frame_pair(
            frame_pair.frames,
            sessions,
            input_names,
            model_imgsz,
            clock,
            executor,
            serial=serial,
            tracking_threshold=tracking_threshold,
        ),
    )


def _select_duplicate_box(boxes: list[list[float]]) -> list[float]:
    defect_boxes = [box for box in boxes if int(box[5]) == DEFECT_CLASS_ID]
    candidates = defect_boxes or boxes
    return list(max(candidates, key=lambda box: float(box[4])))


def deduplicate_yolo_boxes(
    boxes: list[list[float]],
    *,
    iou_threshold: float = DUPLICATE_BOX_IOU_THRESHOLD,
) -> list[list[float]]:
    groups: list[list[list[float]]] = []
    for box in boxes:
        candidate = list(box)
        for group in groups:
            if any(base.box_iou(candidate, existing) >= iou_threshold for existing in group):
                group.append(candidate)
                break
        else:
            groups.append([candidate])

    selected_boxes = [_select_duplicate_box(group) for group in groups]
    selected_boxes.sort(key=lambda box: float(box[4]), reverse=True)
    return selected_boxes


def did_reach_actuation_line(
    previous_box: list[float] | None,
    current_box: list[float],
    *,
    axis: str,
    line_coordinate: float,
    min_axis_motion_px: float = 1.0,
) -> bool:
    if did_cross_reference_line(
        previous_box,
        current_box,
        axis=axis,
        line_coordinate=line_coordinate,
    ):
        return True
    current_value = base.box_center_value(current_box, axis)
    if previous_box is None:
        return current_value >= line_coordinate

    previous_value = base.box_center_value(previous_box, axis)
    axis_delta = current_value - previous_value
    if abs(axis_delta) < min_axis_motion_px:
        return False
    if axis_delta > 0.0:
        return previous_value >= line_coordinate and current_value >= line_coordinate
    return previous_value <= line_coordinate and current_value <= line_coordinate


@dataclass
class CameraObservationSummary:
    observation_count: int = 0
    class_peak_scores: dict[int, float] = field(default_factory=dict)
    best_class_id: int | None = None
    best_score: float | None = None
    first_seen_at: float | None = None
    last_seen_at: float | None = None

    def add(self, class_id: int, confidence: float, timestamp: float) -> None:
        score = float(confidence)
        self.observation_count += 1
        self.class_peak_scores[class_id] = max(
            score,
            float(self.class_peak_scores.get(class_id, 0.0)),
        )
        if self.first_seen_at is None:
            self.first_seen_at = timestamp
        self.last_seen_at = timestamp

        should_replace_best = self.best_score is None or score > self.best_score
        if (
            not should_replace_best
            and self.best_score is not None
            and math.isclose(score, self.best_score, rel_tol=0.0, abs_tol=1e-9)
        ):
            should_replace_best = (
                self.best_class_id != DEFECT_CLASS_ID and class_id == DEFECT_CLASS_ID
            )
        if should_replace_best:
            self.best_class_id = class_id
            self.best_score = score


def copy_camera_observation_summary(
    summary: CameraObservationSummary,
) -> CameraObservationSummary:
    return CameraObservationSummary(
        observation_count=summary.observation_count,
        class_peak_scores=dict(summary.class_peak_scores),
        best_class_id=summary.best_class_id,
        best_score=summary.best_score,
        first_seen_at=summary.first_seen_at,
        last_seen_at=summary.last_seen_at,
    )


def copy_camera_summaries(
    summaries: dict[int, CameraObservationSummary],
) -> dict[int, CameraObservationSummary]:
    return {
        camera_index: copy_camera_observation_summary(summary)
        for camera_index, summary in summaries.items()
    }


@dataclass
class TrackedCap:
    event_id: int
    created_at: float
    last_seen_at: float
    camera_indices: set[int] = field(default_factory=set)
    latest_box_by_camera: dict[int, list[float]] = field(default_factory=dict)
    box_history_by_camera: dict[int, list[list[float]]] = field(default_factory=dict)
    track_ids_by_camera: dict[int, set[int]] = field(default_factory=dict)
    active_track_keys: set[tuple[int, int]] = field(default_factory=set)
    camera_summaries: dict[int, CameraObservationSummary] = field(default_factory=dict)
    anchor_time: float | None = None
    anchor_camera_index: int | None = None
    actuation_time: float | None = None
    actuation_camera_index: int | None = None
    actuation_camera_summaries: dict[int, CameraObservationSummary] = field(default_factory=dict)
    trigger_decision: TrackedCapDecision | None = None
    review_capture_submitted: bool = False
    line_picture_submitted: bool = False
    latest_preview_frame: object | None = None
    latest_raw_frames: list[object] = field(default_factory=list)
    latest_annotated_frames: list[object] = field(default_factory=list)
    observation_log: list[dict] = field(default_factory=list)
    raw_frames_at_actuation: list[object] = field(default_factory=list)
    annotated_frames_at_actuation: list[object] = field(default_factory=list)
    preview_at_actuation: object | None = None
    debug_frame_snapshots: list[FrameSnapshot] = field(default_factory=list)
    post_actuation_snapshot_count: int = 0

    def refresh_actuation_snapshot(self) -> None:
        if self.actuation_time is not None:
            self.actuation_camera_summaries = copy_camera_summaries(self.camera_summaries)

    def add_observation(
        self,
        observation: TrackObservation,
        *,
        timing_camera_index: int,
        anchor_axis: str,
        anchor_line_ratio: float,
    ) -> None:
        track_key = (observation.camera_index, observation.track_id)
        previous_box = self.latest_box_by_camera.get(observation.camera_index)
        camera_was_new = observation.camera_index not in self.camera_indices

        self.created_at = min(self.created_at, observation.timestamp)
        self.last_seen_at = max(self.last_seen_at, observation.timestamp)
        self.camera_indices.add(observation.camera_index)
        self.latest_box_by_camera[observation.camera_index] = list(observation.box)
        box_history = self.box_history_by_camera.setdefault(observation.camera_index, [])
        box_history.append(list(observation.box))
        del box_history[:-2]
        self.track_ids_by_camera.setdefault(observation.camera_index, set()).add(
            observation.track_id
        )
        self.active_track_keys.add(track_key)

        summary = self.camera_summaries.setdefault(
            observation.camera_index,
            CameraObservationSummary(),
        )
        score_value = float(observation.box[4])
        summary.add(observation.class_id, score_value, observation.timestamp)

        class_id_value = int(observation.class_id)
        self.observation_log.append(
            {
                "camera_index": int(observation.camera_index),
                "track_id": int(observation.track_id),
                "timestamp": float(observation.timestamp),
                "box": [float(value) for value in observation.box[:4]],
                "score": score_value,
                "class_id": class_id_value,
                "class_name": class_name(class_id_value),
                "frame_size": [int(observation.frame_size[0]), int(observation.frame_size[1])],
            }
        )

        del timing_camera_index

        if self.actuation_time is not None:
            if camera_was_new:
                self.refresh_actuation_snapshot()
            elif math.isclose(
                observation.timestamp,
                self.actuation_time,
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                self._add_actuation_observation(observation)
            return

        line_coordinate = reference_coordinate(
            observation.frame_size,
            anchor_axis,
            anchor_line_ratio,
        )
        if did_reach_actuation_line(
            previous_box,
            observation.box,
            axis=anchor_axis,
            line_coordinate=line_coordinate,
        ):
            self.anchor_time = observation.timestamp
            self.anchor_camera_index = observation.camera_index
            self.actuation_time = observation.timestamp
            self.actuation_camera_index = observation.camera_index
            self.actuation_camera_summaries = {}
            self._add_actuation_observation(observation)

    def _add_actuation_observation(self, observation: TrackObservation) -> None:
        summary = self.actuation_camera_summaries.setdefault(
            observation.camera_index,
            CameraObservationSummary(),
        )
        summary.add(
            observation.class_id,
            float(observation.box[4]),
            observation.timestamp,
        )

    def close_track(self, camera_index: int, track_id: int) -> None:
        self.active_track_keys.discard((camera_index, track_id))

    def update_review_frames(
        self,
        raw_frames,
        preview_frame,
        annotated_frames=None,
    ) -> None:
        self.latest_raw_frames = copy_frames(raw_frames)
        self.latest_preview_frame = copy_frame(preview_frame)
        if annotated_frames is not None:
            self.latest_annotated_frames = copy_frames(annotated_frames)
        if self.actuation_time is not None and not self.raw_frames_at_actuation:
            self.raw_frames_at_actuation = copy_frames(raw_frames)
            self.preview_at_actuation = copy_frame(preview_frame)
            if annotated_frames is not None:
                self.annotated_frames_at_actuation = copy_frames(annotated_frames)

    def append_debug_frame_snapshot(
        self,
        snapshot: FrameSnapshot,
        *,
        max_snapshots: int | None = None,
    ) -> None:
        self.debug_frame_snapshots.append(copy_frame_snapshot(snapshot))
        if self.actuation_time is not None and snapshot.timestamp > self.actuation_time:
            self.post_actuation_snapshot_count += 1
        if max_snapshots is not None and max_snapshots >= 0:
            max_count = int(max_snapshots)
            if max_count == 0:
                self.debug_frame_snapshots.clear()
            else:
                del self.debug_frame_snapshots[:-max_count]


class TrackedCapManager:
    def __init__(
        self,
        merge_window_seconds: float,
        *,
        camera_count: int,
        timing_camera_index: int,
        anchor_axis: str,
        anchor_line_ratio: float,
        finalize_quiet_seconds: float | None = None,
        same_camera_min_size_ratio: float = 0.5,
    ):
        self.merge_window_seconds = float(merge_window_seconds)
        self.camera_count = int(camera_count)
        self.finalize_quiet_seconds = float(
            finalize_quiet_seconds
            if finalize_quiet_seconds is not None
            else self.merge_window_seconds
        )
        self.timing_camera_index = int(timing_camera_index)
        self.anchor_axis = anchor_axis
        self.anchor_line_ratio = float(anchor_line_ratio)
        self.same_camera_min_size_ratio = float(same_camera_min_size_ratio)
        self._open_caps: list[TrackedCap] = []
        self._track_to_cap: dict[tuple[int, int], TrackedCap] = {}
        self._next_event_id = 1

    def update(
        self,
        observations: list[TrackObservation],
        closed_tracks: list[ClosedTrack],
    ) -> list[TrackedCap]:
        touched_caps: list[TrackedCap] = []
        seen_cap_ids: set[int] = set()

        for observation in observations:
            tracked_cap = self._find_match(observation)
            if tracked_cap is None:
                tracked_cap = TrackedCap(
                    event_id=self._next_event_id,
                    created_at=observation.timestamp,
                    last_seen_at=observation.timestamp,
                )
                self._next_event_id += 1
                self._open_caps.append(tracked_cap)

            tracked_cap.add_observation(
                observation,
                timing_camera_index=self.timing_camera_index,
                anchor_axis=self.anchor_axis,
                anchor_line_ratio=self.anchor_line_ratio,
            )
            self._track_to_cap[(observation.camera_index, observation.track_id)] = tracked_cap
            self._mark_recent(tracked_cap)
            if tracked_cap.event_id not in seen_cap_ids:
                seen_cap_ids.add(tracked_cap.event_id)
                touched_caps.append(tracked_cap)

        for closed_track in closed_tracks:
            track_key = (closed_track.camera_index, closed_track.track_id)
            tracked_cap = self._track_to_cap.pop(track_key, None)
            if tracked_cap is None:
                continue
            tracked_cap.close_track(closed_track.camera_index, closed_track.track_id)
        return touched_caps

    def open_caps(self) -> tuple[TrackedCap, ...]:
        return tuple(self._open_caps)

    def pop_finalized(self, now: float) -> list[TrackedCap]:
        finalized: list[TrackedCap] = []
        remaining: list[TrackedCap] = []
        for tracked_cap in self._open_caps:
            if tracked_cap.active_track_keys:
                remaining.append(tracked_cap)
                continue
            quiet_seconds = self.finalize_quiet_seconds
            if len(tracked_cap.camera_indices) < self.camera_count:
                quiet_seconds = max(quiet_seconds, self.merge_window_seconds)
            if (now - tracked_cap.last_seen_at) < quiet_seconds:
                remaining.append(tracked_cap)
                continue
            finalized.append(tracked_cap)
        self._open_caps = remaining
        return finalized

    def _find_match(self, observation: TrackObservation) -> TrackedCap | None:
        direct_match = self._track_to_cap.get((observation.camera_index, observation.track_id))
        if direct_match is not None:
            return direct_match

        for tracked_cap in reversed(self._open_caps):
            if observation.camera_index in tracked_cap.camera_indices:
                if base.boxes_follow_same_cap_trajectory(
                    observation.box,
                    tracked_cap.box_history_by_camera.get(
                        observation.camera_index,
                        [],
                    ),
                    axis=self.anchor_axis,
                    min_size_ratio=self.same_camera_min_size_ratio,
                ):
                    return tracked_cap
                continue

        cross_camera_candidates: list[TrackedCap] = []
        for tracked_cap in self._open_caps:
            if observation.camera_index in tracked_cap.camera_indices:
                continue
            if (observation.timestamp - tracked_cap.last_seen_at) > self.merge_window_seconds:
                continue
            cross_camera_candidates.append(tracked_cap)

        if not cross_camera_candidates:
            return None

        handoff_candidates = [
            tracked_cap
            for tracked_cap in cross_camera_candidates
            if tracked_cap.camera_indices
        ]
        if handoff_candidates:
            return max(handoff_candidates, key=lambda tracked_cap: tracked_cap.last_seen_at)
        return max(cross_camera_candidates, key=lambda tracked_cap: tracked_cap.last_seen_at)

    def _mark_recent(self, tracked_cap: TrackedCap) -> None:
        try:
            self._open_caps.remove(tracked_cap)
        except ValueError:
            pass
        self._open_caps.append(tracked_cap)


def build_debug_frame_burst(
    tracked_cap: TrackedCap,
    *,
    before_frames: int,
    after_frames: int,
) -> list[FrameSnapshot]:
    before_count = max(0, int(before_frames))
    after_count = max(0, int(after_frames))
    snapshots = tracked_cap.debug_frame_snapshots
    if tracked_cap.actuation_time is None:
        selected = snapshots[-before_count:] if before_count else []
        return [copy_frame_snapshot(snapshot) for snapshot in selected]

    actuation_time = float(tracked_cap.actuation_time)
    before_candidates = [
        snapshot for snapshot in snapshots if snapshot.timestamp <= actuation_time
    ]
    before_or_at = before_candidates[-before_count:] if before_count else []
    after = [
        snapshot for snapshot in snapshots if snapshot.timestamp > actuation_time
    ][:after_count]
    return [copy_frame_snapshot(snapshot) for snapshot in [*before_or_at, *after]]


def build_debug_frame_snapshots(
    tracked_cap: TrackedCap,
    *,
    clock: RuntimeClock,
    before_frames: int,
    after_frames: int,
) -> list[DebugFrameSnapshot]:
    burst = build_debug_frame_burst(
        tracked_cap,
        before_frames=before_frames,
        after_frames=after_frames,
    )
    debug_snapshots: list[DebugFrameSnapshot] = []
    for snapshot in burst:
        offset_ms = (
            None
            if tracked_cap.actuation_time is None
            else (snapshot.timestamp - tracked_cap.actuation_time) * 1000.0
        )
        debug_snapshots.append(
            DebugFrameSnapshot(
                frame_index=int(snapshot.frame_index),
                timestamp_monotonic=float(snapshot.timestamp),
                timestamp_iso=clock.format(snapshot.timestamp),
                offset_from_actuation_ms=offset_ms,
                raw_frames=copy_frames(snapshot.raw_frames),
                annotated_frames=copy_frames(snapshot.annotated_frames),
                boxes_by_camera=copy_boxes_by_camera(snapshot.boxes_by_camera),
                read_duration_ms=snapshot.read_duration_ms,
                frame_interval_ms=snapshot.frame_interval_ms,
                inference_ms_by_camera=[
                    float(value) for value in snapshot.inference_ms_by_camera
                ],
                processing_duration_ms=snapshot.processing_duration_ms,
                pair_skew_ms=snapshot.pair_skew_ms,
                pair_sequences=[int(sequence) for sequence in snapshot.pair_sequences],
            )
        )
    return debug_snapshots


def build_camera_vote(summary: CameraObservationSummary | None, *, camera_index: int) -> CameraVote:
    if (
        summary is None
        or summary.observation_count <= 0
        or summary.best_class_id is None
        or summary.best_score is None
    ):
        return CameraVote(
            camera_index=camera_index,
            class_id=None,
            class_name=None,
            score=None,
            observation_count=0 if summary is None else summary.observation_count,
            first_seen_at=None if summary is None else summary.first_seen_at,
            last_seen_at=None if summary is None else summary.last_seen_at,
        )

    return CameraVote(
        camera_index=camera_index,
        class_id=summary.best_class_id,
        class_name=class_name(summary.best_class_id),
        score=summary.best_score,
        observation_count=summary.observation_count,
        first_seen_at=summary.first_seen_at,
        last_seen_at=summary.last_seen_at,
    )


def build_cap_evaluation(
    tracked_cap: TrackedCap,
    *,
    camera_count: int,
) -> CapEvaluation:
    return build_cap_evaluation_from_summaries(
        tracked_cap.camera_summaries,
        camera_count=camera_count,
    )


def build_actuation_evaluation(
    tracked_cap: TrackedCap,
    *,
    camera_count: int,
) -> CapEvaluation | None:
    if tracked_cap.actuation_time is None:
        return None
    return build_cap_evaluation_from_summaries(
        tracked_cap.actuation_camera_summaries,
        camera_count=camera_count,
    )


def build_cap_evaluation_from_summaries(
    camera_summaries: dict[int, CameraObservationSummary],
    *,
    camera_count: int,
) -> CapEvaluation:
    camera_votes = {
        camera_index: build_camera_vote(
            camera_summaries.get(camera_index),
            camera_index=camera_index,
        )
        for camera_index in range(camera_count)
    }

    total_observations = 0
    class_scores = {class_id: 0.0 for class_id in range(len(CLASS_NAMES))}
    for summary in camera_summaries.values():
        total_observations += summary.observation_count
        for class_id, score in summary.class_peak_scores.items():
            if 0 <= class_id < len(CLASS_NAMES):
                class_scores[class_id] = max(class_scores[class_id], float(score))

    usable_camera_votes = {
        camera_index: vote
        for camera_index, vote in camera_votes.items()
        if vote.class_id is not None
    }
    return CapEvaluation(
        total_observations=total_observations,
        class_scores=class_scores,
        camera_votes=camera_votes,
        usable_camera_votes=usable_camera_votes,
    )


def is_merge_complete(
    tracked_cap: TrackedCap,
    *,
    camera_count: int,
    decision_ready_time: float,
    merge_window_seconds: float,
) -> bool:
    return len(tracked_cap.camera_indices) >= camera_count or (
        decision_ready_time - tracked_cap.last_seen_at
    ) >= merge_window_seconds


def maybe_build_trigger_decision(
    tracked_cap: TrackedCap,
    *,
    evaluation: CapEvaluation,
    timing_camera_index: int,
    decision_ready_time: float,
    camera_count: int,
    merge_window_seconds: float,
    nozzle_distance_mm: float,
    belt_speed_mm_per_s: float,
    trigger_offset_s: float,
    latency_compensation_ms: float,
    reject_threshold: float = DEFECT_REJECT_THRESHOLD,
) -> TrackedCapDecision | None:
    if evaluation.total_observations <= 0 or evaluation.dirt_score < float(reject_threshold):
        return None

    if not is_merge_complete(
        tracked_cap,
        camera_count=camera_count,
        decision_ready_time=decision_ready_time,
        merge_window_seconds=merge_window_seconds,
    ):
        return None

    anchor_time, anchor_source = resolve_anchor_time(
        tracked_cap,
        timing_camera_index=timing_camera_index,
    )
    return build_tracked_cap_decision(
        result="trigger",
        final_class_name=class_name(DEFECT_CLASS_ID),
        final_score=evaluation.dirt_score,
        decision_source="highest_defect_threshold",
        evaluation=evaluation,
        anchor_time=anchor_time,
        anchor_source=anchor_source,
        decision_ready_time=decision_ready_time,
        nozzle_distance_mm=nozzle_distance_mm,
        belt_speed_mm_per_s=belt_speed_mm_per_s,
        trigger_offset_s=trigger_offset_s,
        latency_compensation_ms=latency_compensation_ms,
        review_reason="trigger",
    )


def decide_decision_ready(
    tracked_cap: TrackedCap,
    *,
    camera_count: int,
    timing_camera_index: int,
    decision_ready_time: float,
    merge_window_seconds: float,
    nozzle_distance_mm: float,
    belt_speed_mm_per_s: float,
    trigger_offset_s: float,
    latency_compensation_ms: float,
    reject_threshold: float = DEFECT_REJECT_THRESHOLD,
) -> TrackedCapDecision | None:
    if tracked_cap.trigger_decision is not None:
        return tracked_cap.trigger_decision

    evaluation = build_actuation_evaluation(
        tracked_cap,
        camera_count=camera_count,
    )
    if evaluation is None:
        return None

    return maybe_build_trigger_decision(
        tracked_cap,
        evaluation=evaluation,
        timing_camera_index=timing_camera_index,
        decision_ready_time=decision_ready_time,
        camera_count=camera_count,
        merge_window_seconds=merge_window_seconds,
        nozzle_distance_mm=nozzle_distance_mm,
        belt_speed_mm_per_s=belt_speed_mm_per_s,
        trigger_offset_s=trigger_offset_s,
        latency_compensation_ms=latency_compensation_ms,
        reject_threshold=reject_threshold,
    )


def decide_tracked_cap(
    tracked_cap: TrackedCap,
    *,
    camera_count: int,
    timing_camera_index: int,
    decision_time: float,
    merge_window_seconds: float,
    nozzle_distance_mm: float,
    belt_speed_mm_per_s: float,
    trigger_offset_s: float,
    latency_compensation_ms: float,
    reject_threshold: float = DEFECT_REJECT_THRESHOLD,
) -> TrackedCapDecision:
    if tracked_cap.trigger_decision is not None:
        return tracked_cap.trigger_decision

    evaluation = build_cap_evaluation(
        tracked_cap,
        camera_count=camera_count,
    )
    anchor_time, anchor_source = resolve_anchor_time(
        tracked_cap,
        timing_camera_index=timing_camera_index,
    )
    if evaluation.total_observations <= 0:
        return build_tracked_cap_decision(
            result="skip",
            final_class_name=None,
            final_score=None,
            decision_source="no_observations",
            evaluation=evaluation,
            anchor_time=anchor_time,
            anchor_source=anchor_source,
            decision_ready_time=decision_time,
            nozzle_distance_mm=nozzle_distance_mm,
            belt_speed_mm_per_s=belt_speed_mm_per_s,
            trigger_offset_s=trigger_offset_s,
            latency_compensation_ms=latency_compensation_ms,
        )

    actuation_evaluation = build_actuation_evaluation(
        tracked_cap,
        camera_count=camera_count,
    )
    if actuation_evaluation is None:
        final_class_name = class_name(DEFECT_CLASS_ID) if evaluation.dirt_score else class_name(0)
        review_reason = (
            "missed_actuation"
            if evaluation.dirt_score >= float(reject_threshold)
            else None
        )
        return build_tracked_cap_decision(
            result="skip",
            final_class_name=final_class_name,
            final_score=evaluation.dirt_score,
            decision_source="no_actuation_crossing",
            evaluation=evaluation,
            anchor_time=anchor_time,
            anchor_source=anchor_source,
            decision_ready_time=decision_time,
            nozzle_distance_mm=nozzle_distance_mm,
            belt_speed_mm_per_s=belt_speed_mm_per_s,
            trigger_offset_s=trigger_offset_s,
            latency_compensation_ms=latency_compensation_ms,
            review_reason=review_reason,
        )

    trigger_decision = maybe_build_trigger_decision(
        tracked_cap,
        evaluation=actuation_evaluation,
        timing_camera_index=timing_camera_index,
        decision_ready_time=decision_time,
        camera_count=camera_count,
        merge_window_seconds=merge_window_seconds,
        nozzle_distance_mm=nozzle_distance_mm,
        belt_speed_mm_per_s=belt_speed_mm_per_s,
        trigger_offset_s=trigger_offset_s,
        latency_compensation_ms=latency_compensation_ms,
        reject_threshold=reject_threshold,
    )
    if trigger_decision is not None:
        return trigger_decision

    review_reason = (
        "dirty_before_clean_actuation"
        if evaluation.dirt_score >= float(reject_threshold)
        and actuation_evaluation.dirt_score < float(reject_threshold)
        else None
    )
    return build_tracked_cap_decision(
        result="skip",
        final_class_name=class_name(0),
        final_score=actuation_evaluation.dirt_score,
        decision_source="defect_below_threshold_at_actuation",
        evaluation=actuation_evaluation,
        anchor_time=anchor_time,
        anchor_source=anchor_source,
        decision_ready_time=decision_time,
        nozzle_distance_mm=nozzle_distance_mm,
        belt_speed_mm_per_s=belt_speed_mm_per_s,
        trigger_offset_s=trigger_offset_s,
        latency_compensation_ms=latency_compensation_ms,
        review_reason=review_reason,
    )


def _summary_to_payload(
    summary: CameraObservationSummary | None,
    *,
    clock: RuntimeClock,
) -> dict:
    if summary is None:
        return {
            "observation_count": 0,
            "first_seen_at_iso": None,
            "last_seen_at_iso": None,
            "class_peak_scores": {},
            "best_class_id": None,
            "best_class_name": None,
            "best_score": None,
        }
    return {
        "observation_count": int(summary.observation_count),
        "first_seen_at_iso": (
            clock.format(summary.first_seen_at) if summary.first_seen_at is not None else None
        ),
        "last_seen_at_iso": (
            clock.format(summary.last_seen_at) if summary.last_seen_at is not None else None
        ),
        "class_peak_scores": {
            str(class_id): round_float(float(score), 6)
            for class_id, score in summary.class_peak_scores.items()
        },
        "best_class_id": (
            None if summary.best_class_id is None else int(summary.best_class_id)
        ),
        "best_class_name": (
            None if summary.best_class_id is None else class_name(int(summary.best_class_id))
        ),
        "best_score": (
            None if summary.best_score is None else round_float(float(summary.best_score), 6)
        ),
    }


def _build_observation_log_payload(
    tracked_cap: TrackedCap,
    *,
    clock: RuntimeClock,
) -> list[dict]:
    observations: list[dict] = []
    for entry in tracked_cap.observation_log:
        observations.append(
            {
                "camera_index": entry["camera_index"],
                "track_id": entry["track_id"],
                "timestamp_monotonic": round_float(entry["timestamp"], 6),
                "timestamp_iso": clock.format(entry["timestamp"]),
                "box": [round_float(value, 3) for value in entry["box"]],
                "score": round_float(entry["score"], 6),
                "class_id": entry["class_id"],
                "class_name": entry["class_name"],
                "frame_size": entry["frame_size"],
            }
        )
    return observations


def _build_event_json_payload(
    tracked_cap: TrackedCap,
    decision: TrackedCapDecision,
    *,
    clock: RuntimeClock,
    model_path: str,
    args: argparse.Namespace,
    evaluation: CapEvaluation,
    cam_list: list,
    camera_properties: list[CameraProperties] | None = None,
    performance_snapshot: RuntimePerformanceSnapshot | None = None,
) -> dict:
    actuation_evaluation = build_actuation_evaluation(
        tracked_cap, camera_count=max(len(decision.camera_votes), 2)
    )
    track_ids_serial = {
        str(camera_index): sorted(int(track_id) for track_id in track_ids)
        for camera_index, track_ids in tracked_cap.track_ids_by_camera.items()
    }
    camera_summaries_payload = {
        str(camera_index): _summary_to_payload(summary, clock=clock)
        for camera_index, summary in tracked_cap.camera_summaries.items()
    }
    camera_summaries_at_actuation_payload = (
        {
            str(camera_index): _summary_to_payload(summary, clock=clock)
            for camera_index, summary in tracked_cap.actuation_camera_summaries.items()
        }
        if tracked_cap.actuation_camera_summaries
        else None
    )
    camera_votes_payload = {
        str(camera_index): camera_vote_payload(vote, clock)
        for camera_index, vote in decision.camera_votes.items()
    }
    physical_delay_s = calculate_trigger_delay(
        float(args.nozzle_distance_mm),
        float(args.belt_speed_mm_per_s),
        float(args.trigger_offset_s),
    )
    requested_delay_s = compute_requested_trigger_delay(
        float(args.nozzle_distance_mm),
        float(args.belt_speed_mm_per_s),
        float(args.trigger_offset_s),
        float(args.latency_compensation_ms),
    )

    return {
        "event_id": int(tracked_cap.event_id),
        "recorded_at": clock.format(decision.decision_ready_time),
        "result": decision.result,
        "review_reason": decision.review_reason,
        "decision_source": decision.decision_source,
        "final_class": decision.final_class_name,
        "final_score": (
            None if decision.final_score is None else round_float(decision.final_score, 6)
        ),
        "model_path": model_path,
        "cameras": [str(label) for label in cam_list],
        "decision": {
            "anchor_time_iso": clock.format(decision.anchor_time),
            "anchor_source": decision.anchor_source,
            "decision_ready_time_iso": clock.format(decision.decision_ready_time),
            "requested_fire_time_iso": clock.format(decision.requested_fire_time),
            "computed_trigger_delay_s": round_float(decision.trigger_delay_s, 6),
            "anchor_to_requested_ms": round_float(
                (decision.requested_fire_time - decision.anchor_time) * 1000.0, 3
            )
            if decision.anchor_time is not None
            else None,
            "anchor_to_decision_ready_ms": round_float(
                (decision.decision_ready_time - decision.anchor_time) * 1000.0, 3
            )
            if decision.anchor_time is not None
            else None,
        },
        "trigger_formula": {
            "nozzle_distance_mm": round_float(float(args.nozzle_distance_mm), 3),
            "belt_speed_mm_per_s": round_float(float(args.belt_speed_mm_per_s), 3),
            "trigger_offset_s": round_float(float(args.trigger_offset_s), 6),
            "latency_compensation_ms": round_float(float(args.latency_compensation_ms), 3),
            "physical_delay_s": round_float(physical_delay_s, 6),
            "requested_delay_s": round_float(requested_delay_s, 6),
        },
        "tracked_cap": {
            "created_at_iso": clock.format(tracked_cap.created_at),
            "last_seen_at_iso": clock.format(tracked_cap.last_seen_at),
            "camera_indices": sorted(int(idx) for idx in tracked_cap.camera_indices),
            "track_ids_by_camera": track_ids_serial,
            "anchor_camera_index": tracked_cap.anchor_camera_index,
            "actuation_camera_index": tracked_cap.actuation_camera_index,
            "actuation_time_iso": (
                clock.format(tracked_cap.actuation_time)
                if tracked_cap.actuation_time is not None
                else None
            ),
        },
        "evaluation": {
            "total_observations": int(evaluation.total_observations),
            "class_scores": {
                str(class_id): round_float(float(score), 6)
                for class_id, score in evaluation.class_scores.items()
            },
            "dirt_score": round_float(float(evaluation.dirt_score), 6),
            "undefected_score": round_float(float(evaluation.undefected_score), 6),
        },
        "evaluation_at_actuation": (
            None
            if actuation_evaluation is None
            else {
                "total_observations": int(actuation_evaluation.total_observations),
                "class_scores": {
                    str(class_id): round_float(float(score), 6)
                    for class_id, score in actuation_evaluation.class_scores.items()
                },
                "dirt_score": round_float(float(actuation_evaluation.dirt_score), 6),
                "undefected_score": round_float(
                    float(actuation_evaluation.undefected_score), 6
                ),
            }
        ),
        "camera_summaries_at_decision": camera_summaries_payload,
        "camera_summaries_at_actuation": camera_summaries_at_actuation_payload,
        "camera_votes": camera_votes_payload,
        "observations": _build_observation_log_payload(tracked_cap, clock=clock),
        "settings": {
            "merge_window_ms": float(args.merge_window_ms),
            "finalize_quiet_ms": float(args.finalize_quiet_ms),
            "timing_camera_index": int(args.timing_camera),
            "anchor_axis": args.anchor_axis,
            "anchor_line_ratio": float(args.anchor_line_ratio),
            "tracking_detection_threshold": float(
                getattr(args, "tracking_threshold", TRACKING_DETECTION_THRESHOLD)
            ),
            "defect_reject_threshold": float(
                getattr(args, "reject_threshold", DEFECT_REJECT_THRESHOLD)
            ),
            "global_detection_threshold": float(
                getattr(args, "global_threshold", GLOBAL_DETECTION_THRESHOLD)
            ),
            "duplicate_box_iou_threshold": DUPLICATE_BOX_IOU_THRESHOLD,
            "track_iou": float(args.track_iou),
            "max_missing_frames": int(args.max_missing_frames),
            "pair_max_skew_ms": float(
                getattr(args, "pair_max_skew_ms", DEFAULT_PAIR_MAX_SKEW_MS)
            ),
            "debug_burst_before_frames": int(
                getattr(args, "debug_burst_before_frames", DEFAULT_DEBUG_BURST_BEFORE_FRAMES)
            ),
            "debug_burst_after_frames": int(
                getattr(args, "debug_burst_after_frames", DEFAULT_DEBUG_BURST_AFTER_FRAMES)
            ),
        },
        "runtime": {
            "camera_properties": [
                camera_properties_payload(properties)
                for properties in (camera_properties or [])
            ],
            "performance": performance_snapshot_payload(performance_snapshot),
        },
        "artifacts": {},
    }


def submit_review_capture(
    writer: DebugCaptureWriter,
    tracked_cap: TrackedCap,
    decision: TrackedCapDecision,
    *,
    clock: RuntimeClock,
    model_path: str,
    args: argparse.Namespace,
    cam_list: list,
    camera_properties: list[CameraProperties] | None = None,
    performance_snapshot: RuntimePerformanceSnapshot | None = None,
) -> None:
    if tracked_cap.review_capture_submitted or decision.review_reason is None:
        return

    evaluation = build_cap_evaluation(
        tracked_cap,
        camera_count=max(len(decision.camera_votes), 2),
    )

    preview_for_debug = (
        tracked_cap.preview_at_actuation
        if tracked_cap.preview_at_actuation is not None
        else tracked_cap.latest_preview_frame
    )
    annotated_for_debug = (
        tracked_cap.annotated_frames_at_actuation
        if tracked_cap.annotated_frames_at_actuation
        else tracked_cap.latest_annotated_frames
    )
    raw_for_pictures = (
        tracked_cap.raw_frames_at_actuation
        if tracked_cap.raw_frames_at_actuation
        else tracked_cap.latest_raw_frames
    )

    json_payload = _build_event_json_payload(
        tracked_cap,
        decision,
        clock=clock,
        model_path=model_path,
        args=args,
        evaluation=evaluation,
        cam_list=cam_list,
        camera_properties=camera_properties,
        performance_snapshot=performance_snapshot,
    )
    frame_burst = build_debug_frame_snapshots(
        tracked_cap,
        clock=clock,
        before_frames=getattr(
            args,
            "debug_burst_before_frames",
            DEFAULT_DEBUG_BURST_BEFORE_FRAMES,
        ),
        after_frames=getattr(
            args,
            "debug_burst_after_frames",
            DEFAULT_DEBUG_BURST_AFTER_FRAMES,
        ),
    )

    writer.submit(
        DebugCaptureTask(
            event_id=tracked_cap.event_id,
            recorded_at=clock.format(decision.decision_ready_time),
            result=decision.result,
            review_reason=decision.review_reason,
            decision_source=decision.decision_source,
            final_class=decision.final_class_name,
            final_score=decision.final_score,
            score_summary=format_class_scores(evaluation.class_scores),
            cam0_vote=format_camera_vote(decision.camera_votes.get(0)) or "",
            cam1_vote=format_camera_vote(decision.camera_votes.get(1)) or "",
            model_path=model_path,
            preview_frame=copy_frame(preview_for_debug),
            annotated_frames=copy_frames(annotated_for_debug),
            raw_frames=copy_frames(raw_for_pictures),
            frame_burst=frame_burst,
            json_payload=json_payload,
        )
    )
    tracked_cap.review_capture_submitted = True


def complete_paired_frames(
    frames: list[object],
    *,
    camera_count: int = 2,
) -> list[object]:
    if len(frames) < camera_count:
        return []
    paired_frames = list(frames[:camera_count])
    if any(frame is None for frame in paired_frames):
        return []
    return copy_frames(paired_frames)


def submit_line_picture_capture(
    writer: DebugCaptureWriter,
    tracked_cap: TrackedCap,
    decision: TrackedCapDecision,
    *,
    clock: RuntimeClock,
    args: argparse.Namespace,
) -> None:
    if tracked_cap.line_picture_submitted or tracked_cap.actuation_time is None:
        return

    raw_frames = complete_paired_frames(tracked_cap.raw_frames_at_actuation)
    if not raw_frames:
        raw_frames = complete_paired_frames(tracked_cap.latest_raw_frames)
    if not raw_frames:
        log_fn = getattr(writer, "_log", None)
        if callable(log_fn):
            log_fn(
                f"[PICTURE][WARN] event={tracked_cap.event_id} skipped paired save; "
                "missing complete cam0/cam1 raw frames"
            )
        return

    submit_line_picture = getattr(writer, "submit_line_picture", None)
    if submit_line_picture is None:
        return

    submit_line_picture(
        LinePictureTask(
            event_id=tracked_cap.event_id,
            recorded_at=clock.format(tracked_cap.actuation_time),
            result=decision.result,
            final_class=decision.final_class_name,
            final_score=decision.final_score,
            decision_source=decision.decision_source,
            cam0_vote=format_camera_vote(decision.camera_votes.get(0)) or "",
            cam1_vote=format_camera_vote(decision.camera_votes.get(1)) or "",
            raw_frames=raw_frames,
            anchor_axis=args.anchor_axis,
            anchor_line_ratio=float(args.anchor_line_ratio),
        )
    )
    tracked_cap.line_picture_submitted = True


def validate_args(args: argparse.Namespace) -> None:
    if len(args.cams) != 2:
        raise ValueError("Exactly two cameras are required for this runtime")
    if not 0.0 <= args.tracking_threshold <= 1.0:
        raise ValueError("--tracking-threshold must be between 0 and 1")
    if not 0.0 <= args.reject_threshold <= 1.0:
        raise ValueError("--reject-threshold must be between 0 and 1")
    if not 0.0 <= args.track_iou <= 1.0:
        raise ValueError("--track-iou must be between 0 and 1")
    if args.max_missing_frames < 0:
        raise ValueError("--max-missing-frames must be 0 or greater")
    if args.merge_window_ms < 0:
        raise ValueError("--merge-window-ms must be 0 or greater")
    if args.finalize_quiet_ms < 0:
        raise ValueError("--finalize-quiet-ms must be 0 or greater")
    if args.trigger_duration <= 0:
        raise ValueError("--trigger-duration must be greater than 0")
    if args.trigger_min_gap < 0:
        raise ValueError("--trigger-min-gap must be 0 or greater")
    if args.timing_camera not in (0, 1):
        raise ValueError("--timing-camera must be 0 or 1")
    if args.anchor_axis not in {"x", "y"}:
        raise ValueError("--anchor-axis must be x or y")
    if not 0.0 <= args.anchor_line_ratio <= 1.0:
        raise ValueError("--anchor-line-ratio must be between 0 and 1")
    if args.nozzle_distance_mm < 0:
        raise ValueError("--nozzle-distance-mm must be 0 or greater")
    if args.belt_speed_mm_per_s <= 0:
        raise ValueError("--belt-speed-mm-per-s must be greater than 0")
    if args.latency_compensation_ms < 0:
        raise ValueError("--latency-compensation-ms must be 0 or greater")
    if args.debug_burst_before_frames < 0:
        raise ValueError("--debug-burst-before-frames must be 0 or greater")
    if args.debug_burst_after_frames < 0:
        raise ValueError("--debug-burst-after-frames must be 0 or greater")
    if args.onnx_intra_op_threads < 1:
        raise ValueError("--onnx-intra-op-threads must be 1 or greater")
    if args.perf_log_interval_s < 0:
        raise ValueError("--perf-log-interval-s must be 0 or greater")
    if args.pair_max_skew_ms < 0:
        raise ValueError("--pair-max-skew-ms must be 0 or greater")
    if args.save_queue_warning_threshold < 0:
        raise ValueError("--save-queue-warning-threshold must be 0 or greater")
    args.pixel_format = normalize_camera_pixel_format(args.pixel_format)
    if args.pixel_format != DEFAULT_CAMERA_PIXEL_FORMAT:
        raise ValueError(
            f"--pixel-format must be {DEFAULT_CAMERA_PIXEL_FORMAT} for Arducam B0495 cameras"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run two-class cap detection with separate tracking/reject thresholds, "
            "debug burst capture, timing review logging, and GPIO triggering."
        )
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="model filename or path to a two-class .onnx file",
    )
    parser.add_argument(
        "--cams",
        nargs=2,
        default=["0", "1"],
        help="two camera indices or device paths",
    )
    parser.add_argument(
        "--res",
        type=int,
        nargs=2,
        default=list(DEFAULT_CAMERA_RESOLUTION),
        help="capture W H",
    )
    parser.add_argument("--fps", type=int, default=DEFAULT_CAMERA_FPS)
    parser.add_argument(
        "--pixel-format",
        default=DEFAULT_CAMERA_PIXEL_FORMAT,
        help="V4L2 pixel format to force on the camera hardware (default: YUYV)",
    )
    parser.add_argument(
        "--exposure",
        type=int,
        default=8,
        help="exposure_time_absolute (default=8)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="override inference image size; by default it matches the model input",
    )
    parser.add_argument("--no-display", action="store_true", help="run without OpenCV GUI")
    parser.add_argument(
        "--tracking-threshold",
        type=float,
        default=TRACKING_DETECTION_THRESHOLD,
        help="minimum model score used for tracking detections",
    )
    parser.add_argument(
        "--reject-threshold",
        type=float,
        default=DEFECT_REJECT_THRESHOLD,
        help="minimum dirt_defect score at actuation required to trigger reject",
    )
    parser.add_argument(
        "--trigger-pin",
        default=DEFAULT_TRIGGER_PIN,
        help=(
            "Jetson.GPIO output pin to pulse for reject actuation "
            f"(default: {DEFAULT_TRIGGER_PIN}; GPIO-09 uses CVM naming)"
        ),
    )
    parser.add_argument("--trigger-duration", type=float, default=0.3)
    parser.add_argument("--trigger-min-gap", type=float, default=0.0)
    parser.add_argument("--track-iou", type=float, default=0.3)
    parser.add_argument("--max-missing-frames", type=int, default=1)
    parser.add_argument("--merge-window-ms", type=float, default=150.0)
    parser.add_argument("--finalize-quiet-ms", type=float, default=DEFAULT_FINALIZE_QUIET_MS)
    parser.add_argument("--timing-camera", type=int, default=0)
    parser.add_argument("--anchor-axis", choices=["x", "y"], default="x")
    parser.add_argument("--anchor-line-ratio", type=float, default=0.5)
    parser.add_argument(
        "--nozzle-distance-mm",
        type=float,
        default=DEFAULT_NOZZLE_DISTANCE_MM,
    )
    parser.add_argument(
        "--belt-speed-mm-per-s",
        type=float,
        default=DEFAULT_BELT_SPEED_MM_PER_S,
    )
    parser.add_argument("--trigger-offset-s", type=float, default=DEFAULT_TRIGGER_OFFSET_S)
    parser.add_argument(
        "--latency-compensation-ms",
        type=float,
        default=DEFAULT_LATENCY_COMPENSATION_MS,
    )
    parser.add_argument(
        "--serial-inference",
        action="store_true",
        help="run camera inference serially; useful as a rollback/debug mode",
    )
    parser.add_argument(
        "--onnx-intra-op-threads",
        type=int,
        default=DEFAULT_ONNX_INTRA_OP_THREADS,
        help="CPU threads each ONNX session may use",
    )
    parser.add_argument(
        "--perf-log-interval-s",
        type=float,
        default=DEFAULT_PERF_LOG_INTERVAL_S,
        help="seconds between processed-frame performance summaries; 0 disables summaries",
    )
    parser.add_argument(
        "--pair-max-skew-ms",
        type=float,
        default=DEFAULT_PAIR_MAX_SKEW_MS,
        help="maximum timestamp difference allowed between cam0 and cam1 frames",
    )
    parser.add_argument(
        "--debug-burst-before-frames",
        type=int,
        default=DEFAULT_DEBUG_BURST_BEFORE_FRAMES,
        help="number of at-or-before-actuation frames to save in each debug burst",
    )
    parser.add_argument(
        "--debug-burst-after-frames",
        type=int,
        default=DEFAULT_DEBUG_BURST_AFTER_FRAMES,
        help="number of post-actuation frames to save before trigger debug capture is written",
    )
    parser.add_argument("--timing-log-dir", default=DEFAULT_TIMING_LOG_DIR)
    parser.add_argument(
        "--review-dir",
        default=DEFAULT_DEBUG_DIR,
        help="alias of --debug-dir; kept for backward compatibility",
    )
    parser.add_argument(
        "--debug-dir",
        default=None,
        help=(
            "destination for per-event debug artifacts "
            "(JSON, annotated previews, manifest). Defaults to resources/debugging/."
        ),
    )
    parser.add_argument(
        "--pictures-dir",
        default=DEFAULT_PICTURES_DIR,
        help="destination for raw cap frames (no annotations) for model training",
    )
    parser.add_argument(
        "--session-log-dir",
        default=DEFAULT_SESSION_LOG_DIR,
        help="destination for per-run session log files (console mirrors)",
    )
    parser.add_argument(
        "--simulate-gpio",
        action="store_true",
        help="use a no-op GPIO pin implementation for development",
    )
    parser.add_argument(
        "--save-queue-warning-threshold",
        type=int,
        default=DEFAULT_SAVE_QUEUE_WARNING_THRESHOLD,
        help="warn when debug/picture save queue reaches this backlog; 0 disables",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_arg_parser()
    return parser.parse_args(argv)


def run_detection(
    args: argparse.Namespace,
    *,
    stop_event=None,
    preview_callback: Callable[[object], None] | None = None,
    history_callback: Callable[[DetectionHistoryRecord], None] | None = None,
    timing_log_callback: Callable[[TimingLogRecord], None] | None = None,
    performance_callback: Callable[[RuntimePerformanceSnapshot], None] | None = None,
    log_fn: Callable[..., None] = print,
    pin_factory=GPIOOutputPin,
    time_fn: Callable[[], float] = time.monotonic,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> None:
    import cv2
    import onnxruntime as ort

    validate_args(args)
    clock = RuntimeClock(time_fn=time_fn)
    model_path, preset_imgsz = resolve_model_path(args.model)

    debug_dir = getattr(args, "debug_dir", None) or getattr(
        args, "review_dir", DEFAULT_DEBUG_DIR
    )
    pictures_dir = getattr(args, "pictures_dir", DEFAULT_PICTURES_DIR)
    session_log_dir = getattr(args, "session_log_dir", DEFAULT_SESSION_LOG_DIR)
    session_log_path = os.path.join(
        session_log_dir,
        f"session_{datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')}.log",
    )
    try:
        log_fn = SessionLogTee(log_fn, session_log_path)
    except OSError:
        pass

    timing_logger = TimingCsvLogger(args.timing_log_dir)
    review_writer = ReviewCaptureWriter(
        debug_dir,
        pictures_dir=pictures_dir,
        queue_warning_threshold=args.save_queue_warning_threshold,
        log_fn=log_fn,
    )
    merge_window_seconds = args.merge_window_ms / 1000.0
    finalize_quiet_seconds = args.finalize_quiet_ms / 1000.0

    camera_sources, device_paths = parse_cameras(args.cams)
    trackers = [
        CameraLifecycleTracker(args.track_iou, args.max_missing_frames)
        for _ in camera_sources
    ]
    cap_manager = TrackedCapManager(
        merge_window_seconds,
        camera_count=len(camera_sources),
        timing_camera_index=args.timing_camera,
        anchor_axis=args.anchor_axis,
        anchor_line_ratio=args.anchor_line_ratio,
        finalize_quiet_seconds=finalize_quiet_seconds,
    )

    active_pin_factory = NullGPIOOutputPin if args.simulate_gpio else pin_factory
    scheduler = RejectScheduler(
        trigger_pin=args.trigger_pin,
        trigger_duration=args.trigger_duration,
        trigger_min_gap=args.trigger_min_gap,
        pin_factory=active_pin_factory,
        log_fn=log_fn,
        time_fn=time_fn,
        sleep_fn=sleep_fn,
    )

    cameras = []
    camera_readers: list[LatestFrameCameraReader] = []
    camera_properties: list[CameraProperties] = []
    inference_executor: concurrent.futures.ThreadPoolExecutor | None = None
    display_opened = False
    show_opencv_preview = not args.no_display

    try:
        width, height = args.res
        for device_path in device_paths:
            set_camera_format(
                device_path,
                width,
                height,
                args.fps,
                pixel_format=args.pixel_format,
                log_fn=log_fn,
            )
            set_camera_controls(device_path, args.exposure, log_fn=log_fn)

        for camera_index, camera_source in enumerate(camera_sources):
            camera = open_cam(
                camera_source,
                width,
                height,
                args.fps,
                args.pixel_format,
            )
            cameras.append(camera)
            properties = read_camera_properties(
                camera,
                camera_index=camera_index,
                source=camera_source,
                requested_width=width,
                requested_height=height,
                requested_fps=args.fps,
            )
            camera_properties.append(properties)
            log_camera_properties(properties, log_fn=log_fn)

        camera_readers = [
            LatestFrameCameraReader(
                camera,
                camera_index=camera_index,
                target_fps=args.fps,
                time_fn=time_fn,
                sleep_fn=sleep_fn,
            )
            for camera_index, camera in enumerate(cameras)
        ]
        for reader in camera_readers:
            reader.start()

        if show_opencv_preview:
            cv2.namedWindow("Cap Line Runtime V2", cv2.WINDOW_NORMAL)
            display_opened = True

        sessions = [
            create_onnx_session(ort, model_path, args.onnx_intra_op_threads)
            for _ in camera_sources
        ]
        input_metas = [session.get_inputs()[0] for session in sessions]
        input_names = [input_meta.name for input_meta in input_metas]
        model_imgsz = resolve_imgsz(input_metas[0], args.imgsz, preset_imgsz)
        for input_meta in input_metas[1:]:
            other_imgsz = resolve_imgsz(input_meta, args.imgsz, preset_imgsz)
            if other_imgsz != model_imgsz:
                raise ValueError("Camera ONNX sessions resolved different input sizes")
        if not args.serial_inference:
            inference_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=len(camera_sources)
            )
        physical_delay_s = calculate_trigger_delay(
            args.nozzle_distance_mm,
            args.belt_speed_mm_per_s,
            args.trigger_offset_s,
        )
        requested_delay_s = compute_requested_trigger_delay(
            args.nozzle_distance_mm,
            args.belt_speed_mm_per_s,
            args.trigger_offset_s,
            args.latency_compensation_ms,
        )

        log_fn(f"Using model: {model_path} (imgsz={model_imgsz})")
        log_fn(f"Tracking detection threshold: {args.tracking_threshold:.3f}")
        log_fn(f"Defect reject threshold: {args.reject_threshold:.3f}")
        log_fn(
            "Inference: "
            + (
                "serial"
                if args.serial_inference
                else f"parallel sessions={len(sessions)} "
                f"onnx_intra_op_threads={args.onnx_intra_op_threads}"
            )
        )
        log_fn(
            "Trigger formula: "
            f"physical = {args.nozzle_distance_mm:.3f}mm / {args.belt_speed_mm_per_s:.3f}mm/s "
            f"+ {args.trigger_offset_s:.3f}s = {physical_delay_s:.3f}s; "
            f"requested = {physical_delay_s:.3f}s - {args.latency_compensation_ms:.1f}ms = "
            f"{requested_delay_s:.3f}s"
        )
        log_fn(
            "Decision policy: "
            f"highest_defect_score>={args.reject_threshold:.3f} "
            f"merge_window_ms={args.merge_window_ms:.1f} "
            f"finalize_quiet_ms={args.finalize_quiet_ms:.1f}"
        )
        log_fn(f"Pair sync: max_skew_ms={args.pair_max_skew_ms:.1f}")
        log_fn(
            "Debug burst: "
            f"before_frames={args.debug_burst_before_frames} "
            f"after_frames={args.debug_burst_after_frames}"
        )
        log_fn(f"Timing logs: {timing_logger.directory}")
        log_fn(f"Debug captures: {getattr(review_writer, 'debug_dir', review_writer.directory)}")
        log_fn(f"Raw pictures: {getattr(review_writer, 'pictures_dir', pictures_dir)}")
        if isinstance(log_fn, SessionLogTee):
            log_fn(f"Session log: {log_fn.log_path}")
        log_fn(
            "GPIO backend: "
            + ("simulation" if args.simulate_gpio else scheduler.backend_name)
        )
        performance_stats = RuntimePerformanceStats(
            args.perf_log_interval_s,
            camera_count=len(camera_sources),
            start_time=clock.monotonic(),
        )
        pairing_stats = PairingStats()
        mismatched_camera_indexes = [
            properties.camera_index
            for properties in camera_properties
            if camera_properties_mismatch(properties)
        ]
        last_trigger_anchor_time: float | None = None

        def queue_trigger_decision(
            tracked_cap: TrackedCap,
            decision: TrackedCapDecision,
        ) -> None:
            nonlocal last_trigger_anchor_time
            if tracked_cap.trigger_decision is not None:
                return
            if (
                last_trigger_anchor_time is not None
                and abs(decision.anchor_time - last_trigger_anchor_time)
                < merge_window_seconds
            ):
                tracked_cap.trigger_decision = decision
                log_fn(
                    f"[CAP] event={tracked_cap.event_id} suppressed duplicate belt trigger "
                    f"anchor={clock.format(decision.anchor_time)} "
                    f"last_anchor={clock.format(last_trigger_anchor_time)}"
                )
                return

            tracked_cap.trigger_decision = decision
            last_trigger_anchor_time = decision.anchor_time

            timing_record = build_timing_log_record(
                tracked_cap,
                decision,
                cam_list=camera_sources,
                clock=clock,
                decision_time=decision.decision_ready_time,
                nozzle_distance_mm=args.nozzle_distance_mm,
                belt_speed_mm_per_s=args.belt_speed_mm_per_s,
                trigger_offset_s=args.trigger_offset_s,
                latency_compensation_ms=args.latency_compensation_ms,
            )

            def on_trigger_complete(
                execution: RejectExecution,
                record: TimingLogRecord = timing_record,
                trigger_decision: TrackedCapDecision = decision,
                event_id: int = tracked_cap.event_id,
                recorded_at: str = clock.format(decision.decision_ready_time),
            ) -> None:
                record.queued_at = clock.format(execution.queued_at)
                record.trigger_on_time = clock.format(execution.trigger_on_time)
                record.trigger_off_time = clock.format(execution.trigger_off_time)
                record.anchor_to_actual_on_ms = (
                    execution.trigger_on_time - trigger_decision.anchor_time
                ) * 1000.0
                record.scheduler_late_ms = max(
                    0.0,
                    execution.trigger_on_time - execution.requested_fire_time,
                ) * 1000.0
                record.pulse_duration_ms = (
                    execution.trigger_off_time - execution.trigger_on_time
                ) * 1000.0
                timing_logger.log(record)
                if timing_log_callback is not None:
                    timing_log_callback(record)

                if hasattr(review_writer, "write_trigger_completion"):
                    fire_payload = {
                        "event_id": int(event_id),
                        "recorded_at": recorded_at,
                        "anchor_time_iso": clock.format(trigger_decision.anchor_time),
                        "requested_fire_time_iso": clock.format(
                            execution.requested_fire_time
                        ),
                        "queued_at_iso": clock.format(execution.queued_at),
                        "trigger_on_time_iso": clock.format(execution.trigger_on_time),
                        "trigger_off_time_iso": clock.format(execution.trigger_off_time),
                        "anchor_to_actual_on_ms": round_float(
                            record.anchor_to_actual_on_ms, 3
                        ),
                        "scheduler_late_ms": round_float(record.scheduler_late_ms, 3),
                        "pulse_duration_ms": round_float(record.pulse_duration_ms, 3),
                        "trigger_formula": {
                            "nozzle_distance_mm": round_float(
                                float(args.nozzle_distance_mm), 3
                            ),
                            "belt_speed_mm_per_s": round_float(
                                float(args.belt_speed_mm_per_s), 3
                            ),
                            "trigger_offset_s": round_float(
                                float(args.trigger_offset_s), 6
                            ),
                            "latency_compensation_ms": round_float(
                                float(args.latency_compensation_ms), 3
                            ),
                            "computed_trigger_delay_s": round_float(
                                trigger_decision.trigger_delay_s, 6
                            ),
                        },
                    }
                    review_writer.write_trigger_completion(
                        event_id=event_id,
                        recorded_at=recorded_at,
                        payload=fire_payload,
                    )

            enqueue_result = scheduler.enqueue(
                tracked_cap.event_id,
                decision.requested_fire_time,
                completion_callback=on_trigger_complete,
            )
            camera_vote_text = ", ".join(
                f"{camera_index}={format_camera_vote(vote) or 'none'}"
                for camera_index, vote in sorted(decision.camera_votes.items())
            )
            log_fn(
                f"[CAP] event={tracked_cap.event_id} result=trigger "
                f"class={decision.final_class_name} score={decision.final_score:.3f} "
                f"source={decision.decision_source} votes=[{camera_vote_text}] "
                f"anchor={clock.format(decision.anchor_time)} "
                f"fire_at={clock.format(decision.requested_fire_time)} "
                f"queue={enqueue_result.queue_depth}"
            )
            submit_line_picture_capture(
                review_writer,
                tracked_cap,
                decision,
                clock=clock,
                args=args,
            )

        def maybe_submit_trigger_capture(
            tracked_cap: TrackedCap,
            *,
            force: bool = False,
        ) -> None:
            decision = tracked_cap.trigger_decision
            if decision is None or tracked_cap.review_capture_submitted:
                return
            required_after_frames = max(0, int(args.debug_burst_after_frames))
            if not force and tracked_cap.post_actuation_snapshot_count < required_after_frames:
                return
            submit_review_capture(
                review_writer,
                tracked_cap,
                decision,
                clock=clock,
                model_path=model_path,
                args=args,
                cam_list=camera_sources,
                camera_properties=camera_properties,
                performance_snapshot=performance_stats.latest_snapshot,
            )

        frame_index = 0
        previous_frame_time: float | None = None
        last_processed_sequences: list[int] | None = None

        while True:
            if stop_event is not None and stop_event.is_set():
                break

            frame_pair: FramePair | None = None
            while frame_pair is None:
                if stop_event is not None and stop_event.is_set():
                    break
                latest_frames = [reader.latest() for reader in camera_readers]
                frame_pair = select_synchronized_frame_pair(
                    latest_frames,
                    last_processed_sequences,
                    max_skew_ms=args.pair_max_skew_ms,
                    pairing_stats=pairing_stats,
                    log_fn=log_fn,
                )
                if frame_pair is not None:
                    last_processed_sequences = list(frame_pair.sequences)
                    break
                sleep_fn(0.001)
            if frame_pair is None:
                break

            frames = frame_pair.frames
            frame_time = frame_pair.pair_timestamp
            read_duration_ms = frame_pair.read_duration_ms
            capture_sequences_by_camera = list(frame_pair.sequences)
            frame_interval_ms = (
                None
                if previous_frame_time is None
                else (frame_time - previous_frame_time) * 1000.0
            )
            previous_frame_time = frame_time

            all_observations: list[TrackObservation] = []
            all_closed_tracks: list[ClosedTrack] = []

            paired_inference = infer_paired_frame(
                frame_pair,
                sessions,
                input_names,
                model_imgsz,
                clock,
                inference_executor,
                serial=args.serial_inference,
                tracking_threshold=args.tracking_threshold,
            )
            all_boxes_by_camera = paired_inference.boxes_by_camera
            inference_ms_by_camera = paired_inference.inference_ms_by_camera

            for camera_index, boxes in enumerate(all_boxes_by_camera):
                frame = frames[camera_index]
                track_update = trackers[camera_index].update(
                    camera_index,
                    boxes,
                    frame_time,
                    frame_size=(int(frame.shape[1]), int(frame.shape[0])),
                )
                all_observations.extend(track_update.observations)
                all_closed_tracks.extend(track_update.closed_tracks)

            touched_caps = cap_manager.update(all_observations, all_closed_tracks)

            annotated_frames = []
            for camera_index, (frame, boxes) in enumerate(zip(frames, all_boxes_by_camera)):
                annotated = draw_boxes(frame.copy(), boxes)
                draw_anchor_line(annotated, args.anchor_axis, args.anchor_line_ratio)
                annotated_frames.append(annotated)

            preview = compose_preview(annotated_frames)
            for tracked_cap in touched_caps:
                tracked_cap.update_review_frames(
                    frames,
                    preview,
                    annotated_frames=annotated_frames,
                )
            processing_duration_ms = (clock.monotonic() - frame_time) * 1000.0
            frame_snapshot = FrameSnapshot(
                frame_index=frame_index,
                timestamp=frame_time,
                raw_frames=frames,
                annotated_frames=annotated_frames,
                boxes_by_camera=all_boxes_by_camera,
                read_duration_ms=read_duration_ms,
                frame_interval_ms=frame_interval_ms,
                inference_ms_by_camera=inference_ms_by_camera,
                processing_duration_ms=processing_duration_ms,
                pair_skew_ms=frame_pair.skew_ms,
                pair_sequences=list(frame_pair.sequences),
            )
            max_debug_snapshots = (
                max(0, int(args.debug_burst_before_frames))
                + max(0, int(args.debug_burst_after_frames))
                + 1
            )
            for tracked_cap in cap_manager.open_caps():
                tracked_cap.append_debug_frame_snapshot(
                    frame_snapshot,
                    max_snapshots=max_debug_snapshots,
                )
            performance_snapshot = performance_stats.record(
                now=clock.monotonic(),
                frame_interval_ms=frame_interval_ms,
                read_duration_ms=read_duration_ms,
                inference_ms_by_camera=inference_ms_by_camera,
                processing_duration_ms=processing_duration_ms,
                capture_sequences_by_camera=capture_sequences_by_camera,
                pair_skew_ms=frame_pair.skew_ms,
                pairing_stats=pairing_stats,
            )
            if performance_snapshot is not None:
                log_performance_snapshot(performance_snapshot, log_fn=log_fn)
                if performance_callback is not None:
                    performance_callback(performance_snapshot)
                if mismatched_camera_indexes:
                    log_fn(
                        "[CAMERA][WARN] persistent camera format mismatch "
                        f"indexes={mismatched_camera_indexes}"
                    )
            frame_index += 1

            decision_ready_time = clock.monotonic()
            for tracked_cap in cap_manager.open_caps():
                if tracked_cap.trigger_decision is not None:
                    continue
                decision = decide_decision_ready(
                    tracked_cap,
                    camera_count=len(camera_sources),
                    timing_camera_index=args.timing_camera,
                    decision_ready_time=decision_ready_time,
                    merge_window_seconds=merge_window_seconds,
                    nozzle_distance_mm=args.nozzle_distance_mm,
                    belt_speed_mm_per_s=args.belt_speed_mm_per_s,
                    trigger_offset_s=args.trigger_offset_s,
                    latency_compensation_ms=args.latency_compensation_ms,
                    reject_threshold=args.reject_threshold,
                )
                if decision is not None and decision.result == "trigger":
                    queue_trigger_decision(tracked_cap, decision)

            for tracked_cap in cap_manager.open_caps():
                maybe_submit_trigger_capture(tracked_cap)

            for tracked_cap in cap_manager.pop_finalized(decision_ready_time):
                decision_time = clock.monotonic()
                decision = decide_tracked_cap(
                    tracked_cap,
                    camera_count=len(camera_sources),
                    timing_camera_index=args.timing_camera,
                    decision_time=decision_time,
                    merge_window_seconds=merge_window_seconds,
                    nozzle_distance_mm=args.nozzle_distance_mm,
                    belt_speed_mm_per_s=args.belt_speed_mm_per_s,
                    trigger_offset_s=args.trigger_offset_s,
                    latency_compensation_ms=args.latency_compensation_ms,
                    reject_threshold=args.reject_threshold,
                )

                if decision.result == "trigger" and tracked_cap.trigger_decision is None:
                    queue_trigger_decision(tracked_cap, decision)
                maybe_submit_trigger_capture(tracked_cap, force=True)

                history_record = build_history_record(
                    tracked_cap,
                    decision,
                    cam_list=camera_sources,
                    clock=clock,
                    decision_time=decision_time,
                )
                if history_callback is not None:
                    history_callback(history_record)

                if decision.result == "skip":
                    submit_line_picture_capture(
                        review_writer,
                        tracked_cap,
                        decision,
                        clock=clock,
                        args=args,
                    )
                    submit_review_capture(
                        review_writer,
                        tracked_cap,
                        decision,
                        clock=clock,
                        model_path=model_path,
                        args=args,
                        cam_list=camera_sources,
                        camera_properties=camera_properties,
                        performance_snapshot=performance_stats.latest_snapshot,
                    )
                    timing_record = build_timing_log_record(
                        tracked_cap,
                        decision,
                        cam_list=camera_sources,
                        clock=clock,
                        decision_time=decision_time,
                        nozzle_distance_mm=args.nozzle_distance_mm,
                        belt_speed_mm_per_s=args.belt_speed_mm_per_s,
                        trigger_offset_s=args.trigger_offset_s,
                        latency_compensation_ms=args.latency_compensation_ms,
                    )
                    timing_logger.log(timing_record)
                    if timing_log_callback is not None:
                        timing_log_callback(timing_record)
                    camera_vote_text = ", ".join(
                        f"{camera_index}={format_camera_vote(vote) or 'none'}"
                        for camera_index, vote in sorted(decision.camera_votes.items())
                    )
                    score = decision.final_score if decision.final_score is not None else 0.0
                    log_fn(
                        f"[CAP] event={tracked_cap.event_id} result=skip "
                        f"class={decision.final_class_name or 'none'} "
                        f"score={score:.3f} "
                        f"source={decision.decision_source} votes=[{camera_vote_text}]"
                    )

            if preview_callback is not None and preview is not None:
                preview_callback(preview)

            if show_opencv_preview and preview is not None:
                cv2.imshow("Cap Line Runtime V2", preview)
                key = cv2.waitKey(1) & 0xFF
            else:
                key = -1

            if key == ord("q"):
                break
    finally:
        if "maybe_submit_trigger_capture" in locals():
            for tracked_cap in cap_manager.open_caps():
                maybe_submit_trigger_capture(tracked_cap, force=True)

        for reader in camera_readers:
            reader.stop()

        for camera in cameras:
            camera.release()

        if inference_executor is not None:
            inference_executor.shutdown(wait=True)

        if display_opened:
            cv2.destroyAllWindows()

        review_writer.close()
        scheduler.close()


def main() -> None:
    run_detection(parse_args())


if __name__ == "__main__":
    main()
