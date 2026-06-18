#!/usr/bin/env python3
"""
Two-class cap inspection runtime for conveyor deployment.

- Runs a two-class YOLO ONNX model (`undefected`, `dirt_defect`).
- Tracks one physical cap across frames and both cameras so repeated detections
  do not trigger GPIO multiple times.
- Uses a dedicated GPIO scheduler thread so close rejects remain distinct.
- Writes an append-only daily CSV timing log for later trigger tuning.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import math
import os
import re
from queue import Queue
import subprocess
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from itertools import count
from typing import Callable

from gpio_output import DEFAULT_TRIGGER_PIN, GPIOOutputPin


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CLASS_NAMES = ["undefected", "dirt_defect"]
DEFECT_CLASS_ID = 1
DEFAULT_MODEL = "dirtv2.onnx"
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
DEFAULT_TIMING_LOG_DIR = os.path.join(SCRIPT_DIR, "data", "timing_logs")
DEFAULT_REVIEW_DIR = os.path.join(SCRIPT_DIR, "data", "review_captures")
DEFAULT_CONFIDENCE = 0.60
DEFAULT_NOZZLE_DISTANCE_MM = 430.0
DEFAULT_BELT_SPEED_MM_PER_S = 275.0
DEFAULT_TRIGGER_OFFSET_S = -0.23
DEFAULT_DEFECT_MIN_SCORE = 0.80
DEFAULT_DEFECT_MARGIN = 0.08
DEFAULT_SINGLE_CAMERA_DEFECT_SCORE = 0.97
DEFAULT_FINALIZE_QUIET_MS = 30.0
DEFAULT_LATENCY_COMPENSATION_MS = 50.0
DEFAULT_CAMERA_RESOLUTION = [960, 600]
DEFAULT_CAMERA_FPS = 60
DEFAULT_CAMERA_PIXEL_FORMAT = "YUYV"
SUPPORTED_CAMERA_PIXEL_FORMATS = frozenset({"YUYV", "YUY2"})
TIMING_LOG_HEADERS = [
    "recorded_at",
    "event_id",
    "final_result",
    "final_class",
    "cam0_vote",
    "cam1_vote",
    "cam0_observation_count",
    "cam1_observation_count",
    "cam0_first_seen_at",
    "cam0_last_seen_at",
    "cam1_first_seen_at",
    "cam1_last_seen_at",
    "anchor_camera",
    "anchor_time",
    "decision_time",
    "decision_ready_time",
    "queued_at",
    "requested_fire_time",
    "trigger_on_time",
    "trigger_off_time",
    "nozzle_distance_mm",
    "belt_speed_mm_per_s",
    "trigger_offset_s",
    "latency_compensation_ms",
    "computed_trigger_delay_s",
    "anchor_to_decision_ready_ms",
    "anchor_to_requested_ms",
    "anchor_to_actual_on_ms",
    "scheduler_late_ms",
    "pulse_duration_ms",
]
REVIEW_MANIFEST_HEADERS = [
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
    "raw_cam0_path",
    "raw_cam1_path",
]


def class_name(class_id: int | None) -> str | None:
    if class_id is None:
        return None
    if 0 <= class_id < len(CLASS_NAMES):
        return CLASS_NAMES[class_id]
    return f"class{class_id}"


def round_float(value: float | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def copy_frame(frame):
    if frame is None:
        return None
    if hasattr(frame, "copy"):
        return frame.copy()
    return frame


def copy_frames(frames) -> list[object]:
    return [copy_frame(frame) for frame in frames]


class RuntimeClock:
    def __init__(
        self,
        *,
        time_fn: Callable[[], float] = time.monotonic,
        now_fn: Callable[[], datetime] | None = None,
    ):
        self._time_fn = time_fn
        self._now_fn = now_fn or (lambda: datetime.now().astimezone())
        self._base_monotonic = float(self._time_fn())
        self._base_wall = self._now_fn()

    def monotonic(self) -> float:
        return float(self._time_fn())

    def to_datetime(self, monotonic_value: float) -> datetime:
        return self._base_wall + timedelta(
            seconds=float(monotonic_value) - self._base_monotonic
        )

    def format(self, monotonic_value: float | None, *, timespec: str = "milliseconds") -> str:
        if monotonic_value is None:
            return ""
        return self.to_datetime(monotonic_value).isoformat(timespec=timespec)


class NullGPIOOutputPin:
    backend_name = "null"

    def __init__(self, pin):
        self.pin = pin

    def on(self) -> None:
        return None

    def off(self) -> None:
        return None

    def close(self) -> None:
        return None


@dataclass(frozen=True)
class DetectionHistoryRecord:
    recorded_at: str
    runtime_event_id: int
    result: str
    final_class_name: str | None
    final_score: float | None
    decision_source: str
    camera_labels: list[str]
    camera_votes: dict[str, dict[str, object]]
    anchor_time: str | None
    trigger_delay_s: float | None


@dataclass
class TimingLogRecord:
    recorded_at: str
    event_id: int
    final_result: str
    final_class: str | None
    cam0_vote: str
    cam1_vote: str
    cam0_observation_count: int
    cam1_observation_count: int
    cam0_first_seen_at: str
    cam0_last_seen_at: str
    cam1_first_seen_at: str
    cam1_last_seen_at: str
    anchor_camera: str
    anchor_time: str
    decision_time: str
    decision_ready_time: str = ""
    queued_at: str = ""
    requested_fire_time: str = ""
    trigger_on_time: str = ""
    trigger_off_time: str = ""
    nozzle_distance_mm: float = 0.0
    belt_speed_mm_per_s: float = 0.0
    trigger_offset_s: float = 0.0
    latency_compensation_ms: float | None = None
    computed_trigger_delay_s: float | None = None
    anchor_to_decision_ready_ms: float | None = None
    anchor_to_requested_ms: float | None = None
    anchor_to_actual_on_ms: float | None = None
    scheduler_late_ms: float | None = None
    pulse_duration_ms: float | None = None

    def to_row(self) -> dict[str, object]:
        def optional_number(value: float | None, digits: int = 6):
            if value is None:
                return ""
            return round_float(value, digits)

        return {
            "recorded_at": self.recorded_at,
            "event_id": self.event_id,
            "final_result": self.final_result,
            "final_class": self.final_class or "",
            "cam0_vote": self.cam0_vote,
            "cam1_vote": self.cam1_vote,
            "cam0_observation_count": self.cam0_observation_count,
            "cam1_observation_count": self.cam1_observation_count,
            "cam0_first_seen_at": self.cam0_first_seen_at,
            "cam0_last_seen_at": self.cam0_last_seen_at,
            "cam1_first_seen_at": self.cam1_first_seen_at,
            "cam1_last_seen_at": self.cam1_last_seen_at,
            "anchor_camera": self.anchor_camera,
            "anchor_time": self.anchor_time,
            "decision_time": self.decision_time,
            "decision_ready_time": self.decision_ready_time,
            "queued_at": self.queued_at,
            "requested_fire_time": self.requested_fire_time,
            "trigger_on_time": self.trigger_on_time,
            "trigger_off_time": self.trigger_off_time,
            "nozzle_distance_mm": round_float(self.nozzle_distance_mm, 3) or 0.0,
            "belt_speed_mm_per_s": round_float(self.belt_speed_mm_per_s, 3) or 0.0,
            "trigger_offset_s": round_float(self.trigger_offset_s, 6) or 0.0,
            "latency_compensation_ms": optional_number(self.latency_compensation_ms, 3),
            "computed_trigger_delay_s": optional_number(self.computed_trigger_delay_s, 6),
            "anchor_to_decision_ready_ms": optional_number(self.anchor_to_decision_ready_ms, 3),
            "anchor_to_requested_ms": optional_number(self.anchor_to_requested_ms, 3),
            "anchor_to_actual_on_ms": optional_number(self.anchor_to_actual_on_ms, 3),
            "scheduler_late_ms": optional_number(self.scheduler_late_ms, 3),
            "pulse_duration_ms": optional_number(self.pulse_duration_ms, 3),
        }


class TimingCsvLogger:
    def __init__(self, directory: str = DEFAULT_TIMING_LOG_DIR):
        self.directory = os.path.abspath(directory)
        os.makedirs(self.directory, exist_ok=True)
        self._lock = threading.Lock()

    def _file_path_for_record(self, record: TimingLogRecord) -> str:
        day = (record.recorded_at or datetime.now().astimezone().isoformat()).split("T", 1)[0]
        return os.path.join(self.directory, f"{day}.csv")

    def log(self, record: TimingLogRecord) -> str:
        file_path = self._file_path_for_record(record)
        with self._lock:
            needs_header = not os.path.exists(file_path) or os.path.getsize(file_path) == 0
            with open(file_path, "a", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=TIMING_LOG_HEADERS)
                if needs_header:
                    writer.writeheader()
                writer.writerow(record.to_row())
        return file_path


@dataclass(frozen=True)
class ReviewCaptureTask:
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
    raw_frames: list[object] = field(default_factory=list)


class ReviewCaptureWriter:
    def __init__(
        self,
        directory: str = DEFAULT_REVIEW_DIR,
        *,
        write_image_fn: Callable[[object, str], bool] | None = None,
        log_fn: Callable[..., None] = print,
    ):
        self.directory = os.path.abspath(directory)
        os.makedirs(self.directory, exist_ok=True)
        self._write_image_fn = write_image_fn or self._default_write_image
        self._log = log_fn
        self._queue: Queue[ReviewCaptureTask | None] = Queue()
        self._thread = threading.Thread(
            target=self._run,
            name="cap-line-review-capture",
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

    def submit(self, task: ReviewCaptureTask) -> int:
        self._queue.put(task)
        queue_depth = self._queue.qsize()
        self._log(
            f"[REVIEW] queued event={task.event_id} reason={task.review_reason} backlog={queue_depth}"
        )
        return queue_depth

    def _record_day(self, recorded_at: str) -> str:
        if recorded_at:
            return recorded_at.split("T", 1)[0]
        return datetime.now().astimezone().strftime("%Y-%m-%d")

    def _timestamp_label(self, recorded_at: str) -> str:
        if recorded_at:
            try:
                return datetime.fromisoformat(recorded_at).strftime("%Y%m%d_%H%M%S_%f")[:-3]
            except ValueError:
                pass
        return datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")[:-3]

    def _manifest_path(self, day: str) -> str:
        return os.path.join(self.directory, f"{day}.csv")

    def _write_manifest_row(
        self,
        task: ReviewCaptureTask,
        *,
        preview_path: str,
        raw_paths: list[str],
    ) -> str:
        day = self._record_day(task.recorded_at)
        manifest_path = self._manifest_path(day)
        needs_header = not os.path.exists(manifest_path) or os.path.getsize(manifest_path) == 0
        with open(manifest_path, "a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=REVIEW_MANIFEST_HEADERS)
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
                    "final_score": "" if task.final_score is None else round_float(task.final_score, 6),
                    "score_summary": task.score_summary,
                    "cam0_vote": task.cam0_vote,
                    "cam1_vote": task.cam1_vote,
                    "model_path": task.model_path,
                    "preview_path": preview_path,
                    "raw_cam0_path": raw_paths[0] if len(raw_paths) > 0 else "",
                    "raw_cam1_path": raw_paths[1] if len(raw_paths) > 1 else "",
                }
            )
        return manifest_path

    def _run(self) -> None:
        while True:
            task = self._queue.get()
            if task is None:
                self._queue.task_done()
                break

            try:
                day = self._record_day(task.recorded_at)
                day_dir = os.path.join(self.directory, day)
                os.makedirs(day_dir, exist_ok=True)
                prefix = f"{self._timestamp_label(task.recorded_at)}_event{task.event_id}_{task.review_reason}"
                preview_path = ""
                if task.preview_frame is not None:
                    preview_path = os.path.join(day_dir, f"{prefix}_preview.jpg")
                    if not self._write_image_fn(task.preview_frame, preview_path):
                        raise RuntimeError(f"Failed to write review preview: {preview_path}")

                raw_paths: list[str] = []
                for camera_index, frame in enumerate(task.raw_frames):
                    raw_path = os.path.join(day_dir, f"{prefix}_cam{camera_index}.jpg")
                    if not self._write_image_fn(frame, raw_path):
                        raise RuntimeError(f"Failed to write review raw frame: {raw_path}")
                    raw_paths.append(raw_path)

                manifest_path = self._write_manifest_row(
                    task,
                    preview_path=preview_path,
                    raw_paths=raw_paths,
                )
                self._log(
                    f"[REVIEW] saved event={task.event_id} reason={task.review_reason} manifest={manifest_path}"
                )
            except Exception as exc:
                self._log(f"[REVIEW] error event={task.event_id} {exc}")
            finally:
                self._queue.task_done()

    def close(self) -> None:
        self._queue.put(None)
        self._thread.join()


def infer_model_imgsz_from_name(model_path: str) -> int | None:
    stem = os.path.splitext(os.path.basename(model_path))[0]
    for part in stem.replace("-", "_").split("_"):
        if part.isdigit():
            return int(part)
    return None


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


def parse_cameras(cam_args: list[str]) -> tuple[list[object], list[str]]:
    camera_sources: list[object] = []
    device_paths: list[str] = []

    for cam in cam_args:
        try:
            cam_index = int(cam)
            camera_sources.append(cam_index)
            device_paths.append(f"/dev/video{cam_index}")
        except ValueError:
            camera_sources.append(cam)
            device_paths.append(cam)

    return camera_sources, device_paths


def set_camera_controls(device_path: str, exposure_value: int, *, log_fn: Callable[..., None] = print) -> None:
    log_fn(f"Configuring camera {device_path}...")
    try:
        subprocess.run(
            ["v4l2-ctl", "-d", device_path, "-c", "auto_exposure=1"],
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        log_fn(f"Failed to set auto_exposure=1 on {device_path}: {exc}")

    try:
        subprocess.run(
            ["v4l2-ctl", "-d", device_path, "-c", f"exposure_time_absolute={exposure_value}"],
            check=True,
        )
        log_fn(f"{device_path}: exposure_time_absolute={exposure_value}")
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        log_fn(f"Failed to set exposure_time_absolute on {device_path}: {exc}")


def _run_v4l2_ctl(
    device_path: str,
    *args: str,
    check: bool = True,
) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(
            ["v4l2-ctl", "-d", device_path, *args],
            check=check,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _read_v4l2_video_format(device_path: str) -> tuple[int | None, int | None, str | None, float | None]:
    fmt_result = _run_v4l2_ctl(device_path, "--get-fmt-video", check=False)
    parm_result = _run_v4l2_ctl(device_path, "--get-parm", check=False)
    width = height = None
    pixel_format = None
    fps = None

    if fmt_result and fmt_result.stdout:
        size_match = re.search(r"Width/Height\s*:\s*(\d+)/(\d+)", fmt_result.stdout)
        if size_match:
            width = int(size_match.group(1))
            height = int(size_match.group(2))
        format_match = re.search(r"Pixel Format\s*:\s*'([A-Z0-9]+)'", fmt_result.stdout)
        if format_match:
            pixel_format = format_match.group(1)

    if parm_result and parm_result.stdout:
        fps_match = re.search(
            r"Frames per second:\s*([0-9.]+)",
            parm_result.stdout,
        )
        if fps_match:
            fps = float(fps_match.group(1))

    return width, height, pixel_format, fps


def normalize_camera_pixel_format(pixel_format: str) -> str:
    normalized = str(pixel_format).strip().upper()
    if normalized in SUPPORTED_CAMERA_PIXEL_FORMATS:
        return DEFAULT_CAMERA_PIXEL_FORMAT
    return normalized


def _pixel_formats_match(requested: str, actual: str | None) -> bool:
    if actual is None:
        return True
    return normalize_camera_pixel_format(actual) == normalize_camera_pixel_format(requested)


def set_camera_format(
    device_path: str,
    width: int,
    height: int,
    fps: int,
    *,
    pixel_format: str = DEFAULT_CAMERA_PIXEL_FORMAT,
    log_fn: Callable[..., None] = print,
) -> None:
    pixel_format = normalize_camera_pixel_format(pixel_format)
    commands = [
        [
            "v4l2-ctl",
            "-d",
            device_path,
            (
                f"--set-fmt-video=width={int(width)},height={int(height)},"
                f"pixelformat={pixel_format}"
            ),
        ],
        ["v4l2-ctl", "-d", device_path, f"--set-parm={int(fps)}"],
    ]
    for command in commands:
        try:
            subprocess.run(command, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            log_fn(f"Failed to set camera format on {device_path}: {exc}")
            return

    actual_width, actual_height, actual_format, actual_fps = _read_v4l2_video_format(device_path)
    mismatch = (
        actual_width != int(width)
        or actual_height != int(height)
        or not _pixel_formats_match(pixel_format, actual_format)
        or (actual_fps is not None and abs(actual_fps - float(fps)) >= 1.0)
    )
    if mismatch:
        log_fn(
            f"[CAMERA][WARN] {device_path}: re-applying requested format "
            f"{int(width)}x{int(height)}@{int(fps)} {pixel_format}; "
            f"hardware reported "
            f"{actual_width}x{actual_height}@{actual_fps} {actual_format}"
        )
        for command in commands:
            try:
                subprocess.run(command, check=True)
            except (subprocess.CalledProcessError, FileNotFoundError) as exc:
                log_fn(f"Failed to re-apply camera format on {device_path}: {exc}")
                return
        actual_width, actual_height, actual_format, actual_fps = _read_v4l2_video_format(
            device_path
        )
        if (
            actual_width != int(width)
            or actual_height != int(height)
            or not _pixel_formats_match(pixel_format, actual_format)
            or (actual_fps is not None and abs(actual_fps - float(fps)) >= 1.0)
        ):
            log_fn(
                f"[CAMERA][WARN] {device_path}: hardware still reports "
                f"{actual_width}x{actual_height}@{actual_fps} {actual_format} "
                f"after forcing {int(width)}x{int(height)}@{int(fps)} {pixel_format}"
            )

    log_fn(
        f"{device_path}: forced format "
        f"{int(width)}x{int(height)}@{int(fps)} {pixel_format}"
    )


def open_cam(
    src: object,
    width: int | None = None,
    height: int | None = None,
    fps: int | None = None,
    pixel_format: str = DEFAULT_CAMERA_PIXEL_FORMAT,
):
    pixel_format = normalize_camera_pixel_format(pixel_format)
    import cv2

    cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
    if pixel_format:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*pixel_format))
    if width is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height is not None:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if fps is not None:
        cap.set(cv2.CAP_PROP_FPS, fps)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera source {src}")
    return cap


def resolve_imgsz(input_meta, requested_imgsz: int | None = None, preset_imgsz: int | None = None) -> int:
    input_shape = getattr(input_meta, "shape", None)
    fixed_imgsz = None
    if isinstance(input_shape, (list, tuple)) and len(input_shape) >= 4:
        height = input_shape[-2]
        width = input_shape[-1]
        if isinstance(height, int) and isinstance(width, int) and height == width:
            fixed_imgsz = int(height)

    if requested_imgsz is not None:
        if fixed_imgsz is not None and requested_imgsz != fixed_imgsz:
            raise ValueError(
                f"--imgsz {requested_imgsz} does not match model input size {fixed_imgsz}"
            )
        return int(requested_imgsz)

    if fixed_imgsz is not None:
        return fixed_imgsz
    if preset_imgsz is not None:
        return int(preset_imgsz)

    raise ValueError(
        "Could not infer model input size automatically. Pass --imgsz explicitly."
    )


def letterbox_resize(image_bgr, new_shape: tuple[int, int] = (640, 640), color=(114, 114, 114)):
    import cv2

    original_height, original_width = image_bgr.shape[:2]
    scale = min(new_shape[0] / original_height, new_shape[1] / original_width)
    resized_width = int(round(original_width * scale))
    resized_height = int(round(original_height * scale))

    resized_image = cv2.resize(
        image_bgr,
        (resized_width, resized_height),
        interpolation=cv2.INTER_LINEAR,
    )
    pad_width = new_shape[1] - resized_width
    pad_height = new_shape[0] - resized_height
    pad_left = int(round(pad_width / 2 - 0.1))
    pad_right = int(round(pad_width / 2 + 0.1))
    pad_top = int(round(pad_height / 2 - 0.1))
    pad_bottom = int(round(pad_height / 2 + 0.1))

    padded_image = cv2.copyMakeBorder(
        resized_image,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=color,
    )
    return padded_image, scale, (pad_left, pad_top)


def preprocess(frame, img_size: int = 640):
    import cv2
    import numpy as np

    letterboxed_bgr, resize_scale, padding = letterbox_resize(
        frame,
        new_shape=(img_size, img_size),
        color=(114, 114, 114),
    )
    img = cv2.cvtColor(letterboxed_bgr, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img, {
        "scale": float(resize_scale),
        "pad_left": int(padding[0]),
        "pad_top": int(padding[1]),
        "frame_shape": frame.shape,
        "img_size": int(img_size),
    }


def postprocess(output, preprocess_meta, conf_threshold: float = DEFAULT_CONFIDENCE) -> list[list[float]]:
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

    scale = preprocess_meta["scale"]
    pad_left = preprocess_meta["pad_left"]
    pad_top = preprocess_meta["pad_top"]
    frame_h, frame_w = preprocess_meta["frame_shape"][:2]
    img_size = preprocess_meta["img_size"]

    kept_boxes: list[list[float]] = []
    for detection in detections:
        x1, y1, x2, y2, score, class_id_value = detection[:6]
        score = float(score)
        if score < conf_threshold:
            continue

        coords = np.array([x1, y1, x2, y2], dtype=np.float32)
        if float(np.max(np.abs(coords))) <= 1.5:
            coords[[0, 2]] *= img_size
            coords[[1, 3]] *= img_size

        x1, y1, x2, y2 = coords.tolist()
        x1 = (x1 - pad_left) / scale
        y1 = (y1 - pad_top) / scale
        x2 = (x2 - pad_left) / scale
        y2 = (y2 - pad_top) / scale

        x1 = max(0.0, min(frame_w - 1.0, x1))
        y1 = max(0.0, min(frame_h - 1.0, y1))
        x2 = max(0.0, min(frame_w - 1.0, x2))
        y2 = max(0.0, min(frame_h - 1.0, y2))
        if x2 <= x1 or y2 <= y1:
            continue

        class_id = int(round(float(class_id_value)))
        if class_id < 0 or class_id >= len(CLASS_NAMES):
            continue

        kept_boxes.append([x1, y1, x2, y2, score, class_id])

    kept_boxes.sort(key=lambda item: item[4], reverse=True)
    return kept_boxes


def clip_box_xyxy(x1: float, y1: float, x2: float, y2: float, frame_shape) -> tuple[int, int, int, int]:
    frame_h, frame_w = frame_shape[:2]
    x1 = max(0, min(frame_w - 1, int(round(x1))))
    y1 = max(0, min(frame_h - 1, int(round(y1))))
    x2 = max(0, min(frame_w - 1, int(round(x2))))
    y2 = max(0, min(frame_h - 1, int(round(y2))))
    return x1, y1, x2, y2


def draw_anchor_line(frame, axis: str, line_ratio: float, color=(255, 215, 0)):
    import cv2

    frame_h, frame_w = frame.shape[:2]
    if axis == "x":
        x = max(0, min(frame_w - 1, int(round(frame_w * line_ratio))))
        cv2.line(frame, (x, 0), (x, frame_h - 1), color, 2)
    else:
        y = max(0, min(frame_h - 1, int(round(frame_h * line_ratio))))
        cv2.line(frame, (0, y), (frame_w - 1, y), color, 2)
    return frame


def draw_boxes(frame, boxes: list[list[float]]):
    import cv2

    colors = {
        0: (0, 200, 0),
        1: (0, 0, 255),
    }

    for box in boxes:
        x1, y1, x2, y2, conf, cls = box
        class_id = int(cls)
        x1, y1, x2, y2 = clip_box_xyxy(x1, y1, x2, y2, frame.shape)
        if x2 <= x1 or y2 <= y1:
            continue

        color = colors.get(class_id, (255, 255, 255))
        label = f"{class_name(class_id)}:{conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            label,
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
        )
    return frame


def compose_preview(frames, pad: int = 6):
    import cv2
    import numpy as np

    if not frames:
        return None
    if len(frames) == 1:
        return frames[0]

    height = min(frame.shape[0] for frame in frames)
    resized = [
        cv2.resize(
            frame,
            (int(frame.shape[1] * height / frame.shape[0]), height),
            interpolation=cv2.INTER_AREA,
        )
        for frame in frames
    ]

    if pad <= 0:
        return np.hstack(resized)

    spacer = np.zeros((height, pad, 3), dtype=np.uint8)
    stacked = []
    for index, frame in enumerate(resized):
        if index > 0:
            stacked.append(spacer)
        stacked.append(frame)
    return np.hstack(stacked)


DEFAULT_LIVE_PREVIEW_FPS = 30.0
DEFAULT_LIVE_PREVIEW_MAX_EXTRAPOLATION_S = 0.35
ONNX_PROVIDER_PREFERENCE = (
    "TensorrtExecutionProvider",
    "CUDAExecutionProvider",
    "CPUExecutionProvider",
)


def create_onnx_session(ort, model_path: str, intra_op_threads: int):
    session_options = None
    if hasattr(ort, "SessionOptions"):
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = max(1, int(intra_op_threads))
        session_options.inter_op_num_threads = 1

    available_providers = list(getattr(ort, "get_available_providers", lambda: [])())
    providers = [
        provider
        for provider in ONNX_PROVIDER_PREFERENCE
        if provider in available_providers
    ]
    if not providers:
        providers = ["CPUExecutionProvider"]

    if session_options is None:
        return ort.InferenceSession(model_path, providers=providers)
    return ort.InferenceSession(
        model_path,
        sess_options=session_options,
        providers=providers,
    )


@dataclass(frozen=True)
class LivePreviewOverlaySnapshot:
    boxes_by_camera: list[list[list[float]]]
    timestamps: list[float | None] = field(default_factory=list)
    sequences: list[int | None] = field(default_factory=list)


def copy_preview_boxes_by_camera(
    boxes_by_camera: list[list[list[float]]],
) -> list[list[list[float]]]:
    return [
        [[float(value) for value in box] for box in camera_boxes]
        for camera_boxes in boxes_by_camera
    ]


def live_preview_snapshot_from_frame_pair(
    frame_pair: object,
    boxes_by_camera: list[list[list[float]]],
) -> LivePreviewOverlaySnapshot:
    timestamps = [
        None if timestamp is None else float(timestamp)
        for timestamp in list(getattr(frame_pair, "timestamps", []))
    ]
    sequences = [
        None if sequence is None else int(sequence)
        for sequence in list(getattr(frame_pair, "sequences", []))
    ]
    return LivePreviewOverlaySnapshot(
        boxes_by_camera=copy_preview_boxes_by_camera(boxes_by_camera),
        timestamps=timestamps,
        sequences=sequences,
    )


def list_get_or_none(values: list, index: int):
    if index < 0 or index >= len(values):
        return None
    return values[index]


def shifted_box(box: list[float], shift_x: float, shift_y: float) -> list[float]:
    predicted = [float(value) for value in box]
    predicted[0] += shift_x
    predicted[1] += shift_y
    predicted[2] += shift_x
    predicted[3] += shift_y
    return predicted


def match_previous_preview_box(
    current_box: list[float],
    previous_boxes: list[list[float]],
) -> list[float] | None:
    same_class_boxes = [
        box
        for box in previous_boxes
        if len(box) >= 6 and int(box[5]) == int(current_box[5])
    ]
    candidates = same_class_boxes or previous_boxes
    if not candidates:
        return None

    plausible_candidates = [
        box
        for box in candidates
        if boxes_look_like_same_cap(
            current_box,
            box,
            min_iou=0.01,
            max_center_distance=3.0,
            min_size_ratio=0.45,
        )
    ]
    if not plausible_candidates:
        return None

    return min(plausible_candidates, key=lambda box: normalized_center_distance(current_box, box))


def predict_preview_boxes_for_timestamp(
    current_boxes: list[list[float]],
    previous_boxes: list[list[float]],
    *,
    current_timestamp: float | None,
    previous_timestamp: float | None,
    target_timestamp: float | None,
    max_extrapolation_s: float = DEFAULT_LIVE_PREVIEW_MAX_EXTRAPOLATION_S,
) -> list[list[float]]:
    copied_current_boxes = [[float(value) for value in box] for box in current_boxes]
    if current_timestamp is None or target_timestamp is None:
        return copied_current_boxes

    extrapolation_s = float(target_timestamp) - float(current_timestamp)
    if extrapolation_s <= 0.0:
        return copied_current_boxes
    if extrapolation_s > float(max_extrapolation_s):
        return []
    if previous_timestamp is None:
        return copied_current_boxes

    history_s = float(current_timestamp) - float(previous_timestamp)
    if history_s <= 0.0:
        return copied_current_boxes

    predicted_boxes = []
    for box in copied_current_boxes:
        previous_box = match_previous_preview_box(box, previous_boxes)
        if previous_box is None:
            predicted_boxes.append(box)
            continue

        current_center_x, current_center_y = box_center(box)
        previous_center_x, previous_center_y = box_center(previous_box)
        shift_x = ((current_center_x - previous_center_x) / history_s) * extrapolation_s
        shift_y = ((current_center_y - previous_center_y) / history_s) * extrapolation_s
        predicted_boxes.append(shifted_box(box, shift_x, shift_y))

    return predicted_boxes


class LivePreviewPublisher:
    """Publish smooth camera previews while detection runs at a lower rate."""

    def __init__(
        self,
        camera_readers,
        preview_callback: Callable[[object], None],
        *,
        anchor_axis: str,
        anchor_line_ratio: float,
        target_fps: float = DEFAULT_LIVE_PREVIEW_FPS,
        stop_event=None,
        time_fn: Callable[[], float] = time.monotonic,
        sleep_fn: Callable[[float], None] = time.sleep,
    ):
        self._camera_readers = list(camera_readers)
        self._preview_callback = preview_callback
        self._anchor_axis = anchor_axis
        self._anchor_line_ratio = anchor_line_ratio
        self._target_fps = float(target_fps)
        self._external_stop_event = stop_event
        self._stop_event = threading.Event()
        self._time_fn = time_fn
        self._sleep_fn = sleep_fn
        self._overlay_lock = threading.Lock()
        self._overlay_snapshot: LivePreviewOverlaySnapshot | None = None
        self._previous_overlay_snapshot: LivePreviewOverlaySnapshot | None = None
        self._thread = threading.Thread(
            target=self._run,
            name="cap-line-live-preview",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join()

    def update_overlay(
        self,
        frame_pair_or_boxes: object,
        boxes_by_camera: list[list[list[float]]] | None = None,
    ) -> None:
        if boxes_by_camera is None:
            snapshot = LivePreviewOverlaySnapshot(
                boxes_by_camera=copy_preview_boxes_by_camera(frame_pair_or_boxes)
            )
        else:
            snapshot = live_preview_snapshot_from_frame_pair(
                frame_pair_or_boxes,
                boxes_by_camera,
            )
        with self._overlay_lock:
            self._previous_overlay_snapshot = self._overlay_snapshot
            self._overlay_snapshot = snapshot

    def _should_stop(self) -> bool:
        if self._stop_event.is_set():
            return True
        return (
            self._external_stop_event is not None
            and self._external_stop_event.is_set()
        )

    def _run(self) -> None:
        min_interval_s = (
            0.0
            if self._target_fps <= 0.0
            else 1.0 / self._target_fps
        )
        while not self._should_stop():
            loop_started_at = self._time_fn()
            latest_frames = [reader.latest() for reader in self._camera_readers]
            if all(frame is not None for frame in latest_frames):
                frames = [frame.frame for frame in latest_frames]
                with self._overlay_lock:
                    overlay_snapshot = self._overlay_snapshot
                    previous_overlay_snapshot = self._previous_overlay_snapshot
                overlay = self._overlay_for_latest_frames(
                    latest_frames,
                    overlay_snapshot,
                    previous_overlay_snapshot,
                )

                annotated_frames = []
                for frame, boxes in zip(frames, overlay):
                    annotated = draw_boxes(frame.copy(), boxes)
                    draw_anchor_line(
                        annotated,
                        self._anchor_axis,
                        self._anchor_line_ratio,
                    )
                    annotated_frames.append(annotated)

                preview = compose_preview(annotated_frames)
                if preview is not None:
                    self._preview_callback(preview)

            if min_interval_s > 0.0 and not self._should_stop():
                elapsed_s = self._time_fn() - loop_started_at
                remaining_s = min_interval_s - elapsed_s
                if remaining_s > 0.0:
                    self._sleep_fn(remaining_s)

    def _overlay_for_latest_frames(
        self,
        latest_frames: list[object],
        overlay_snapshot: LivePreviewOverlaySnapshot | None,
        previous_overlay_snapshot: LivePreviewOverlaySnapshot | None,
    ) -> list[list[list[float]]]:
        if overlay_snapshot is None:
            return [[] for _frame in latest_frames]
        if len(overlay_snapshot.boxes_by_camera) != len(latest_frames):
            return [[] for _frame in latest_frames]

        overlay = []
        for camera_index, captured_frame in enumerate(latest_frames):
            current_boxes = overlay_snapshot.boxes_by_camera[camera_index]
            previous_boxes = (
                []
                if previous_overlay_snapshot is None
                or len(previous_overlay_snapshot.boxes_by_camera) <= camera_index
                else previous_overlay_snapshot.boxes_by_camera[camera_index]
            )
            overlay.append(
                predict_preview_boxes_for_timestamp(
                    current_boxes,
                    previous_boxes,
                    current_timestamp=list_get_or_none(
                        overlay_snapshot.timestamps,
                        camera_index,
                    ),
                    previous_timestamp=(
                        None
                        if previous_overlay_snapshot is None
                        else list_get_or_none(
                            previous_overlay_snapshot.timestamps,
                            camera_index,
                        )
                    ),
                    target_timestamp=getattr(captured_frame, "timestamp", None),
                )
            )
        return overlay


def box_iou(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a[:4]
    bx1, by1, bx2, by2 = box_b[:4]

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union_area = area_a + area_b - inter_area
    if union_area <= 0.0:
        return 0.0

    return inter_area / union_area


def box_center(box: list[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = box[:4]
    return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)


def box_center_value(box: list[float], axis: str) -> float:
    center_x, center_y = box_center(box)
    return center_x if axis == "x" else center_y


def box_edge_values(box: list[float], axis: str) -> tuple[float, float]:
    x1, y1, x2, y2 = box[:4]
    if axis == "x":
        return float(x1), float(x2)
    return float(y1), float(y2)


def box_spans_line_coordinate(
    box: list[float],
    *,
    axis: str,
    line_coordinate: float,
) -> bool:
    leading_edge, trailing_edge = box_edge_values(box, axis)
    lower_edge = min(leading_edge, trailing_edge)
    upper_edge = max(leading_edge, trailing_edge)
    return lower_edge <= line_coordinate <= upper_edge


def opposite_axis(axis: str) -> str:
    return "y" if axis == "x" else "x"


def box_size_along_axis(box: list[float], axis: str) -> float:
    x1, y1, x2, y2 = box[:4]
    if axis == "x":
        return max(0.0, x2 - x1)
    return max(0.0, y2 - y1)


def box_area(box: list[float]) -> float:
    x1, y1, x2, y2 = box[:4]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def normalized_center_distance(box_a: list[float], box_b: list[float]) -> float:
    ax, ay = box_center(box_a)
    bx, by = box_center(box_b)
    distance = math.hypot(ax - bx, ay - by)
    scale = max(
        1.0,
        box_a[2] - box_a[0],
        box_a[3] - box_a[1],
        box_b[2] - box_b[0],
        box_b[3] - box_b[1],
    )
    return distance / scale


def box_size_ratio(box_a: list[float], box_b: list[float]) -> float:
    area_a = box_area(box_a)
    area_b = box_area(box_b)
    largest = max(area_a, area_b, 1.0)
    smallest = min(area_a, area_b)
    return smallest / largest


def boxes_look_like_same_cap(
    box_a: list[float],
    box_b: list[float],
    *,
    min_iou: float,
    max_center_distance: float,
    min_size_ratio: float,
) -> bool:
    if box_iou(box_a, box_b) >= min_iou:
        return True
    if box_size_ratio(box_a, box_b) < min_size_ratio:
        return False
    return normalized_center_distance(box_a, box_b) <= max_center_distance


def boxes_follow_same_cap_trajectory(
    candidate_box: list[float],
    box_history: list[list[float]],
    *,
    axis: str,
    min_size_ratio: float,
    max_cross_axis_distance: float = 0.65,
    max_axis_progress: float = 2.0,
    max_axis_prediction_error: float = 1.0,
    min_axis_motion_px: float = 2.0,
) -> bool:
    if len(box_history) < 2:
        return False

    previous_box = box_history[-2]
    latest_box = box_history[-1]
    if box_size_ratio(candidate_box, latest_box) < min_size_ratio:
        return False

    previous_axis = box_center_value(previous_box, axis)
    latest_axis = box_center_value(latest_box, axis)
    candidate_axis = box_center_value(candidate_box, axis)
    axis_delta = latest_axis - previous_axis
    if abs(axis_delta) < min_axis_motion_px:
        return False

    direction = 1.0 if axis_delta > 0.0 else -1.0
    progress = (candidate_axis - latest_axis) * direction
    if progress <= 0.0:
        return False

    cross_axis = opposite_axis(axis)
    latest_cross = box_center_value(latest_box, cross_axis)
    candidate_cross = box_center_value(candidate_box, cross_axis)
    cross_scale = max(
        1.0,
        box_size_along_axis(latest_box, cross_axis),
        box_size_along_axis(candidate_box, cross_axis),
    )
    if abs(candidate_cross - latest_cross) / cross_scale > max_cross_axis_distance:
        return False

    axis_scale = max(
        1.0,
        box_size_along_axis(latest_box, axis),
        box_size_along_axis(candidate_box, axis),
    )
    if progress > max(axis_scale * max_axis_progress, abs(axis_delta) * max_axis_progress):
        return False

    predicted_axis = latest_axis + axis_delta
    prediction_error = abs(candidate_axis - predicted_axis)
    max_prediction_error = max(
        axis_scale * max_axis_prediction_error,
        abs(axis_delta) * max_axis_prediction_error,
    )
    return prediction_error <= max_prediction_error


def reference_coordinate(frame_size: tuple[int, int], axis: str, line_ratio: float) -> float:
    width, height = frame_size
    size = width if axis == "x" else height
    return float(size) * float(line_ratio)


def did_cross_reference_line(
    previous_box: list[float] | None,
    current_box: list[float],
    *,
    axis: str,
    line_coordinate: float,
) -> bool:
    if not box_spans_line_coordinate(
        current_box,
        axis=axis,
        line_coordinate=line_coordinate,
    ):
        return False

    if previous_box is None:
        return True

    if box_spans_line_coordinate(
        previous_box,
        axis=axis,
        line_coordinate=line_coordinate,
    ):
        return False

    current_value = box_center_value(current_box, axis)
    previous_value = box_center_value(previous_box, axis)
    lower = min(previous_value, current_value)
    upper = max(previous_value, current_value)
    return lower <= line_coordinate <= upper


@dataclass
class CameraTrack:
    track_id: int
    box: list[float]
    last_seen_at: float
    missed_frames: int = 0


@dataclass
class TrackObservation:
    camera_index: int
    track_id: int
    class_id: int
    box: list[float]
    timestamp: float
    frame_size: tuple[int, int]


@dataclass
class ClosedTrack:
    camera_index: int
    track_id: int
    box: list[float]
    last_seen_at: float


@dataclass
class TrackUpdate:
    observations: list[TrackObservation] = field(default_factory=list)
    closed_tracks: list[ClosedTrack] = field(default_factory=list)


class CameraLifecycleTracker:
    def __init__(
        self,
        iou_threshold: float,
        max_missing_frames: int,
        center_distance_threshold: float = 1.35,
        min_size_ratio: float = 0.45,
    ):
        self.iou_threshold = float(iou_threshold)
        self.max_missing_frames = int(max_missing_frames)
        self.center_distance_threshold = float(center_distance_threshold)
        self.min_size_ratio = float(min_size_ratio)
        self._tracks: dict[int, CameraTrack] = {}
        self._next_track_id = 1

    def update(
        self,
        camera_index: int,
        boxes: list[list[float]],
        timestamp: float,
        frame_size: tuple[int, int],
    ) -> TrackUpdate:
        match_candidates: list[tuple[float, float, int, int]] = []
        for detection_index, box in enumerate(boxes):
            for track_id, track in self._tracks.items():
                if not boxes_look_like_same_cap(
                    box,
                    track.box,
                    min_iou=self.iou_threshold,
                    max_center_distance=self.center_distance_threshold,
                    min_size_ratio=self.min_size_ratio,
                ):
                    continue
                iou = box_iou(box, track.box)
                distance = normalized_center_distance(box, track.box)
                match_candidates.append((iou, -distance, track_id, detection_index))

        match_candidates.sort(reverse=True)

        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()
        observations: list[TrackObservation] = []
        closed_tracks: list[ClosedTrack] = []

        for _, _, track_id, detection_index in match_candidates:
            if track_id in matched_tracks or detection_index in matched_detections:
                continue
            box = list(boxes[detection_index])
            track = self._tracks[track_id]
            track.box = box
            track.last_seen_at = timestamp
            track.missed_frames = 0
            matched_tracks.add(track_id)
            matched_detections.add(detection_index)
            observations.append(
                TrackObservation(
                    camera_index=camera_index,
                    track_id=track_id,
                    class_id=int(box[5]),
                    box=box,
                    timestamp=timestamp,
                    frame_size=frame_size,
                )
            )

        for track_id in list(self._tracks):
            if track_id in matched_tracks:
                continue

            track = self._tracks[track_id]
            track.missed_frames += 1
            if track.missed_frames > self.max_missing_frames:
                closed_tracks.append(
                    ClosedTrack(
                        camera_index=camera_index,
                        track_id=track_id,
                        box=list(track.box),
                        last_seen_at=track.last_seen_at,
                    )
                )
                del self._tracks[track_id]

        for detection_index, box in enumerate(boxes):
            if detection_index in matched_detections:
                continue

            track_id = self._next_track_id
            self._next_track_id += 1
            stored_box = list(box)
            self._tracks[track_id] = CameraTrack(
                track_id=track_id,
                box=stored_box,
                last_seen_at=timestamp,
            )
            observations.append(
                TrackObservation(
                    camera_index=camera_index,
                    track_id=track_id,
                    class_id=int(stored_box[5]),
                    box=stored_box,
                    timestamp=timestamp,
                    frame_size=frame_size,
                )
            )

        return TrackUpdate(observations=observations, closed_tracks=closed_tracks)


@dataclass
class CameraObservationSummary:
    observation_count: int = 0
    class_confidence_totals: dict[int, float] = field(default_factory=dict)
    first_seen_at: float | None = None
    last_seen_at: float | None = None

    def add(self, class_id: int, confidence: float, timestamp: float) -> None:
        self.observation_count += 1
        self.class_confidence_totals[class_id] = (
            self.class_confidence_totals.get(class_id, 0.0) + float(confidence)
        )
        if self.first_seen_at is None:
            self.first_seen_at = timestamp
        self.last_seen_at = timestamp


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
    trigger_decision: "TrackedCapDecision | None" = None
    review_capture_submitted: bool = False
    latest_preview_frame: object | None = None
    latest_raw_frames: list[object] = field(default_factory=list)

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
        summary.add(observation.class_id, float(observation.box[4]), observation.timestamp)

        if observation.camera_index != timing_camera_index or self.anchor_time is not None:
            return

        line_coordinate = reference_coordinate(
            observation.frame_size,
            anchor_axis,
            anchor_line_ratio,
        )
        if did_cross_reference_line(
            previous_box,
            observation.box,
            axis=anchor_axis,
            line_coordinate=line_coordinate,
        ):
            self.anchor_time = observation.timestamp
            self.anchor_camera_index = observation.camera_index

    def close_track(self, camera_index: int, track_id: int) -> None:
        self.active_track_keys.discard((camera_index, track_id))

    def update_review_frames(self, raw_frames, preview_frame) -> None:
        self.latest_raw_frames = copy_frames(raw_frames)
        self.latest_preview_frame = copy_frame(preview_frame)


@dataclass(frozen=True)
class CameraVote:
    camera_index: int
    class_id: int | None
    class_name: str | None
    score: float | None
    observation_count: int
    first_seen_at: float | None
    last_seen_at: float | None


@dataclass(frozen=True)
class CapEvaluation:
    total_observations: int
    class_scores: dict[int, float]
    camera_votes: dict[int, CameraVote]
    usable_camera_votes: dict[int, CameraVote]

    @property
    def dirt_score(self) -> float:
        return float(self.class_scores.get(DEFECT_CLASS_ID, 0.0))

    @property
    def undefected_score(self) -> float:
        return float(self.class_scores.get(0, 0.0))


@dataclass(frozen=True)
class TrackedCapDecision:
    result: str
    final_class_name: str | None
    final_score: float | None
    decision_source: str
    camera_votes: dict[int, CameraVote]
    anchor_time: float
    anchor_source: str
    decision_ready_time: float
    latency_compensation_ms: float
    trigger_delay_s: float
    requested_fire_time: float
    review_reason: str | None = None


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
                if boxes_follow_same_cap_trajectory(
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

            if (observation.timestamp - tracked_cap.last_seen_at) <= self.merge_window_seconds:
                return tracked_cap

        return None

    def _mark_recent(self, tracked_cap: TrackedCap) -> None:
        try:
            self._open_caps.remove(tracked_cap)
        except ValueError:
            pass
        self._open_caps.append(tracked_cap)


def calculate_trigger_delay(
    nozzle_distance_mm: float,
    belt_speed_mm_per_s: float,
    trigger_offset_s: float,
) -> float:
    if belt_speed_mm_per_s <= 0:
        raise ValueError("belt_speed_mm_per_s must be greater than 0")
    if nozzle_distance_mm < 0:
        raise ValueError("nozzle_distance_mm must be 0 or greater")
    return (float(nozzle_distance_mm) / float(belt_speed_mm_per_s)) + float(trigger_offset_s)


def compute_class_scores(
    summaries: list[CameraObservationSummary | None],
) -> tuple[int, dict[int, float]]:
    total_observations = 0
    class_confidence_totals = {class_id: 0.0 for class_id in range(len(CLASS_NAMES))}

    for summary in summaries:
        if summary is None or summary.observation_count <= 0:
            continue
        total_observations += summary.observation_count
        for class_id, confidence_total in summary.class_confidence_totals.items():
            if 0 <= class_id < len(CLASS_NAMES):
                class_confidence_totals[class_id] += float(confidence_total)

    if total_observations <= 0:
        return 0, {class_id: 0.0 for class_id in range(len(CLASS_NAMES))}

    return total_observations, {
        class_id: confidence_total / total_observations
        for class_id, confidence_total in class_confidence_totals.items()
    }


def build_camera_vote(
    summary: CameraObservationSummary | None,
    *,
    camera_index: int,
    tie_abs_tol: float = 1e-9,
) -> CameraVote:
    if summary is None or summary.observation_count <= 0:
        return CameraVote(
            camera_index=camera_index,
            class_id=None,
            class_name=None,
            score=None,
            observation_count=0,
            first_seen_at=None,
            last_seen_at=None,
        )

    _, class_scores = compute_class_scores([summary])
    winning_score = max(class_scores.values())
    winning_ids = [
        class_id
        for class_id, score in class_scores.items()
        if math.isclose(score, winning_score, rel_tol=0.0, abs_tol=tie_abs_tol)
    ]
    if len(winning_ids) != 1:
        return CameraVote(
            camera_index=camera_index,
            class_id=None,
            class_name=None,
            score=None,
            observation_count=summary.observation_count,
            first_seen_at=summary.first_seen_at,
            last_seen_at=summary.last_seen_at,
        )

    winner_class_id = winning_ids[0]
    return CameraVote(
        camera_index=camera_index,
        class_id=winner_class_id,
        class_name=class_name(winner_class_id),
        score=winning_score,
        observation_count=summary.observation_count,
        first_seen_at=summary.first_seen_at,
        last_seen_at=summary.last_seen_at,
    )


def resolve_anchor_time(tracked_cap: TrackedCap, *, timing_camera_index: int) -> tuple[float, str]:
    if tracked_cap.anchor_time is not None:
        return tracked_cap.anchor_time, "anchor_line"

    timing_summary = tracked_cap.camera_summaries.get(timing_camera_index)
    if timing_summary is not None and timing_summary.first_seen_at is not None:
        return timing_summary.first_seen_at, "timing_camera_first_seen"

    first_seen_candidates = [
        summary.first_seen_at
        for summary in tracked_cap.camera_summaries.values()
        if summary.first_seen_at is not None
    ]
    if first_seen_candidates:
        return min(first_seen_candidates), "cap_first_seen"

    return tracked_cap.created_at, "cap_created_at"


def build_cap_evaluation(
    tracked_cap: TrackedCap,
    *,
    camera_count: int,
) -> CapEvaluation:
    camera_votes = {
        camera_index: build_camera_vote(
            tracked_cap.camera_summaries.get(camera_index),
            camera_index=camera_index,
        )
        for camera_index in range(camera_count)
    }
    total_observations, class_scores = compute_class_scores(
        list(tracked_cap.camera_summaries.values())
    )
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


def compute_requested_trigger_delay(
    nozzle_distance_mm: float,
    belt_speed_mm_per_s: float,
    trigger_offset_s: float,
    latency_compensation_ms: float,
) -> float:
    physical_delay_s = calculate_trigger_delay(
        nozzle_distance_mm,
        belt_speed_mm_per_s,
        trigger_offset_s,
    )
    return max(0.0, physical_delay_s - (float(latency_compensation_ms) / 1000.0))


def build_tracked_cap_decision(
    *,
    result: str,
    final_class_name: str | None,
    final_score: float | None,
    decision_source: str,
    evaluation: CapEvaluation,
    anchor_time: float,
    anchor_source: str,
    decision_ready_time: float,
    nozzle_distance_mm: float,
    belt_speed_mm_per_s: float,
    trigger_offset_s: float,
    latency_compensation_ms: float,
    review_reason: str | None = None,
) -> TrackedCapDecision:
    trigger_delay_s = compute_requested_trigger_delay(
        nozzle_distance_mm,
        belt_speed_mm_per_s,
        trigger_offset_s,
        latency_compensation_ms,
    )
    return TrackedCapDecision(
        result=result,
        final_class_name=final_class_name,
        final_score=final_score,
        decision_source=decision_source,
        camera_votes=evaluation.camera_votes,
        anchor_time=anchor_time,
        anchor_source=anchor_source,
        decision_ready_time=decision_ready_time,
        latency_compensation_ms=float(latency_compensation_ms),
        trigger_delay_s=trigger_delay_s,
        requested_fire_time=anchor_time + trigger_delay_s,
        review_reason=review_reason,
    )


def maybe_build_trigger_decision(
    tracked_cap: TrackedCap,
    *,
    evaluation: CapEvaluation,
    timing_camera_index: int,
    decision_ready_time: float,
    camera_count: int,
    merge_window_seconds: float,
    defect_min_score: float,
    defect_margin: float,
    single_camera_defect_score: float,
    nozzle_distance_mm: float,
    belt_speed_mm_per_s: float,
    trigger_offset_s: float,
    latency_compensation_ms: float,
) -> TrackedCapDecision | None:
    if evaluation.total_observations <= 0:
        return None

    anchor_time, anchor_source = resolve_anchor_time(
        tracked_cap,
        timing_camera_index=timing_camera_index,
    )
    usable_camera_votes = evaluation.usable_camera_votes
    dirt_score = evaluation.dirt_score
    undefected_score = evaluation.undefected_score

    if len(usable_camera_votes) >= camera_count:
        if (
            dirt_score > undefected_score
            and dirt_score >= defect_min_score
            and (dirt_score - undefected_score) >= defect_margin
        ):
            return build_tracked_cap_decision(
                result="trigger",
                final_class_name=class_name(DEFECT_CLASS_ID),
                final_score=dirt_score,
                decision_source="dual_camera_margin_trigger",
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
        return None

    if len(usable_camera_votes) != 1:
        return None

    sole_vote = next(iter(usable_camera_votes.values()))
    if sole_vote.class_id != DEFECT_CLASS_ID:
        return None

    if (decision_ready_time - tracked_cap.last_seen_at) < merge_window_seconds:
        return None

    if dirt_score < single_camera_defect_score:
        return None

    return build_tracked_cap_decision(
        result="trigger",
        final_class_name=class_name(DEFECT_CLASS_ID),
        final_score=dirt_score,
        decision_source="single_camera_high_conf_trigger",
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
    defect_min_score: float,
    defect_margin: float,
    single_camera_defect_score: float,
    nozzle_distance_mm: float,
    belt_speed_mm_per_s: float,
    trigger_offset_s: float,
    latency_compensation_ms: float,
) -> TrackedCapDecision | None:
    if tracked_cap.trigger_decision is not None:
        return tracked_cap.trigger_decision

    evaluation = build_cap_evaluation(
        tracked_cap,
        camera_count=camera_count,
    )
    return maybe_build_trigger_decision(
        tracked_cap,
        evaluation=evaluation,
        timing_camera_index=timing_camera_index,
        decision_ready_time=decision_ready_time,
        camera_count=camera_count,
        merge_window_seconds=merge_window_seconds,
        defect_min_score=defect_min_score,
        defect_margin=defect_margin,
        single_camera_defect_score=single_camera_defect_score,
        nozzle_distance_mm=nozzle_distance_mm,
        belt_speed_mm_per_s=belt_speed_mm_per_s,
        trigger_offset_s=trigger_offset_s,
        latency_compensation_ms=latency_compensation_ms,
    )


def decide_tracked_cap(
    tracked_cap: TrackedCap,
    *,
    camera_count: int,
    timing_camera_index: int,
    decision_time: float,
    merge_window_seconds: float,
    defect_min_score: float,
    defect_margin: float,
    single_camera_defect_score: float,
    nozzle_distance_mm: float,
    belt_speed_mm_per_s: float,
    trigger_offset_s: float,
    latency_compensation_ms: float,
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
    trigger_decision = maybe_build_trigger_decision(
        tracked_cap,
        evaluation=evaluation,
        timing_camera_index=timing_camera_index,
        decision_ready_time=decision_time,
        camera_count=camera_count,
        merge_window_seconds=merge_window_seconds,
        defect_min_score=defect_min_score,
        defect_margin=defect_margin,
        single_camera_defect_score=single_camera_defect_score,
        nozzle_distance_mm=nozzle_distance_mm,
        belt_speed_mm_per_s=belt_speed_mm_per_s,
        trigger_offset_s=trigger_offset_s,
        latency_compensation_ms=latency_compensation_ms,
    )
    if trigger_decision is not None:
        return trigger_decision

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

    usable_camera_votes = evaluation.usable_camera_votes
    review_reason = None
    decision_source = "clean_bias_skip"
    if len(usable_camera_votes) == 1:
        sole_vote = next(iter(usable_camera_votes.values()))
        if sole_vote.class_id == DEFECT_CLASS_ID and evaluation.dirt_score < single_camera_defect_score:
            decision_source = "challenged_clean"
            review_reason = "challenged_clean"

    return build_tracked_cap_decision(
        result="skip",
        final_class_name=class_name(0),
        final_score=evaluation.undefected_score,
        decision_source=decision_source,
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


def format_camera_vote(vote: CameraVote) -> str:
    if vote.class_name is None or vote.score is None:
        return ""
    return f"{vote.class_name}:{vote.score:.3f}"


def camera_vote_payload(vote: CameraVote, clock: RuntimeClock) -> dict[str, object]:
    return {
        "class_name": vote.class_name,
        "score": round_float(vote.score, 6),
        "observation_count": vote.observation_count,
        "first_seen_at": clock.format(vote.first_seen_at) if vote.first_seen_at is not None else None,
        "last_seen_at": clock.format(vote.last_seen_at) if vote.last_seen_at is not None else None,
    }


def build_history_record(
    tracked_cap: TrackedCap,
    decision: TrackedCapDecision,
    *,
    cam_list: list[object],
    clock: RuntimeClock,
    decision_time: float,
) -> DetectionHistoryRecord:
    camera_votes = {
        str(cam_list[camera_index]): camera_vote_payload(vote, clock)
        for camera_index, vote in decision.camera_votes.items()
    }
    camera_labels = [str(cam_list[camera_index]) for camera_index in sorted(tracked_cap.camera_indices)]
    return DetectionHistoryRecord(
        recorded_at=clock.format(decision_time),
        runtime_event_id=tracked_cap.event_id,
        result=decision.result,
        final_class_name=decision.final_class_name,
        final_score=round_float(decision.final_score, 6),
        decision_source=decision.decision_source,
        camera_labels=camera_labels,
        camera_votes=camera_votes,
        anchor_time=clock.format(decision.anchor_time),
        trigger_delay_s=round_float(decision.trigger_delay_s, 6),
    )


def build_timing_log_record(
    tracked_cap: TrackedCap,
    decision: TrackedCapDecision,
    *,
    cam_list: list[object],
    clock: RuntimeClock,
    decision_time: float,
    nozzle_distance_mm: float,
    belt_speed_mm_per_s: float,
    trigger_offset_s: float,
    latency_compensation_ms: float,
) -> TimingLogRecord:
    cam0_vote = decision.camera_votes.get(0)
    cam1_vote = decision.camera_votes.get(1)
    return TimingLogRecord(
        recorded_at=clock.format(decision_time),
        event_id=tracked_cap.event_id,
        final_result=decision.result,
        final_class=decision.final_class_name,
        cam0_vote=format_camera_vote(cam0_vote) if cam0_vote is not None else "",
        cam1_vote=format_camera_vote(cam1_vote) if cam1_vote is not None else "",
        cam0_observation_count=cam0_vote.observation_count if cam0_vote is not None else 0,
        cam1_observation_count=cam1_vote.observation_count if cam1_vote is not None else 0,
        cam0_first_seen_at=clock.format(cam0_vote.first_seen_at) if cam0_vote is not None else "",
        cam0_last_seen_at=clock.format(cam0_vote.last_seen_at) if cam0_vote is not None else "",
        cam1_first_seen_at=clock.format(cam1_vote.first_seen_at) if cam1_vote is not None else "",
        cam1_last_seen_at=clock.format(cam1_vote.last_seen_at) if cam1_vote is not None else "",
        anchor_camera=(
            str(cam_list[tracked_cap.anchor_camera_index])
            if tracked_cap.anchor_camera_index is not None
            else decision.anchor_source
        ),
        anchor_time=clock.format(decision.anchor_time),
        decision_time=clock.format(decision_time),
        decision_ready_time=clock.format(decision.decision_ready_time),
        requested_fire_time=clock.format(decision.requested_fire_time),
        nozzle_distance_mm=float(nozzle_distance_mm),
        belt_speed_mm_per_s=float(belt_speed_mm_per_s),
        trigger_offset_s=float(trigger_offset_s),
        latency_compensation_ms=float(latency_compensation_ms),
        computed_trigger_delay_s=float(decision.trigger_delay_s),
        anchor_to_decision_ready_ms=(decision.decision_ready_time - decision.anchor_time) * 1000.0,
        anchor_to_requested_ms=(decision.requested_fire_time - decision.anchor_time) * 1000.0,
    )


def format_class_scores(class_scores: dict[int, float]) -> str:
    parts = []
    for class_id in range(len(CLASS_NAMES)):
        parts.append(f"{class_name(class_id)}:{class_scores.get(class_id, 0.0):.3f}")
    return ", ".join(parts)


def submit_review_capture(
    writer: ReviewCaptureWriter,
    tracked_cap: TrackedCap,
    decision: TrackedCapDecision,
    *,
    clock: RuntimeClock,
    model_path: str,
) -> None:
    if tracked_cap.review_capture_submitted or decision.review_reason is None:
        return

    evaluation = build_cap_evaluation(tracked_cap, camera_count=2)
    writer.submit(
        ReviewCaptureTask(
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
            preview_frame=copy_frame(tracked_cap.latest_preview_frame),
            raw_frames=copy_frames(tracked_cap.latest_raw_frames),
        )
    )
    tracked_cap.review_capture_submitted = True


@dataclass(frozen=True)
class RejectEnqueueResult:
    queue_depth: int
    queued_at: float
    requested_fire_time: float


@dataclass(frozen=True)
class RejectExecution:
    event_id: int
    queued_at: float
    requested_fire_time: float
    trigger_on_time: float
    trigger_off_time: float


@dataclass(order=True)
class RejectJob:
    requested_fire_time: float
    sequence: int
    event_id: int = field(compare=False)
    queued_at: float = field(compare=False)
    completion_callback: Callable[[RejectExecution], None] | None = field(compare=False, default=None)


class RejectScheduler:
    def __init__(
        self,
        trigger_pin,
        trigger_duration: float,
        trigger_min_gap: float = 0.0,
        *,
        pin_factory=GPIOOutputPin,
        log_fn: Callable[..., None] = print,
        time_fn: Callable[[], float] = time.monotonic,
        sleep_fn: Callable[[float], None] = time.sleep,
    ):
        if trigger_duration <= 0:
            raise ValueError("trigger_duration must be greater than 0")
        if trigger_min_gap < 0:
            raise ValueError("trigger_min_gap must be 0 or greater")

        self.trigger_pin = trigger_pin
        self.trigger_duration = float(trigger_duration)
        self.trigger_min_gap = float(trigger_min_gap)
        self._pin = pin_factory(self.trigger_pin)
        self.backend_name = getattr(self._pin, "backend_name", "hardware")
        self._log = log_fn
        self._time_fn = time_fn
        self._sleep_fn = sleep_fn
        self._jobs: list[RejectJob] = []
        self._sequence = count()
        self._cv = threading.Condition()
        self._stop = False
        self._last_pulse_end: float | None = None
        self._thread = threading.Thread(
            target=self._run,
            name="cap-line-reject-scheduler",
            daemon=True,
        )
        self._thread.start()

    @property
    def queue_depth(self) -> int:
        with self._cv:
            return len(self._jobs)

    def enqueue(
        self,
        event_id: int,
        requested_fire_time: float,
        *,
        completion_callback: Callable[[RejectExecution], None] | None = None,
    ) -> RejectEnqueueResult:
        queued_at = float(self._time_fn())
        job = RejectJob(
            requested_fire_time=float(requested_fire_time),
            sequence=next(self._sequence),
            event_id=int(event_id),
            queued_at=queued_at,
            completion_callback=completion_callback,
        )

        with self._cv:
            if self._stop:
                raise RuntimeError("reject scheduler is closed")
            heapq.heappush(self._jobs, job)
            queue_depth = len(self._jobs)
            self._cv.notify_all()

        self._log(
            "[TRIGGER] QUEUED "
            f"event={job.event_id} requested_at={job.requested_fire_time:.6f} "
            f"requested_in={job.requested_fire_time - queued_at:.3f}s "
            f"queue={queue_depth} duration={self.trigger_duration:.3f}s"
        )
        return RejectEnqueueResult(
            queue_depth=queue_depth,
            queued_at=queued_at,
            requested_fire_time=job.requested_fire_time,
        )

    def _run(self) -> None:
        try:
            while True:
                with self._cv:
                    while not self._stop and not self._jobs:
                        self._cv.wait()
                    if self._stop:
                        break

                    job = self._jobs[0]
                    earliest_start = job.requested_fire_time
                    if self._last_pulse_end is not None:
                        earliest_start = max(
                            earliest_start,
                            self._last_pulse_end + self.trigger_min_gap,
                        )

                    now = float(self._time_fn())
                    wait_time = earliest_start - now
                    if wait_time > 0:
                        self._cv.wait(timeout=wait_time)
                        continue

                    heapq.heappop(self._jobs)
                    remaining_depth = len(self._jobs)

                trigger_on_time = max(
                    job.requested_fire_time,
                    float(self._time_fn()),
                    (self._last_pulse_end + self.trigger_min_gap)
                    if self._last_pulse_end is not None
                    else float("-inf"),
                )
                self._pin.on()
                self._log(
                    "[TRIGGER] ON "
                    f"event={job.event_id} requested_at={job.requested_fire_time:.6f} "
                    f"actual_at={trigger_on_time:.6f} "
                    f"late_by={max(0.0, trigger_on_time - job.requested_fire_time):.6f}s "
                    f"queue={remaining_depth} duration={self.trigger_duration:.3f}s"
                )
                self._sleep_fn(self.trigger_duration)
                trigger_off_time = float(self._time_fn())
                self._pin.off()
                self._last_pulse_end = trigger_off_time
                self._log(
                    "[TRIGGER] OFF "
                    f"event={job.event_id} actual_at={trigger_off_time:.6f} queue={remaining_depth}"
                )

                if job.completion_callback is not None:
                    job.completion_callback(
                        RejectExecution(
                            event_id=job.event_id,
                            queued_at=job.queued_at,
                            requested_fire_time=job.requested_fire_time,
                            trigger_on_time=trigger_on_time,
                            trigger_off_time=trigger_off_time,
                        )
                    )
        finally:
            try:
                self._pin.off()
            finally:
                self._pin.close()

    def close(self) -> None:
        with self._cv:
            if self._stop:
                return
            self._stop = True
            self._jobs.clear()
            self._cv.notify_all()
        self._thread.join()


def validate_args(args: argparse.Namespace) -> None:
    if len(args.cams) != 2:
        raise ValueError("Exactly two cameras are required for this runtime")
    if not 0.0 <= args.conf <= 1.0:
        raise ValueError("--conf must be between 0 and 1")
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
    if not 0.0 <= args.defect_min_score <= 1.0:
        raise ValueError("--defect-min-score must be between 0 and 1")
    if not 0.0 <= args.defect_margin <= 1.0:
        raise ValueError("--defect-margin must be between 0 and 1")
    if not 0.0 <= args.single_camera_defect_score <= 1.0:
        raise ValueError("--single-camera-defect-score must be between 0 and 1")
    if args.nozzle_distance_mm < 0:
        raise ValueError("--nozzle-distance-mm must be 0 or greater")
    if args.belt_speed_mm_per_s <= 0:
        raise ValueError("--belt-speed-mm-per-s must be greater than 0")
    if args.latency_compensation_ms < 0:
        raise ValueError("--latency-compensation-ms must be 0 or greater")
    args.pixel_format = normalize_camera_pixel_format(args.pixel_format)
    if args.pixel_format != DEFAULT_CAMERA_PIXEL_FORMAT:
        raise ValueError(
            f"--pixel-format must be {DEFAULT_CAMERA_PIXEL_FORMAT} for Arducam B0495 cameras"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run two-class cap detection, tracking, timing review logging, and GPIO triggering."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="model filename or path to a two-class .onnx file",
    )
    parser.add_argument(
        "--cams",
        nargs=2,
        default=["0", "3"],
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
    parser.add_argument("--conf", type=float, default=DEFAULT_CONFIDENCE)
    parser.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="override inference image size; by default it matches the model input",
    )
    parser.add_argument("--no-display", action="store_true", help="run without OpenCV GUI")
    parser.add_argument(
        "--trigger-pin",
        default=DEFAULT_TRIGGER_PIN,
        help=(
            "Jetson.GPIO output pin to pulse for reject actuation "
            f"(default: {DEFAULT_TRIGGER_PIN}; GPIO09 is BOARD physical pin 7)"
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
    parser.add_argument("--defect-min-score", type=float, default=DEFAULT_DEFECT_MIN_SCORE)
    parser.add_argument("--defect-margin", type=float, default=DEFAULT_DEFECT_MARGIN)
    parser.add_argument(
        "--single-camera-defect-score",
        type=float,
        default=DEFAULT_SINGLE_CAMERA_DEFECT_SCORE,
    )
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
    parser.add_argument("--timing-log-dir", default=DEFAULT_TIMING_LOG_DIR)
    parser.add_argument("--review-dir", default=DEFAULT_REVIEW_DIR)
    parser.add_argument(
        "--simulate-gpio",
        action="store_true",
        help="use a no-op GPIO pin implementation for development",
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
    timing_logger = TimingCsvLogger(args.timing_log_dir)
    review_writer = ReviewCaptureWriter(args.review_dir, log_fn=log_fn)
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

        for camera_source in camera_sources:
            cameras.append(
                open_cam(camera_source, width, height, args.fps, args.pixel_format)
            )

        if show_opencv_preview:
            cv2.namedWindow("Cap Line Runtime", cv2.WINDOW_NORMAL)
            display_opened = True

        session = create_onnx_session(ort, model_path, args.onnx_intra_op_threads)
        input_meta = session.get_inputs()[0]
        input_name = input_meta.name
        model_imgsz = resolve_imgsz(input_meta, args.imgsz, preset_imgsz)
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
        log_fn(f"Confidence threshold: {args.conf:.3f}")
        log_fn(
            "Trigger formula: "
            f"physical = {args.nozzle_distance_mm:.3f}mm / {args.belt_speed_mm_per_s:.3f}mm/s "
            f"+ {args.trigger_offset_s:.3f}s = {physical_delay_s:.3f}s; "
            f"requested = {physical_delay_s:.3f}s - {args.latency_compensation_ms:.1f}ms = "
            f"{requested_delay_s:.3f}s"
        )
        log_fn(
            "Decision policy: "
            f"defect_min_score={args.defect_min_score:.3f} "
            f"defect_margin={args.defect_margin:.3f} "
            f"single_camera_defect_score={args.single_camera_defect_score:.3f} "
            f"merge_window_ms={args.merge_window_ms:.1f} "
            f"finalize_quiet_ms={args.finalize_quiet_ms:.1f}"
        )
        log_fn(f"Timing logs: {timing_logger.directory}")
        log_fn(f"Review captures: {review_writer.directory}")
        log_fn(
            "GPIO backend: "
            + ("simulation" if args.simulate_gpio else scheduler.backend_name)
        )

        def queue_trigger_decision(
            tracked_cap: TrackedCap,
            decision: TrackedCapDecision,
        ) -> None:
            tracked_cap.trigger_decision = decision
            submit_review_capture(
                review_writer,
                tracked_cap,
                decision,
                clock=clock,
                model_path=model_path,
            )

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

            enqueue_result = scheduler.enqueue(
                tracked_cap.event_id,
                decision.requested_fire_time,
                completion_callback=on_trigger_complete,
            )
            timing_record.queued_at = clock.format(enqueue_result.queued_at)
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

        while True:
            if stop_event is not None and stop_event.is_set():
                break

            frames = []
            ok_read = True
            for camera in cameras:
                ok, frame = camera.read()
                if not ok or frame is None:
                    ok_read = False
                    break
                frames.append(frame)

            frame_time = clock.monotonic()
            if not ok_read or not frames:
                if stop_event is not None and stop_event.is_set():
                    break
                sleep_fn(0.01)
                continue

            all_observations: list[TrackObservation] = []
            all_closed_tracks: list[ClosedTrack] = []
            all_boxes_by_camera: list[list[list[float]]] = []

            for camera_index, frame in enumerate(frames):
                input_tensor, preprocess_meta = preprocess(frame, model_imgsz)
                output = session.run(None, {input_name: input_tensor})[0]
                boxes = postprocess(output, preprocess_meta, args.conf)
                all_boxes_by_camera.append(boxes)

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
                if camera_index == args.timing_camera:
                    draw_anchor_line(annotated, args.anchor_axis, args.anchor_line_ratio)
                annotated_frames.append(annotated)

            preview = compose_preview(annotated_frames)
            for tracked_cap in touched_caps:
                tracked_cap.update_review_frames(frames, preview)

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
                    defect_min_score=args.defect_min_score,
                    defect_margin=args.defect_margin,
                    single_camera_defect_score=args.single_camera_defect_score,
                    nozzle_distance_mm=args.nozzle_distance_mm,
                    belt_speed_mm_per_s=args.belt_speed_mm_per_s,
                    trigger_offset_s=args.trigger_offset_s,
                    latency_compensation_ms=args.latency_compensation_ms,
                )
                if decision is not None and decision.result == "trigger":
                    queue_trigger_decision(tracked_cap, decision)

            for tracked_cap in cap_manager.pop_finalized(decision_ready_time):
                decision_time = clock.monotonic()
                decision = decide_tracked_cap(
                    tracked_cap,
                    camera_count=len(camera_sources),
                    timing_camera_index=args.timing_camera,
                    decision_time=decision_time,
                    merge_window_seconds=merge_window_seconds,
                    defect_min_score=args.defect_min_score,
                    defect_margin=args.defect_margin,
                    single_camera_defect_score=args.single_camera_defect_score,
                    nozzle_distance_mm=args.nozzle_distance_mm,
                    belt_speed_mm_per_s=args.belt_speed_mm_per_s,
                    trigger_offset_s=args.trigger_offset_s,
                    latency_compensation_ms=args.latency_compensation_ms,
                )

                if decision.result == "trigger" and tracked_cap.trigger_decision is None:
                    queue_trigger_decision(tracked_cap, decision)

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
                    submit_review_capture(
                        review_writer,
                        tracked_cap,
                        decision,
                        clock=clock,
                        model_path=model_path,
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
                    log_fn(
                        f"[CAP] event={tracked_cap.event_id} result=skip "
                        f"class={decision.final_class_name or 'none'} "
                        f"score={(decision.final_score if decision.final_score is not None else 0.0):.3f} "
                        f"source={decision.decision_source} votes=[{camera_vote_text}]"
                    )

            if preview_callback is not None and preview is not None:
                preview_callback(preview)

            if show_opencv_preview and preview is not None:
                cv2.imshow("Cap Line Runtime", preview)
                key = cv2.waitKey(1) & 0xFF
            else:
                key = -1

            if key == ord("q"):
                break
    finally:
        for camera in cameras:
            camera.release()

        if display_opened:
            cv2.destroyAllWindows()

        review_writer.close()
        scheduler.close()


def main() -> None:
    run_detection(parse_args())


if __name__ == "__main__":
    main()
