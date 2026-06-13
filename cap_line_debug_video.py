#!/usr/bin/env python3
"""
Record a full-session diagnostic video for the cap-line runtime.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from queue import Queue
from typing import Callable

import cap_line_runtime
from cap_line_runtime import DetectionHistoryRecord

try:
    import cv2
except ModuleNotFoundError:
    cv2 = None


DEFAULT_RECORD_DIR = os.path.join(cap_line_runtime.SCRIPT_DIR, "data", "debug_recordings")
DEBUG_EVENT_HEADERS = [
    "recorded_at",
    "runtime_event_id",
    "result",
    "final_class_name",
    "final_score",
    "decision_source",
    "anchor_time",
    "trigger_delay_s",
    "camera_labels_json",
    "camera_votes_json",
]


def require_cv2():
    if cv2 is None:
        raise RuntimeError(
            "OpenCV is required for cap_line_debug_video.py. Install python3-opencv first."
        )
    return cv2


@dataclass(frozen=True)
class RecordingSessionPaths:
    directory: str
    basename: str
    requested_video_path: str
    events_csv_path: str
    session_json_path: str


@dataclass(frozen=True)
class DebugRecordingResult:
    directory: str
    basename: str
    video_path: str
    requested_video_path: str
    events_csv_path: str
    session_json_path: str
    video_codec: str | None
    frame_count: int


def normalize_basename(basename: str | None, session_now: datetime) -> str:
    if basename is not None:
        candidate = os.path.splitext(os.path.basename(basename.strip()))[0]
        if candidate:
            return candidate
    timestamp = session_now.strftime("%Y%m%d_%H%M%S_%f")[:-3]
    return f"{timestamp}_cap_line_debug"


def prepare_session_paths(
    record_dir: str,
    *,
    basename: str | None,
    session_now: datetime,
) -> RecordingSessionPaths:
    root_dir = os.path.abspath(record_dir)
    day_dir = os.path.join(root_dir, session_now.strftime("%Y-%m-%d"))
    os.makedirs(day_dir, exist_ok=True)

    resolved_basename = normalize_basename(basename, session_now)
    return RecordingSessionPaths(
        directory=day_dir,
        basename=resolved_basename,
        requested_video_path=os.path.join(day_dir, f"{resolved_basename}.mp4"),
        events_csv_path=os.path.join(day_dir, f"{resolved_basename}_events.csv"),
        session_json_path=os.path.join(day_dir, f"{resolved_basename}_session.json"),
    )


class DebugEventLogger:
    def __init__(self, file_path: str):
        self.file_path = os.path.abspath(file_path)
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        self._ensure_file()

    def _ensure_file(self) -> None:
        if os.path.exists(self.file_path) and os.path.getsize(self.file_path) > 0:
            return
        with open(self.file_path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=DEBUG_EVENT_HEADERS)
            writer.writeheader()

    def log(self, record: DetectionHistoryRecord) -> None:
        with self._lock:
            with open(self.file_path, "a", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=DEBUG_EVENT_HEADERS)
                writer.writerow(
                    {
                        "recorded_at": record.recorded_at,
                        "runtime_event_id": record.runtime_event_id,
                        "result": record.result,
                        "final_class_name": record.final_class_name or "",
                        "final_score": (
                            ""
                            if record.final_score is None
                            else cap_line_runtime.round_float(record.final_score, 6)
                        ),
                        "decision_source": record.decision_source,
                        "anchor_time": record.anchor_time or "",
                        "trigger_delay_s": (
                            ""
                            if record.trigger_delay_s is None
                            else cap_line_runtime.round_float(record.trigger_delay_s, 6)
                        ),
                        "camera_labels_json": json.dumps(record.camera_labels),
                        "camera_votes_json": json.dumps(record.camera_votes, sort_keys=True),
                    }
                )

    def close(self) -> None:
        return None


class DecisionOverlayState:
    def __init__(
        self,
        *,
        model_label: str,
        camera_labels: list[str],
        started_at: datetime,
        monotonic_fn: Callable[[], float] = time.monotonic,
    ):
        self.model_label = model_label
        self.camera_labels = [str(label) for label in camera_labels]
        self.started_at = started_at
        self._monotonic_fn = monotonic_fn
        self._started_monotonic = float(monotonic_fn())
        self._lock = threading.Lock()
        self._latest_record: DetectionHistoryRecord | None = None

    def update_from_record(self, record: DetectionHistoryRecord) -> None:
        with self._lock:
            self._latest_record = record

    def build_lines(self) -> list[str]:
        with self._lock:
            latest_record = self._latest_record

        elapsed_seconds = max(0.0, float(self._monotonic_fn()) - self._started_monotonic)
        lines = [
            f"Session: {self.started_at.isoformat(timespec='seconds')} (+{elapsed_seconds:.1f}s)",
            f"Model: {self.model_label}",
            f"Cameras: {', '.join(self.camera_labels)}",
        ]
        if latest_record is None:
            lines.append("Latest event: waiting")
            return lines

        score_text = (
            "-"
            if latest_record.final_score is None
            else f"{float(latest_record.final_score):.3f}"
        )
        lines.append(
            "Latest event: "
            f"{latest_record.runtime_event_id} "
            f"{latest_record.result} "
            f"{latest_record.final_class_name or 'none'} "
            f"score={score_text}"
        )
        lines.append(f"Decision source: {latest_record.decision_source}")

        ordered_labels = [str(label) for label in latest_record.camera_labels]
        if not ordered_labels:
            ordered_labels = sorted(str(label) for label in latest_record.camera_votes.keys())
        for label in self._ordered_camera_labels(ordered_labels, latest_record.camera_votes):
            lines.append(f"Cam {label}: {self._format_vote(latest_record.camera_votes.get(label))}")
        return lines

    def _ordered_camera_labels(
        self,
        primary_labels: list[str],
        camera_votes: dict[str, dict[str, object]],
    ) -> list[str]:
        ordered: list[str] = []
        seen: set[str] = set()
        for label in primary_labels + self.camera_labels + sorted(camera_votes.keys()):
            text = str(label)
            if text in seen:
                continue
            seen.add(text)
            ordered.append(text)
        return ordered

    @staticmethod
    def _format_vote(vote: dict[str, object] | None) -> str:
        if not vote:
            return "none"
        class_name = str(vote.get("class_name") or "none")
        score = vote.get("score")
        observation_count = vote.get("observation_count")
        parts = [class_name]
        if score is not None:
            parts.append(f"score={float(score):.3f}")
        if observation_count not in (None, ""):
            parts.append(f"obs={int(observation_count)}")
        return " ".join(parts)


class DebugVideoRecorder:
    def __init__(
        self,
        directory: str,
        basename: str,
        fps: float,
        *,
        writer_factory=None,
        fourcc_fn=None,
        log_fn: Callable[..., None] = print,
    ):
        self.directory = os.path.abspath(directory)
        self.basename = basename
        self.fps = max(1.0, float(fps))
        self.requested_video_path = os.path.join(self.directory, f"{self.basename}.mp4")
        self.video_path: str | None = None
        self.video_codec: str | None = None
        self.frame_count = 0

        os.makedirs(self.directory, exist_ok=True)
        cv2_module = None
        if writer_factory is None or fourcc_fn is None:
            cv2_module = require_cv2()
        self._writer_factory = writer_factory or cv2_module.VideoWriter
        self._fourcc_fn = fourcc_fn or cv2_module.VideoWriter_fourcc
        self._log = log_fn
        self._queue: Queue[object | None] = Queue()
        self._closed = False
        self._error: Exception | None = None
        self._writer = None
        self._thread = threading.Thread(
            target=self._run,
            name="cap-line-debug-video",
            daemon=True,
        )
        self._thread.start()

    def submit(self, frame) -> None:
        if frame is None or self._closed:
            return
        self._raise_if_failed()
        self._queue.put(cap_line_runtime.copy_frame(frame))

    def _raise_if_failed(self) -> None:
        if self._error is not None:
            raise RuntimeError(f"Debug video recording failed: {self._error}") from self._error

    def _open_writer(self, frame):
        frame_height, frame_width = int(frame.shape[0]), int(frame.shape[1])
        writer_specs = [
            ("mp4v", ".mp4"),
            ("MJPG", ".avi"),
        ]
        for codec_name, extension in writer_specs:
            output_path = os.path.join(self.directory, f"{self.basename}{extension}")
            fourcc = self._fourcc_fn(*codec_name)
            writer = self._writer_factory(
                output_path,
                fourcc,
                self.fps,
                (frame_width, frame_height),
            )
            is_opened = True
            if hasattr(writer, "isOpened"):
                is_opened = bool(writer.isOpened())
            if is_opened:
                self.video_path = output_path
                self.video_codec = codec_name
                self._log(f"[DEBUG] recording video={output_path} codec={codec_name}")
                return writer
            if hasattr(writer, "release"):
                writer.release()
        raise RuntimeError("Failed to open a debug video writer with mp4v or MJPG")

    def _run(self) -> None:
        while True:
            frame = self._queue.get()
            try:
                if frame is None:
                    break
                if self._error is not None:
                    continue
                if self._writer is None:
                    self._writer = self._open_writer(frame)
                self._writer.write(frame)
                self.frame_count += 1
            except Exception as exc:
                self._error = exc
            finally:
                self._queue.task_done()

        if self._writer is not None and hasattr(self._writer, "release"):
            self._writer.release()

    def close(self) -> None:
        if self._closed:
            self._raise_if_failed()
            return
        self._closed = True
        self._queue.put(None)
        self._thread.join()
        self._raise_if_failed()


def draw_debug_overlay(frame, lines: list[str], *, cv2_module=None):
    if frame is None:
        return None
    cv2_module = cv2_module or require_cv2()
    if not lines:
        return frame

    overlay = frame.copy()
    frame_height, frame_width = int(frame.shape[0]), int(frame.shape[1])
    padding = 16
    line_height = 24
    box_height = min(frame_height, padding * 2 + line_height * len(lines))

    cv2_module.rectangle(
        overlay,
        (0, 0),
        (frame_width, box_height),
        (0, 0, 0),
        -1,
    )
    blended = cv2_module.addWeighted(overlay, 0.45, frame, 0.55, 0)
    if hasattr(frame, "__setitem__"):
        frame[:, :] = blended
    else:
        frame = blended

    y = padding + 6
    for line in lines:
        cv2_module.putText(
            frame,
            line,
            (padding, y),
            cv2_module.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2_module.LINE_AA,
        )
        y += line_height
    return frame


def build_arg_parser() -> argparse.ArgumentParser:
    parser = cap_line_runtime.build_arg_parser()
    parser.description = "Run cap-line detection and record a full-session diagnostic video."
    parser.add_argument("--record-dir", default=DEFAULT_RECORD_DIR)
    parser.add_argument(
        "--basename",
        default=None,
        help="optional output basename; defaults to a timestamped cap-line debug name",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_arg_parser().parse_args(argv)


def _session_manifest(
    args: argparse.Namespace,
    session_paths: RecordingSessionPaths,
    *,
    started_at: datetime,
    ended_at: datetime | None,
    model_path: str,
    preset_imgsz: int | None,
    recorder,
    status: str,
) -> dict[str, object]:
    video_path = getattr(recorder, "video_path", None) or session_paths.requested_video_path
    return {
        "status": status,
        "started_at": started_at.isoformat(timespec="milliseconds"),
        "ended_at": ended_at.isoformat(timespec="milliseconds") if ended_at is not None else None,
        "output_directory": session_paths.directory,
        "basename": session_paths.basename,
        "requested_video_path": session_paths.requested_video_path,
        "video_path": video_path,
        "video_codec": getattr(recorder, "video_codec", None),
        "events_csv_path": session_paths.events_csv_path,
        "session_json_path": session_paths.session_json_path,
        "frame_count": int(getattr(recorder, "frame_count", 0)),
        "display_enabled": not bool(args.no_display),
        "runtime": {
            "model": args.model,
            "resolved_model_path": model_path,
            "preset_imgsz": preset_imgsz,
            "cams": [str(camera) for camera in args.cams],
            "res": [int(value) for value in args.res],
            "fps": int(args.fps),
            "conf": float(args.conf),
            "imgsz": args.imgsz,
            "timing_camera": int(args.timing_camera),
            "anchor_axis": args.anchor_axis,
            "anchor_line_ratio": float(args.anchor_line_ratio),
            "defect_min_score": float(args.defect_min_score),
            "defect_margin": float(args.defect_margin),
            "single_camera_defect_score": float(args.single_camera_defect_score),
            "merge_window_ms": float(args.merge_window_ms),
            "finalize_quiet_ms": float(args.finalize_quiet_ms),
            "nozzle_distance_mm": float(args.nozzle_distance_mm),
            "belt_speed_mm_per_s": float(args.belt_speed_mm_per_s),
            "trigger_offset_s": float(args.trigger_offset_s),
            "latency_compensation_ms": float(args.latency_compensation_ms),
            "timing_log_dir": os.path.abspath(args.timing_log_dir),
            "review_dir": os.path.abspath(args.review_dir),
            "simulate_gpio": bool(args.simulate_gpio),
        },
    }


def write_session_manifest(file_path: str, payload: dict[str, object]) -> None:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def run_debug_recording(
    args: argparse.Namespace,
    *,
    detector_runner: Callable[..., None] = cap_line_runtime.run_detection,
    recorder_factory=None,
    event_logger_factory=None,
    overlay_renderer: Callable[[object, list[str]], object] = draw_debug_overlay,
    now_fn: Callable[[], datetime] | None = None,
    monotonic_fn: Callable[[], float] = time.monotonic,
    log_fn: Callable[..., None] = print,
) -> DebugRecordingResult:
    session_started_at = (now_fn or (lambda: datetime.now().astimezone()))()
    session_paths = prepare_session_paths(
        args.record_dir,
        basename=args.basename,
        session_now=session_started_at,
    )
    model_path, preset_imgsz = cap_line_runtime.resolve_model_path(args.model)
    overlay_state = DecisionOverlayState(
        model_label=model_path,
        camera_labels=[str(camera) for camera in args.cams],
        started_at=session_started_at,
        monotonic_fn=monotonic_fn,
    )
    if recorder_factory is None:
        recorder_factory = (
            lambda directory, basename, fps, **kwargs: DebugVideoRecorder(
                directory,
                basename,
                fps,
                log_fn=kwargs.get("log_fn", print),
            )
        )
    if event_logger_factory is None:
        event_logger_factory = DebugEventLogger

    recorder = recorder_factory(
        session_paths.directory,
        session_paths.basename,
        float(args.fps),
        log_fn=log_fn,
    )
    event_logger = event_logger_factory(session_paths.events_csv_path)
    write_session_manifest(
        session_paths.session_json_path,
        _session_manifest(
            args,
            session_paths,
            started_at=session_started_at,
            ended_at=None,
            model_path=model_path,
            preset_imgsz=preset_imgsz,
            recorder=recorder,
            status="running",
        ),
    )

    def preview_callback(preview_frame) -> None:
        if preview_frame is None:
            return
        decorated_frame = overlay_renderer(preview_frame, overlay_state.build_lines())
        if decorated_frame is not None:
            recorder.submit(decorated_frame)

    def history_callback(record: DetectionHistoryRecord) -> None:
        overlay_state.update_from_record(record)
        event_logger.log(record)

    try:
        detector_runner(
            args,
            preview_callback=preview_callback,
            history_callback=history_callback,
            log_fn=log_fn,
        )
    finally:
        active_exception = sys.exc_info()[0] is not None
        close_error = None
        try:
            recorder.close()
        except Exception as exc:
            close_error = exc
            if active_exception:
                log_fn(f"[DEBUG] recorder close error: {exc}")
        finally:
            try:
                event_logger.close()
            finally:
                write_session_manifest(
                    session_paths.session_json_path,
                    _session_manifest(
                        args,
                        session_paths,
                        started_at=session_started_at,
                        ended_at=(now_fn or (lambda: datetime.now().astimezone()))(),
                        model_path=model_path,
                        preset_imgsz=preset_imgsz,
                        recorder=recorder,
                        status="error" if active_exception or close_error is not None else "completed",
                    ),
                )
        if close_error is not None and not active_exception:
            raise close_error

    result = DebugRecordingResult(
        directory=session_paths.directory,
        basename=session_paths.basename,
        video_path=getattr(recorder, "video_path", None) or session_paths.requested_video_path,
        requested_video_path=session_paths.requested_video_path,
        events_csv_path=session_paths.events_csv_path,
        session_json_path=session_paths.session_json_path,
        video_codec=getattr(recorder, "video_codec", None),
        frame_count=int(getattr(recorder, "frame_count", 0)),
    )
    log_fn(f"[DEBUG] saved session manifest={result.session_json_path}")
    log_fn(f"[DEBUG] saved event log={result.events_csv_path}")
    log_fn(f"[DEBUG] saved video={result.video_path}")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        run_debug_recording(args)
    except KeyboardInterrupt:
        print("Interrupted by user.")
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
