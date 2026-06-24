from __future__ import annotations

import argparse
import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from gpio_output import DEFAULT_TRIGGER_PIN


SCRIPT_DIR = Path(__file__).resolve().parent.parent
CLASS_NAMES = ("undefected", "dirt_defect")
DEFECT_CLASS_ID = 1
DEFAULT_MODEL = "dirtv2.onnx"
DEFAULT_CAMERA_RESOLUTION = (960, 600)
DEFAULT_CAMERA_FPS = 60
DEFAULT_CAMERA_PIXEL_FORMAT = "YUYV"
DEFAULT_MIRROR_CAMERAS = (False, True)
TRACKING_DETECTION_THRESHOLD = 0.45
DEFECT_REJECT_THRESHOLD = 0.45
DEFAULT_PAIR_MAX_SKEW_MS = 40.0
DEFAULT_DEBUG_BURST_BEFORE_FRAMES = 3
DEFAULT_DEBUG_BURST_AFTER_FRAMES = 3
DEFAULT_ONNX_INTRA_OP_THREADS = max(1, (os.cpu_count() or 2) // 2)
DEFAULT_PERF_LOG_INTERVAL_S = 5.0
DEFAULT_NOZZLE_DISTANCE_MM = 430.0
DEFAULT_BELT_SPEED_MM_PER_S = 275.0
DEFAULT_TRIGGER_OFFSET_S = -0.23
DEFAULT_FINALIZE_QUIET_MS = 30.0
DEFAULT_LATENCY_COMPENSATION_MS = 50.0
DEFAULT_PREVIEW_LATENCY_COMPENSATION_MS = 0.0
DEFAULT_LIVE_PREVIEW_FPS = 30.0
DEFAULT_ANCHOR_LINE_RATIO = 0.75
DEFAULT_ACTUATION_SNAPSHOT_HOLD_MS = 450.0
DEFAULT_SAVE_QUEUE_WARNING_THRESHOLD = 25
DEFAULT_CAPTURE_BUFFER_FRAMES = 8
DEFAULT_DECISION_DEADLINE_GUARD_MS = 25.0
DEFAULT_ACTUATION_WINDOW_MS = 100.0
DEFAULT_ACTUATION_PREDICTION_HORIZON_MS = 120.0


@dataclass(frozen=True)
class RuntimeConfig:
    model: str = DEFAULT_MODEL
    cameras: tuple[str, str] = ("0", "3")
    mirror_cameras: tuple[bool, bool] = DEFAULT_MIRROR_CAMERAS
    resolution: tuple[int, int] = DEFAULT_CAMERA_RESOLUTION
    target_fps: int = DEFAULT_CAMERA_FPS
    pixel_format: str = DEFAULT_CAMERA_PIXEL_FORMAT
    exposure: int = 8
    imgsz: int | None = None
    no_display: bool = False
    tracking_threshold: float = TRACKING_DETECTION_THRESHOLD
    reject_threshold: float = DEFECT_REJECT_THRESHOLD
    trigger_pin: str | int = DEFAULT_TRIGGER_PIN
    trigger_duration: float = 0.3
    trigger_min_gap: float = 0.0
    track_iou: float = 0.3
    max_missing_frames: int = 1
    merge_window_ms: float = 150.0
    finalize_quiet_ms: float = DEFAULT_FINALIZE_QUIET_MS
    timing_camera: int = 0
    anchor_axis: str = "x"
    anchor_line_ratio: float = DEFAULT_ANCHOR_LINE_RATIO
    nozzle_distance_mm: float = DEFAULT_NOZZLE_DISTANCE_MM
    belt_speed_mm_per_s: float = DEFAULT_BELT_SPEED_MM_PER_S
    trigger_offset_s: float = DEFAULT_TRIGGER_OFFSET_S
    latency_compensation_ms: float = DEFAULT_LATENCY_COMPENSATION_MS
    preview_latency_compensation_ms: float = DEFAULT_PREVIEW_LATENCY_COMPENSATION_MS
    actuation_snapshot_hold_ms: float = DEFAULT_ACTUATION_SNAPSHOT_HOLD_MS
    serial_inference: bool = False
    onnx_intra_op_threads: int = DEFAULT_ONNX_INTRA_OP_THREADS
    perf_log_interval_s: float = DEFAULT_PERF_LOG_INTERVAL_S
    live_preview_fps: float = DEFAULT_LIVE_PREVIEW_FPS
    pair_max_skew_ms: float = DEFAULT_PAIR_MAX_SKEW_MS
    capture_buffer_frames: int = DEFAULT_CAPTURE_BUFFER_FRAMES
    single_camera_wait_ms: float | None = None
    decision_deadline_guard_ms: float = DEFAULT_DECISION_DEADLINE_GUARD_MS
    actuation_window_ms: float = DEFAULT_ACTUATION_WINDOW_MS
    actuation_prediction_horizon_ms: float = DEFAULT_ACTUATION_PREDICTION_HORIZON_MS
    log_skip_events: bool = True
    debug_burst_before_frames: int = DEFAULT_DEBUG_BURST_BEFORE_FRAMES
    debug_burst_after_frames: int = DEFAULT_DEBUG_BURST_AFTER_FRAMES
    timing_log_dir: str = str(SCRIPT_DIR / "data" / "timing_logs_v3")
    debug_dir: str = str(SCRIPT_DIR / "resources" / "debugging_v3")
    pictures_dir: str = str(SCRIPT_DIR / "resources" / "pictures_v3")
    session_log_dir: str = str(SCRIPT_DIR / "resources" / "debugging_v3" / "sessions")
    simulate_gpio: bool = os.name == "nt"
    save_queue_warning_threshold: int = DEFAULT_SAVE_QUEUE_WARNING_THRESHOLD

    @classmethod
    def defaults(cls) -> "RuntimeConfig":
        return cls()

    def to_json_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["cameras"] = list(self.cameras)
        data["mirror_cameras"] = list(self.mirror_cameras)
        data["resolution"] = list(self.resolution)
        return data

    @classmethod
    def from_json_dict(cls, data: dict[str, Any]) -> "RuntimeConfig":
        defaults = cls.defaults()
        allowed = defaults.to_json_dict()
        merged = {**allowed, **{key: value for key, value in data.items() if key in allowed}}
        if float(merged.get("anchor_line_ratio", defaults.anchor_line_ratio)) <= 0.60:
            merged["anchor_line_ratio"] = defaults.anchor_line_ratio
        merged["cameras"] = tuple(str(value) for value in merged["cameras"])  # type: ignore[assignment]
        mirror_cameras = list(merged.get("mirror_cameras", defaults.mirror_cameras))
        if len(mirror_cameras) != 2:
            mirror_cameras = list(defaults.mirror_cameras)
        merged["mirror_cameras"] = tuple(bool(value) for value in mirror_cameras)  # type: ignore[assignment]
        merged["resolution"] = tuple(int(value) for value in merged["resolution"])  # type: ignore[assignment]
        return cls(**merged)


def normalize_pixel_format(pixel_format: str) -> str:
    normalized = str(pixel_format).strip().upper()
    return "YUYV" if normalized == "YUY2" else normalized


def validate_config(config: RuntimeConfig) -> None:
    if len(config.cameras) != 2:
        raise ValueError("V3 requires exactly two cameras")
    if len(config.mirror_cameras) != 2:
        raise ValueError("mirror_cameras must contain exactly two values")
    if int(config.resolution[0]) <= 0 or int(config.resolution[1]) <= 0:
        raise ValueError("resolution must be positive")
    if int(config.target_fps) <= 0:
        raise ValueError("target_fps must be greater than 0")
    if normalize_pixel_format(config.pixel_format) != DEFAULT_CAMERA_PIXEL_FORMAT:
        raise ValueError("--pixel-format must be YUYV for Arducam B0495 cameras")
    for name, value in (
        ("tracking_threshold", config.tracking_threshold),
        ("reject_threshold", config.reject_threshold),
        ("anchor_line_ratio", config.anchor_line_ratio),
        ("track_iou", config.track_iou),
    ):
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1")
    if config.trigger_duration <= 0:
        raise ValueError("trigger_duration must be greater than 0")
    if config.trigger_min_gap < 0:
        raise ValueError("trigger_min_gap must be 0 or greater")
    if config.max_missing_frames < 0:
        raise ValueError("max_missing_frames must be 0 or greater")
    if config.capture_buffer_frames < 1:
        raise ValueError("capture_buffer_frames must be at least 1")
    if config.merge_window_ms < 0 or config.finalize_quiet_ms < 0:
        raise ValueError("merge/finalize windows must be 0 or greater")
    if config.single_camera_wait_ms is not None and config.single_camera_wait_ms < 0:
        raise ValueError("single_camera_wait_ms must be 0 or greater")
    for name, value in (
        ("decision_deadline_guard_ms", config.decision_deadline_guard_ms),
        ("actuation_window_ms", config.actuation_window_ms),
        ("actuation_prediction_horizon_ms", config.actuation_prediction_horizon_ms),
    ):
        if value < 0:
            raise ValueError(f"{name} must be 0 or greater")
    if config.timing_camera not in (0, 1):
        raise ValueError("timing_camera must be 0 or 1")
    if config.anchor_axis not in {"x", "y"}:
        raise ValueError("anchor_axis must be x or y")
    if config.belt_speed_mm_per_s <= 0:
        raise ValueError("belt_speed_mm_per_s must be greater than 0")
    if config.latency_compensation_ms < 0:
        raise ValueError("latency_compensation_ms must be 0 or greater")
    if config.preview_latency_compensation_ms < 0:
        raise ValueError("preview_latency_compensation_ms must be 0 or greater")
    if config.actuation_snapshot_hold_ms < 0:
        raise ValueError("actuation_snapshot_hold_ms must be 0 or greater")


def build_arg_parser() -> argparse.ArgumentParser:
    defaults = RuntimeConfig.defaults()
    parser = argparse.ArgumentParser(
        description="Run standalone V3 RGB cap detection, tracking, preview, and GPIO triggering."
    )
    parser.add_argument("--model", default=defaults.model)
    parser.add_argument("--cams", nargs=2, default=list(defaults.cameras))
    parser.add_argument("--mirror-camera-0", action=argparse.BooleanOptionalAction, default=defaults.mirror_cameras[0])
    parser.add_argument("--mirror-camera-1", action=argparse.BooleanOptionalAction, default=defaults.mirror_cameras[1])
    parser.add_argument("--res", type=int, nargs=2, default=list(defaults.resolution))
    parser.add_argument("--target-fps", "--fps", type=int, default=defaults.target_fps)
    parser.add_argument("--pixel-format", default=defaults.pixel_format)
    parser.add_argument("--exposure", type=int, default=defaults.exposure)
    parser.add_argument("--imgsz", type=int, default=defaults.imgsz)
    parser.add_argument("--no-display", action="store_true", default=defaults.no_display)
    parser.add_argument("--tracking-threshold", type=float, default=defaults.tracking_threshold)
    parser.add_argument("--reject-threshold", type=float, default=defaults.reject_threshold)
    parser.add_argument("--trigger-pin", default=defaults.trigger_pin)
    parser.add_argument("--trigger-duration", type=float, default=defaults.trigger_duration)
    parser.add_argument("--trigger-min-gap", type=float, default=defaults.trigger_min_gap)
    parser.add_argument("--track-iou", type=float, default=defaults.track_iou)
    parser.add_argument("--max-missing-frames", type=int, default=defaults.max_missing_frames)
    parser.add_argument("--merge-window-ms", type=float, default=defaults.merge_window_ms)
    parser.add_argument("--finalize-quiet-ms", type=float, default=defaults.finalize_quiet_ms)
    parser.add_argument("--timing-camera", type=int, default=defaults.timing_camera)
    parser.add_argument("--anchor-axis", choices=["x", "y"], default=defaults.anchor_axis)
    parser.add_argument("--anchor-line-ratio", type=float, default=defaults.anchor_line_ratio)
    parser.add_argument("--nozzle-distance-mm", type=float, default=defaults.nozzle_distance_mm)
    parser.add_argument("--belt-speed-mm-per-s", type=float, default=defaults.belt_speed_mm_per_s)
    parser.add_argument("--trigger-offset-s", type=float, default=defaults.trigger_offset_s)
    parser.add_argument("--latency-compensation-ms", type=float, default=defaults.latency_compensation_ms)
    parser.add_argument("--preview-latency-compensation-ms", type=float, default=defaults.preview_latency_compensation_ms)
    parser.add_argument("--actuation-snapshot-hold-ms", type=float, default=defaults.actuation_snapshot_hold_ms)
    parser.add_argument("--serial-inference", action="store_true", default=defaults.serial_inference)
    parser.add_argument("--onnx-intra-op-threads", type=int, default=defaults.onnx_intra_op_threads)
    parser.add_argument("--perf-log-interval-s", type=float, default=defaults.perf_log_interval_s)
    parser.add_argument("--live-preview-fps", type=float, default=defaults.live_preview_fps)
    parser.add_argument("--pair-max-skew-ms", type=float, default=defaults.pair_max_skew_ms)
    parser.add_argument("--capture-buffer-frames", type=int, default=defaults.capture_buffer_frames)
    parser.add_argument("--single-camera-wait-ms", type=float, default=defaults.single_camera_wait_ms)
    parser.add_argument("--decision-deadline-guard-ms", type=float, default=defaults.decision_deadline_guard_ms)
    parser.add_argument("--actuation-window-ms", type=float, default=defaults.actuation_window_ms)
    parser.add_argument("--actuation-prediction-horizon-ms", type=float, default=defaults.actuation_prediction_horizon_ms)
    parser.add_argument("--log-skip-events", action=argparse.BooleanOptionalAction, default=defaults.log_skip_events)
    parser.add_argument("--debug-burst-before-frames", type=int, default=defaults.debug_burst_before_frames)
    parser.add_argument("--debug-burst-after-frames", type=int, default=defaults.debug_burst_after_frames)
    parser.add_argument("--timing-log-dir", default=defaults.timing_log_dir)
    parser.add_argument("--debug-dir", default=defaults.debug_dir)
    parser.add_argument("--pictures-dir", default=defaults.pictures_dir)
    parser.add_argument("--session-log-dir", default=defaults.session_log_dir)
    parser.add_argument("--simulate-gpio", action="store_true", default=defaults.simulate_gpio)
    parser.add_argument("--save-queue-warning-threshold", type=int, default=defaults.save_queue_warning_threshold)
    return parser


def config_from_args(args: argparse.Namespace) -> RuntimeConfig:
    config = RuntimeConfig(
        model=args.model,
        cameras=tuple(str(value) for value in args.cams),  # type: ignore[arg-type]
        mirror_cameras=(bool(args.mirror_camera_0), bool(args.mirror_camera_1)),
        resolution=(int(args.res[0]), int(args.res[1])),
        target_fps=int(args.target_fps),
        pixel_format=normalize_pixel_format(args.pixel_format),
        exposure=int(args.exposure),
        imgsz=args.imgsz,
        no_display=bool(args.no_display),
        tracking_threshold=float(args.tracking_threshold),
        reject_threshold=float(args.reject_threshold),
        trigger_pin=args.trigger_pin,
        trigger_duration=float(args.trigger_duration),
        trigger_min_gap=float(args.trigger_min_gap),
        track_iou=float(args.track_iou),
        max_missing_frames=int(args.max_missing_frames),
        merge_window_ms=float(args.merge_window_ms),
        finalize_quiet_ms=float(args.finalize_quiet_ms),
        timing_camera=int(args.timing_camera),
        anchor_axis=str(args.anchor_axis),
        anchor_line_ratio=float(args.anchor_line_ratio),
        nozzle_distance_mm=float(args.nozzle_distance_mm),
        belt_speed_mm_per_s=float(args.belt_speed_mm_per_s),
        trigger_offset_s=float(args.trigger_offset_s),
        latency_compensation_ms=float(args.latency_compensation_ms),
        preview_latency_compensation_ms=float(args.preview_latency_compensation_ms),
        actuation_snapshot_hold_ms=float(args.actuation_snapshot_hold_ms),
        serial_inference=bool(args.serial_inference),
        onnx_intra_op_threads=int(args.onnx_intra_op_threads),
        perf_log_interval_s=float(args.perf_log_interval_s),
        live_preview_fps=float(args.live_preview_fps),
        pair_max_skew_ms=float(args.pair_max_skew_ms),
        capture_buffer_frames=int(args.capture_buffer_frames),
        single_camera_wait_ms=None if args.single_camera_wait_ms is None else float(args.single_camera_wait_ms),
        decision_deadline_guard_ms=float(args.decision_deadline_guard_ms),
        actuation_window_ms=float(args.actuation_window_ms),
        actuation_prediction_horizon_ms=float(args.actuation_prediction_horizon_ms),
        log_skip_events=bool(args.log_skip_events),
        debug_burst_before_frames=int(args.debug_burst_before_frames),
        debug_burst_after_frames=int(args.debug_burst_after_frames),
        timing_log_dir=str(args.timing_log_dir),
        debug_dir=str(args.debug_dir),
        pictures_dir=str(args.pictures_dir),
        session_log_dir=str(args.session_log_dir),
        simulate_gpio=bool(args.simulate_gpio),
        save_queue_warning_threshold=int(args.save_queue_warning_threshold),
    )
    validate_config(config)
    return config


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_arg_parser().parse_args(argv)
