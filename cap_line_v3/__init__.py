"""Standalone V3 cap-line runtime package."""

from .config import RuntimeConfig, build_arg_parser, config_from_args, parse_args, replace
from .decision import TrackedCap, TrackedCapManager, decide_decision_ready, decide_tracked_cap
from .pairing import default_single_camera_wait_ms, select_capture_batch, select_synchronized_frame_pair
from .preview import overlay_stale_timeout_s, predict_preview_overlay, resolve_preview_views
from .runtime import LatestFrameCameraReader, LivePreviewPublisher, mirror_frame_horizontal, postprocess, preprocess, run_detection
from .types import (
    CaptureBatch,
    CapturedFrame,
    DetectionHistoryRecord,
    DetectionPacket,
    FramePair,
    PairDropStats,
    RuntimeCallbacks,
    RuntimePerformanceSnapshot,
    TimingLogRecord,
    TrackObservation,
    TrackedCapDecision,
)

__all__ = [
    "CapturedFrame",
    "CaptureBatch",
    "DetectionHistoryRecord",
    "DetectionPacket",
    "FramePair",
    "LatestFrameCameraReader",
    "LivePreviewPublisher",
    "PairDropStats",
    "RuntimeCallbacks",
    "RuntimeConfig",
    "RuntimePerformanceSnapshot",
    "TimingLogRecord",
    "TrackObservation",
    "TrackedCap",
    "TrackedCapManager",
    "TrackedCapDecision",
    "build_arg_parser",
    "config_from_args",
    "decide_decision_ready",
    "decide_tracked_cap",
    "default_single_camera_wait_ms",
    "overlay_stale_timeout_s",
    "mirror_frame_horizontal",
    "parse_args",
    "predict_preview_overlay",
    "resolve_preview_views",
    "postprocess",
    "preprocess",
    "replace",
    "run_detection",
    "select_capture_batch",
    "select_synchronized_frame_pair",
]
