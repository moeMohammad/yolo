"""Standalone V3 cap-line runtime package."""

from .config import RuntimeConfig, build_arg_parser, config_from_args, parse_args, replace
from .decision import TrackedCap, TrackedCapManager, decide_decision_ready, decide_tracked_cap
from .pairing import select_synchronized_frame_pair
from .preview import overlay_stale_timeout_s, predict_preview_overlay, resolve_preview_views
from .runtime import LatestFrameCameraReader, LivePreviewPublisher, mirror_frame_horizontal, postprocess, preprocess, run_detection
from .types import (
    CapturedFrame,
    DetectionHistoryRecord,
    DetectionPacket,
    FramePair,
    RuntimeCallbacks,
    RuntimePerformanceSnapshot,
    TimingLogRecord,
    TrackObservation,
    TrackedCapDecision,
)

__all__ = [
    "CapturedFrame",
    "DetectionHistoryRecord",
    "DetectionPacket",
    "FramePair",
    "LatestFrameCameraReader",
    "LivePreviewPublisher",
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
    "overlay_stale_timeout_s",
    "mirror_frame_horizontal",
    "parse_args",
    "predict_preview_overlay",
    "resolve_preview_views",
    "postprocess",
    "preprocess",
    "replace",
    "run_detection",
    "select_synchronized_frame_pair",
]
