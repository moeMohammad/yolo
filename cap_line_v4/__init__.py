"""Standalone v4 cap-inspection runtime package.

A fresh, deliberately small implementation: two cameras viewing one cap from two
angles, one ONNX model, per-camera tracking with a defect-wins (OR) decision, and
a single global cooldown that guarantees one air pulse per physical cap.
"""

from .actuation import NullGPIOOutputPin, RejectScheduler
from .config import (
    CLASS_NAMES,
    DEFECT_CLASS_ID,
    DEFAULT_MODEL,
    RuntimeConfig,
    build_arg_parser,
    class_name,
    config_from_args,
    parse_args,
    validate_config,
)
from .decision import CapEvent, CapEventManager
from .model import postprocess, preprocess, resolve_imgsz, resolve_model_path
from .runtime import Clock, CameraWorker, compose_preview, draw_boxes, run_detection
from .tracking import CameraTracker, Track, box_iou
from .types import Box, CapEventRecord, CapturedFrame, PerfSnapshot, RuntimeCallbacks

__all__ = [
    "Box",
    "CLASS_NAMES",
    "CameraTracker",
    "CameraWorker",
    "CapEvent",
    "CapEventManager",
    "CapEventRecord",
    "CapturedFrame",
    "Clock",
    "DEFECT_CLASS_ID",
    "DEFAULT_MODEL",
    "NullGPIOOutputPin",
    "PerfSnapshot",
    "RejectScheduler",
    "RuntimeCallbacks",
    "RuntimeConfig",
    "Track",
    "box_iou",
    "build_arg_parser",
    "class_name",
    "compose_preview",
    "config_from_args",
    "draw_boxes",
    "parse_args",
    "postprocess",
    "preprocess",
    "resolve_imgsz",
    "resolve_model_path",
    "run_detection",
    "validate_config",
]
