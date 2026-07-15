"""Standalone v6 cap-inspection runtime package.

Jetson-targeted cap inspection with physical track qualification, temporal
defect confirmation, presence-cycle idempotency, latest-frame capture, and a
stale-safe reject scheduler. GPIO defaults to ``gpio_output.py`` on BOARD pin 7
(GPIO09); the Raspberry Pi backend remains selectable.
"""

from .actuation import NullGPIOOutputPin, RejectScheduler
from .config import (
    CLASS_NAMES,
    DEFECT_CLASS_ID,
    DEFAULT_MODEL,
    GPIO_BACKENDS,
    RuntimeConfig,
    build_arg_parser,
    class_name,
    config_from_args,
    parse_args,
    validate_config,
)
from .decision import CapEvent, CapEventManager
from .model import deduplicate_boxes, postprocess, preprocess, resolve_imgsz, resolve_model_path
from .runtime import Clock, CameraWorker, compose_preview, draw_boxes, resolve_pin_factory, run_detection
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
    "GPIO_BACKENDS",
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
    "deduplicate_boxes",
    "parse_args",
    "postprocess",
    "preprocess",
    "resolve_imgsz",
    "resolve_model_path",
    "resolve_pin_factory",
    "run_detection",
    "validate_config",
]
