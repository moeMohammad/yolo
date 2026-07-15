"""Standalone v7 cap-inspection runtime package.

Two-stage Jetson-targeted cap inspection: a single-class cap detector plus a
binary dirt crop classifier, with probability-based track voting (consecutive
dirty frames AND trimmed-mean P(dirt)) replacing v6's per-frame class latch.
Physical track qualification, presence-cycle idempotency, latest-frame capture,
and the stale-safe reject scheduler carry over from v6 unchanged. GPIO defaults
to ``gpio_output.py`` on BOARD pin 7 (GPIO09); the Raspberry Pi backend remains
selectable.
"""

from .actuation import NullGPIOOutputPin, RejectScheduler
from .config import (
    CLASS_NAMES,
    CLASSIFIER_DIRT_INDEX,
    DEFECT_CLASS_ID,
    DEFAULT_CLASSIFIER_MODEL,
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
from .model import (
    classifier_postprocess,
    classifier_preprocess,
    classify_dirt_probability,
    crop_cap_region,
    deduplicate_boxes,
    postprocess,
    preprocess,
    resolve_imgsz,
    resolve_model_path,
)
from .runtime import Clock, CameraWorker, compose_preview, draw_boxes, resolve_pin_factory, run_detection
from .tracking import CameraTracker, Track, box_iou, trimmed_mean
from .types import Box, CapEventRecord, CapturedFrame, PerfSnapshot, RuntimeCallbacks

__all__ = [
    "Box",
    "CLASS_NAMES",
    "CLASSIFIER_DIRT_INDEX",
    "CameraTracker",
    "CameraWorker",
    "CapEvent",
    "CapEventManager",
    "CapEventRecord",
    "CapturedFrame",
    "Clock",
    "DEFECT_CLASS_ID",
    "DEFAULT_CLASSIFIER_MODEL",
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
    "classifier_postprocess",
    "classifier_preprocess",
    "classify_dirt_probability",
    "compose_preview",
    "config_from_args",
    "crop_cap_region",
    "draw_boxes",
    "deduplicate_boxes",
    "parse_args",
    "postprocess",
    "preprocess",
    "resolve_imgsz",
    "resolve_model_path",
    "resolve_pin_factory",
    "run_detection",
    "trimmed_mean",
    "validate_config",
]
