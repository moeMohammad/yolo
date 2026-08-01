"""Slim v7 runtime configuration.

v7 replaces the single 2-class detector with the two-stage pipeline trained in
``detectx/two_stage_training.ipynb``:

1. ``model`` — a single-class *cap* detector (``cap_detector_640.onnx``). Its
   only job is presence/geometry; it knows nothing about dirt.
2. ``classifier_model`` — a binary crop classifier
   (``dirt_classifier_384.onnx``) that scores each tracked cap crop with
   ``P(dirt)``. Output index 0 is ``dirt_defect`` (softmax baked into the
   export, plain RGB/255 input — verified against the training notebook).

The per-frame class vote of v6 becomes a per-frame *probability* vote:
a track is a defect only when at least ``min_defect_frames`` consecutive
observations score ``P(dirt) >= frame_dirt_threshold`` AND the trimmed-mean
``P(dirt)`` over the whole track reaches ``track_dirt_threshold``. Thresholds
come from the cap-level sweep in ``training_summary.json`` and must be
re-calibrated whenever either model is retrained.

Everything else (Jetson GPIO defaults, physical track qualification,
presence-cycle idempotency, stale-safe scheduler) is unchanged from v6.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from gpio_output import GPIO09


SCRIPT_DIR = Path(__file__).resolve().parent.parent
CLASS_NAMES = ("undefected", "dirt_defect")
UNDEFECTED_CLASS_ID = 0
DEFECT_CLASS_ID = 1
# The classifier ONNX emits [P(dirt_defect), P(undefected)] (alphabetical
# training-folder order, verified on the exported model).
CLASSIFIER_DIRT_INDEX = 0

GPIO_BACKENDS = ("rpi", "jetson")
SETTINGS_SCHEMA_VERSION = 3

DEFAULT_MODEL = "cap_detector_640.onnx"
DEFAULT_CLASSIFIER_MODEL = "dirt_classifier_384.onnx"
DEFAULT_CAMERAS = ("0", "2")
# The saved footage travels right -> left after the recorder's configured
# transforms. It does not prove the orientation of each camera's raw live
# frames, so mirror calibration remains an explicit deployment check.
DEFAULT_MIRROR_CAMERAS = (False, False)
DEFAULT_RESOLUTION = (960, 600)
# The cameras in the recorded rig deliver about 30 real frames/s. Requesting
# 60 produced misleading video timestamps and extra USB/capture pressure while
# providing no additional observations.
DEFAULT_TARGET_FPS = 30
DEFAULT_EXPOSURE = 8
DEFAULT_PIXEL_FORMAT = "YUYV"
DEFAULT_ONNX_INTRA_OP_THREADS = max(1, (os.cpu_count() or 2) // 2)
# Stage 1: minimum detector confidence for a box to count as a cap at all.
DEFAULT_DETECT_THRESHOLD = 0.25
# Stage 2, per frame: a crop scoring at least this P(dirt) is a dirty frame.
DEFAULT_FRAME_DIRT_THRESHOLD = 0.50
# Stage 2, per track: trimmed-mean P(dirt) across the track must also reach
# this before the track is a defect (operating point from the test-set sweep).
DEFAULT_TRACK_DIRT_THRESHOLD = 0.45
# Crop margin around the detector box before square-padding, matching the
# training-crop generation in build_two_stage_dataset.py exactly.
DEFAULT_CROP_MARGIN = 0.10
# Only classify frames whose box center lies in the central band of the frame
# along the belt axis (fraction of the frame dimension). The classifier was
# trained on center-frame captures; entry/exit perspectives are out-of-domain
# and score spuriously high P(dirt), so edge frames contribute no evidence.
DEFAULT_CLASSIFY_BAND_RATIO = 0.75
# Classify at most this many boxes per frame (normally one cap is visible).
DEFAULT_MAX_CLASSIFIED_BOXES = 2
DEFAULT_DUPLICATE_IOU_THRESHOLD = 0.65
DEFAULT_TRACK_IOU = 0.3
# Must exceed the worst processed-frame interval or tracks fragment mid-cap;
# must stay below the gap between consecutive caps at one camera.
DEFAULT_TRACK_TIMEOUT_MS = 350.0
# A detector confidence score is not proof that a physical cap is present.
# Qualify tracks using multiple frames and conveyor-like motion before they are
# allowed to reach the event manager / air scheduler.
DEFAULT_MIN_TRACK_FRAMES = 2
DEFAULT_MIN_TRACK_TRAVEL_RATIO = 0.35
DEFAULT_MIN_TRACK_DIRECTIONALITY = 0.60
DEFAULT_MIN_DEFECT_FRAMES = 2
# A low-throughput Jetson may observe a fast cap only once on each side of the
# gate. Requiring two observations per side made the system mathematically
# incapable of seeing caps below roughly 7.5 processed FPS.
DEFAULT_MIN_LINE_SIDE_FRAMES = 1
# Never silently call a cap clean when the classifier did not produce enough
# evidence. In production an unknown inspection is rejected fail-closed.
DEFAULT_MIN_CLASSIFIED_FRAMES = 2
DEFAULT_REQUIRED_INSPECTED_CAMERAS = 2
DEFAULT_REJECT_UNINSPECTED = True
DEFAULT_PRESENCE_LINE_AXIS = "x"
DEFAULT_PRESENCE_LINE_RATIO = 0.50
DEFAULT_PRESENCE_DIRECTION = "negative"
DEFAULT_MAX_TRACK_GAP_MS = 300.0
# Near-simultaneous line crossings in the same perpendicular image band retain
# one camera-local presence-cycle id, making actuation idempotent across
# fragments without conflating concurrently visible caps in different bands.
DEFAULT_PRESENCE_CLEAR_MS = 150.0
DEFAULT_FIRE_DELAY_S = 0.0
# Two finished tracks are the same physical cap when their last_seen times are
# within this window (cross-camera exit skew), regardless of reporting lag.
DEFAULT_MERGE_WINDOW_MS = 150.0
# Hard once-per-cap guarantee: suppress any fire whose reference is closer than
# this to the previous fire's reference. A pulse is trigger_duration=0.3 s long,
# so two distinguishable caps cannot reach the nozzle 250 ms apart.
DEFAULT_MIN_FIRE_INTERVAL_MS = 250.0
DEFAULT_GPIO_BACKEND = "jetson"
DEFAULT_TRIGGER_PIN = GPIO09  # Legacy constant name; physical Jetson BOARD pin 7.
DEFAULT_TRIGGER_DURATION = 0.3
DEFAULT_TRIGGER_MIN_GAP = 0.0
# Runnable jobs older than this are backlog, not timely reject commands. This
# prevents a burst of false events from continuing to pulse after the scene is
# clear. Age starts when the job becomes runnable, not at cap last_seen.
DEFAULT_TRIGGER_MAX_QUEUE_AGE_MS = 250.0
DEFAULT_TRIGGER_MAX_LATENESS_MS = 500.0
DEFAULT_MAX_FRAME_AGE_MS = 500.0
DEFAULT_CAMERA_READ_TIMEOUT_S = 2.0
DEFAULT_SIMULATE_GPIO = True
DEFAULT_LIVE_PREVIEW_FPS = 30.0
DEFAULT_DB_PATH = str(SCRIPT_DIR / "data" / "cap_line_history_v7.sqlite3")


def class_name(class_id: int | None) -> str | None:
    if class_id is None:
        return None
    if 0 <= int(class_id) < len(CLASS_NAMES):
        return CLASS_NAMES[int(class_id)]
    return f"class_{int(class_id)}"


def normalize_pixel_format(pixel_format: str) -> str:
    normalized = str(pixel_format).strip().upper()
    return "YUYV" if normalized == "YUY2" else normalized


def normalize_gpio_backend(backend: str) -> str:
    normalized = str(backend).strip().lower()
    return normalized if normalized in GPIO_BACKENDS else DEFAULT_GPIO_BACKEND


@dataclass(frozen=True)
class RuntimeConfig:
    settings_schema_version: int = SETTINGS_SCHEMA_VERSION
    model: str = DEFAULT_MODEL
    classifier_model: str = DEFAULT_CLASSIFIER_MODEL
    cameras: tuple[str, str] = DEFAULT_CAMERAS
    mirror_cameras: tuple[bool, bool] = DEFAULT_MIRROR_CAMERAS
    resolution: tuple[int, int] = DEFAULT_RESOLUTION
    target_fps: int = DEFAULT_TARGET_FPS
    exposure: int = DEFAULT_EXPOSURE
    pixel_format: str = DEFAULT_PIXEL_FORMAT
    imgsz: int | None = None
    classifier_imgsz: int | None = None
    onnx_intra_op_threads: int = DEFAULT_ONNX_INTRA_OP_THREADS
    detect_threshold: float = DEFAULT_DETECT_THRESHOLD
    frame_dirt_threshold: float = DEFAULT_FRAME_DIRT_THRESHOLD
    track_dirt_threshold: float = DEFAULT_TRACK_DIRT_THRESHOLD
    crop_margin: float = DEFAULT_CROP_MARGIN
    classify_band_ratio: float = DEFAULT_CLASSIFY_BAND_RATIO
    max_classified_boxes: int = DEFAULT_MAX_CLASSIFIED_BOXES
    duplicate_iou_threshold: float = DEFAULT_DUPLICATE_IOU_THRESHOLD
    track_iou: float = DEFAULT_TRACK_IOU
    track_timeout_ms: float = DEFAULT_TRACK_TIMEOUT_MS
    min_track_frames: int = DEFAULT_MIN_TRACK_FRAMES
    min_track_travel_ratio: float = DEFAULT_MIN_TRACK_TRAVEL_RATIO
    min_track_directionality: float = DEFAULT_MIN_TRACK_DIRECTIONALITY
    min_defect_frames: int = DEFAULT_MIN_DEFECT_FRAMES
    min_line_side_frames: int = DEFAULT_MIN_LINE_SIDE_FRAMES
    min_classified_frames: int = DEFAULT_MIN_CLASSIFIED_FRAMES
    required_inspected_cameras: int = DEFAULT_REQUIRED_INSPECTED_CAMERAS
    reject_uninspected: bool = DEFAULT_REJECT_UNINSPECTED
    presence_line_axis: str = DEFAULT_PRESENCE_LINE_AXIS
    presence_line_ratio: float = DEFAULT_PRESENCE_LINE_RATIO
    presence_direction: str = DEFAULT_PRESENCE_DIRECTION
    max_track_gap_ms: float = DEFAULT_MAX_TRACK_GAP_MS
    presence_clear_ms: float = DEFAULT_PRESENCE_CLEAR_MS
    fire_delay_s: float = DEFAULT_FIRE_DELAY_S
    merge_window_ms: float = DEFAULT_MERGE_WINDOW_MS
    min_fire_interval_ms: float = DEFAULT_MIN_FIRE_INTERVAL_MS
    gpio_backend: str = DEFAULT_GPIO_BACKEND
    trigger_pin: str | int = DEFAULT_TRIGGER_PIN
    trigger_duration: float = DEFAULT_TRIGGER_DURATION
    trigger_min_gap: float = DEFAULT_TRIGGER_MIN_GAP
    trigger_max_queue_age_ms: float = DEFAULT_TRIGGER_MAX_QUEUE_AGE_MS
    trigger_max_lateness_ms: float = DEFAULT_TRIGGER_MAX_LATENESS_MS
    max_frame_age_ms: float = DEFAULT_MAX_FRAME_AGE_MS
    camera_read_timeout_s: float = DEFAULT_CAMERA_READ_TIMEOUT_S
    # Fresh and migrated installs start electrically disarmed. The operator
    # must calibrate a viable gate-to-nozzle delay before choosing real GPIO.
    simulate_gpio: bool = DEFAULT_SIMULATE_GPIO
    live_preview_fps: float = DEFAULT_LIVE_PREVIEW_FPS
    db_path: str = DEFAULT_DB_PATH
    no_display: bool = False

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
        """Build a config from a (possibly partial / legacy) settings dict.

        Unknown keys are dropped so a v5/v6 settings file can be pointed at v7
        without crashing; missing keys fall back to defaults. A settings dict
        without ``classifier_model`` predates the two-stage pipeline, so its
        ``model`` value names a single-stage 2-class detector — silently
        running that as the cap detector would break inspection, so both model
        fields fall back to the v7 defaults in that case.
        """

        defaults = cls.defaults()
        allowed = defaults.to_json_dict()
        data = dict(data)
        try:
            source_schema = int(data.get("settings_schema_version", 0) or 0)
        except (TypeError, ValueError):
            source_schema = 0

        # v7 originally shipped with geometry and throughput defaults that were
        # later proven wrong on the recorded rig. The ignored JSON file survives
        # a git pull, so merely changing dataclass defaults never fixed an
        # already-deployed Jetson. Migrate only exact old shipped values; custom
        # operator values are otherwise preserved.
        if source_schema < SETTINGS_SCHEMA_VERSION:
            legacy_camera_pair = tuple(str(value) for value in data.get("cameras", ())) == ("0", "3")
            known_legacy_profile = (
                legacy_camera_pair
                and data.get("target_fps") == 60
                and data.get("track_timeout_ms") == 250.0
                and data.get("max_track_gap_ms") == 250.0
                and data.get("min_track_frames") == 4
                and data.get("min_defect_frames") == 3
            )
            if known_legacy_profile:
                data["cameras"] = list(DEFAULT_CAMERAS)
            if known_legacy_profile and str(data.get("presence_direction", "")).lower() == "positive":
                # Camera 0 in the recorded rig proves negative-x motion, but
                # those recordings do not prove camera 2's *raw* orientation:
                # the recorder may already have applied its saved mirror flag.
                # Correct the bad shared direction while preserving the
                # operator's per-camera mirror calibration.
                data["presence_direction"] = DEFAULT_PRESENCE_DIRECTION
            old_defaults = {
                "target_fps": (60,),
                "classify_band_ratio": (0.60,),
                "simulate_gpio": (False,),
                # Schema 2 briefly used 500/400; migrate both it and the
                # original 250/250 defaults while preserving custom tuning.
                "track_timeout_ms": (250.0, 500.0),
                "max_track_gap_ms": (250.0, 400.0),
                "min_track_frames": (4,),
                "min_defect_frames": (3,),
            }
            new_defaults = {
                "target_fps": DEFAULT_TARGET_FPS,
                "classify_band_ratio": DEFAULT_CLASSIFY_BAND_RATIO,
                "simulate_gpio": DEFAULT_SIMULATE_GPIO,
                "track_timeout_ms": DEFAULT_TRACK_TIMEOUT_MS,
                "max_track_gap_ms": DEFAULT_MAX_TRACK_GAP_MS,
                "min_track_frames": DEFAULT_MIN_TRACK_FRAMES,
                "min_defect_frames": DEFAULT_MIN_DEFECT_FRAMES,
            }
            for key, old_values in old_defaults.items():
                if key in data and data[key] in old_values:
                    data[key] = new_defaults[key]
            data["settings_schema_version"] = SETTINGS_SCHEMA_VERSION
        if "classifier_model" not in data:
            data.pop("model", None)
            data.pop("imgsz", None)
        merged = {**allowed, **{key: value for key, value in data.items() if key in allowed}}
        if "max_track_gap_ms" not in data:
            merged["max_track_gap_ms"] = min(
                float(defaults.max_track_gap_ms),
                float(merged["track_timeout_ms"]),
            )
        camera_values = merged.get("cameras")
        if not isinstance(camera_values, (list, tuple)) or len(camera_values) != 2:
            camera_values = defaults.cameras
        merged["cameras"] = tuple(str(value) for value in camera_values)
        mirror = list(merged.get("mirror_cameras", defaults.mirror_cameras))
        if len(mirror) != 2:
            mirror = list(defaults.mirror_cameras)
        merged["mirror_cameras"] = tuple(bool(value) for value in mirror)
        resolution_values = merged.get("resolution")
        if not isinstance(resolution_values, (list, tuple)) or len(resolution_values) != 2:
            resolution_values = defaults.resolution
        merged["resolution"] = tuple(int(value) for value in resolution_values)
        merged["pixel_format"] = normalize_pixel_format(merged["pixel_format"])
        merged["gpio_backend"] = normalize_gpio_backend(merged["gpio_backend"])
        merged["presence_line_axis"] = str(merged["presence_line_axis"]).lower()
        merged["presence_direction"] = str(merged["presence_direction"]).lower()
        return cls(**merged)


def validate_config(config: RuntimeConfig, *, require_actuation_ready: bool = True) -> None:
    if len(config.cameras) != 2:
        raise ValueError("v7 requires exactly two cameras")
    if len(config.mirror_cameras) != 2:
        raise ValueError("mirror_cameras must contain exactly two values")
    if not str(config.classifier_model).strip():
        raise ValueError("classifier_model must not be empty")
    if int(config.resolution[0]) <= 0 or int(config.resolution[1]) <= 0:
        raise ValueError("resolution must be positive")
    if int(config.target_fps) <= 0:
        raise ValueError("target_fps must be greater than 0")
    if not 0.0 <= float(config.detect_threshold) <= 1.0:
        raise ValueError("detect_threshold must be between 0 and 1")
    if not 0.0 <= float(config.frame_dirt_threshold) <= 1.0:
        raise ValueError("frame_dirt_threshold must be between 0 and 1")
    if not 0.0 <= float(config.track_dirt_threshold) <= 1.0:
        raise ValueError("track_dirt_threshold must be between 0 and 1")
    if not 0.0 <= float(config.crop_margin) <= 1.0:
        raise ValueError("crop_margin must be between 0 and 1")
    if not 0.0 < float(config.classify_band_ratio) <= 1.0:
        raise ValueError("classify_band_ratio must be greater than 0 and at most 1")
    if int(config.max_classified_boxes) < 1:
        raise ValueError("max_classified_boxes must be at least 1")
    if config.classifier_imgsz is not None and int(config.classifier_imgsz) <= 0:
        raise ValueError("classifier_imgsz must be positive when set")
    if not 0.0 <= float(config.duplicate_iou_threshold) <= 1.0:
        raise ValueError("duplicate_iou_threshold must be between 0 and 1")
    if not 0.0 <= float(config.track_iou) <= 1.0:
        raise ValueError("track_iou must be between 0 and 1")
    if float(config.track_timeout_ms) <= 0:
        raise ValueError("track_timeout_ms must be greater than 0")
    if int(config.min_track_frames) < 2:
        raise ValueError("min_track_frames must be at least 2")
    if float(config.min_track_travel_ratio) < 0:
        raise ValueError("min_track_travel_ratio must be 0 or greater")
    if not 0.0 <= float(config.min_track_directionality) <= 1.0:
        raise ValueError("min_track_directionality must be between 0 and 1")
    if int(config.min_defect_frames) < 2:
        raise ValueError("min_defect_frames must be at least 2")
    if int(config.min_line_side_frames) < 1:
        raise ValueError("min_line_side_frames must be at least 1")
    if int(config.min_classified_frames) < 1:
        raise ValueError("min_classified_frames must be at least 1")
    if not 1 <= int(config.required_inspected_cameras) <= len(config.cameras):
        raise ValueError("required_inspected_cameras must be between 1 and the camera count")
    if str(config.presence_line_axis).lower() not in {"x", "y"}:
        raise ValueError("presence_line_axis must be x or y")
    if not 0.0 <= float(config.presence_line_ratio) <= 1.0:
        raise ValueError("presence_line_ratio must be between 0 and 1")
    if str(config.presence_direction).lower() not in {"positive", "negative", "either"}:
        raise ValueError("presence_direction must be positive, negative, or either")
    if float(config.max_track_gap_ms) <= 0:
        raise ValueError("max_track_gap_ms must be greater than 0")
    if float(config.max_track_gap_ms) > float(config.track_timeout_ms):
        raise ValueError("max_track_gap_ms cannot exceed track_timeout_ms")
    if float(config.presence_clear_ms) < 0:
        raise ValueError("presence_clear_ms must be 0 or greater")
    if float(config.fire_delay_s) < 0:
        raise ValueError("fire_delay_s must be 0 or greater")
    if require_actuation_ready and not config.simulate_gpio:
        minimum_decision_delay_s = float(config.track_timeout_ms) / 1000.0
        if config.reject_uninspected:
            minimum_decision_delay_s += float(config.merge_window_ms) / 1000.0
        if float(config.fire_delay_s) <= minimum_decision_delay_s:
            raise ValueError(
                "real GPIO requires fire_delay_s to exceed the minimum decision horizon "
                f"({minimum_decision_delay_s:.3f}s with the current tracking settings); "
                "calibrate the physical gate-to-nozzle delay with simulate_gpio enabled first"
            )
    if float(config.merge_window_ms) < 0:
        raise ValueError("merge_window_ms must be 0 or greater")
    if float(config.min_fire_interval_ms) < 0:
        raise ValueError("min_fire_interval_ms must be 0 or greater")
    if config.gpio_backend not in GPIO_BACKENDS:
        raise ValueError(f"gpio_backend must be one of {GPIO_BACKENDS}")
    if float(config.trigger_duration) <= 0:
        raise ValueError("trigger_duration must be greater than 0")
    if float(config.trigger_min_gap) < 0:
        raise ValueError("trigger_min_gap must be 0 or greater")
    if float(config.trigger_max_queue_age_ms) < 0:
        raise ValueError("trigger_max_queue_age_ms must be 0 or greater")
    if float(config.trigger_max_lateness_ms) < 0:
        raise ValueError("trigger_max_lateness_ms must be 0 or greater")
    if float(config.max_frame_age_ms) <= 0:
        raise ValueError("max_frame_age_ms must be greater than 0")
    if float(config.camera_read_timeout_s) <= 0:
        raise ValueError("camera_read_timeout_s must be greater than 0")
    if float(config.live_preview_fps) < 0:
        raise ValueError("live_preview_fps must be 0 or greater")


def build_arg_parser() -> argparse.ArgumentParser:
    defaults = RuntimeConfig.defaults()
    parser = argparse.ArgumentParser(
        description="Run the standalone v7 cap inspection runtime (two cameras, two-stage model, one nozzle)."
    )
    parser.add_argument("--model", default=defaults.model, help="single-class cap detector ONNX")
    parser.add_argument("--classifier-model", default=defaults.classifier_model, help="binary dirt classifier ONNX")
    parser.add_argument("--cams", nargs=2, default=list(defaults.cameras))
    parser.add_argument("--mirror-camera-0", action=argparse.BooleanOptionalAction, default=defaults.mirror_cameras[0])
    parser.add_argument("--mirror-camera-1", action=argparse.BooleanOptionalAction, default=defaults.mirror_cameras[1])
    parser.add_argument("--res", type=int, nargs=2, default=list(defaults.resolution))
    parser.add_argument("--target-fps", "--fps", type=int, default=defaults.target_fps)
    parser.add_argument("--exposure", type=int, default=defaults.exposure)
    parser.add_argument("--pixel-format", default=defaults.pixel_format)
    parser.add_argument("--imgsz", type=int, default=defaults.imgsz)
    parser.add_argument("--classifier-imgsz", type=int, default=defaults.classifier_imgsz)
    parser.add_argument("--onnx-intra-op-threads", type=int, default=defaults.onnx_intra_op_threads)
    parser.add_argument("--detect-threshold", type=float, default=defaults.detect_threshold)
    parser.add_argument("--frame-dirt-threshold", type=float, default=defaults.frame_dirt_threshold)
    parser.add_argument("--track-dirt-threshold", type=float, default=defaults.track_dirt_threshold)
    parser.add_argument("--crop-margin", type=float, default=defaults.crop_margin)
    parser.add_argument("--classify-band-ratio", type=float, default=defaults.classify_band_ratio)
    parser.add_argument("--max-classified-boxes", type=int, default=defaults.max_classified_boxes)
    parser.add_argument("--duplicate-iou-threshold", type=float, default=defaults.duplicate_iou_threshold)
    parser.add_argument("--track-iou", type=float, default=defaults.track_iou)
    parser.add_argument("--track-timeout-ms", type=float, default=defaults.track_timeout_ms)
    parser.add_argument("--min-track-frames", type=int, default=defaults.min_track_frames)
    parser.add_argument("--min-track-travel-ratio", type=float, default=defaults.min_track_travel_ratio)
    parser.add_argument("--min-track-directionality", type=float, default=defaults.min_track_directionality)
    parser.add_argument("--min-defect-frames", type=int, default=defaults.min_defect_frames)
    parser.add_argument("--min-line-side-frames", type=int, default=defaults.min_line_side_frames)
    parser.add_argument("--min-classified-frames", type=int, default=defaults.min_classified_frames)
    parser.add_argument(
        "--required-inspected-cameras", type=int, default=defaults.required_inspected_cameras
    )
    parser.add_argument(
        "--reject-uninspected",
        action=argparse.BooleanOptionalAction,
        default=defaults.reject_uninspected,
    )
    parser.add_argument("--presence-line-axis", choices=["x", "y"], default=defaults.presence_line_axis)
    parser.add_argument("--presence-line-ratio", type=float, default=defaults.presence_line_ratio)
    parser.add_argument(
        "--presence-direction",
        choices=["positive", "negative", "either"],
        default=defaults.presence_direction,
    )
    parser.add_argument("--max-track-gap-ms", type=float, default=defaults.max_track_gap_ms)
    parser.add_argument("--presence-clear-ms", type=float, default=defaults.presence_clear_ms)
    parser.add_argument("--fire-delay-s", type=float, default=defaults.fire_delay_s)
    parser.add_argument("--merge-window-ms", type=float, default=defaults.merge_window_ms)
    parser.add_argument("--min-fire-interval-ms", type=float, default=defaults.min_fire_interval_ms)
    parser.add_argument("--gpio-backend", choices=list(GPIO_BACKENDS), default=defaults.gpio_backend)
    parser.add_argument("--trigger-pin", default=defaults.trigger_pin)
    parser.add_argument("--trigger-duration", type=float, default=defaults.trigger_duration)
    parser.add_argument("--trigger-min-gap", type=float, default=defaults.trigger_min_gap)
    parser.add_argument("--trigger-max-queue-age-ms", type=float, default=defaults.trigger_max_queue_age_ms)
    parser.add_argument("--trigger-max-lateness-ms", type=float, default=defaults.trigger_max_lateness_ms)
    parser.add_argument("--max-frame-age-ms", type=float, default=defaults.max_frame_age_ms)
    parser.add_argument("--camera-read-timeout-s", type=float, default=defaults.camera_read_timeout_s)
    parser.add_argument(
        "--simulate-gpio",
        action=argparse.BooleanOptionalAction,
        default=defaults.simulate_gpio,
    )
    parser.add_argument("--live-preview-fps", type=float, default=defaults.live_preview_fps)
    parser.add_argument("--db-path", default=defaults.db_path)
    parser.add_argument("--no-display", action="store_true", default=defaults.no_display)
    return parser


def config_from_args(args: argparse.Namespace) -> RuntimeConfig:
    config = RuntimeConfig(
        model=args.model,
        classifier_model=args.classifier_model,
        cameras=tuple(str(value) for value in args.cams),  # type: ignore[arg-type]
        mirror_cameras=(bool(args.mirror_camera_0), bool(args.mirror_camera_1)),
        resolution=(int(args.res[0]), int(args.res[1])),
        target_fps=int(args.target_fps),
        exposure=int(args.exposure),
        pixel_format=normalize_pixel_format(args.pixel_format),
        imgsz=args.imgsz,
        classifier_imgsz=args.classifier_imgsz,
        onnx_intra_op_threads=int(args.onnx_intra_op_threads),
        detect_threshold=float(args.detect_threshold),
        frame_dirt_threshold=float(args.frame_dirt_threshold),
        track_dirt_threshold=float(args.track_dirt_threshold),
        crop_margin=float(args.crop_margin),
        classify_band_ratio=float(args.classify_band_ratio),
        max_classified_boxes=int(args.max_classified_boxes),
        duplicate_iou_threshold=float(args.duplicate_iou_threshold),
        track_iou=float(args.track_iou),
        track_timeout_ms=float(args.track_timeout_ms),
        min_track_frames=int(args.min_track_frames),
        min_track_travel_ratio=float(args.min_track_travel_ratio),
        min_track_directionality=float(args.min_track_directionality),
        min_defect_frames=int(args.min_defect_frames),
        min_line_side_frames=int(args.min_line_side_frames),
        min_classified_frames=int(args.min_classified_frames),
        required_inspected_cameras=int(args.required_inspected_cameras),
        reject_uninspected=bool(args.reject_uninspected),
        presence_line_axis=str(args.presence_line_axis).lower(),
        presence_line_ratio=float(args.presence_line_ratio),
        presence_direction=str(args.presence_direction).lower(),
        max_track_gap_ms=float(args.max_track_gap_ms),
        presence_clear_ms=float(args.presence_clear_ms),
        fire_delay_s=float(args.fire_delay_s),
        merge_window_ms=float(args.merge_window_ms),
        min_fire_interval_ms=float(args.min_fire_interval_ms),
        gpio_backend=normalize_gpio_backend(args.gpio_backend),
        trigger_pin=args.trigger_pin,
        trigger_duration=float(args.trigger_duration),
        trigger_min_gap=float(args.trigger_min_gap),
        trigger_max_queue_age_ms=float(args.trigger_max_queue_age_ms),
        trigger_max_lateness_ms=float(args.trigger_max_lateness_ms),
        max_frame_age_ms=float(args.max_frame_age_ms),
        camera_read_timeout_s=float(args.camera_read_timeout_s),
        simulate_gpio=bool(args.simulate_gpio),
        live_preview_fps=float(args.live_preview_fps),
        db_path=str(args.db_path),
        no_display=bool(args.no_display),
    )
    validate_config(config)
    return config


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_arg_parser().parse_args(argv)
