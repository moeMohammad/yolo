"""Slim v6 runtime configuration.

Identical to v5's config except the GPIO defaults: v6 targets the Jetson Nano
rig, so ``gpio_backend`` defaults to ``"jetson"`` and ``trigger_pin`` to BOARD
pin 7 (GPIO09), driven by the untouched ``gpio_output.py``. Everything else —
dirtv7 model and the double-trigger fix knobs (``merge_window_ms`` keyed to
physical cap-exit times plus the ``min_fire_interval_ms`` post-fire
refractory) — is unchanged from v5.
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

GPIO_BACKENDS = ("rpi", "jetson")

DEFAULT_MODEL = "dirtv7.onnx"
DEFAULT_CAMERAS = ("0", "3")
DEFAULT_MIRROR_CAMERAS = (False, True)
DEFAULT_RESOLUTION = (960, 600)
DEFAULT_TARGET_FPS = 60
DEFAULT_EXPOSURE = 8
DEFAULT_PIXEL_FORMAT = "YUYV"
DEFAULT_ONNX_INTRA_OP_THREADS = max(1, (os.cpu_count() or 2) // 2)
DEFAULT_REJECT_THRESHOLD = 0.45
DEFAULT_TRACK_IOU = 0.3
# Must exceed ~2x the worst processed-frame interval or tracks fragment
# mid-cap; must stay below the gap between consecutive caps at one camera.
DEFAULT_TRACK_TIMEOUT_MS = 150.0
DEFAULT_FIRE_DELAY_S = 0.0
# Two finished tracks are the same physical cap when their last_seen times are
# within this window (cross-camera exit skew), regardless of reporting lag.
DEFAULT_MERGE_WINDOW_MS = 150.0
# Hard once-per-cap guarantee: suppress any fire whose reference is closer than
# this to the previous fire's reference. A pulse is trigger_duration=0.3 s long,
# so two distinguishable caps cannot reach the nozzle 250 ms apart.
DEFAULT_MIN_FIRE_INTERVAL_MS = 250.0
DEFAULT_GPIO_BACKEND = "jetson"
DEFAULT_TRIGGER_PIN = GPIO09  # Jetson Nano BOARD pin 7
DEFAULT_TRIGGER_DURATION = 0.3
DEFAULT_TRIGGER_MIN_GAP = 0.0
DEFAULT_LIVE_PREVIEW_FPS = 30.0
DEFAULT_DB_PATH = str(SCRIPT_DIR / "data" / "cap_line_history_v6.sqlite3")


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
    model: str = DEFAULT_MODEL
    cameras: tuple[str, str] = DEFAULT_CAMERAS
    mirror_cameras: tuple[bool, bool] = DEFAULT_MIRROR_CAMERAS
    resolution: tuple[int, int] = DEFAULT_RESOLUTION
    target_fps: int = DEFAULT_TARGET_FPS
    exposure: int = DEFAULT_EXPOSURE
    pixel_format: str = DEFAULT_PIXEL_FORMAT
    imgsz: int | None = None
    onnx_intra_op_threads: int = DEFAULT_ONNX_INTRA_OP_THREADS
    reject_threshold: float = DEFAULT_REJECT_THRESHOLD
    track_iou: float = DEFAULT_TRACK_IOU
    track_timeout_ms: float = DEFAULT_TRACK_TIMEOUT_MS
    fire_delay_s: float = DEFAULT_FIRE_DELAY_S
    merge_window_ms: float = DEFAULT_MERGE_WINDOW_MS
    min_fire_interval_ms: float = DEFAULT_MIN_FIRE_INTERVAL_MS
    gpio_backend: str = DEFAULT_GPIO_BACKEND
    trigger_pin: str | int = DEFAULT_TRIGGER_PIN
    trigger_duration: float = DEFAULT_TRIGGER_DURATION
    trigger_min_gap: float = DEFAULT_TRIGGER_MIN_GAP
    simulate_gpio: bool = False
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

        Unknown keys are dropped so a v3/v4 settings file can be pointed at v6
        without crashing; missing keys fall back to defaults. The v4 key
        ``global_cooldown_ms`` maps onto ``merge_window_ms`` when the new key
        is absent.
        """

        defaults = cls.defaults()
        allowed = defaults.to_json_dict()
        data = dict(data)
        if "merge_window_ms" not in data and "global_cooldown_ms" in data:
            data["merge_window_ms"] = data["global_cooldown_ms"]
        merged = {**allowed, **{key: value for key, value in data.items() if key in allowed}}
        merged["cameras"] = tuple(str(value) for value in merged["cameras"])[:2]
        mirror = list(merged.get("mirror_cameras", defaults.mirror_cameras))
        if len(mirror) != 2:
            mirror = list(defaults.mirror_cameras)
        merged["mirror_cameras"] = tuple(bool(value) for value in mirror)
        merged["resolution"] = tuple(int(value) for value in merged["resolution"])[:2]
        merged["pixel_format"] = normalize_pixel_format(merged["pixel_format"])
        merged["gpio_backend"] = normalize_gpio_backend(merged["gpio_backend"])
        return cls(**merged)


def validate_config(config: RuntimeConfig) -> None:
    if len(config.cameras) != 2:
        raise ValueError("v6 requires exactly two cameras")
    if len(config.mirror_cameras) != 2:
        raise ValueError("mirror_cameras must contain exactly two values")
    if int(config.resolution[0]) <= 0 or int(config.resolution[1]) <= 0:
        raise ValueError("resolution must be positive")
    if int(config.target_fps) <= 0:
        raise ValueError("target_fps must be greater than 0")
    if not 0.0 <= float(config.reject_threshold) <= 1.0:
        raise ValueError("reject_threshold must be between 0 and 1")
    if not 0.0 <= float(config.track_iou) <= 1.0:
        raise ValueError("track_iou must be between 0 and 1")
    if float(config.track_timeout_ms) < 0:
        raise ValueError("track_timeout_ms must be 0 or greater")
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
    if float(config.live_preview_fps) < 0:
        raise ValueError("live_preview_fps must be 0 or greater")


def build_arg_parser() -> argparse.ArgumentParser:
    defaults = RuntimeConfig.defaults()
    parser = argparse.ArgumentParser(
        description="Run the standalone v6 cap inspection runtime (two cameras, one model, one nozzle)."
    )
    parser.add_argument("--model", default=defaults.model)
    parser.add_argument("--cams", nargs=2, default=list(defaults.cameras))
    parser.add_argument("--mirror-camera-0", action=argparse.BooleanOptionalAction, default=defaults.mirror_cameras[0])
    parser.add_argument("--mirror-camera-1", action=argparse.BooleanOptionalAction, default=defaults.mirror_cameras[1])
    parser.add_argument("--res", type=int, nargs=2, default=list(defaults.resolution))
    parser.add_argument("--target-fps", "--fps", type=int, default=defaults.target_fps)
    parser.add_argument("--exposure", type=int, default=defaults.exposure)
    parser.add_argument("--pixel-format", default=defaults.pixel_format)
    parser.add_argument("--imgsz", type=int, default=defaults.imgsz)
    parser.add_argument("--onnx-intra-op-threads", type=int, default=defaults.onnx_intra_op_threads)
    parser.add_argument("--reject-threshold", type=float, default=defaults.reject_threshold)
    parser.add_argument("--track-iou", type=float, default=defaults.track_iou)
    parser.add_argument("--track-timeout-ms", type=float, default=defaults.track_timeout_ms)
    parser.add_argument("--fire-delay-s", type=float, default=defaults.fire_delay_s)
    parser.add_argument("--merge-window-ms", type=float, default=defaults.merge_window_ms)
    parser.add_argument("--min-fire-interval-ms", type=float, default=defaults.min_fire_interval_ms)
    parser.add_argument("--gpio-backend", choices=list(GPIO_BACKENDS), default=defaults.gpio_backend)
    parser.add_argument("--trigger-pin", default=defaults.trigger_pin)
    parser.add_argument("--trigger-duration", type=float, default=defaults.trigger_duration)
    parser.add_argument("--trigger-min-gap", type=float, default=defaults.trigger_min_gap)
    parser.add_argument("--simulate-gpio", action="store_true", default=defaults.simulate_gpio)
    parser.add_argument("--live-preview-fps", type=float, default=defaults.live_preview_fps)
    parser.add_argument("--db-path", default=defaults.db_path)
    parser.add_argument("--no-display", action="store_true", default=defaults.no_display)
    return parser


def config_from_args(args: argparse.Namespace) -> RuntimeConfig:
    config = RuntimeConfig(
        model=args.model,
        cameras=tuple(str(value) for value in args.cams),  # type: ignore[arg-type]
        mirror_cameras=(bool(args.mirror_camera_0), bool(args.mirror_camera_1)),
        resolution=(int(args.res[0]), int(args.res[1])),
        target_fps=int(args.target_fps),
        exposure=int(args.exposure),
        pixel_format=normalize_pixel_format(args.pixel_format),
        imgsz=args.imgsz,
        onnx_intra_op_threads=int(args.onnx_intra_op_threads),
        reject_threshold=float(args.reject_threshold),
        track_iou=float(args.track_iou),
        track_timeout_ms=float(args.track_timeout_ms),
        fire_delay_s=float(args.fire_delay_s),
        merge_window_ms=float(args.merge_window_ms),
        min_fire_interval_ms=float(args.min_fire_interval_ms),
        gpio_backend=normalize_gpio_backend(args.gpio_backend),
        trigger_pin=args.trigger_pin,
        trigger_duration=float(args.trigger_duration),
        trigger_min_gap=float(args.trigger_min_gap),
        simulate_gpio=bool(args.simulate_gpio),
        live_preview_fps=float(args.live_preview_fps),
        db_path=str(args.db_path),
        no_display=bool(args.no_display),
    )
    validate_config(config)
    return config


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_arg_parser().parse_args(argv)
