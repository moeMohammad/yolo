#!/usr/bin/env python3
"""Capture paired snapshots from the V3 camera setup into new_dataset."""

from __future__ import annotations

import argparse
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from cap_line_v3.config import RuntimeConfig, normalize_pixel_format, validate_config
from cap_line_v3 import runtime

try:
    import cv2
except ImportError:
    cv2 = None


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "new_dataset"
SPACE_KEY = ord(" ")
ESC_KEY = 27


def require_cv2():
    if cv2 is None:
        raise RuntimeError("OpenCV is not installed. Install `python3-opencv` before running this script.")
    return cv2


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than 0")
    return parsed


def non_negative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be 0 or greater")
    return parsed


def build_arg_parser() -> argparse.ArgumentParser:
    defaults = RuntimeConfig.defaults()
    parser = argparse.ArgumentParser(
        description="Open both V3 cameras and save paired snapshots to new_dataset when Space is pressed."
    )
    parser.add_argument("--cams", nargs=2, default=list(defaults.cameras), help="two camera indices or device paths")
    parser.add_argument("--res", type=positive_int, nargs=2, default=list(defaults.resolution), metavar=("W", "H"))
    parser.add_argument("--target-fps", "--fps", type=positive_int, default=defaults.target_fps)
    parser.add_argument("--pixel-format", default=defaults.pixel_format)
    parser.add_argument("--exposure", type=positive_int, default=defaults.exposure)
    parser.add_argument("--mirror-camera-0", action=argparse.BooleanOptionalAction, default=defaults.mirror_cameras[0])
    parser.add_argument("--mirror-camera-1", action=argparse.BooleanOptionalAction, default=defaults.mirror_cameras[1])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prefix", default="capture")
    parser.add_argument("--warmup-seconds", type=non_negative_float, default=1.0)
    parser.add_argument("--jpeg-quality", type=positive_int, default=95)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_arg_parser().parse_args(argv)


def config_from_args(args: argparse.Namespace) -> RuntimeConfig:
    config = replace(
        RuntimeConfig.defaults(),
        cameras=tuple(str(value) for value in args.cams),
        mirror_cameras=(bool(args.mirror_camera_0), bool(args.mirror_camera_1)),
        resolution=(int(args.res[0]), int(args.res[1])),
        target_fps=int(args.target_fps),
        pixel_format=normalize_pixel_format(args.pixel_format),
        exposure=int(args.exposure),
    )
    validate_config(config)
    return config


def sanitize_camera_label(camera_label: object) -> str:
    text = str(camera_label)
    return "".join(ch if ch.isalnum() else "_" for ch in text).strip("_") or "camera"


def save_frames(
    frames: list,
    cam_list: list[object],
    output_dir: Path,
    prefix: str,
    capture_index: int,
    jpeg_quality: int,
    *,
    timestamp: str | None = None,
    cv2_module=None,
) -> list[Path]:
    cv2_module = cv2_module or require_cv2()
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    saved_paths: list[Path] = []

    for frame, camera_label in zip(frames, cam_list):
        camera_suffix = sanitize_camera_label(camera_label)
        image_path = output_dir / f"{timestamp}_{prefix}_{capture_index:06d}_cam_{camera_suffix}.jpg"
        ok = cv2_module.imwrite(
            str(image_path),
            frame,
            [int(cv2_module.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
        )
        if not ok:
            raise RuntimeError(f"Failed to save image to {image_path}")
        saved_paths.append(image_path)

    return saved_paths


def open_configured_cameras(config: RuntimeConfig):
    cam_list, device_paths = runtime.parse_cameras(config.cameras)
    width, height = config.resolution

    for device_path in device_paths:
        runtime.set_camera_format(
            device_path,
            width,
            height,
            config.target_fps,
            pixel_format=config.pixel_format,
        )
        runtime.set_camera_controls(device_path, config.exposure)

    cameras = []
    try:
        cameras = [
            runtime.open_cam(source, width, height, config.target_fps, config.pixel_format)
            for source in cam_list
        ]
        runtime.validate_opened_cameras(cameras, cam_list, device_paths)
    except Exception:
        for camera in cameras:
            camera.release()
        raise

    return cam_list, cameras


def read_frames(cameras: list, mirror_cameras: tuple[bool, bool]) -> list:
    frames = []
    for index, camera in enumerate(cameras):
        ok, frame = camera.read()
        if not ok or frame is None:
            raise RuntimeError(f"Failed to read a frame from camera {index}")
        if index < len(mirror_cameras) and mirror_cameras[index]:
            frame = runtime.mirror_frame_horizontal(frame)
        frames.append(frame)
    return frames


def capture_loop(args: argparse.Namespace) -> int:
    cv2_module = require_cv2()
    config = config_from_args(args)
    output_dir = Path(args.output_dir).expanduser().resolve()
    cam_list, cameras = open_configured_cameras(config)
    window_names = [f"V3 camera {index} ({label})" for index, label in enumerate(cam_list)]
    capture_index = 0

    try:
        for window_name in window_names:
            cv2_module.namedWindow(window_name, cv2_module.WINDOW_NORMAL)

        if args.warmup_seconds:
            print(f"Warming cameras for {float(args.warmup_seconds):.1f}s...")
            time.sleep(float(args.warmup_seconds))

        print(f"Saving paired snapshots to: {output_dir}")
        print("Press Space to save both cameras. Press q or Esc to quit.")
        print(
            f"V3 settings: cams={list(config.cameras)} res={config.resolution[0]}x{config.resolution[1]} "
            f"fps={config.target_fps} pixel_format={config.pixel_format} exposure={config.exposure} "
            f"mirror={list(config.mirror_cameras)}"
        )

        wait_ms = max(1, int(round(1000 / max(1, int(config.target_fps)))))
        while True:
            frames = read_frames(cameras, config.mirror_cameras)
            for window_name, frame in zip(window_names, frames):
                cv2_module.imshow(window_name, frame)

            key = cv2_module.waitKey(wait_ms) & 0xFF
            if key == SPACE_KEY:
                capture_index += 1
                saved_paths = save_frames(
                    frames,
                    cam_list,
                    output_dir,
                    args.prefix,
                    capture_index,
                    args.jpeg_quality,
                    cv2_module=cv2_module,
                )
                print(f"[{capture_index}] saved {len(saved_paths)} image(s)")
                for image_path in saved_paths:
                    print(f"  {image_path}")
            elif key in (ESC_KEY, ord("q"), ord("Q")):
                break

    except KeyboardInterrupt:
        print("\nCapture stopped by user.")
    finally:
        for camera in cameras:
            camera.release()
        cv2_module.destroyAllWindows()

    print(f"Saved {capture_index * len(cam_list)} image(s) to {output_dir}")
    return 0


def main() -> int:
    return capture_loop(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
