#!/usr/bin/env python3
"""
Capture frames from both cameras into label_data for later labeling.

This mirrors the camera defaults from full_run_alt.py but skips all model,
detection, tracking, and GPIO logic. It only opens both cameras and saves
every captured frame to disk.
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import cap_line_runtime as runtime

try:
    import cv2
except ImportError:
    cv2 = None


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "label_data"


def require_cv2():
    if cv2 is None:
        raise RuntimeError(
            "OpenCV is not installed. Install `python3-opencv` before running this script."
        )
    return cv2


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than 0")
    return parsed


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than 0")
    return parsed


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be 0 or greater")
    return parsed


def non_negative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be 0 or greater")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture frames from both cameras into label_data for later labeling."
    )
    parser.add_argument(
        "--cams",
        nargs="+",
        default=["0", "3"],
        help="camera indices or device paths (default: 0 3)",
    )
    parser.add_argument(
        "--res",
        type=positive_int,
        nargs=2,
        default=list(runtime.DEFAULT_CAMERA_RESOLUTION),
        metavar=("W", "H"),
        help=(
            "capture width and height "
            f"(default: {runtime.DEFAULT_CAMERA_RESOLUTION[0]} "
            f"{runtime.DEFAULT_CAMERA_RESOLUTION[1]})"
        ),
    )
    parser.add_argument(
        "--fps",
        type=positive_float,
        default=float(runtime.DEFAULT_CAMERA_FPS),
        help=f"target capture rate in frames per second (default: {runtime.DEFAULT_CAMERA_FPS})",
    )
    parser.add_argument(
        "--pixel-format",
        default=runtime.DEFAULT_CAMERA_PIXEL_FORMAT,
        help="V4L2 pixel format to force on the camera hardware (default: YUYV)",
    )
    parser.add_argument(
        "--exposure",
        type=positive_int,
        default=8,
        help="exposure_time_absolute for each camera (default: 8)",
    )
    parser.add_argument(
        "--count",
        type=non_negative_int,
        default=0,
        help="number of capture cycles before stopping; 0 runs until Ctrl+C",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"directory where images are saved (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--prefix",
        default="label",
        help="filename prefix for saved images (default: label)",
    )
    parser.add_argument(
        "--warmup-seconds",
        type=non_negative_float,
        default=2.0,
        help="camera warm-up time before saving frames (default: 2.0)",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=positive_int,
        default=95,
        help="JPEG quality used when saving images (default: 95)",
    )
    return parser.parse_args(argv)


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
) -> list[Path]:
    cv2_module = require_cv2()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    saved_paths: list[Path] = []

    for frame, camera_label in zip(frames, cam_list):
        camera_suffix = sanitize_camera_label(camera_label)
        image_path = output_dir / (
            f"{timestamp}_{prefix}_{capture_index:06d}_cam_{camera_suffix}.jpg"
        )
        ok = cv2_module.imwrite(
            str(image_path),
            frame,
            [int(cv2_module.IMWRITE_JPEG_QUALITY), jpeg_quality],
        )
        if not ok:
            raise RuntimeError(f"Failed to save image to {image_path}")
        saved_paths.append(image_path)

    return saved_paths


def capture_loop(args: argparse.Namespace) -> int:
    args.pixel_format = runtime.normalize_camera_pixel_format(args.pixel_format)
    if args.pixel_format != runtime.DEFAULT_CAMERA_PIXEL_FORMAT:
        raise ValueError(
            f"--pixel-format must be {runtime.DEFAULT_CAMERA_PIXEL_FORMAT} "
            "for Arducam B0495 cameras"
        )

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cam_list, device_paths = runtime.parse_cameras(args.cams)
    width, height = args.res

    for device_path in device_paths:
        runtime.set_camera_format(
            device_path,
            width,
            height,
            int(args.fps),
            pixel_format=args.pixel_format,
        )
        runtime.set_camera_controls(device_path, args.exposure)

    cameras = []
    try:
        for cam in cam_list:
            cameras.append(
                runtime.open_cam(
                    cam,
                    width,
                    height,
                    int(args.fps),
                    args.pixel_format,
                )
            )
    except Exception:
        for camera in cameras:
            camera.release()
        raise

    if args.warmup_seconds > 0:
        time.sleep(args.warmup_seconds)

    interval = 1.0 / args.fps
    limit = None if args.count == 0 else args.count
    capture_count = 0
    next_capture_at = time.monotonic()
    progress_every = max(1, int(round(args.fps)))

    print(f"Saving images to: {output_dir}")
    print(f"Cameras: {cam_list}")
    print(f"Resolution: {width}x{height}")
    print(f"Target capture rate: {args.fps:.2f} FPS")
    if limit is None:
        print("Press Ctrl+C to stop.")
    else:
        print(f"Stopping after {limit} capture cycle(s).")

    try:
        while limit is None or capture_count < limit:
            now = time.monotonic()
            if now < next_capture_at:
                time.sleep(next_capture_at - now)

            frames = []
            for camera in cameras:
                ok, frame = camera.read()
                if not ok or frame is None:
                    raise RuntimeError("Failed to read a frame from one of the cameras")
                frames.append(frame)

            capture_count += 1
            saved_paths = save_frames(
                frames,
                cam_list,
                output_dir,
                args.prefix,
                capture_count,
                args.jpeg_quality,
            )

            if capture_count <= 3 or capture_count % progress_every == 0:
                print(f"[{capture_count}] saved {len(saved_paths)} images")
                for image_path in saved_paths:
                    print(f"  {image_path}")

            next_capture_at += interval
            if next_capture_at < time.monotonic():
                next_capture_at = time.monotonic()

    except KeyboardInterrupt:
        print("\nCapture stopped by user.")

    finally:
        for camera in cameras:
            camera.release()

    print(
        f"Saved {capture_count * len(cam_list)} image(s) from {len(cam_list)} camera(s) to {output_dir}"
    )
    return 0


def main() -> int:
    args = parse_args()
    return capture_loop(args)


if __name__ == "__main__":
    raise SystemExit(main())
