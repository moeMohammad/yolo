#!/usr/bin/env python3
"""Record both v6 camera feeds to separate video files.

The camera setup mirrors the cap-line v6 defaults, except that the default
device indices are 0 and 2. Both files are finalized when the program exits
normally, receives Ctrl+C/SIGTERM, or the preview is closed with q/Esc.
"""

from __future__ import annotations

import argparse
import signal
import threading
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from cap_line_v6 import runtime
from cap_line_v6.config import RuntimeConfig, normalize_pixel_format, validate_config

try:
    import cv2
except ImportError:
    cv2 = None


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CAMERAS = ("0", "2")
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "recorded_videos"
DEFAULT_CODEC = "mp4v"
DEFAULT_EXTENSION = ".mp4"
PREVIEW_WINDOW = "V6 dual-camera recorder"
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


def build_arg_parser() -> argparse.ArgumentParser:
    defaults = RuntimeConfig.defaults()
    parser = argparse.ArgumentParser(
        description="Record the two v6 camera feeds to separate MP4 files until the program is stopped."
    )
    parser.add_argument(
        "--cams",
        nargs=2,
        default=list(DEFAULT_CAMERAS),
        metavar=("CAM0", "CAM1"),
        help="two camera indices or device paths (default: 0 2)",
    )
    parser.add_argument("--res", type=positive_int, nargs=2, default=list(defaults.resolution), metavar=("W", "H"))
    parser.add_argument("--target-fps", "--fps", type=positive_int, default=defaults.target_fps)
    parser.add_argument("--pixel-format", default=defaults.pixel_format)
    parser.add_argument("--exposure", type=positive_int, default=defaults.exposure)
    parser.add_argument(
        "--mirror-camera-0", action=argparse.BooleanOptionalAction, default=defaults.mirror_cameras[0]
    )
    parser.add_argument(
        "--mirror-camera-1", action=argparse.BooleanOptionalAction, default=defaults.mirror_cameras[1]
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prefix", default="recording")
    parser.add_argument("--codec", default=DEFAULT_CODEC, help="four-character OpenCV video codec (default: mp4v)")
    parser.add_argument("--no-display", action="store_true", default=defaults.no_display)
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
        no_display=bool(args.no_display),
    )
    validate_config(config)
    return config


def sanitize_label(value: object) -> str:
    text = str(value)
    return "".join(character if character.isalnum() else "_" for character in text).strip("_") or "camera"


def build_output_paths(
    output_dir: Path,
    prefix: str,
    camera_labels: list[object],
    *,
    timestamp: str | None = None,
) -> list[Path]:
    timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    return [
        output_dir / f"{timestamp}_{prefix}_cam_{sanitize_label(camera_label)}{DEFAULT_EXTENSION}"
        for camera_label in camera_labels
    ]


def open_configured_cameras(config: RuntimeConfig):
    camera_sources, device_paths = runtime.parse_cameras(config.cameras)
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
            for source in camera_sources
        ]
        runtime.validate_opened_cameras(cameras, camera_sources, device_paths)
    except Exception:
        for camera in cameras:
            camera.release()
        raise

    return camera_sources, cameras


def create_video_writers(
    output_paths: list[Path],
    resolution: tuple[int, int],
    fps: int,
    codec: str,
    *,
    cv2_module=None,
):
    cv2_module = cv2_module or require_cv2()
    if len(codec) != 4:
        raise ValueError("--codec must contain exactly four characters")

    fourcc = cv2_module.VideoWriter_fourcc(*codec)
    writers = []
    try:
        for output_path in output_paths:
            writer = cv2_module.VideoWriter(str(output_path), fourcc, float(fps), tuple(resolution))
            writers.append(writer)
            if hasattr(writer, "isOpened") and not writer.isOpened():
                raise RuntimeError(f"Unable to open video writer for {output_path}")
    except Exception:
        for writer in writers:
            writer.release()
        raise
    return writers


def prepare_frame(frame, resolution: tuple[int, int], mirror_horizontal: bool, *, cv2_module=None):
    cv2_module = cv2_module or require_cv2()
    if mirror_horizontal:
        frame = runtime.mirror_frame_horizontal(frame)

    width, height = resolution
    if tuple(frame.shape[:2]) != (height, width):
        frame = cv2_module.resize(frame, (width, height))
    return frame


def record_loop(args: argparse.Namespace, *, stop_event: threading.Event | None = None) -> int:
    cv2_module = require_cv2()
    config = config_from_args(args)
    stop_event = stop_event or threading.Event()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    camera_sources, cameras = open_configured_cameras(config)
    output_paths = build_output_paths(output_dir, str(args.prefix), camera_sources)
    writers = []
    frame_counts = [0] * len(cameras)
    started_at = time.monotonic()

    try:
        writers = create_video_writers(
            output_paths,
            config.resolution,
            config.target_fps,
            str(args.codec),
            cv2_module=cv2_module,
        )

        if not config.no_display:
            cv2_module.namedWindow(PREVIEW_WINDOW, cv2_module.WINDOW_NORMAL)

        print(
            f"Recording cameras {list(config.cameras)} at {config.resolution[0]}x{config.resolution[1]} "
            f"{config.target_fps} FPS, {config.pixel_format}, exposure={config.exposure}, "
            f"mirror={list(config.mirror_cameras)}"
        )
        for output_path in output_paths:
            print(f"  {output_path}")
        print("Press q or Esc in the preview, or Ctrl+C in the terminal, to stop and save both videos.")

        while not stop_event.is_set():
            frames = []
            for index, camera in enumerate(cameras):
                ok, frame = camera.read()
                if not ok or frame is None:
                    raise RuntimeError(f"Failed to read a frame from camera {index} (source {camera_sources[index]!r})")
                frame = prepare_frame(
                    frame,
                    config.resolution,
                    config.mirror_cameras[index],
                    cv2_module=cv2_module,
                )
                writers[index].write(frame)
                frame_counts[index] += 1
                frames.append(frame)

            if not config.no_display:
                preview = runtime.compose_preview(frames)
                if preview is not None:
                    cv2_module.imshow(PREVIEW_WINDOW, preview)
                key = cv2_module.waitKey(1) & 0xFF
                if key in (ESC_KEY, ord("q"), ord("Q")):
                    stop_event.set()

    except KeyboardInterrupt:
        stop_event.set()
        print("\nStop requested; finalizing videos...")
    finally:
        for camera in cameras:
            camera.release()
        for writer in writers:
            writer.release()
        if not config.no_display:
            cv2_module.destroyAllWindows()

    elapsed = max(0.0, time.monotonic() - started_at)
    print(f"Saved both videos after {elapsed:.1f} seconds:")
    for output_path, frame_count in zip(output_paths, frame_counts):
        print(f"  {output_path} ({frame_count} frames)")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    stop_event = threading.Event()

    def request_stop(signum, _frame) -> None:
        print(f"\nReceived signal {signum}; finalizing videos...")
        stop_event.set()

    previous_handlers = {}
    for signal_name in ("SIGINT", "SIGTERM"):
        current_signal = getattr(signal, signal_name, None)
        if current_signal is not None:
            previous_handlers[current_signal] = signal.signal(current_signal, request_stop)

    try:
        return record_loop(args, stop_event=stop_event)
    finally:
        for current_signal, previous_handler in previous_handlers.items():
            signal.signal(current_signal, previous_handler)


if __name__ == "__main__":
    raise SystemExit(main())
