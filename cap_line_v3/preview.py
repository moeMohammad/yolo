from __future__ import annotations

from dataclasses import dataclass

from .config import DEFAULT_ANCHOR_LINE_RATIO
from .geometry import (
    box_center,
    box_crossed_line_between,
    box_spans_line_coordinate,
    boxes_look_like_same_cap,
    frame_line_coordinate,
)
from .types import Box, CapturedFrame, DetectionPacket


def overlay_stale_timeout_s(target_fps: int | float) -> float:
    fps = max(1.0, float(target_fps))
    return min(0.35, max(0.10, 3.0 / fps))


def _camera_inference_s(packet: DetectionPacket, camera_index: int) -> float:
    if camera_index >= len(packet.inference_ms_by_camera):
        return 0.0
    return max(0.0, float(packet.inference_ms_by_camera[camera_index]) / 1000.0)


def _overlay_timeout_s(
    current_packet: DetectionPacket,
    previous_packet: DetectionPacket | None,
    camera_index: int,
    *,
    base_timeout_s: float,
) -> float:
    timeout_s = max(base_timeout_s, _camera_inference_s(current_packet, camera_index) + base_timeout_s)
    if previous_packet is not None and camera_index < len(previous_packet.frame_pair.timestamps):
        history_s = (
            float(current_packet.frame_pair.timestamps[camera_index])
            - float(previous_packet.frame_pair.timestamps[camera_index])
        )
        if history_s > 0.0:
            timeout_s = max(timeout_s, history_s * 1.5)
    return min(1.25, timeout_s)


def _frame_size(frame) -> tuple[int, int] | None:
    shape = getattr(frame, "shape", None)
    if shape is None or len(shape) < 2:
        return None
    return int(shape[1]), int(shape[0])


def _actuation_line_boxes(
    captured: CapturedFrame,
    boxes: tuple[Box, ...],
    *,
    previous_boxes: tuple[Box, ...] = (),
    anchor_axis: str,
    anchor_line_ratio: float,
) -> tuple[Box, ...]:
    frame_size = _frame_size(captured.frame)
    if frame_size is None:
        return ()
    line_coordinate = frame_line_coordinate(
        frame_size,
        axis=anchor_axis,
        ratio=anchor_line_ratio,
    )
    actuation_boxes = []
    for box in boxes:
        if box_spans_line_coordinate(
            box,
            axis=anchor_axis,
            line_coordinate=line_coordinate,
        ):
            actuation_boxes.append(box)
            continue
        previous_box = _match_previous_box(box, previous_boxes)
        if previous_box is None:
            continue
        if box_crossed_line_between(
            previous_box,
            box,
            axis=anchor_axis,
            line_coordinate=line_coordinate,
        ):
            actuation_boxes.append(box)
    return tuple(actuation_boxes)


def _match_previous_box(box: Box, previous_boxes: tuple[Box, ...]) -> Box | None:
    plausible = tuple(candidate for candidate in previous_boxes if boxes_look_like_same_cap(box, candidate))
    if not plausible:
        return None
    center_x, center_y = box_center(box)
    return min(
        plausible,
        key=lambda candidate: abs(box_center(candidate)[0] - center_x)
        + abs(box_center(candidate)[1] - center_y),
    )


@dataclass(frozen=True)
class CameraPreviewView:
    captured: CapturedFrame
    boxes: tuple[Box, ...]


def resolve_preview_views(
    previous_packet: DetectionPacket | None,
    current_packet: DetectionPacket | None,
    live_frames: tuple[CapturedFrame, ...],
    *,
    target_fps: int | float,
    anchor_axis: str = "x",
    anchor_line_ratio: float = DEFAULT_ANCHOR_LINE_RATIO,
    preview_latency_compensation_ms: int | float = 0.0,
) -> tuple[CameraPreviewView, ...]:
    if current_packet is None:
        return tuple(CameraPreviewView(frame, ()) for frame in live_frames)
    if len(current_packet.boxes_by_camera) != len(live_frames):
        return tuple(CameraPreviewView(frame, ()) for frame in live_frames)

    base_timeout_s = overlay_stale_timeout_s(target_fps)
    current_timestamps = current_packet.frame_pair.timestamps
    views: list[CameraPreviewView] = []
    for camera_index, live_frame in enumerate(live_frames):
        detection_frame = current_packet.frame_pair.frames[camera_index]
        current_timestamp = current_timestamps[camera_index]
        previous_boxes = (
            previous_packet.boxes_by_camera[camera_index]
            if previous_packet is not None and camera_index < len(previous_packet.boxes_by_camera)
            else ()
        )
        actuation_boxes = _actuation_line_boxes(
            detection_frame,
            current_packet.boxes_by_camera[camera_index],
            previous_boxes=previous_boxes,
            anchor_axis=anchor_axis,
            anchor_line_ratio=anchor_line_ratio,
        )
        if not actuation_boxes:
            views.append(CameraPreviewView(live_frame, ()))
            continue

        overlay_age_s = float(live_frame.timestamp) - float(current_timestamp)
        overlay_age_s += max(0.0, float(preview_latency_compensation_ms)) / 1000.0
        timeout_s = _overlay_timeout_s(
            current_packet,
            previous_packet,
            camera_index,
            base_timeout_s=base_timeout_s,
        )
        if overlay_age_s > timeout_s:
            views.append(CameraPreviewView(live_frame, ()))
            continue

        views.append(CameraPreviewView(detection_frame, actuation_boxes))
    return tuple(views)


def predict_preview_overlay(
    previous_packet: DetectionPacket | None,
    current_packet: DetectionPacket | None,
    live_frames: tuple[CapturedFrame, ...],
    *,
    target_fps: int | float,
    anchor_axis: str = "x",
    anchor_line_ratio: float = DEFAULT_ANCHOR_LINE_RATIO,
    preview_latency_compensation_ms: int | float = 0.0,
) -> tuple[tuple[Box, ...], ...]:
    return tuple(
        view.boxes
        for view in resolve_preview_views(
            previous_packet,
            current_packet,
            live_frames,
            target_fps=target_fps,
            anchor_axis=anchor_axis,
            anchor_line_ratio=anchor_line_ratio,
            preview_latency_compensation_ms=preview_latency_compensation_ms,
        )
    )
