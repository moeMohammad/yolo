from __future__ import annotations

from dataclasses import dataclass

from .geometry import box_center, box_size_ratio, boxes_look_like_same_cap
from .types import Box, CapturedFrame, DetectionPacket

MAX_PREVIEW_EXTRAPOLATION_S = 0.35


def overlay_stale_timeout_s(target_fps: int | float) -> float:
    fps = max(1.0, float(target_fps))
    return min(0.35, max(0.10, 3.0 / fps))


def _camera_inference_s(packet: DetectionPacket, camera_index: int) -> float:
    if camera_index >= len(packet.inference_ms_by_camera):
        return 0.0
    return max(0.0, float(packet.inference_ms_by_camera[camera_index]) / 1000.0)


def _shift_box(box: Box, shift_x: float, shift_y: float) -> Box:
    return (
        box[0] + shift_x,
        box[1] + shift_y,
        box[2] + shift_x,
        box[3] + shift_y,
        box[4],
        int(box[5]),
    )


def _match_previous_box(box: Box, previous_boxes: tuple[Box, ...]) -> Box | None:
    same_class = tuple(candidate for candidate in previous_boxes if int(candidate[5]) == int(box[5]))
    candidates = same_class or previous_boxes
    plausible = tuple(candidate for candidate in candidates if boxes_look_like_same_cap(box, candidate))
    if not plausible and len(candidates) == 1 and box_size_ratio(box, candidates[0]) >= 0.25:
        return candidates[0]
    if not plausible:
        return None
    current_x, current_y = box_center(box)
    return min(
        plausible,
        key=lambda candidate: abs(box_center(candidate)[0] - current_x)
        + abs(box_center(candidate)[1] - current_y),
    )


def _predict_shifted_boxes(
    current_boxes: tuple[Box, ...],
    previous_boxes: tuple[Box, ...],
    *,
    history_s: float,
    extrapolation_s: float,
) -> tuple[Box, ...]:
    if extrapolation_s <= 0.0 or history_s <= 0.0:
        return current_boxes
    predicted_boxes = []
    for box in current_boxes:
        previous_box = _match_previous_box(box, previous_boxes)
        if previous_box is None:
            predicted_boxes.append(box)
            continue
        current_center = box_center(box)
        previous_center = box_center(previous_box)
        shift_x = ((current_center[0] - previous_center[0]) / history_s) * extrapolation_s
        shift_y = ((current_center[1] - previous_center[1]) / history_s) * extrapolation_s
        predicted_boxes.append(_shift_box(box, shift_x, shift_y))
    return tuple(predicted_boxes)


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
    preview_latency_compensation_ms: int | float = 0.0,
) -> tuple[CameraPreviewView, ...]:
    if current_packet is None:
        return tuple(CameraPreviewView(frame, ()) for frame in live_frames)
    if len(current_packet.boxes_by_camera) != len(live_frames):
        return tuple(CameraPreviewView(frame, ()) for frame in live_frames)

    base_timeout_s = overlay_stale_timeout_s(target_fps)
    current_timestamps = current_packet.frame_pair.timestamps
    previous_timestamps = previous_packet.frame_pair.timestamps if previous_packet is not None else ()
    views: list[CameraPreviewView] = []
    for camera_index, live_frame in enumerate(live_frames):
        detection_frame = current_packet.frame_pair.frames[camera_index]
        current_timestamp = current_timestamps[camera_index]
        overlay_age_s = float(live_frame.timestamp) - float(current_timestamp)
        overlay_age_s += max(0.0, float(preview_latency_compensation_ms)) / 1000.0
        timeout_s = _overlay_timeout_s(
            current_packet,
            previous_packet,
            camera_index,
            base_timeout_s=base_timeout_s,
        )
        current_boxes = current_packet.boxes_by_camera[camera_index]
        if overlay_age_s > timeout_s:
            views.append(CameraPreviewView(live_frame, ()))
            continue

        if int(live_frame.sequence) <= int(detection_frame.sequence):
            views.append(CameraPreviewView(detection_frame, current_boxes))
            continue

        can_extrapolate = (
            previous_packet is not None
            and camera_index < len(previous_packet.boxes_by_camera)
            and camera_index < len(previous_timestamps)
        )
        if not can_extrapolate:
            views.append(CameraPreviewView(detection_frame, current_boxes))
            continue

        previous_timestamp = previous_timestamps[camera_index]
        history_s = float(current_timestamp) - float(previous_timestamp)
        extrapolation_s = min(max(0.0, overlay_age_s), MAX_PREVIEW_EXTRAPOLATION_S)
        predicted_boxes = _predict_shifted_boxes(
            current_boxes,
            previous_packet.boxes_by_camera[camera_index],
            history_s=history_s,
            extrapolation_s=extrapolation_s,
        )
        views.append(CameraPreviewView(live_frame, predicted_boxes))
    return tuple(views)


def predict_preview_overlay(
    previous_packet: DetectionPacket | None,
    current_packet: DetectionPacket | None,
    live_frames: tuple[CapturedFrame, ...],
    *,
    target_fps: int | float,
    preview_latency_compensation_ms: int | float = 0.0,
) -> tuple[tuple[Box, ...], ...]:
    return tuple(
        view.boxes
        for view in resolve_preview_views(
            previous_packet,
            current_packet,
            live_frames,
            target_fps=target_fps,
            preview_latency_compensation_ms=preview_latency_compensation_ms,
        )
    )
