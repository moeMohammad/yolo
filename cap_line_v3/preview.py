from __future__ import annotations

from .geometry import box_center, boxes_look_like_same_cap
from .types import Box, CapturedFrame, DetectionPacket


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
    if not plausible:
        return None
    current_x, current_y = box_center(box)
    return min(
        plausible,
        key=lambda candidate: abs(box_center(candidate)[0] - current_x)
        + abs(box_center(candidate)[1] - current_y),
    )


def predict_preview_overlay(
    previous_packet: DetectionPacket | None,
    current_packet: DetectionPacket | None,
    live_frames: tuple[CapturedFrame, ...],
    *,
    target_fps: int | float,
) -> tuple[tuple[Box, ...], ...]:
    if current_packet is None:
        return tuple(() for _frame in live_frames)
    if len(current_packet.boxes_by_camera) != len(live_frames):
        return tuple(() for _frame in live_frames)

    base_timeout_s = overlay_stale_timeout_s(target_fps)
    current_timestamps = current_packet.frame_pair.timestamps
    previous_timestamps = previous_packet.frame_pair.timestamps if previous_packet is not None else ()
    predicted_by_camera = []
    for camera_index, live_frame in enumerate(live_frames):
        current_timestamp = current_timestamps[camera_index]
        overlay_age_s = float(live_frame.timestamp) - float(current_timestamp)
        timeout_s = max(base_timeout_s, _camera_inference_s(current_packet, camera_index) + base_timeout_s)
        if previous_packet is not None and camera_index < len(previous_timestamps):
            history_s = float(current_timestamp) - float(previous_timestamps[camera_index])
            if history_s > 0.0:
                timeout_s = max(timeout_s, history_s * 1.5)
        timeout_s = min(1.25, timeout_s)
        if overlay_age_s > timeout_s:
            predicted_by_camera.append(())
            continue
        current_boxes = current_packet.boxes_by_camera[camera_index]
        if previous_packet is None or camera_index >= len(previous_packet.boxes_by_camera):
            predicted_by_camera.append(current_boxes)
            continue
        previous_timestamp = previous_timestamps[camera_index]
        history_s = float(current_timestamp) - float(previous_timestamp)
        if overlay_age_s <= 0.0 or history_s <= 0.0:
            predicted_by_camera.append(current_boxes)
            continue
        previous_boxes = previous_packet.boxes_by_camera[camera_index]
        predicted_boxes = []
        for box in current_boxes:
            previous_box = _match_previous_box(box, previous_boxes)
            if previous_box is None:
                predicted_boxes.append(box)
                continue
            current_center = box_center(box)
            previous_center = box_center(previous_box)
            shift_x = ((current_center[0] - previous_center[0]) / history_s) * overlay_age_s
            shift_y = ((current_center[1] - previous_center[1]) / history_s) * overlay_age_s
            predicted_boxes.append(_shift_box(box, shift_x, shift_y))
        predicted_by_camera.append(tuple(predicted_boxes))
    return tuple(predicted_by_camera)
