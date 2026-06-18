from __future__ import annotations

import math

from .config import CLASS_NAMES
from .types import Box


def class_name(class_id: int | None) -> str | None:
    if class_id is None:
        return None
    if 0 <= int(class_id) < len(CLASS_NAMES):
        return CLASS_NAMES[int(class_id)]
    return f"class_{int(class_id)}"


def box_center(box: Box) -> tuple[float, float]:
    x1, y1, x2, y2 = box[:4]
    return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)


def box_area(box: Box) -> float:
    x1, y1, x2, y2 = box[:4]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def box_iou(box_a: Box, box_b: Box) -> float:
    ax1, ay1, ax2, ay2 = box_a[:4]
    bx1, by1, bx2, by2 = box_b[:4]
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    if inter <= 0.0:
        return 0.0
    union = box_area(box_a) + box_area(box_b) - inter
    return 0.0 if union <= 0.0 else inter / union


def normalized_center_distance(box_a: Box, box_b: Box) -> float:
    ax, ay = box_center(box_a)
    bx, by = box_center(box_b)
    scale = max(
        1.0,
        box_a[2] - box_a[0],
        box_a[3] - box_a[1],
        box_b[2] - box_b[0],
        box_b[3] - box_b[1],
    )
    return math.hypot(ax - bx, ay - by) / scale


def box_size_ratio(box_a: Box, box_b: Box) -> float:
    largest = max(box_area(box_a), box_area(box_b), 1.0)
    smallest = min(box_area(box_a), box_area(box_b))
    return smallest / largest


def boxes_look_like_same_cap(box_a: Box, box_b: Box) -> bool:
    if box_iou(box_a, box_b) >= 0.01:
        return True
    if box_size_ratio(box_a, box_b) < 0.45:
        return False
    return normalized_center_distance(box_a, box_b) <= 3.0


def box_spans_line_coordinate(box: Box, *, axis: str, line_coordinate: float) -> bool:
    x1, y1, x2, y2 = box[:4]
    start, end = (x1, x2) if axis == "x" else (y1, y2)
    return min(start, end) <= line_coordinate <= max(start, end)


def frame_line_coordinate(frame_size: tuple[int, int], *, axis: str, ratio: float) -> float:
    width, height = frame_size
    return float(width if axis == "x" else height) * float(ratio)
