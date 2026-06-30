"""Per-camera cap tracking.

Each camera is tracked completely independently (there is no cross-camera pixel
matching in v4 - the two cameras view the cap from different angles). A track
accumulates frames of one physical cap as it moves through the field of view and
finishes when it goes unmatched for ``track_timeout_ms``.

Decision rule is pure single-frame "defect wins" (OR): a single ``dirt_defect``
frame makes the whole track defective. Detections are already confidence-filtered
by ``model.postprocess`` before they reach here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import count

from .config import DEFECT_CLASS_ID, UNDEFECTED_CLASS_ID
from .types import Box


# When boxes from consecutive frames don't overlap (a fast cap on a fast belt),
# fall back to nearest-centroid association if the gap is within this many box
# widths. Generous enough for fast motion, tight enough not to bridge the spaced
# out next cap.
CENTROID_MATCH_GATE = 2.0


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


def box_center(box: Box) -> tuple[float, float]:
    x1, y1, x2, y2 = box[:4]
    return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)


def normalized_center_distance(box_a: Box, box_b: Box) -> float:
    ax, ay = box_center(box_a)
    bx, by = box_center(box_b)
    scale = max(1.0, box_a[2] - box_a[0], box_a[3] - box_a[1], box_b[2] - box_b[0], box_b[3] - box_b[1])
    return math.hypot(ax - bx, ay - by) / scale


@dataclass
class Track:
    """One physical cap as seen by one camera across consecutive frames."""

    track_id: int
    camera_index: int
    first_seen: float
    last_seen: float
    frame_count: int
    last_box: Box
    is_defect: bool = False
    best_defect_conf: float = 0.0
    best_undefected_conf: float = 0.0

    def observe(self, box: Box, timestamp: float) -> None:
        self.last_seen = float(timestamp)
        self.last_box = box
        self.frame_count += 1
        confidence = float(box[4])
        if int(box[5]) == DEFECT_CLASS_ID:
            self.is_defect = True  # defect-wins: one defect frame is enough
            self.best_defect_conf = max(self.best_defect_conf, confidence)
        else:
            self.best_undefected_conf = max(self.best_undefected_conf, confidence)

    @property
    def winning_class_id(self) -> int:
        return DEFECT_CLASS_ID if self.is_defect else UNDEFECTED_CLASS_ID

    @property
    def winning_confidence(self) -> float:
        return self.best_defect_conf if self.is_defect else self.best_undefected_conf


class CameraTracker:
    """Greedy associator + lifecycle manager for one camera's tracks."""

    def __init__(self, camera_index: int, *, track_iou: float, track_timeout_s: float):
        self.camera_index = int(camera_index)
        self.track_iou = float(track_iou)
        self.track_timeout_s = float(track_timeout_s)
        self._tracks: list[Track] = []
        self._counter = count(1)

    @property
    def active_tracks(self) -> tuple[Track, ...]:
        return tuple(self._tracks)

    def update(self, boxes, timestamp: float) -> None:
        """Associate this frame's detections to existing tracks (greedy)."""

        boxes = list(boxes)
        if not boxes:
            return
        matched_tracks: set[int] = set()
        matched_dets: set[int] = set()

        # 1) Greedy IoU association, highest overlap first.
        iou_pairs = [
            (box_iou(track.last_box, box), ti, di)
            for ti, track in enumerate(self._tracks)
            for di, box in enumerate(boxes)
        ]
        iou_pairs.sort(key=lambda item: item[0], reverse=True)
        for iou, ti, di in iou_pairs:
            if iou < self.track_iou:
                break  # sorted desc: nothing left meets the threshold
            if ti in matched_tracks or di in matched_dets:
                continue
            self._tracks[ti].observe(boxes[di], timestamp)
            matched_tracks.add(ti)
            matched_dets.add(di)

        # 2) Nearest-centroid fallback for detections IoU missed (fast motion).
        remaining_tracks = [ti for ti in range(len(self._tracks)) if ti not in matched_tracks]
        remaining_dets = [di for di in range(len(boxes)) if di not in matched_dets]
        if remaining_tracks and remaining_dets:
            dist_pairs = [
                (normalized_center_distance(self._tracks[ti].last_box, boxes[di]), ti, di)
                for ti in remaining_tracks
                for di in remaining_dets
            ]
            dist_pairs.sort(key=lambda item: item[0])
            for distance, ti, di in dist_pairs:
                if distance > CENTROID_MATCH_GATE:
                    break
                if ti in matched_tracks or di in matched_dets:
                    continue
                self._tracks[ti].observe(boxes[di], timestamp)
                matched_tracks.add(ti)
                matched_dets.add(di)

        # 3) Anything still unmatched becomes a new track.
        for di, box in enumerate(boxes):
            if di in matched_dets:
                continue
            self._tracks.append(
                Track(
                    track_id=next(self._counter),
                    camera_index=self.camera_index,
                    first_seen=float(timestamp),
                    last_seen=float(timestamp),
                    frame_count=1,
                    last_box=box,
                    is_defect=int(box[5]) == DEFECT_CLASS_ID,
                    best_defect_conf=float(box[4]) if int(box[5]) == DEFECT_CLASS_ID else 0.0,
                    best_undefected_conf=float(box[4]) if int(box[5]) != DEFECT_CLASS_ID else 0.0,
                )
            )

    def collect_finished(self, now: float) -> list[Track]:
        """Return and remove tracks unmatched for >= ``track_timeout_s``."""

        finished = [track for track in self._tracks if float(now) - track.last_seen >= self.track_timeout_s]
        if finished:
            finished_ids = {id(track) for track in finished}
            self._tracks = [track for track in self._tracks if id(track) not in finished_ids]
        return finished

    def flush(self) -> list[Track]:
        """Return and remove all remaining tracks (used at shutdown)."""

        remaining = self._tracks
        self._tracks = []
        return remaining
