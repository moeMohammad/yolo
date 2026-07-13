"""Per-camera cap tracking.

Each camera is tracked completely independently (there is no cross-camera pixel
matching in v6 - the two cameras view the cap from different angles). A track
accumulates frames of one physical cap as it moves through the field of view and
finishes when it goes unmatched for ``track_timeout_ms``. Association is greedy
IoU first, then a velocity-gated centroid fallback for fast motion — gated so
the next cap entering right behind a departed one starts a fresh track instead
of silently extending (and immortalizing) the old one.

Safety qualification is deliberately independent of model confidence: a track
must persist across several frames and move coherently like a conveyor-carried
cap before the event manager accepts it. Dirt must also persist for consecutive
frames; a single 0.9+ hallucination can no longer poison a clean cap.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import count

from .config import DEFECT_CLASS_ID, UNDEFECTED_CLASS_ID
from .types import Box


# When boxes from consecutive frames don't overlap (a fast cap on a fast belt),
# fall back to nearest-centroid association if the gap is within this many box
# widths. Only used while a track has no velocity estimate yet (single
# observation); afterwards the predictive gate below takes over.
CENTROID_MATCH_GATE = 2.0

# Once a track has a velocity estimate, the fallback matches against the
# *predicted* center instead of the last one, so the acceptance gate can be much
# tighter. Anything further off the prediction is a different cap — most
# importantly the next cap entering right behind one that just left view, which
# the loose centroid gate used to absorb into the old track (chaining a
# continuous feed of caps into one track that never times out, so no cap event
# and no fire ever happens).
PREDICTED_MATCH_GATE = 0.8
# Two tracks that cross at nearly the same time are fragments of one physical
# presence only when they also occupy the same lane/perpendicular image band.
PRESENCE_CROSS_LANE_GATE = 1.5


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
    velocity: tuple[float, float] | None = None  # px/s, from the last two observations
    first_box: Box | None = None
    path_length_px: float = 0.0
    defect_frame_count: int = 0
    undefected_frame_count: int = 0
    consecutive_defect_frames: int = 0
    max_consecutive_defect_frames: int = 0
    min_defect_frames: int = 3
    presence_cycle_id: int | None = None
    crossed_presence_line: bool = False
    presence_crossed_at: float | None = None
    presence_cross_coordinate: float | None = None
    presence_cross_scale: float = 1.0
    line_negative_frames: int = 0
    line_positive_frames: int = 0
    motion_axis: str = "x"
    motion_direction: str = "positive"
    largest_observation_gap_s: float = 0.0
    max_observation_gap_s: float = 0.500

    def observe(
        self,
        box: Box,
        timestamp: float,
        *,
        presence_axis: str = "x",
        presence_line: float | None = None,
        presence_direction: str = "positive",
    ) -> None:
        self.motion_axis = "y" if str(presence_axis).lower() == "y" else "x"
        self.motion_direction = str(presence_direction).lower()
        if self.first_box is None:
            self.first_box = self.last_box
        dt = float(timestamp) - self.last_seen
        if dt > 0.0:
            self.largest_observation_gap_s = max(self.largest_observation_gap_s, dt)
        if dt > self.max_observation_gap_s:
            self.consecutive_defect_frames = 0
        prev_cx, prev_cy = box_center(self.last_box)
        new_cx, new_cy = box_center(box)
        self.path_length_px += math.hypot(new_cx - prev_cx, new_cy - prev_cy)
        if dt > 0.0:
            self.velocity = ((new_cx - prev_cx) / dt, (new_cy - prev_cy) / dt)
        if presence_line is not None:
            previous_value = prev_cx if presence_axis == "x" else prev_cy
            current_value = new_cx if presence_axis == "x" else new_cy
            previous_delta = previous_value - presence_line
            current_delta = current_value - presence_line
            if current_delta < 0.0:
                self.line_negative_frames += 1
            elif current_delta > 0.0:
                self.line_positive_frames += 1
            # Require observed center motion from one side to the other (or
            # from one side exactly onto the line). Merely having a large box
            # overlap the line is not evidence that a cap traversed it.
            if (
                previous_delta * current_delta < 0.0
                or (current_delta == 0.0 and previous_delta != 0.0)
            ):
                self.crossed_presence_line = True
                if self.presence_crossed_at is None:
                    self.presence_crossed_at = float(timestamp)
                    if presence_axis == "x":
                        self.presence_cross_coordinate = new_cy
                        self.presence_cross_scale = max(1.0, float(box[3]) - float(box[1]))
                    else:
                        self.presence_cross_coordinate = new_cx
                        self.presence_cross_scale = max(1.0, float(box[2]) - float(box[0]))
        self.last_seen = float(timestamp)
        self.last_box = box
        self.frame_count += 1
        confidence = float(box[4])
        if int(box[5]) == DEFECT_CLASS_ID:
            self.defect_frame_count += 1
            self.consecutive_defect_frames += 1
            self.max_consecutive_defect_frames = max(
                self.max_consecutive_defect_frames,
                self.consecutive_defect_frames,
            )
            # A single high-confidence frame is not sufficient: the real rig
            # produces 0.9+ dirt hallucinations on an empty belt.
            if self.max_consecutive_defect_frames >= self.min_defect_frames:
                self.is_defect = True
            self.best_defect_conf = max(self.best_defect_conf, confidence)
        else:
            self.undefected_frame_count += 1
            self.consecutive_defect_frames = 0
            self.best_undefected_conf = max(self.best_undefected_conf, confidence)

    def mark_missed(self) -> None:
        """Break temporal dirt confirmation on a processed empty/miss frame."""

        self.consecutive_defect_frames = 0

    @property
    def travel_ratio(self) -> float:
        first_box = self.first_box or self.last_box
        first_cx, first_cy = box_center(first_box)
        last_cx, last_cy = box_center(self.last_box)
        displacement = math.hypot(last_cx - first_cx, last_cy - first_cy)
        scale = max(
            1.0,
            first_box[2] - first_box[0],
            first_box[3] - first_box[1],
            self.last_box[2] - self.last_box[0],
            self.last_box[3] - self.last_box[1],
        )
        return displacement / scale

    @property
    def motion_directionality(self) -> float:
        """Fraction of observed motion that progressed along the belt axis."""

        first_box = self.first_box or self.last_box
        first_cx, first_cy = box_center(first_box)
        last_cx, last_cy = box_center(self.last_box)
        signed_displacement = (last_cy - first_cy) if self.motion_axis == "y" else (last_cx - first_cx)
        if self.motion_direction == "negative":
            displacement = max(0.0, -signed_displacement)
        elif self.motion_direction == "either":
            displacement = abs(signed_displacement)
        else:
            displacement = max(0.0, signed_displacement)
        if self.path_length_px <= 0.0:
            return 0.0
        return min(1.0, displacement / self.path_length_px)

    def qualifies_as_cap(
        self,
        *,
        min_frames: int,
        min_travel_ratio: float,
        min_directionality: float,
        max_observation_gap: float | None = None,
        require_line_crossing: bool = True,
    ) -> bool:
        """Return whether observations look like a conveyor-carried cap."""

        return (
            self.frame_count >= int(min_frames)
            and self.travel_ratio >= float(min_travel_ratio)
            and self.motion_directionality >= float(min_directionality)
            and (
                max_observation_gap is None
                or self.largest_observation_gap_s <= float(max_observation_gap)
            )
            and (self.crossed_presence_line or not require_line_crossing)
            and (
                not require_line_crossing
                or (self.line_negative_frames >= 2 and self.line_positive_frames >= 2)
            )
        )

    @property
    def winning_class_id(self) -> int:
        return DEFECT_CLASS_ID if self.is_defect else UNDEFECTED_CLASS_ID

    @property
    def winning_confidence(self) -> float:
        return self.best_defect_conf if self.is_defect else self.best_undefected_conf


class CameraTracker:
    """Greedy associator + lifecycle manager for one camera's tracks."""

    def __init__(
        self,
        camera_index: int,
        *,
        track_iou: float,
        track_timeout_s: float,
        min_defect_frames: int = 3,
        presence_clear_s: float = 0.350,
        min_track_frames: int = 4,
        min_track_travel_ratio: float = 0.35,
        min_track_directionality: float = 0.60,
        presence_line_axis: str = "x",
        presence_line_ratio: float = 0.50,
        presence_direction: str = "positive",
        max_track_gap_s: float = 0.500,
    ):
        self.camera_index = int(camera_index)
        self.track_iou = float(track_iou)
        self.track_timeout_s = float(track_timeout_s)
        self.min_defect_frames = max(2, int(min_defect_frames))
        self.presence_clear_s = max(0.0, float(presence_clear_s))
        self.min_track_frames = max(2, int(min_track_frames))
        self.min_track_travel_ratio = max(0.0, float(min_track_travel_ratio))
        self.min_track_directionality = max(0.0, min(1.0, float(min_track_directionality)))
        self.presence_line_axis = "y" if str(presence_line_axis).lower() == "y" else "x"
        self.presence_line_ratio = max(0.0, min(1.0, float(presence_line_ratio)))
        direction = str(presence_direction).lower()
        self.presence_direction = direction if direction in {"positive", "negative", "either"} else "positive"
        self.max_track_gap_s = max(0.001, float(max_track_gap_s))
        self._tracks: list[Track] = []
        self._counter = count(1)
        self._presence_counter = count(1)
        self._recent_crossings: list[tuple[float, float, float, int]] = []

    @property
    def active_tracks(self) -> tuple[Track, ...]:
        return tuple(self._tracks)

    def update(self, boxes, timestamp: float, frame_size: tuple[int, int] | None = None) -> None:
        """Associate this frame's detections to existing tracks (greedy)."""

        boxes = list(boxes)
        presence_line = None
        if frame_size is not None:
            width, height = frame_size
            dimension = width if self.presence_line_axis == "x" else height
            presence_line = float(dimension) * self.presence_line_ratio
        matched_tracks: set[int] = set()
        matched_dets: set[int] = set()
        # A detection arriving after the configured timeout belongs to a new
        # physical presence even if its box overlaps perfectly. Keep expired
        # tracks alive only so collect_finished() can hand them to the manager;
        # never allow association to revive them.
        association_tracks = [
            ti
            for ti, track in enumerate(self._tracks)
            if float(timestamp) - track.last_seen < self.track_timeout_s
        ]

        # 1) Greedy IoU association, highest overlap first.
        iou_pairs = [
            (box_iou(track.last_box, box), ti, di)
            for ti in association_tracks
            for track in (self._tracks[ti],)
            for di, box in enumerate(boxes)
        ]
        iou_pairs.sort(key=lambda item: item[0], reverse=True)
        for iou, ti, di in iou_pairs:
            if iou < self.track_iou:
                break  # sorted desc: nothing left meets the threshold
            if ti in matched_tracks or di in matched_dets:
                continue
            self._tracks[ti].observe(
                boxes[di],
                timestamp,
                presence_axis=self.presence_line_axis,
                presence_line=presence_line,
                presence_direction=self.presence_direction,
            )
            matched_tracks.add(ti)
            matched_dets.add(di)

        # 2) Nearest-centroid fallback for detections IoU missed (fast motion).
        #    Velocity-gated: with a motion estimate the detection must sit near
        #    the *predicted* center, so the next cap entering behind a departed
        #    one starts its own track instead of extending the old one.
        remaining_tracks = [ti for ti in association_tracks if ti not in matched_tracks]
        remaining_dets = [di for di in range(len(boxes)) if di not in matched_dets]
        if remaining_tracks and remaining_dets:
            candidates = []
            for ti in remaining_tracks:
                for di in remaining_dets:
                    distance, gate = self._fallback_candidate(self._tracks[ti], boxes[di], timestamp)
                    if distance <= gate:
                        candidates.append((distance, ti, di))
            candidates.sort(key=lambda item: item[0])
            for distance, ti, di in candidates:
                if ti in matched_tracks or di in matched_dets:
                    continue
                self._tracks[ti].observe(
                    boxes[di],
                    timestamp,
                    presence_axis=self.presence_line_axis,
                    presence_line=presence_line,
                    presence_direction=self.presence_direction,
                )
                matched_tracks.add(ti)
                matched_dets.add(di)

        # A defect streak means consecutive *processed* observations. Any
        # successful frame in which an existing track was not matched breaks
        # that streak, including a completely empty frame.
        for ti, track in enumerate(self._tracks):
            if ti not in matched_tracks:
                track.mark_missed()

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
                    is_defect=False,
                    best_defect_conf=float(box[4]) if int(box[5]) == DEFECT_CLASS_ID else 0.0,
                    best_undefected_conf=float(box[4]) if int(box[5]) != DEFECT_CLASS_ID else 0.0,
                    first_box=box,
                    defect_frame_count=1 if int(box[5]) == DEFECT_CLASS_ID else 0,
                    undefected_frame_count=1 if int(box[5]) != DEFECT_CLASS_ID else 0,
                    consecutive_defect_frames=1 if int(box[5]) == DEFECT_CLASS_ID else 0,
                    max_consecutive_defect_frames=1 if int(box[5]) == DEFECT_CLASS_ID else 0,
                    min_defect_frames=self.min_defect_frames,
                    presence_cycle_id=None,
                    # One observation cannot prove that the center traversed
                    # the inspection line, even when the box spans it.
                    crossed_presence_line=False,
                    line_negative_frames=(
                        1
                        if presence_line is not None
                        and (
                            box_center(box)[0] if self.presence_line_axis == "x" else box_center(box)[1]
                        )
                        < presence_line
                        else 0
                    ),
                    line_positive_frames=(
                        1
                        if presence_line is not None
                        and (
                            box_center(box)[0] if self.presence_line_axis == "x" else box_center(box)[1]
                        )
                        > presence_line
                        else 0
                    ),
                    motion_axis=self.presence_line_axis,
                    motion_direction=self.presence_direction,
                    max_observation_gap_s=self.max_track_gap_s,
                )
            )
        self._assign_presence_cycles(float(timestamp))

    def _assign_presence_cycles(self, timestamp: float) -> None:
        """Assign spatial-temporal crossing ids to qualified observations.

        Static high-confidence hallucinations must neither open nor hold a
        presence cycle; otherwise one persistent false box would suppress every
        later real cap after the first consumed cycle. Separate caps visible at
        the same time must also remain separate when they occupy different
        perpendicular image bands.
        """

        observed_now = [
            track
            for track in self._tracks
            if math.isclose(track.last_seen, timestamp, rel_tol=0.0, abs_tol=1e-9)
            and track.qualifies_as_cap(
                min_frames=self.min_track_frames,
                min_travel_ratio=self.min_track_travel_ratio,
                min_directionality=self.min_track_directionality,
                max_observation_gap=self.max_track_gap_s,
                require_line_crossing=True,
            )
        ]
        oldest_allowed = float(timestamp) - self.presence_clear_s
        self._recent_crossings = [
            crossing for crossing in self._recent_crossings if crossing[0] >= oldest_allowed
        ]
        for track in observed_now:
            if track.presence_cycle_id is None:
                crossed_at = float(timestamp if track.presence_crossed_at is None else track.presence_crossed_at)
                coordinate = float(
                    0.0 if track.presence_cross_coordinate is None else track.presence_cross_coordinate
                )
                scale = max(1.0, float(track.presence_cross_scale))
                matching_cycles = [
                    (abs(crossed_at - other_time), cycle_id)
                    for other_time, other_coordinate, other_scale, cycle_id in self._recent_crossings
                    if abs(crossed_at - other_time) <= self.presence_clear_s
                    and abs(coordinate - other_coordinate)
                    <= PRESENCE_CROSS_LANE_GATE * max(scale, other_scale)
                ]
                if matching_cycles:
                    _distance, cycle_id = min(matching_cycles)
                else:
                    cycle_id = next(self._presence_counter)
                track.presence_cycle_id = cycle_id
                self._recent_crossings.append((crossed_at, coordinate, scale, cycle_id))

    def _fallback_candidate(self, track: Track, box: Box, timestamp: float) -> tuple[float, float]:
        """Normalized distance and acceptance gate for a non-IoU fallback match."""

        if track.velocity is None:
            return normalized_center_distance(track.last_box, box), CENTROID_MATCH_GATE
        dt = max(0.0, float(timestamp) - track.last_seen)
        last_cx, last_cy = box_center(track.last_box)
        predicted_cx = last_cx + track.velocity[0] * dt
        predicted_cy = last_cy + track.velocity[1] * dt
        det_cx, det_cy = box_center(box)
        scale = max(
            1.0,
            track.last_box[2] - track.last_box[0],
            track.last_box[3] - track.last_box[1],
            box[2] - box[0],
            box[3] - box[1],
        )
        distance = math.hypot(det_cx - predicted_cx, det_cy - predicted_cy) / scale
        return distance, PREDICTED_MATCH_GATE

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
