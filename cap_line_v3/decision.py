from __future__ import annotations

from dataclasses import dataclass, field

from .config import DEFECT_CLASS_ID, RuntimeConfig
from .geometry import (
    box_center_value,
    box_crossed_line_between,
    box_size_along_axis,
    boxes_look_like_same_cap,
    class_name,
    frame_line_coordinate,
)
from .types import CameraObservationSummary, CameraVote, CapEvaluation, TrackObservation, TrackedCapDecision


def calculate_trigger_delay(config: RuntimeConfig) -> float:
    return float(config.nozzle_distance_mm) / float(config.belt_speed_mm_per_s) + float(config.trigger_offset_s)


def compute_requested_trigger_delay(config: RuntimeConfig) -> float:
    return calculate_trigger_delay(config) - float(config.latency_compensation_ms) / 1000.0


@dataclass
class TrackedCap:
    event_id: int
    created_at: float
    last_seen_at: float
    camera_summaries: dict[int, CameraObservationSummary] = field(default_factory=dict)
    actuation_camera_summaries: dict[int, CameraObservationSummary] = field(default_factory=dict)
    camera_indices: set[int] = field(default_factory=set)
    latest_box_by_camera: dict[int, tuple[float, float, float, float, float, int]] = field(default_factory=dict)
    box_history_by_camera: dict[int, list[tuple[float, float, float, float, float, int]]] = field(default_factory=dict)
    anchor_time: float | None = None
    anchor_camera_index: int | None = None
    actuation_time: float | None = None
    actuation_camera_index: int | None = None
    trigger_decision: TrackedCapDecision | None = None

    def add_observation(
        self,
        observation: TrackObservation,
        *,
        anchor_axis: str = "x",
        anchor_line_ratio: float = 0.5,
    ) -> None:
        previous_box = self.latest_box_by_camera.get(int(observation.camera_index))
        self.last_seen_at = max(float(self.last_seen_at), float(observation.timestamp))
        self.camera_indices.add(int(observation.camera_index))
        self.latest_box_by_camera[int(observation.camera_index)] = observation.box
        box_history = self.box_history_by_camera.setdefault(int(observation.camera_index), [])
        box_history.append(observation.box)
        del box_history[:-3]
        summary = self.camera_summaries.setdefault(
            int(observation.camera_index),
            CameraObservationSummary(),
        )
        summary.add(observation.class_id, observation.confidence, observation.timestamp)
        if self.anchor_time is None or observation.timestamp < self.anchor_time:
            self.anchor_time = float(observation.timestamp)
            self.anchor_camera_index = int(observation.camera_index)

        line_coordinate = frame_line_coordinate(
            observation.frame_size,
            axis=anchor_axis,
            ratio=anchor_line_ratio,
        )
        at_actuation = observation.at_actuation_line or (
            previous_box is not None
            and box_crossed_line_between(
                previous_box,
                observation.box,
                axis=anchor_axis,
                line_coordinate=line_coordinate,
            )
        )

        if at_actuation:
            actuation_summary = self.actuation_camera_summaries.setdefault(
                int(observation.camera_index),
                CameraObservationSummary(),
            )
            actuation_summary.add(
                observation.class_id,
                observation.confidence,
                observation.timestamp,
            )
            if self.actuation_time is None or observation.timestamp < self.actuation_time:
                self.actuation_time = float(observation.timestamp)
                self.actuation_camera_index = int(observation.camera_index)


def _moves_backward_against_history(
    box: tuple[float, float, float, float, float, int],
    history: list[tuple[float, float, float, float, float, int]],
    *,
    axis: str,
) -> bool:
    if len(history) < 2:
        return False
    previous_box = history[-2]
    latest_box = history[-1]
    previous_center = box_center_value(previous_box, axis)
    latest_center = box_center_value(latest_box, axis)
    candidate_center = box_center_value(box, axis)
    direction = latest_center - previous_center
    if abs(direction) < 1.0:
        return False
    sign = 1.0 if direction > 0.0 else -1.0
    progress = (candidate_center - latest_center) * sign
    size = max(1.0, box_size_along_axis(latest_box, axis), box_size_along_axis(box, axis))
    return progress < -0.5 * size


class TrackedCapManager:
    def __init__(
        self,
        *,
        camera_count: int,
        merge_window_seconds: float,
        finalize_quiet_seconds: float,
        anchor_axis: str,
        anchor_line_ratio: float,
    ):
        self.camera_count = int(camera_count)
        self.merge_window_seconds = float(merge_window_seconds)
        self.finalize_quiet_seconds = float(finalize_quiet_seconds)
        self.anchor_axis = anchor_axis
        self.anchor_line_ratio = float(anchor_line_ratio)
        self._open_caps: list[TrackedCap] = []
        self._next_event_id = 1

    def update(self, observations: list[TrackObservation]) -> list[TrackedCap]:
        touched_caps: list[TrackedCap] = []
        touched_ids: set[int] = set()
        matched_camera_keys: set[tuple[int, int]] = set()
        for observation in observations:
            tracked_cap = self._find_match(observation, matched_camera_keys=matched_camera_keys)
            if tracked_cap is None:
                tracked_cap = TrackedCap(
                    event_id=self._next_event_id,
                    created_at=observation.timestamp,
                    last_seen_at=observation.timestamp,
                )
                self._next_event_id += 1
                self._open_caps.append(tracked_cap)
            tracked_cap.add_observation(
                observation,
                anchor_axis=self.anchor_axis,
                anchor_line_ratio=self.anchor_line_ratio,
            )
            matched_camera_keys.add((tracked_cap.event_id, int(observation.camera_index)))
            self._mark_recent(tracked_cap)
            if tracked_cap.event_id not in touched_ids:
                touched_ids.add(tracked_cap.event_id)
                touched_caps.append(tracked_cap)
        return touched_caps

    def open_caps(self) -> tuple[TrackedCap, ...]:
        return tuple(self._open_caps)

    def pop_finalized(self, now: float) -> list[TrackedCap]:
        finalized = []
        remaining = []
        quiet_seconds = max(self.finalize_quiet_seconds, self.merge_window_seconds)
        for tracked_cap in self._open_caps:
            if float(now) - float(tracked_cap.last_seen_at) < quiet_seconds:
                remaining.append(tracked_cap)
                continue
            finalized.append(tracked_cap)
        self._open_caps = remaining
        return finalized

    def _find_match(
        self,
        observation: TrackObservation,
        *,
        matched_camera_keys: set[tuple[int, int]],
    ) -> TrackedCap | None:
        camera_index = int(observation.camera_index)
        for tracked_cap in reversed(self._open_caps):
            if (tracked_cap.event_id, camera_index) in matched_camera_keys:
                continue
            latest_box = tracked_cap.latest_box_by_camera.get(camera_index)
            if latest_box is None:
                continue
            history = tracked_cap.box_history_by_camera.get(camera_index, [])
            if _moves_backward_against_history(observation.box, history, axis=self.anchor_axis):
                continue
            if boxes_look_like_same_cap(observation.box, latest_box):
                return tracked_cap

        cross_camera_candidates = [
            tracked_cap
            for tracked_cap in self._open_caps
            if (tracked_cap.event_id, camera_index) not in matched_camera_keys
            and camera_index not in tracked_cap.camera_indices
            and (float(observation.timestamp) - float(tracked_cap.last_seen_at)) <= self.merge_window_seconds
        ]
        if not cross_camera_candidates:
            return None
        return max(cross_camera_candidates, key=lambda tracked_cap: tracked_cap.last_seen_at)

    def _mark_recent(self, tracked_cap: TrackedCap) -> None:
        try:
            self._open_caps.remove(tracked_cap)
        except ValueError:
            pass
        self._open_caps.append(tracked_cap)


def build_camera_vote(summary: CameraObservationSummary | None, *, camera_index: int) -> CameraVote:
    if summary is None or summary.observation_count <= 0:
        return CameraVote(camera_index, None, None, 0)
    return CameraVote(
        camera_index=camera_index,
        class_id=summary.best_class_id,
        score=summary.best_score,
        observation_count=summary.observation_count,
    )


def build_evaluation(
    summaries: dict[int, CameraObservationSummary],
    *,
    camera_count: int,
) -> CapEvaluation:
    camera_votes = {
        camera_index: build_camera_vote(summaries.get(camera_index), camera_index=camera_index)
        for camera_index in range(camera_count)
    }
    total = sum(summary.observation_count for summary in summaries.values())
    class_scores = {0: 0.0, 1: 0.0}
    for summary in summaries.values():
        for class_id, score in summary.class_peak_scores.items():
            class_scores[int(class_id)] = max(class_scores.get(int(class_id), 0.0), float(score))
    return CapEvaluation(total_observations=total, class_scores=class_scores, camera_votes=camera_votes)


def _anchor_time(tracked_cap: TrackedCap) -> float:
    if tracked_cap.actuation_time is not None:
        return float(tracked_cap.actuation_time)
    if tracked_cap.anchor_time is not None:
        return float(tracked_cap.anchor_time)
    return float(tracked_cap.created_at)


def _build_decision(
    *,
    result: str,
    final_class_name: str | None,
    final_score: float | None,
    decision_source: str,
    evaluation: CapEvaluation,
    anchor_time: float,
    decision_ready_time: float,
    config: RuntimeConfig,
    review_reason: str | None = None,
) -> TrackedCapDecision:
    trigger_delay_s = compute_requested_trigger_delay(config)
    return TrackedCapDecision(
        result=result,
        final_class_name=final_class_name,
        final_score=final_score,
        decision_source=decision_source,
        camera_votes=evaluation.camera_votes,
        anchor_time=anchor_time,
        decision_ready_time=decision_ready_time,
        trigger_delay_s=trigger_delay_s,
        requested_fire_time=anchor_time + trigger_delay_s,
        review_reason=review_reason,
    )


def _merge_complete(
    tracked_cap: TrackedCap,
    *,
    camera_count: int,
    decision_ready_time: float,
    merge_window_seconds: float,
) -> bool:
    return len(tracked_cap.camera_indices) >= camera_count or (
        float(decision_ready_time) - float(tracked_cap.last_seen_at)
    ) >= float(merge_window_seconds)


def decide_decision_ready(
    tracked_cap: TrackedCap,
    *,
    config: RuntimeConfig,
    decision_ready_time: float,
    camera_count: int = 2,
) -> TrackedCapDecision | None:
    if tracked_cap.trigger_decision is not None:
        return tracked_cap.trigger_decision
    if tracked_cap.actuation_time is None:
        return None
    evaluation = build_evaluation(tracked_cap.actuation_camera_summaries, camera_count=camera_count)
    if evaluation.total_observations <= 0 or evaluation.dirt_score < float(config.reject_threshold):
        return None
    if not _merge_complete(
        tracked_cap,
        camera_count=camera_count,
        decision_ready_time=decision_ready_time,
        merge_window_seconds=float(config.merge_window_ms) / 1000.0,
    ):
        return None
    decision = _build_decision(
        result="trigger",
        final_class_name=class_name(DEFECT_CLASS_ID),
        final_score=evaluation.dirt_score,
        decision_source="highest_defect_threshold",
        evaluation=evaluation,
        anchor_time=_anchor_time(tracked_cap),
        decision_ready_time=decision_ready_time,
        config=config,
        review_reason="trigger",
    )
    tracked_cap.trigger_decision = decision
    return decision


def decide_tracked_cap(
    tracked_cap: TrackedCap,
    *,
    config: RuntimeConfig,
    decision_time: float,
    camera_count: int = 2,
) -> TrackedCapDecision:
    if tracked_cap.trigger_decision is not None:
        return tracked_cap.trigger_decision
    trigger = decide_decision_ready(
        tracked_cap,
        config=config,
        decision_ready_time=decision_time,
        camera_count=camera_count,
    )
    if trigger is not None:
        return trigger

    if tracked_cap.actuation_time is None:
        evaluation = build_evaluation({}, camera_count=camera_count)
        return _build_decision(
            result="skip",
            final_class_name=None,
            final_score=None,
            decision_source="no_actuation_crossing",
            evaluation=evaluation,
            anchor_time=_anchor_time(tracked_cap),
            decision_ready_time=decision_time,
            config=config,
        )

    evaluation = build_evaluation(tracked_cap.actuation_camera_summaries, camera_count=camera_count)
    if evaluation.total_observations <= 0:
        return _build_decision(
            result="skip",
            final_class_name=None,
            final_score=None,
            decision_source="no_actuation_observations",
            evaluation=evaluation,
            anchor_time=_anchor_time(tracked_cap),
            decision_ready_time=decision_time,
            config=config,
        )
    final_class_id = DEFECT_CLASS_ID if evaluation.dirt_score >= float(config.reject_threshold) else 0
    return _build_decision(
        result="skip",
        final_class_name=class_name(final_class_id),
        final_score=evaluation.class_scores.get(final_class_id, 0.0),
        decision_source="below_reject_threshold",
        evaluation=evaluation,
        anchor_time=_anchor_time(tracked_cap),
        decision_ready_time=decision_time,
        config=config,
    )
