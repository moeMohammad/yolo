from __future__ import annotations

from dataclasses import dataclass, field

from .config import DEFECT_CLASS_ID, RuntimeConfig
from .geometry import class_name
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
    anchor_time: float | None = None
    anchor_camera_index: int | None = None
    actuation_time: float | None = None
    actuation_camera_index: int | None = None
    trigger_decision: TrackedCapDecision | None = None

    def add_observation(self, observation: TrackObservation) -> None:
        self.last_seen_at = max(float(self.last_seen_at), float(observation.timestamp))
        self.camera_indices.add(int(observation.camera_index))
        summary = self.camera_summaries.setdefault(
            int(observation.camera_index),
            CameraObservationSummary(),
        )
        summary.add(observation.class_id, observation.confidence, observation.timestamp)
        if self.anchor_time is None or observation.timestamp < self.anchor_time:
            self.anchor_time = float(observation.timestamp)
            self.anchor_camera_index = int(observation.camera_index)

        if observation.at_actuation_line:
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
    evaluation = build_evaluation(tracked_cap.camera_summaries, camera_count=camera_count)
    if evaluation.total_observations <= 0:
        return _build_decision(
            result="skip",
            final_class_name=None,
            final_score=None,
            decision_source="no_observations",
            evaluation=evaluation,
            anchor_time=_anchor_time(tracked_cap),
            decision_ready_time=decision_time,
            config=config,
        )
    final_class_id = DEFECT_CLASS_ID if evaluation.dirt_score >= float(config.reject_threshold) else 0
    decision_source = "no_actuation_crossing" if tracked_cap.actuation_time is None else "below_reject_threshold"
    return _build_decision(
        result="skip",
        final_class_name=class_name(final_class_id),
        final_score=evaluation.class_scores.get(final_class_id, 0.0),
        decision_source=decision_source,
        evaluation=evaluation,
        anchor_time=_anchor_time(tracked_cap),
        decision_ready_time=decision_time,
        config=config,
        review_reason="missed_actuation" if final_class_id == DEFECT_CLASS_ID else None,
    )
