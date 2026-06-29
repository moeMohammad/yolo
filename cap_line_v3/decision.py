from __future__ import annotations

import math
from dataclasses import dataclass, field

from .config import DEFAULT_ANCHOR_LINE_RATIO, DEFECT_CLASS_ID, RuntimeConfig
from .geometry import (
    box_center,
    box_center_value,
    box_crossed_line_between,
    box_iou,
    box_size_along_axis,
    box_size_ratio,
    class_name,
    frame_line_coordinate,
    normalized_center_distance,
)
from .types import CameraObservationSummary, CameraVote, CapEvaluation, TrackObservation, TrackedCapDecision


def calculate_trigger_delay(config: RuntimeConfig) -> float:
    return float(config.nozzle_distance_mm) / float(config.belt_speed_mm_per_s) + float(config.trigger_offset_s)


def compute_requested_trigger_delay(config: RuntimeConfig) -> float:
    return calculate_trigger_delay(config) - float(config.latency_compensation_ms) / 1000.0


def _copy_summary(summary: CameraObservationSummary) -> CameraObservationSummary:
    return CameraObservationSummary(
        observation_count=int(summary.observation_count),
        class_peak_scores=dict(summary.class_peak_scores),
        best_class_id=summary.best_class_id,
        best_score=summary.best_score,
        first_seen_at=summary.first_seen_at,
        last_seen_at=summary.last_seen_at,
    )


def _copy_summaries(summaries: dict[int, CameraObservationSummary]) -> dict[int, CameraObservationSummary]:
    return {camera_index: _copy_summary(summary) for camera_index, summary in summaries.items()}


def _add_summary_observation(
    summaries: dict[int, CameraObservationSummary],
    observation: TrackObservation,
) -> None:
    summary = summaries.setdefault(int(observation.camera_index), CameraObservationSummary())
    summary.add(observation.class_id, observation.confidence, observation.timestamp)


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
    observations_by_camera: dict[int, list[TrackObservation]] = field(default_factory=dict)
    missing_frames_by_camera: dict[int, int] = field(default_factory=dict)
    latest_frame_size_by_camera: dict[int, tuple[int, int]] = field(default_factory=dict)
    anchor_time: float | None = None
    anchor_camera_index: int | None = None
    actuation_time: float | None = None
    actuation_camera_index: int | None = None
    predicted_actuation: bool = False
    trigger_decision: TrackedCapDecision | None = None

    def add_observation(
        self,
        observation: TrackObservation,
        *,
        anchor_axis: str = "x",
        anchor_line_ratio: float = DEFAULT_ANCHOR_LINE_RATIO,
        actuation_window_s: float = 0.0,
    ) -> None:
        camera_index = int(observation.camera_index)
        previous_box = self.latest_box_by_camera.get(camera_index)
        self.created_at = min(float(self.created_at), float(observation.timestamp))
        self.last_seen_at = max(float(self.last_seen_at), float(observation.timestamp))
        self.camera_indices.add(camera_index)
        self.latest_box_by_camera[camera_index] = observation.box
        self.latest_frame_size_by_camera[camera_index] = observation.frame_size
        self.missing_frames_by_camera[camera_index] = 0

        box_history = self.box_history_by_camera.setdefault(camera_index, [])
        box_history.append(observation.box)
        del box_history[:-4]

        observation_history = self.observations_by_camera.setdefault(camera_index, [])
        observation_history.append(observation)
        del observation_history[:-12]

        summary = self.camera_summaries.setdefault(camera_index, CameraObservationSummary())
        summary.add(observation.class_id, observation.confidence, observation.timestamp)
        if self.anchor_time is None or observation.timestamp < self.anchor_time:
            self.anchor_time = float(observation.timestamp)
            self.anchor_camera_index = camera_index

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
            if self.actuation_time is None or float(observation.timestamp) < float(self.actuation_time):
                summaries = self._summaries_near_time(
                    observation.timestamp,
                    window_s=max(0.0, float(actuation_window_s)),
                )
                if not summaries:
                    summaries = {camera_index: CameraObservationSummary()}
                    _add_summary_observation(summaries, observation)
                self._set_actuation(
                    observation.timestamp,
                    camera_index=camera_index,
                    predicted=False,
                    summaries=summaries,
                )
            else:
                _add_summary_observation(self.actuation_camera_summaries, observation)

    def mark_missed_camera(self, camera_index: int) -> None:
        camera_index = int(camera_index)
        if camera_index not in self.camera_indices:
            return
        self.missing_frames_by_camera[camera_index] = self.missing_frames_by_camera.get(camera_index, 0) + 1

    def maybe_predict_actuation(
        self,
        *,
        now: float,
        anchor_axis: str,
        anchor_line_ratio: float,
        prediction_horizon_s: float,
        actuation_window_s: float,
    ) -> bool:
        if self.actuation_time is not None:
            return False

        best_prediction: tuple[float, int] | None = None
        for camera_index, observations in self.observations_by_camera.items():
            if len(observations) < 2:
                continue
            previous = observations[-2]
            latest = observations[-1]
            dt = float(latest.timestamp) - float(previous.timestamp)
            if dt <= 0.0:
                continue
            previous_center = box_center_value(previous.box, anchor_axis)
            latest_center = box_center_value(latest.box, anchor_axis)
            velocity = (latest_center - previous_center) / dt
            if abs(velocity) < 1e-6:
                continue
            line_coordinate = frame_line_coordinate(
                latest.frame_size,
                axis=anchor_axis,
                ratio=anchor_line_ratio,
            )
            time_to_line = (line_coordinate - latest_center) / velocity
            if time_to_line < 0.0 or time_to_line > float(prediction_horizon_s):
                continue
            predicted_time = float(latest.timestamp) + float(time_to_line)
            if predicted_time < float(now) - float(prediction_horizon_s):
                continue
            candidate = (predicted_time, int(camera_index))
            if best_prediction is None or candidate[0] < best_prediction[0]:
                best_prediction = candidate

        if best_prediction is None:
            return False

        predicted_time, camera_index = best_prediction
        summaries = self._summaries_near_time(
            predicted_time,
            window_s=max(0.0, float(actuation_window_s)),
        )
        if not summaries:
            summaries = _copy_summaries(self.camera_summaries)
        self._set_actuation(
            predicted_time,
            camera_index=camera_index,
            predicted=True,
            summaries=summaries,
        )
        return True

    def _summaries_near_time(self, timestamp: float, *, window_s: float) -> dict[int, CameraObservationSummary]:
        lower_bound = float(timestamp) - float(window_s)
        summaries: dict[int, CameraObservationSummary] = {}
        for observations in self.observations_by_camera.values():
            for observation in observations:
                if float(observation.timestamp) > float(timestamp):
                    continue
                if float(observation.timestamp) < lower_bound:
                    continue
                _add_summary_observation(summaries, observation)
        return summaries

    def _set_actuation(
        self,
        timestamp: float,
        *,
        camera_index: int,
        predicted: bool,
        summaries: dict[int, CameraObservationSummary],
    ) -> None:
        if self.actuation_time is not None and float(timestamp) >= float(self.actuation_time):
            if not self.actuation_camera_summaries:
                self.actuation_camera_summaries = summaries
            return
        self.actuation_time = float(timestamp)
        self.actuation_camera_index = int(camera_index)
        self.predicted_actuation = bool(predicted)
        self.actuation_camera_summaries = summaries


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


def _boxes_follow_same_cap(
    candidate_box: tuple[float, float, float, float, float, int],
    latest_box: tuple[float, float, float, float, float, int],
    history: list[tuple[float, float, float, float, float, int]],
    *,
    axis: str,
    track_iou: float,
    missing_frames: int,
) -> bool:
    if box_iou(candidate_box, latest_box) >= float(track_iou):
        return True
    if box_size_ratio(candidate_box, latest_box) < 0.35:
        return False
    distance_limit = 3.0 + max(0, int(missing_frames))
    if normalized_center_distance(candidate_box, latest_box) <= distance_limit:
        return True
    if len(history) < 2:
        return False

    previous_box = history[-2]
    latest_center_x, latest_center_y = box_center(latest_box)
    previous_center_x, previous_center_y = box_center(previous_box)
    candidate_center_x, candidate_center_y = box_center(candidate_box)
    predicted_x = latest_center_x + (latest_center_x - previous_center_x)
    predicted_y = latest_center_y + (latest_center_y - previous_center_y)
    scale = max(
        1.0,
        box_size_along_axis(candidate_box, axis),
        box_size_along_axis(latest_box, axis),
        latest_box[3] - latest_box[1],
        candidate_box[3] - candidate_box[1],
    )
    predicted_distance = math.hypot(candidate_center_x - predicted_x, candidate_center_y - predicted_y) / scale
    return predicted_distance <= distance_limit


class TrackedCapManager:
    def __init__(
        self,
        *,
        camera_count: int,
        merge_window_seconds: float,
        finalize_quiet_seconds: float,
        anchor_axis: str,
        anchor_line_ratio: float,
        track_iou: float = 0.3,
        max_missing_frames: int = 1,
        actuation_window_seconds: float = 0.0,
    ):
        self.camera_count = int(camera_count)
        self.merge_window_seconds = float(merge_window_seconds)
        self.finalize_quiet_seconds = float(finalize_quiet_seconds)
        self.anchor_axis = anchor_axis
        self.anchor_line_ratio = float(anchor_line_ratio)
        self.track_iou = float(track_iou)
        self.max_missing_frames = int(max_missing_frames)
        self.actuation_window_seconds = max(0.0, float(actuation_window_seconds))
        self._open_caps: list[TrackedCap] = []
        self._next_event_id = 1

    def update(
        self,
        observations: list[TrackObservation],
        *,
        observed_camera_indices: set[int] | None = None,
    ) -> list[TrackedCap]:
        touched_caps: list[TrackedCap] = []
        touched_ids: set[int] = set()
        touched_camera_keys: set[tuple[int, int]] = set()
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
                actuation_window_s=self.actuation_window_seconds,
            )
            camera_index = int(observation.camera_index)
            matched_camera_keys.add((tracked_cap.event_id, camera_index))
            touched_camera_keys.add((tracked_cap.event_id, camera_index))
            self._mark_recent(tracked_cap)
            if tracked_cap.event_id not in touched_ids:
                touched_ids.add(tracked_cap.event_id)
                touched_caps.append(tracked_cap)

        for tracked_cap in self._open_caps:
            for camera_index in observed_camera_indices or set():
                if (tracked_cap.event_id, int(camera_index)) not in touched_camera_keys:
                    tracked_cap.mark_missed_camera(int(camera_index))
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
            missing_frames = tracked_cap.missing_frames_by_camera.get(camera_index, 0)
            if missing_frames > self.max_missing_frames:
                continue
            history = tracked_cap.box_history_by_camera.get(camera_index, [])
            if _moves_backward_against_history(observation.box, history, axis=self.anchor_axis):
                continue
            if _boxes_follow_same_cap(
                observation.box,
                latest_box,
                history,
                axis=self.anchor_axis,
                track_iou=self.track_iou,
                missing_frames=missing_frames,
            ):
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


def _ensure_predicted_actuation(
    tracked_cap: TrackedCap,
    *,
    config: RuntimeConfig,
    decision_ready_time: float,
) -> None:
    tracked_cap.maybe_predict_actuation(
        now=decision_ready_time,
        anchor_axis=config.anchor_axis,
        anchor_line_ratio=config.anchor_line_ratio,
        prediction_horizon_s=float(config.actuation_prediction_horizon_ms) / 1000.0,
        actuation_window_s=float(config.actuation_window_ms) / 1000.0,
    )


def _deadline_fallback_due(
    decision: TrackedCapDecision,
    *,
    decision_ready_time: float,
    config: RuntimeConfig,
) -> bool:
    remaining_s = float(decision.requested_fire_time) - float(decision_ready_time)
    return remaining_s <= float(config.decision_deadline_guard_ms) / 1000.0


def _has_defect_camera_vote(evaluation: CapEvaluation) -> bool:
    return any(vote.class_id == DEFECT_CLASS_ID for vote in evaluation.camera_votes.values())


def _is_defective_evaluation(evaluation: CapEvaluation, *, config: RuntimeConfig) -> bool:
    return evaluation.dirt_score >= float(config.reject_threshold) or _has_defect_camera_vote(evaluation)


def decide_decision_ready(
    tracked_cap: TrackedCap,
    *,
    config: RuntimeConfig,
    decision_ready_time: float,
    camera_count: int = 2,
) -> TrackedCapDecision | None:
    if tracked_cap.trigger_decision is not None:
        return tracked_cap.trigger_decision

    _ensure_predicted_actuation(tracked_cap, config=config, decision_ready_time=decision_ready_time)
    if tracked_cap.actuation_time is None:
        return None

    evaluation = build_evaluation(tracked_cap.actuation_camera_summaries, camera_count=camera_count)
    if evaluation.total_observations <= 0 or not _is_defective_evaluation(evaluation, config=config):
        return None

    defect_vote_override = evaluation.dirt_score < float(config.reject_threshold)
    if defect_vote_override:
        base_decision_source = (
            "predicted_actuation_camera_defect_vote"
            if tracked_cap.predicted_actuation
            else "camera_defect_vote"
        )
    else:
        base_decision_source = (
            "predicted_actuation_threshold"
            if tracked_cap.predicted_actuation
            else "highest_defect_threshold"
        )
    base_review_reason = "predicted_actuation" if tracked_cap.predicted_actuation else "trigger"
    candidate = _build_decision(
        result="trigger",
        final_class_name=class_name(DEFECT_CLASS_ID),
        final_score=evaluation.dirt_score,
        decision_source=base_decision_source,
        evaluation=evaluation,
        anchor_time=_anchor_time(tracked_cap),
        decision_ready_time=decision_ready_time,
        config=config,
        review_reason=base_review_reason,
    )

    merge_complete = _merge_complete(
        tracked_cap,
        camera_count=camera_count,
        decision_ready_time=decision_ready_time,
        merge_window_seconds=float(config.merge_window_ms) / 1000.0,
    )
    if not merge_complete:
        if not _deadline_fallback_due(candidate, decision_ready_time=decision_ready_time, config=config):
            return None
        candidate = _build_decision(
            result="trigger",
            final_class_name=class_name(DEFECT_CLASS_ID),
            final_score=evaluation.dirt_score,
            decision_source="single_camera_deadline_fallback",
            evaluation=evaluation,
            anchor_time=_anchor_time(tracked_cap),
            decision_ready_time=decision_ready_time,
            config=config,
            review_reason="single_camera_deadline_trigger",
        )

    tracked_cap.trigger_decision = candidate
    return candidate


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

    full_evaluation = build_evaluation(tracked_cap.camera_summaries, camera_count=camera_count)
    if tracked_cap.actuation_time is None:
        if _is_defective_evaluation(full_evaluation, config=config):
            final_class_id = DEFECT_CLASS_ID
        elif full_evaluation.total_observations > 0:
            final_class_id = 0
        else:
            final_class_id = None
        return _build_decision(
            result="skip",
            final_class_name=None if final_class_id is None else class_name(final_class_id),
            final_score=None if final_class_id is None else full_evaluation.class_scores.get(final_class_id, 0.0),
            decision_source="no_actuation_crossing",
            evaluation=full_evaluation,
            anchor_time=_anchor_time(tracked_cap),
            decision_ready_time=decision_time,
            config=config,
            review_reason=(
                "missed_actuation"
                if _is_defective_evaluation(full_evaluation, config=config)
                else None
            ),
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
    final_class_id = DEFECT_CLASS_ID if _is_defective_evaluation(evaluation, config=config) else 0
    review_reason = (
        "dirty_before_clean_actuation"
        if _is_defective_evaluation(full_evaluation, config=config)
        and not _is_defective_evaluation(evaluation, config=config)
        else None
    )
    return _build_decision(
        result="skip",
        final_class_name=class_name(final_class_id),
        final_score=evaluation.class_scores.get(final_class_id, 0.0),
        decision_source="below_reject_threshold",
        evaluation=evaluation,
        anchor_time=_anchor_time(tracked_cap),
        decision_ready_time=decision_time,
        config=config,
        review_reason=review_reason,
    )
