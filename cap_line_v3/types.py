from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


Box = tuple[float, float, float, float, float, int]


@dataclass(frozen=True)
class CapturedFrame:
    camera_index: int
    frame: object
    timestamp: float
    sequence: int
    read_duration_ms: float | None = None


@dataclass(frozen=True)
class FramePair:
    frames: tuple[CapturedFrame, ...]
    pair_timestamp: float
    skew_ms: float

    @property
    def sequences(self) -> tuple[int, ...]:
        return tuple(frame.sequence for frame in self.frames)

    @property
    def images(self) -> tuple[object, ...]:
        return tuple(frame.frame for frame in self.frames)

    @property
    def timestamps(self) -> tuple[float, ...]:
        return tuple(frame.timestamp for frame in self.frames)


@dataclass(frozen=True)
class CaptureBatch:
    frames_by_camera: tuple[CapturedFrame | None, ...]
    batch_timestamp: float
    skew_ms: float | None
    reason: str
    missing_camera_indices: tuple[int, ...] = ()

    @property
    def frames(self) -> tuple[CapturedFrame, ...]:
        return tuple(frame for frame in self.frames_by_camera if frame is not None)

    @property
    def sequences(self) -> tuple[int | None, ...]:
        return tuple(None if frame is None else int(frame.sequence) for frame in self.frames_by_camera)

    @property
    def timestamps(self) -> tuple[float | None, ...]:
        return tuple(None if frame is None else float(frame.timestamp) for frame in self.frames_by_camera)

    @property
    def is_single_camera(self) -> bool:
        return len(self.frames) == 1


@dataclass(frozen=True)
class DetectionPacket:
    frame_pair: FramePair
    boxes_by_camera: tuple[tuple[Box, ...], ...]
    inference_ms_by_camera: tuple[float, ...]
    capture_batch: CaptureBatch | None = None


@dataclass
class PairDropStats:
    stale_sequence: int = 0
    skew: int = 0
    missing_camera: int = 0
    overwritten: int = 0

    def copy(self) -> "PairDropStats":
        return PairDropStats(
            stale_sequence=int(self.stale_sequence),
            skew=int(self.skew),
            missing_camera=int(self.missing_camera),
            overwritten=int(self.overwritten),
        )


@dataclass(frozen=True)
class TrackObservation:
    camera_index: int
    box: Box
    timestamp: float
    frame_size: tuple[int, int]
    at_actuation_line: bool = False
    sequence: int | None = None

    @property
    def class_id(self) -> int:
        return int(self.box[5])

    @property
    def confidence(self) -> float:
        return float(self.box[4])


@dataclass(frozen=True)
class CameraVote:
    camera_index: int
    class_id: int | None
    score: float | None
    observation_count: int


@dataclass(frozen=True)
class CapEvaluation:
    total_observations: int
    class_scores: dict[int, float]
    camera_votes: dict[int, CameraVote]

    @property
    def dirt_score(self) -> float:
        return float(self.class_scores.get(1, 0.0))


@dataclass(frozen=True)
class TrackedCapDecision:
    result: str
    final_class_name: str | None
    final_score: float | None
    decision_source: str
    camera_votes: dict[int, CameraVote]
    anchor_time: float
    decision_ready_time: float
    trigger_delay_s: float
    requested_fire_time: float
    review_reason: str | None = None


@dataclass
class DetectionHistoryRecord:
    recorded_at: str
    runtime_event_id: int
    result: str
    final_class_name: str | None
    final_score: float | None
    decision_source: str
    camera_labels: list[str]
    camera_votes: dict[int, dict[str, object]]
    anchor_time: str | None
    trigger_delay_s: float | None


@dataclass
class TimingLogRecord:
    recorded_at: str
    runtime_event_id: int
    result: str
    final_class_name: str | None
    anchor_time: str | None
    decision_time: str | None
    queued_at: str | None = None
    requested_fire_time: str | None = None
    trigger_on_time: str | None = None
    trigger_off_time: str | None = None
    anchor_to_actual_on_ms: float | None = None
    scheduler_late_ms: float | None = None
    pulse_duration_ms: float | None = None


@dataclass(frozen=True)
class RuntimePerformanceSnapshot:
    frame_count: int
    target_fps: int
    elapsed_s: float
    capture_fps_by_camera: tuple[float | None, ...]
    processed_fps: float
    preview_fps: float
    latest_pair_skew_ms: float | None
    dropped_pairs: int
    overlay_age_ms: float | None
    latest_inference_ms_by_camera: tuple[float | None, ...] = ()
    latest_total_inference_ms: float | None = None
    single_camera_batches: int = 0
    pair_drop_stats: PairDropStats = field(default_factory=PairDropStats)
    actual_camera_fps_by_camera: tuple[float | None, ...] = ()


@dataclass(frozen=True)
class RuntimeCallbacks:
    preview_callback: Callable[[object], None] | None = None
    history_callback: Callable[[DetectionHistoryRecord], None] | None = None
    timing_log_callback: Callable[[TimingLogRecord], None] | None = None
    performance_callback: Callable[[RuntimePerformanceSnapshot], None] | None = None
    log_fn: Callable[..., None] = print


@dataclass
class CameraObservationSummary:
    observation_count: int = 0
    class_peak_scores: dict[int, float] = field(default_factory=dict)
    best_class_id: int | None = None
    best_score: float | None = None
    first_seen_at: float | None = None
    last_seen_at: float | None = None

    def add(self, class_id: int, confidence: float, timestamp: float) -> None:
        score = float(confidence)
        self.observation_count += 1
        self.class_peak_scores[class_id] = max(score, self.class_peak_scores.get(class_id, 0.0))
        if self.first_seen_at is None:
            self.first_seen_at = timestamp
        self.last_seen_at = timestamp
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            self.best_class_id = int(class_id)
