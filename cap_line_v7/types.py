"""Shared dataclasses and type aliases for the v7 cap-inspection runtime.

Same tiny surface as v4: a captured frame, the per-cap event record that is
logged once per physical cap (and updated when asynchronous actuation settles),
a performance snapshot, and the runtime callbacks bundle the UI plugs into.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


# (x1, y1, x2, y2, confidence, class_id) in original-frame pixel coordinates.
Box = tuple[float, float, float, float, float, int]


@dataclass(frozen=True)
class CapturedFrame:
    """A single frame read from one camera."""

    camera_index: int
    frame: object  # numpy BGR image (kept as `object` so tests can use stand-ins)
    timestamp: float
    sequence: int


@dataclass(frozen=True)
class CapEventRecord:
    """One row per *physical* cap, emitted after the cross-camera merge.

    `cameras` is every camera that saw the cap; `flagged_cameras` is the subset
    that classified it as a defect. `requested_fire_time`/`actual_fire_time` are
    human-readable strings (or None for pass caps / not-yet-fired rejects).
    `fire_suppressed` marks a reject whose pulse was deduped by the post-fire
    refractory (the cap was already blown off by the previous fire).
    """

    event_id: int
    recorded_at: str
    result: str  # "reject" | "pass" | "unknown" when fail-closed is disabled
    class_name: str | None
    confidence: float | None
    cameras: list[int]
    flagged_cameras: list[int]
    requested_fire_time: str | None = None
    actual_fire_time: str | None = None
    fire_suppressed: bool = False
    inspection_status: str = "valid"  # "valid" | "unknown"
    fire_status: str = "not_requested"


@dataclass(frozen=True)
class PerfSnapshot:
    """Aggregated, periodically-emitted performance/counters snapshot."""

    elapsed_s: float
    capture_fps_by_camera: tuple[float | None, ...]
    processed_fps_by_camera: tuple[float | None, ...]
    inference_ms_by_camera: tuple[float | None, ...]
    gpio_backend: str
    caps_seen: int
    rejects: int
    filtered_tracks: int = 0
    unknown_inspections: int = 0
    stale_results_by_camera: tuple[int, ...] = ()
    detected_boxes_by_camera: tuple[int, ...] = ()
    max_detector_confidence_by_camera: tuple[float, ...] = ()
    detector_providers: tuple[str, ...] = ()
    classifier_providers: tuple[str, ...] = ()
    throughput_status: str = "warming_up"
    throughput_detail: str = ""


@dataclass(frozen=True)
class RuntimeCallbacks:
    """Hooks the UI (or a test) injects into the runtime.

    ``test_fire_poll`` is polled by the coordinator loop; returning True fires
    one manual pulse through the runtime's own ``RejectScheduler``/pin. This is
    how the UI's Test Fire button works while detection is running — opening a
    second GPIO handle for the same pin would tear the runtime's pin down when
    it closes (Jetson.GPIO cleanup is process-wide per channel).
    """

    preview_callback: Callable[[object], None] | None = None  # composite BGR frame
    history_callback: Callable[[CapEventRecord], None] | None = None
    performance_callback: Callable[[PerfSnapshot], None] | None = None
    log_fn: Callable[..., None] = print
    test_fire_poll: Callable[[], bool] | None = None
