"""Shared dataclasses and type aliases for the v4 cap-inspection runtime.

v4 is intentionally tiny compared to v1-v3: there is no frame-pairing, no anchor
geometry and no belt math, so the only shared types are a captured frame, the
per-cap event record that is logged once per physical cap, a performance
snapshot, and the runtime callbacks bundle the UI plugs into.
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
    """

    event_id: int
    recorded_at: str
    result: str  # "reject" | "pass"
    class_name: str | None
    confidence: float | None
    cameras: list[int]
    flagged_cameras: list[int]
    requested_fire_time: str | None = None
    actual_fire_time: str | None = None


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


@dataclass(frozen=True)
class RuntimeCallbacks:
    """Hooks the UI (or a test) injects into the runtime."""

    preview_callback: Callable[[object], None] | None = None  # composite BGR frame
    history_callback: Callable[[CapEventRecord], None] | None = None
    performance_callback: Callable[[PerfSnapshot], None] | None = None
    log_fn: Callable[..., None] = print
