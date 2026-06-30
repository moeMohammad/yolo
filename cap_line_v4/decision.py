"""Cross-camera de-duplication and the once-per-cap fire guarantee.

Both cameras see the same physical cap from different angles, so both of their
tracks finish at nearly the same time. ``CapEventManager`` collapses those
near-simultaneous finishes into a single *cap event* using a short global
cooldown window, and schedules at most one air pulse per physical cap.

Locking: ``handle_finished_track`` is called from each camera thread,
``flush_expired``/``finalize_all`` from the coordinator thread, and the fire
completion callback from the scheduler thread. All event mutation happens under
``_lock``; user callbacks are always invoked *outside* the lock.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from itertools import count
from typing import Callable

from .config import DEFECT_CLASS_ID, RuntimeConfig, class_name
from .types import CapEventRecord


@dataclass
class CapEvent:
    """Accumulates the (possibly two-camera) state of one physical cap."""

    event_id: int
    opened_at: float  # monotonic time the first finished track opened this event
    last_seen: float  # latest "cap left view" time across merged tracks
    seen_cameras: set[int] = field(default_factory=set)
    flagged_cameras: set[int] = field(default_factory=set)
    is_defect: bool = False
    best_defect_conf: float = 0.0
    best_undefected_conf: float = 0.0
    fired: bool = False
    finalized: bool = False
    requested_fire_time: float | None = None
    actual_fire_time: float | None = None

    def absorb(self, track) -> None:
        self.seen_cameras.add(int(track.camera_index))
        self.last_seen = max(self.last_seen, float(track.last_seen))
        self.best_undefected_conf = max(self.best_undefected_conf, float(track.best_undefected_conf))
        if track.is_defect:
            self.is_defect = True  # defect-wins across cameras too
            self.flagged_cameras.add(int(track.camera_index))
            self.best_defect_conf = max(self.best_defect_conf, float(track.best_defect_conf))


class CapEventManager:
    def __init__(
        self,
        config: RuntimeConfig,
        *,
        scheduler,
        time_fn: Callable[[], float],
        clock=None,
        history_callback: Callable[[CapEventRecord], None] | None = None,
        log_fn: Callable[..., None] = print,
    ):
        self.global_cooldown_s = float(config.global_cooldown_ms) / 1000.0
        self.fire_delay_s = float(config.fire_delay_s)
        self.scheduler = scheduler
        self.time_fn = time_fn
        self.clock = clock
        self.history_callback = history_callback
        self.log_fn = log_fn
        self._lock = threading.Lock()
        self._open_event: CapEvent | None = None
        self._counter = count(1)
        self.caps_seen = 0
        self.rejects = 0

    # -- public API ---------------------------------------------------------

    def handle_finished_track(self, track) -> None:
        """A camera reports one of its tracks has left view (the cap is gone)."""

        emitted: list[CapEventRecord] = []
        with self._lock:
            now = float(self.time_fn())
            event = self._open_event
            if event is not None and (now - event.opened_at) <= self.global_cooldown_s:
                # Within the window -> same physical cap (second camera, or a
                # slightly-late frame from the same camera).
                event.absorb(track)
                if track.is_defect and not event.fired:
                    # First camera saw it clean, this one saw it dirty: fire now.
                    self._schedule_fire_locked(event, track.last_seen)
            else:
                # A genuinely new cap: close out the previous one first.
                if event is not None:
                    emitted.append(self._finalize_locked(event))
                event = CapEvent(event_id=next(self._counter), opened_at=now, last_seen=float(track.last_seen))
                event.absorb(track)
                self._open_event = event
                if track.is_defect:
                    self._schedule_fire_locked(event, track.last_seen)
        for record in emitted:
            self._emit(record)

    def flush_expired(self, now: float) -> None:
        """Finalize (log) the open event once its merge window has elapsed."""

        emitted: list[CapEventRecord] = []
        with self._lock:
            event = self._open_event
            if event is not None and (float(now) - event.opened_at) > self.global_cooldown_s:
                emitted.append(self._finalize_locked(event))
                self._open_event = None
        for record in emitted:
            self._emit(record)

    def finalize_all(self) -> None:
        """Finalize any lingering open event (called at shutdown)."""

        emitted: list[CapEventRecord] = []
        with self._lock:
            if self._open_event is not None:
                emitted.append(self._finalize_locked(self._open_event))
                self._open_event = None
        for record in emitted:
            self._emit(record)

    # -- internals (call with _lock held) -----------------------------------

    def _schedule_fire_locked(self, event: CapEvent, reference_last_seen: float) -> None:
        requested_fire_time = float(reference_last_seen) + self.fire_delay_s
        event.fired = True
        event.requested_fire_time = requested_fire_time
        self.scheduler.enqueue(
            event.event_id,
            requested_fire_time,
            completion_callback=lambda execution, ev=event: self._on_fire_complete(ev, execution),
        )

    def _finalize_locked(self, event: CapEvent) -> CapEventRecord:
        event.finalized = True
        self.caps_seen += 1
        if event.is_defect:
            self.rejects += 1
        return self._build_record_locked(event)

    def _build_record_locked(self, event: CapEvent) -> CapEventRecord:
        if event.is_defect:
            winning_class, confidence = DEFECT_CLASS_ID, event.best_defect_conf
        else:
            winning_class, confidence = 0, event.best_undefected_conf
        return CapEventRecord(
            event_id=event.event_id,
            recorded_at=self._format_time(event.last_seen) or "",
            result="reject" if event.is_defect else "pass",
            class_name=class_name(winning_class),
            confidence=float(confidence),
            cameras=sorted(event.seen_cameras),
            flagged_cameras=sorted(event.flagged_cameras),
            requested_fire_time=self._format_time(event.requested_fire_time),
            actual_fire_time=self._format_time(event.actual_fire_time),
        )

    def _on_fire_complete(self, event: CapEvent, execution) -> None:
        """Scheduler-thread callback: record the actual pulse time.

        If the event was already finalized/logged (fire happens after the merge
        window when ``fire_delay_s`` is large), re-emit the row with the actual
        fire time filled in; otherwise ``_finalize_locked`` will pick it up.
        """

        emitted: list[CapEventRecord] = []
        with self._lock:
            event.actual_fire_time = float(execution.trigger_on_time)
            if event.finalized:
                emitted.append(self._build_record_locked(event))
        for record in emitted:
            self._emit(record)

    # -- helpers ------------------------------------------------------------

    def _format_time(self, value: float | None) -> str | None:
        if value is None:
            return None
        if self.clock is not None:
            return self.clock.format(value)
        return f"{float(value):.6f}"

    def _emit(self, record: CapEventRecord) -> None:
        result = record.result.upper()
        cameras = ",".join(str(index) for index in record.flagged_cameras) or "-"
        self.log_fn(
            f"[CAP] event={record.event_id} {result} "
            f"class={record.class_name} conf={record.confidence:.3f} flagged_by={cameras}"
        )
        if self.history_callback is not None:
            self.history_callback(record)
