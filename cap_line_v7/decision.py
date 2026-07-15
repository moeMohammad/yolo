"""Cross-camera de-duplication and the once-per-cap fire guarantee (v7).

Both cameras see the same physical cap from different angles, so both of their
tracks finish at nearly the same time — but on a Pi 5 the *reporting* of those
finishes can be far apart (slow, jittery CPU inference). v4 merged on the
reporting clock and double-fired when camera B's finish landed outside the
50 ms window. v7 keys everything off the physical exit timestamp
``track.last_seen`` instead, with layered safety (see ``cap_line_v7_PROMPT.md``):

1. Physical qualification: short/static/jittering tracks never become events.
2. Merge window: a finished track joins the open event iff
   ``abs(track.last_seen - event.last_seen) <= merge_window``.
3. Presence-cycle idempotency: fragments from one camera presence can consume
   at most one fire decision, even beyond the time merge window.
4. Post-fire refractory: a new fire is suppressed (never delayed) if its
   ``last_seen`` reference is within ``min_fire_interval`` of the previous
   fire's reference — the hard once-per-cap guarantee.
5. The open event is finalized only once ``now`` is past
   ``event.last_seen + merge_window + track_timeout`` so every finish that
   could still merge has had time to be reported.

Locking: ``handle_finished_track`` is called from each camera thread,
``flush_expired``/``finalize_all`` from the coordinator thread, and the fire
completion callback from the scheduler thread. All event mutation happens under
``_lock``; user callbacks are always invoked *outside* the lock.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from itertools import count
from typing import Callable

from .config import DEFECT_CLASS_ID, RuntimeConfig, class_name
from .types import CapEventRecord


# Monotonic timestamps are binary floats. Nominal decimal millisecond
# boundaries can subtract a fraction of a nanosecond above their configured
# value, which must not bypass a once-per-cap safety boundary.
TIME_BOUNDARY_EPSILON_S = 1e-9


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
    fire_suppressed: bool = False
    finalized: bool = False
    requested_fire_time: float | None = None
    actual_fire_time: float | None = None
    fire_decided: bool = False
    cycle_keys: set[tuple[int, int]] = field(default_factory=set)

    def absorb(self, track) -> None:
        self.absorb_presence(track)
        self.best_undefected_conf = max(self.best_undefected_conf, float(track.best_undefected_conf))
        if track.is_defect:
            self.is_defect = True  # defect-wins across cameras too
            self.flagged_cameras.add(int(track.camera_index))
            self.best_defect_conf = max(self.best_defect_conf, float(track.best_defect_conf))

    def absorb_presence(self, track) -> None:
        """Remember physical identity without changing a finalized verdict."""

        self.seen_cameras.add(int(track.camera_index))
        cycle_id = getattr(track, "presence_cycle_id", None)
        if cycle_id is not None:
            self.cycle_keys.add((int(track.camera_index), int(cycle_id)))
        self.last_seen = max(self.last_seen, float(track.last_seen))


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
        self.merge_window_s = float(config.merge_window_ms) / 1000.0
        self.min_fire_interval_s = float(config.min_fire_interval_ms) / 1000.0
        self.track_timeout_s = float(config.track_timeout_ms) / 1000.0
        self.min_track_frames = int(config.min_track_frames)
        self.min_track_travel_ratio = float(config.min_track_travel_ratio)
        self.min_track_directionality = float(config.min_track_directionality)
        self.max_track_gap_s = float(config.max_track_gap_ms) / 1000.0
        self.fire_delay_s = float(config.fire_delay_s)
        self.scheduler = scheduler
        self.time_fn = time_fn
        self.clock = clock
        self.history_callback = history_callback
        self.log_fn = log_fn
        self._lock = threading.Lock()
        self._open_event: CapEvent | None = None
        # Recently finalized events are terminal decision tombstones.  A slow
        # second-camera worker can report an old track after that cap has
        # already been finalized (and even while a newer cap is open).  Match
        # that fragment by its physical last_seen timestamp, comparing its
        # distance with the open event, so it can neither re-arm a PASS nor
        # contaminate the next cap. The bounded history is ample for
        # camera/inference lag and cannot grow for the process lifetime.
        self._recent_finalized_events: deque[CapEvent] = deque(maxlen=256)
        self._last_fire_ref: float | None = None  # last_seen reference of the last scheduled fire
        self._consumed_cycle_keys: set[tuple[int, int]] = set()
        self._consumed_cycle_order: deque[tuple[int, int]] = deque()
        self._counter = count(1)
        self.caps_seen = 0
        self.rejects = 0
        self.suppressed_fires = 0
        self.filtered_tracks = 0

    # -- public API ---------------------------------------------------------

    def handle_finished_track(self, track) -> None:
        """A camera reports one of its tracks has left view (the cap is gone).

        Merging is keyed to the *physical* exit time ``track.last_seen``, never
        to when this call happens to run, so a slow camera thread reporting the
        same cap late still merges instead of double-firing.
        """

        if not track.qualifies_as_cap(
            min_frames=self.min_track_frames,
            min_travel_ratio=self.min_track_travel_ratio,
            min_directionality=self.min_track_directionality,
            max_observation_gap=self.max_track_gap_s,
            require_line_crossing=True,
        ):
            with self._lock:
                self.filtered_tracks += 1
            self.log_fn(
                f"[FILTER] ignored unconfirmed track camera={track.camera_index} "
                f"frames={track.frame_count} travel={track.travel_ratio:.2f} "
                f"directionality={track.motion_directionality:.2f} "
                f"line_crossed={track.crossed_presence_line} "
                f"line_sides={track.line_negative_frames}/{track.line_positive_frames} "
                f"defect_conf={track.best_defect_conf:.3f}"
            )
            return

        emitted: list[CapEventRecord] = []
        log_messages: list[str] = []
        with self._lock:
            now = float(self.time_fn())
            event = self._open_event
            open_distance = None if event is None else self._track_merge_distance_locked(event, track)
            finalized_match = self._matching_finalized_event_locked(track)
            finalized_event = None if finalized_match is None else finalized_match[0]
            finalized_distance = None if finalized_match is None else finalized_match[1]
            # Select the closest physical exit across both current and
            # terminal events. On an exact tie, prefer the terminal decision:
            # suppressing an ambiguous fragment is safer than re-arming air.
            if finalized_event is not None and (
                open_distance is None or finalized_distance <= open_distance
            ):
                cycle_id = getattr(track, "presence_cycle_id", None)
                if cycle_id is not None:
                    self._remember_consumed_cycle_locked((int(track.camera_index), int(cycle_id)))
                finalized_event.absorb_presence(track)
                if track.is_defect:
                    self.suppressed_fires += 1
                log_messages.append(
                    f"[DEDUP] ignored late camera={track.camera_index} fragment for "
                    f"finalized event={finalized_event.event_id}"
                )
            else:
                if event is not None and open_distance is not None:
                    # Exit times overlap -> same physical cap (second camera,
                    # or a slightly-late frame from the same camera).
                    event.absorb(track)
                    if event.fire_decided:
                        self._consume_event_cycles_locked(event)
                    elif track.is_defect:
                        # First camera saw it clean, this one saw it dirty.
                        message = self._maybe_schedule_fire_locked(event, float(track.last_seen))
                        if message is not None:
                            log_messages.append(message)
                else:
                    # A genuinely new cap: close out the previous one first.
                    if event is not None:
                        emitted.append(self._finalize_locked(event))
                    event = CapEvent(event_id=next(self._counter), opened_at=now, last_seen=float(track.last_seen))
                    event.absorb(track)
                    self._open_event = event
                    if track.is_defect:
                        message = self._maybe_schedule_fire_locked(event, float(track.last_seen))
                        if message is not None:
                            log_messages.append(message)
        for message in log_messages:
            self.log_fn(message)
        for record in emitted:
            self._emit(record)

    def flush_expired(self, now: float) -> None:
        """Finalize (log) the open event once no further merge can arrive.

        A mergeable finish has ``last_seen <= event.last_seen + merge_window``
        and is reported at most ``track_timeout`` after its ``last_seen``, so
        past that horizon the event is complete.
        """

        emitted: list[CapEventRecord] = []
        with self._lock:
            event = self._open_event
            if (
                event is not None
                and float(now)
                > event.last_seen + self.merge_window_s + self.track_timeout_s + TIME_BOUNDARY_EPSILON_S
            ):
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

    def _can_merge_track_locked(self, event: CapEvent, track) -> bool:
        return self._track_merge_distance_locked(event, track) is not None

    def _track_merge_distance_locked(self, event: CapEvent, track) -> float | None:
        distance = abs(float(track.last_seen) - event.last_seen)
        if distance > self.merge_window_s + TIME_BOUNDARY_EPSILON_S:
            return None
        camera_index = int(track.camera_index)
        if camera_index not in event.seen_cameras:
            return distance  # matching exit time from the other camera
        cycle_id = getattr(track, "presence_cycle_id", None)
        if cycle_id is None:
            return distance  # compatibility for externally supplied/legacy tracks
        # Same-camera tracks in different spatial crossing cycles are distinct
        # caps even when they leave at nearly the same time.
        return distance if (camera_index, int(cycle_id)) in event.cycle_keys else None

    def _matching_finalized_event_locked(self, track) -> tuple[CapEvent, float] | None:
        """Return a terminal event matching this late physical track.

        A different camera with the same exit window is the common late-report
        case; same-camera fragments must share their presence cycle when one is
        available. The closest eligible timestamp wins.
        """

        candidates = [
            (event, distance)
            for event in self._recent_finalized_events
            for distance in (self._track_merge_distance_locked(event, track),)
            if distance is not None
        ]
        if not candidates:
            return None
        # Closest timestamp wins; newest event breaks an exact tombstone tie.
        return min(candidates, key=lambda item: (item[1], -item[0].event_id))

    def _maybe_schedule_fire_locked(self, event: CapEvent, reference_last_seen: float) -> str | None:
        """Schedule the fire unless the post-fire refractory suppresses it.

        The refractory is the hard once-per-cap guarantee: whatever leaks past
        the merge window (extreme reporting lag, a fragmented track) cannot
        produce a second pulse for the same cap.
        """

        if event.fire_decided:
            self._consume_event_cycles_locked(event)
            return None
        reused_cycles = event.cycle_keys.intersection(self._consumed_cycle_keys)
        if reused_cycles:
            event.fire_decided = True
            event.fire_suppressed = True
            self.suppressed_fires += 1
            self._consume_event_cycles_locked(event)
            return (
                f"[DEDUP] fire suppressed for event={event.event_id}: "
                f"presence cycle already consumed ({sorted(reused_cycles)})"
            )
        if (
            self._last_fire_ref is not None
            and reference_last_seen - self._last_fire_ref
            <= self.min_fire_interval_s + TIME_BOUNDARY_EPSILON_S
        ):
            event.fire_decided = True
            event.fire_suppressed = True
            self.suppressed_fires += 1
            self._consume_event_cycles_locked(event)
            return (
                f"[DEDUP] fire suppressed for event={event.event_id}: reference "
                f"{(reference_last_seen - self._last_fire_ref) * 1000.0:.0f} ms after the previous fire "
                f"(min_fire_interval={self.min_fire_interval_s * 1000.0:.0f} ms)"
            )
        requested_fire_time = reference_last_seen + self.fire_delay_s
        event.fire_decided = True
        event.fired = True
        event.requested_fire_time = requested_fire_time
        self._last_fire_ref = reference_last_seen
        self._consume_event_cycles_locked(event)
        try:
            self.scheduler.enqueue(
                event.event_id,
                requested_fire_time,
                completion_callback=lambda execution, ev=event: self._on_fire_complete(ev, execution),
            )
        except RuntimeError as exc:
            # Shutdown closes the scheduler before worker joins. A racing
            # decision is terminally suppressed, never allowed to resurrect.
            event.fired = False
            event.fire_suppressed = True
            event.requested_fire_time = None
            self.suppressed_fires += 1
            return f"[REJECT][CANCELLED] event={event.event_id}: {exc}"
        return None

    def _consume_event_cycles_locked(self, event: CapEvent) -> None:
        for key in event.cycle_keys:
            self._remember_consumed_cycle_locked(key)

    def _remember_consumed_cycle_locked(self, key: tuple[int, int]) -> None:
        if key in self._consumed_cycle_keys:
            return
        self._consumed_cycle_keys.add(key)
        self._consumed_cycle_order.append(key)
        while len(self._consumed_cycle_order) > 4096:
            expired = self._consumed_cycle_order.popleft()
            self._consumed_cycle_keys.discard(expired)

    def _finalize_locked(self, event: CapEvent) -> CapEventRecord:
        event.finalized = True
        # A finalized PASS is a decision too. Without consuming its crossing
        # key, a late dirty fragment from that same physical presence could
        # open a new event and fire after the clean cap had already left.
        self._consume_event_cycles_locked(event)
        self._recent_finalized_events.append(event)
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
            fire_suppressed=event.fire_suppressed,
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
        suffix = " (fire deduped)" if record.fire_suppressed else ""
        self.log_fn(
            f"[CAP] event={record.event_id} {result} "
            f"class={record.class_name} conf={record.confidence:.3f} flagged_by={cameras}{suffix}"
        )
        if self.history_callback is not None:
            self.history_callback(record)
