"""Air-nozzle actuation: a time-ordered fire scheduler plus a simulated pin.

Copied (self-contained) from the v3 actuation pattern. The scheduler keeps a heap
keyed by ``requested_fire_time`` and pulses the solenoid on a dedicated thread so
fires happen on time regardless of what the capture/inference threads are doing.
``time_fn`` is injectable for deterministic deadline tests; the pulse wait is
cancel-aware so shutdown can de-energize the valve promptly.

A failed pulse (GPIO error, torn-down pin, callback bug) is logged and skipped —
it must never kill the scheduler thread, because a dead scheduler silently stops
every future fire while the rest of the runtime keeps running.

The queue is fail-safe: newly earlier jobs can preempt a future wait, targets
already covered by an active pulse are coalesced, runnable backlog expires, and
``close`` cancels pending jobs instead of draining them through the valve.
"""

from __future__ import annotations

import heapq
import threading
import time
from dataclasses import dataclass
from itertools import count
from typing import Callable


class NullGPIOOutputPin:
    """Drop-in stand-in for ``GPIOOutputPin`` on machines with no GPIO hardware library."""

    backend_name = "simulation"

    def __init__(self, pin=None):
        self.pin = pin

    def on(self) -> None:
        return None

    def off(self) -> None:
        return None

    def close(self) -> None:
        return None


@dataclass(frozen=True)
class RejectExecution:
    event_id: int
    queued_at: float
    requested_fire_time: float
    trigger_on_time: float
    trigger_off_time: float


@dataclass(frozen=True)
class RejectEnqueueResult:
    queue_depth: int
    queued_at: float
    requested_fire_time: float


class RejectScheduler:
    def __init__(
        self,
        *,
        trigger_pin,
        trigger_duration: float,
        trigger_min_gap: float,
        max_queue_age: float = 0.250,
        max_lateness: float = 0.500,
        pin_factory,
        log_fn: Callable[..., None] = print,
        time_fn: Callable[[], float] = time.monotonic,
        sleep_fn: Callable[[float], None] = time.sleep,
        cancel_event: threading.Event | None = None,
    ):
        self.trigger_duration = float(trigger_duration)
        self.trigger_min_gap = float(trigger_min_gap)
        self.max_queue_age = max(0.0, float(max_queue_age))
        self.max_lateness = max(0.0, float(max_lateness))
        self.log_fn = log_fn
        self.time_fn = time_fn
        self.sleep_fn = sleep_fn
        self.cancel_event = cancel_event
        self.pin = pin_factory(trigger_pin)
        self.backend_name = getattr(self.pin, "backend_name", type(self.pin).__name__)
        self._queue: list[tuple[float, int, int, float, Callable[[RejectExecution], None] | None]] = []
        self._counter = count()
        self._closed = False
        self._close_requested = threading.Event()
        self._condition = threading.Condition()
        self._fatal_lock = threading.Lock()
        self._last_fire_end: float | None = None
        self.fatal_error: str | None = None
        self._close_lock_timeout_s = 0.05
        self._close_join_timeout_s = 2.0
        self.stale_drops = 0
        self.coalesced_fires = 0
        self.cancelled_on_close = 0
        self._thread = threading.Thread(target=self._run, name="cap-line-v7-reject", daemon=True)
        self._thread.start()

    def enqueue(
        self,
        event_id: int,
        requested_fire_time: float,
        *,
        completion_callback: Callable[[RejectExecution], None] | None = None,
    ) -> RejectEnqueueResult:
        queued_at = float(self.time_fn())
        with self._condition:
            if self._closed or self._close_requested.is_set() or self._cancel_requested():
                raise RuntimeError("reject scheduler is closed")
            heapq.heappush(
                self._queue,
                (float(requested_fire_time), next(self._counter), int(event_id), queued_at, completion_callback),
            )
            queue_depth = len(self._queue)
            self._condition.notify_all()
        return RejectEnqueueResult(queue_depth, queued_at, float(requested_fire_time))

    def _cancel_requested(self) -> bool:
        return self.cancel_event is not None and self.cancel_event.is_set()

    def _log(self, message: str) -> None:
        """Operator logging must never delay or disable fail-safe GPIO work."""

        try:
            self.log_fn(message)
        except Exception:
            pass

    def _mark_fatal(self, message: str) -> None:
        """Latch a hardware/scheduler failure without calling user code."""

        with self._fatal_lock:
            if self.fatal_error is None:
                self.fatal_error = str(message)

    def _run(self) -> None:
        try:
            while True:
                coalesced_message = None
                with self._condition:
                    while not self._queue and not self._closed:
                        self._condition.wait(timeout=0.05)
                    if self._closed or self._close_requested.is_set() or self._cancel_requested():
                        return

                    # Peek rather than pop: a newly-enqueued earlier job can
                    # notify this wait and preempt a future job.
                    job = self._queue[0]
                    requested_fire_time, _order, event_id, queued_at, callback = job

                    # A target that occurred while the preceding pulse was ON
                    # has already been physically covered; never pulse twice.
                    if self._last_fire_end is not None and requested_fire_time <= self._last_fire_end:
                        heapq.heappop(self._queue)
                        self.coalesced_fires += 1
                        coalesced_message = (
                            f"[REJECT][COALESCED] event={event_id} target was covered by the previous pulse"
                        )

                    earliest_start = requested_fire_time
                    if self._last_fire_end is not None:
                        earliest_start = max(earliest_start, self._last_fire_end + self.trigger_min_gap)
                if coalesced_message is not None:
                    self._log(coalesced_message)
                    continue

                # Clock and logger callbacks are deliberately outside the
                # condition: close() must never wait behind user code.
                wait_time = earliest_start - float(self.time_fn())
                if wait_time > 0.0:
                    with self._condition:
                        if self._closed or self._close_requested.is_set() or self._cancel_requested():
                            return
                        # An earlier enqueue may have replaced the heap head.
                        if not self._queue or self._queue[0] is not job:
                            continue
                        self._condition.wait(timeout=min(0.05, wait_time))
                    continue
                on_exc = None
                with self._condition:
                    if self._closed or self._close_requested.is_set() or self._cancel_requested():
                        return
                    if not self._queue or self._queue[0] is not job:
                        continue
                    heapq.heappop(self._queue)

                now = float(self.time_fn())
                absolute_lateness = max(0.0, now - float(requested_fire_time))
                if absolute_lateness > self.max_lateness:
                    self.stale_drops += 1
                    self._log(
                        f"[REJECT][STALE] dropped event={event_id}: target is "
                        f"{absolute_lateness * 1000.0:.0f} ms late "
                        f"(limit={self.max_lateness * 1000.0:.0f} ms)"
                    )
                    continue
                runnable_since = max(float(queued_at), float(requested_fire_time))
                queue_age = max(0.0, now - runnable_since)
                if queue_age > self.max_queue_age:
                    self.stale_drops += 1
                    self._log(
                        f"[REJECT][STALE] dropped event={event_id}: runnable in queue for "
                        f"{queue_age * 1000.0:.0f} ms (limit={self.max_queue_age * 1000.0:.0f} ms)"
                    )
                    continue

                # Read the clock outside the state condition. GPIO activation
                # is serialized below, while close's lock-free event can still
                # request cancellation if a driver call stalls.
                pre_on = float(self.time_fn())
                with self._condition:
                    if self._closed or self._close_requested.is_set() or self._cancel_requested():
                        return
                    # Recheck deadlines at the last software-controlled point
                    # before energizing the valve.
                    if pre_on - float(requested_fire_time) > self.max_lateness:
                        self.stale_drops += 1
                        continue
                    if pre_on - max(float(queued_at), float(requested_fire_time)) > self.max_queue_age:
                        self.stale_drops += 1
                        continue
                    # Holding the state lock through the fast GPIO transition
                    # gives close() a clear linearization point. close() uses a
                    # bounded acquisition, so a broken driver still cannot
                    # hang shutdown indefinitely.
                    try:
                        self.pin.on()
                    except Exception as exc:
                        on_exc = exc
                if on_exc is not None:
                    off_exc = None
                    try:
                        self.pin.off()
                    except Exception as failure:
                        off_exc = failure
                        with self._condition:
                            self._closed = True
                            self.cancelled_on_close += len(self._queue)
                            self._queue.clear()
                    self._log(f"[REJECT][ERROR] pulse for event={event_id} failed to turn on: {on_exc}")
                    if off_exc is not None:
                        self._mark_fatal("reject valve state is unknown after ON/OFF failure")
                        self._log(f"[REJECT][ERROR] fail-safe OFF also failed: {off_exc}")
                        return
                    continue
                trigger_on = float(self.time_fn())
                activation_cancelled = (
                    self._closed or self._close_requested.is_set() or self._cancel_requested()
                )
                activation_lateness = trigger_on - float(requested_fire_time)
                activation_queue_age = trigger_on - max(float(queued_at), float(requested_fire_time))
                if (
                    activation_cancelled
                    or activation_lateness > self.max_lateness
                    or activation_queue_age > self.max_queue_age
                ):
                    # A slow/broken driver may apply ON only as it returns. It
                    # is too late to prevent that transition, but immediately
                    # de-energize it, never report completion, and make the
                    # runtime surface that a late/cancelled pulse may have
                    # occurred instead of claiming a clean stop.
                    try:
                        self.pin.off()
                    except Exception:
                        self._mark_fatal(
                            "reject valve activation completed late and could not be forced off"
                        )
                        return
                    self.stale_drops += 1
                    if activation_cancelled:
                        self._mark_fatal(
                            "reject valve activation completed after cancellation; a brief pulse may have occurred"
                        )
                    else:
                        self._mark_fatal(
                            "reject valve activation completed after its safety deadline; a late pulse may have occurred"
                        )
                    return
                # Poll both cancellation sources so UI Stop and explicit close
                # shorten an active pulse rather than waiting out its duration.
                pulse_deadline = time.monotonic() + self.trigger_duration
                while not self._close_requested.is_set() and not self._cancel_requested():
                    remaining = pulse_deadline - time.monotonic()
                    if remaining <= 0.0:
                        break
                    self._close_requested.wait(timeout=min(0.01, remaining))
                try:
                    self.pin.off()
                except Exception as exc:
                    with self._condition:
                        self._closed = True
                        self.cancelled_on_close += len(self._queue)
                        self._queue.clear()
                    self._mark_fatal("reject valve failed to turn off; hardware state is unknown")
                    self._log(f"[REJECT][ERROR] pulse for event={event_id} failed to turn off: {exc}")
                    return
                if self._closed or self._close_requested.is_set() or self._cancel_requested():
                    return
                trigger_off = float(self.time_fn())
                self._last_fire_end = trigger_off

                with self._condition:
                    if self._closed:
                        return
                if callback is not None:
                    try:
                        callback(
                            RejectExecution(
                                event_id=event_id,
                                queued_at=queued_at,
                                requested_fire_time=requested_fire_time,
                                trigger_on_time=trigger_on,
                                trigger_off_time=trigger_off,
                            )
                        )
                    except Exception as exc:
                        self._log(f"[REJECT][ERROR] completion callback for event={event_id} failed: {exc}")
        except BaseException as exc:
            # A dead scheduler would otherwise leave detection apparently
            # healthy while every future reject silently remains unserved.
            self._mark_fatal(f"reject scheduler crashed with {type(exc).__name__}")
        finally:
            with self._condition:
                if not self._closed:
                    self._closed = True
                    self.cancelled_on_close += len(self._queue)
                    self._queue.clear()
                    self._condition.notify_all()
            off_exc = None
            close_exc = None
            try:
                self.pin.off()
            except Exception as exc:
                off_exc = exc
            try:
                self.pin.close()
            except Exception as exc:
                close_exc = exc
            if off_exc is not None:
                self._mark_fatal("reject valve could not be forced off during shutdown")
                self._log(f"[REJECT][ERROR] unable to force pin off during close: {off_exc}")
            if close_exc is not None:
                self._mark_fatal("reject GPIO handle could not be closed cleanly")
                self._log(f"[REJECT][ERROR] unable to close pin: {close_exc}")

    def close(self) -> bool:
        # This lock-free signal also interrupts an active pulse wait. It is set
        # before acquiring the condition in case a GPIO call is slow.
        self._close_requested.set()
        acquired = self._condition.acquire(timeout=self._close_lock_timeout_s)
        if acquired:
            try:
                if not self._closed:
                    self._closed = True
                    self.cancelled_on_close += len(self._queue)
                    self._queue.clear()
                    self._condition.notify_all()
            finally:
                self._condition.release()
        if threading.current_thread() is not self._thread:
            self._thread.join(timeout=self._close_join_timeout_s)
            if self._thread.is_alive():
                # Do not call the configured logger here: this path can itself
                # be caused by a hanging logger, and close must stay bounded.
                self._mark_fatal(
                    "reject scheduler did not stop before the shutdown deadline; hardware state is unknown"
                )
        return self.fatal_error is None and not self._thread.is_alive()
