from __future__ import annotations

import heapq
import threading
import time
from dataclasses import dataclass
from itertools import count
from typing import Callable


class NullGPIOOutputPin:
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
        pin_factory,
        log_fn: Callable[..., None] = print,
        time_fn: Callable[[], float] = time.monotonic,
        sleep_fn: Callable[[float], None] = time.sleep,
    ):
        self.trigger_duration = float(trigger_duration)
        self.trigger_min_gap = float(trigger_min_gap)
        self.log_fn = log_fn
        self.time_fn = time_fn
        self.sleep_fn = sleep_fn
        self.pin = pin_factory(trigger_pin)
        self.backend_name = getattr(self.pin, "backend_name", type(self.pin).__name__)
        self._queue: list[tuple[float, int, int, float, Callable[[RejectExecution], None] | None]] = []
        self._counter = count()
        self._closed = False
        self._condition = threading.Condition()
        self._last_fire_time: float | None = None
        self._thread = threading.Thread(target=self._run, name="cap-line-v3-reject", daemon=True)
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
            heapq.heappush(
                self._queue,
                (float(requested_fire_time), next(self._counter), int(event_id), queued_at, completion_callback),
            )
            queue_depth = len(self._queue)
            self._condition.notify()
        return RejectEnqueueResult(queue_depth, queued_at, float(requested_fire_time))

    def _run(self) -> None:
        while True:
            with self._condition:
                while not self._queue and not self._closed:
                    self._condition.wait(0.05)
                if self._closed and not self._queue:
                    return
                requested_fire_time, _order, event_id, queued_at, callback = heapq.heappop(self._queue)
            if self._last_fire_time is not None:
                requested_fire_time = max(requested_fire_time, self._last_fire_time + self.trigger_min_gap)
            cancelled = False
            while True:
                now = float(self.time_fn())
                if now >= requested_fire_time:
                    break
                if self._closed:
                    cancelled = True
                    break
                self.sleep_fn(min(0.01, requested_fire_time - now))
            if cancelled:
                continue
            trigger_on = float(self.time_fn())
            self.pin.on()
            self.sleep_fn(self.trigger_duration)
            self.pin.off()
            trigger_off = float(self.time_fn())
            self._last_fire_time = trigger_on
            if callback is not None:
                callback(
                    RejectExecution(
                        event_id=event_id,
                        queued_at=queued_at,
                        requested_fire_time=requested_fire_time,
                        trigger_on_time=trigger_on,
                        trigger_off_time=trigger_off,
                    )
                )

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        self._thread.join(timeout=2.0)
        self.pin.close()
