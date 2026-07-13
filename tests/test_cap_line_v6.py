"""Tests for the v6 cap-inspection runtime.

The v4 suite ported to v5 names/defaults, plus regression tests that encode the
double-trigger fix:

- merging keyed to physical ``last_seen`` exit times (a late-*reported* second
  camera still merges instead of double-firing);
- the post-fire refractory (``min_fire_interval_ms``) suppressing anything that
  leaks past the merge window;
- the legacy ``global_cooldown_ms`` settings key mapping onto
  ``merge_window_ms``;
- the Raspberry Pi gpiozero pin driver (with a fake gpiozero module).
"""

from __future__ import annotations

import sqlite3
import sys
import threading
import time
import types
from dataclasses import replace
from itertools import count

import numpy as np
import pytest

from cap_line_v6.actuation import NullGPIOOutputPin, RejectExecution, RejectScheduler
from cap_line_v6.config import RuntimeConfig, validate_config
from cap_line_v6.decision import CapEventManager
from cap_line_v6.model import postprocess
from cap_line_v6.runtime import resolve_pin_factory
from cap_line_v6.tracking import CameraTracker, Track
from cap_line_v6.types import CapEventRecord

import rpi_gpio_output
from rpi_gpio_output import resolve_pin_spec


# --------------------------------------------------------------------------- #
# Helpers / fakes
# --------------------------------------------------------------------------- #

_TRACK_IDS = count(1)


def make_track(
    camera_index: int,
    *,
    last_seen: float,
    is_defect: bool,
    defect_conf: float = 0.0,
    undef_conf: float = 0.0,
    first_seen: float | None = None,
) -> Track:
    return Track(
        track_id=next(_TRACK_IDS),
        camera_index=camera_index,
        first_seen=last_seen if first_seen is None else first_seen,
        last_seen=last_seen,
        frame_count=1,
        last_box=(0.0, 0.0, 10.0, 10.0, defect_conf if is_defect else undef_conf, 1 if is_defect else 0),
        is_defect=is_defect,
        best_defect_conf=defect_conf,
        best_undefected_conf=undef_conf,
    )


class FakeScheduler:
    """Records what would be fired, without any threads or timing."""

    backend_name = "fake"

    def __init__(self):
        self.enqueued: list[tuple[int, float, object]] = []

    def enqueue(self, event_id, requested_fire_time, *, completion_callback=None):
        self.enqueued.append((int(event_id), float(requested_fire_time), completion_callback))

    def close(self):
        return None


def make_manager(scheduler, clock_holder, **overrides):
    overrides.setdefault("merge_window_ms", 150.0)
    overrides.setdefault("min_fire_interval_ms", 250.0)
    overrides.setdefault("track_timeout_ms", 50.0)
    config = replace(RuntimeConfig.defaults(), **overrides)
    records: list[CapEventRecord] = []
    manager = CapEventManager(
        config,
        scheduler=scheduler,
        time_fn=lambda: clock_holder[0],
        history_callback=records.append,
        log_fn=lambda *args, **kwargs: None,
    )
    return manager, records


# --------------------------------------------------------------------------- #
# 1. Tracker association + OR decision
# --------------------------------------------------------------------------- #

def test_tracker_or_decision_marks_track_defective_on_single_defect_frame():
    tracker = CameraTracker(0, track_iou=0.3, track_timeout_s=0.150)
    clean_box = (10.0, 10.0, 50.0, 50.0, 0.90, 0)
    for timestamp in (0.000, 0.016, 0.032, 0.048):
        tracker.update([clean_box], timestamp)
    # A single dirt_defect frame must flip the whole track to defective.
    tracker.update([(11.0, 11.0, 51.0, 51.0, 0.82, 1)], 0.064)

    assert len(tracker.active_tracks) == 1  # all frames associated into one track
    track = tracker.active_tracks[0]
    assert track.frame_count == 5
    assert track.is_defect is True
    assert track.winning_class_id == 1
    assert track.winning_confidence == pytest.approx(0.82)

    finished = tracker.collect_finished(0.064 + 0.150 + 0.001)
    assert len(finished) == 1 and finished[0].is_defect is True


# --------------------------------------------------------------------------- #
# 2. Track finishes on timeout
# --------------------------------------------------------------------------- #

def test_track_finishes_after_timeout():
    tracker = CameraTracker(0, track_iou=0.3, track_timeout_s=0.150)
    tracker.update([(10.0, 10.0, 50.0, 50.0, 0.90, 0)], 1.000)

    # Before the timeout the track is still active and not returned.
    assert tracker.collect_finished(1.000 + 0.100) == []
    assert len(tracker.active_tracks) == 1

    finished = tracker.collect_finished(1.000 + 0.151)
    assert len(finished) == 1
    assert tracker.active_tracks == ()


def test_new_detection_far_away_starts_a_second_track():
    tracker = CameraTracker(0, track_iou=0.3, track_timeout_s=0.150)
    tracker.update([(10.0, 10.0, 50.0, 50.0, 0.9, 0)], 0.0)
    tracker.update([(400.0, 400.0, 440.0, 440.0, 0.9, 0)], 0.016)  # no overlap, far -> new track
    assert len(tracker.active_tracks) == 2


def test_fallback_does_not_chain_next_cap_into_previous_track():
    """Back-to-back caps: when cap 1 leaves view and cap 2 appears one cap-width
    upstream on the next frame, the old loose centroid gate absorbed cap 2 into
    cap 1's track. Chained across a continuous feed, that single track never
    timed out, so no cap ever finished -> no event, no verdict, no fire."""

    tracker = CameraTracker(0, track_iou=0.3, track_timeout_s=0.150)
    # Cap 1 (defective) moves steadily right: 30 px/frame, 60 px wide.
    tracker.update([(700.0, 100.0, 760.0, 160.0, 0.9, 1)], 0.000)
    tracker.update([(730.0, 100.0, 790.0, 160.0, 0.9, 1)], 0.033)
    tracker.update([(760.0, 100.0, 820.0, 160.0, 0.9, 1)], 0.066)
    assert len(tracker.active_tracks) == 1

    # Cap 1 exits; cap 2 appears one cap-width upstream of where cap 1 vanished.
    tracker.update([(700.0, 100.0, 760.0, 160.0, 0.9, 0)], 0.099)
    assert len(tracker.active_tracks) == 2  # a fresh track, not cap 1's

    # Cap 1's track must now time out and finish (defective), while cap 2 lives on.
    finished = tracker.collect_finished(0.066 + 0.150 + 0.001)
    assert len(finished) == 1 and finished[0].is_defect is True
    assert len(tracker.active_tracks) == 1


def test_fallback_predictive_match_keeps_fast_cap_in_one_track():
    """A fast cap (zero IoU between consecutive frames) must still associate:
    the first hop through the loose centroid gate, later hops against the
    velocity-predicted center."""

    tracker = CameraTracker(0, track_iou=0.3, track_timeout_s=0.150)
    tracker.update([(0.0, 100.0, 60.0, 160.0, 0.9, 0)], 0.000)
    tracker.update([(100.0, 100.0, 160.0, 160.0, 0.9, 0)], 0.033)
    tracker.update([(200.0, 100.0, 260.0, 160.0, 0.9, 0)], 0.066)
    tracker.update([(300.0, 100.0, 360.0, 160.0, 0.9, 0)], 0.099)

    assert len(tracker.active_tracks) == 1
    assert tracker.active_tracks[0].frame_count == 4


# --------------------------------------------------------------------------- #
# 3. Fire timing
# --------------------------------------------------------------------------- #

def test_fire_scheduled_at_last_seen_plus_delay():
    scheduler = FakeScheduler()
    clock = [100.0]
    manager, _records = make_manager(scheduler, clock, fire_delay_s=0.25)

    manager.handle_finished_track(make_track(0, last_seen=99.8, is_defect=True, defect_conf=0.9))

    assert len(scheduler.enqueued) == 1
    _event_id, requested_fire_time, _cb = scheduler.enqueued[0]
    assert requested_fire_time == pytest.approx(99.8 + 0.25)


# --------------------------------------------------------------------------- #
# 4. Once-per-cap across cameras
# --------------------------------------------------------------------------- #

def test_two_cameras_same_cap_fire_once():
    scheduler = FakeScheduler()
    clock = [100.0]
    manager, records = make_manager(scheduler, clock, fire_delay_s=0.0)

    manager.handle_finished_track(make_track(0, last_seen=100.0, is_defect=True, defect_conf=0.9))
    clock[0] = 100.020
    manager.handle_finished_track(make_track(1, last_seen=100.020, is_defect=True, defect_conf=0.8))

    assert len(scheduler.enqueued) == 1  # exactly one fire for the one physical cap

    clock[0] = 100.500  # past last_seen + merge_window + track_timeout
    manager.flush_expired(clock[0])
    assert len(records) == 1
    assert records[0].result == "reject"
    assert records[0].cameras == [0, 1]
    assert records[0].flagged_cameras == [0, 1]
    assert manager.caps_seen == 1 and manager.rejects == 1


def test_late_reported_second_camera_still_merges_and_does_not_double_fire():
    """The v4 double-trigger bug: camera B's finish is *reported* long after
    camera A's (slow inference thread), but the cap physically left both views
    at nearly the same time. v5 merges on ``last_seen``, so exactly one fire."""

    scheduler = FakeScheduler()
    clock = [100.0]
    manager, records = make_manager(scheduler, clock, fire_delay_s=0.0)

    manager.handle_finished_track(make_track(0, last_seen=100.0, is_defect=True, defect_conf=0.9))
    clock[0] = 100.300  # reported 300 ms later (way beyond v4's 50 ms window)...
    manager.handle_finished_track(make_track(1, last_seen=100.020, is_defect=True, defect_conf=0.8))

    assert len(scheduler.enqueued) == 1  # ...but the exit times match: same cap

    clock[0] = 100.500
    manager.flush_expired(clock[0])
    assert len(records) == 1
    assert records[0].cameras == [0, 1]


def test_refractory_suppresses_fire_that_leaks_past_merge_window():
    scheduler = FakeScheduler()
    clock = [100.0]
    manager, records = make_manager(scheduler, clock, fire_delay_s=0.0)

    manager.handle_finished_track(make_track(0, last_seen=100.0, is_defect=True, defect_conf=0.9))
    # 200 ms later: outside the 150 ms merge window (-> a second event), but
    # inside the 250 ms refractory (-> the fire is suppressed, not doubled).
    clock[0] = 100.200
    manager.handle_finished_track(make_track(1, last_seen=100.200, is_defect=True, defect_conf=0.8))

    assert len(scheduler.enqueued) == 1
    assert manager.suppressed_fires == 1

    manager.finalize_all()
    assert len(records) == 2  # first event finalized when the second opened
    assert records[0].fire_suppressed is False
    assert records[1].result == "reject" and records[1].fire_suppressed is True
    assert records[1].requested_fire_time is None


def test_clean_then_dirty_within_merge_window_fires_once_and_rejects():
    scheduler = FakeScheduler()
    clock = [100.0]
    manager, records = make_manager(scheduler, clock, fire_delay_s=0.0)

    # First camera sees it clean -> no fire yet.
    manager.handle_finished_track(make_track(0, last_seen=100.0, is_defect=False, undef_conf=0.7))
    assert scheduler.enqueued == []

    # Second camera (same cap) sees it dirty -> must fire now.
    clock[0] = 100.020
    manager.handle_finished_track(make_track(1, last_seen=100.020, is_defect=True, defect_conf=0.85))
    assert len(scheduler.enqueued) == 1
    assert scheduler.enqueued[0][1] == pytest.approx(100.020)  # fire keyed off the defect track

    clock[0] = 100.500
    manager.flush_expired(clock[0])
    assert len(records) == 1
    assert records[0].result == "reject"
    assert records[0].cameras == [0, 1]
    assert records[0].flagged_cameras == [1]


def test_two_separate_caps_beyond_refractory_fire_twice():
    scheduler = FakeScheduler()
    clock = [100.0]
    manager, records = make_manager(scheduler, clock, fire_delay_s=0.0)

    manager.handle_finished_track(make_track(0, last_seen=100.0, is_defect=True, defect_conf=0.9))
    clock[0] = 100.400  # 400 ms later -> a different physical cap, past the refractory
    manager.handle_finished_track(make_track(0, last_seen=100.400, is_defect=True, defect_conf=0.9))

    assert len(scheduler.enqueued) == 2  # one per cap
    # Opening the second cap finalizes (logs) the first.
    assert len(records) == 1 and records[0].event_id == 1
    manager.finalize_all()
    assert len(records) == 2
    assert manager.caps_seen == 2 and manager.rejects == 2
    assert manager.suppressed_fires == 0


# --------------------------------------------------------------------------- #
# 5. Pass caps don't fire
# --------------------------------------------------------------------------- #

def test_pass_cap_schedules_no_fire():
    scheduler = FakeScheduler()
    clock = [100.0]
    manager, records = make_manager(scheduler, clock, fire_delay_s=0.0)

    manager.handle_finished_track(make_track(0, last_seen=100.0, is_defect=False, undef_conf=0.7))
    assert scheduler.enqueued == []

    clock[0] = 100.500
    manager.flush_expired(clock[0])
    assert len(records) == 1
    assert records[0].result == "pass"
    assert records[0].class_name == "undefected"
    assert manager.caps_seen == 1 and manager.rejects == 0


# --------------------------------------------------------------------------- #
# 6. Threshold filtering
# --------------------------------------------------------------------------- #

def test_postprocess_filters_sub_threshold_detections():
    output = np.array(
        [
            [10.0, 10.0, 50.0, 50.0, 0.80, 1.0],  # keep
            [60.0, 60.0, 90.0, 90.0, 0.20, 1.0],  # drop (below reject_threshold)
        ],
        dtype=np.float32,
    )
    meta = {"scale": 1.0, "pad_left": 0, "pad_top": 0, "frame_shape": (100, 100, 3), "img_size": 100}

    boxes = postprocess(output, meta, conf_threshold=0.45)

    assert len(boxes) == 1
    assert boxes[0][4] == pytest.approx(0.80)
    assert int(boxes[0][5]) == 1


# --------------------------------------------------------------------------- #
# Fire completion re-emits the record with the actual fire time
# --------------------------------------------------------------------------- #

def test_fire_completion_updates_record_when_finalized_first():
    scheduler = FakeScheduler()
    clock = [100.0]
    manager, records = make_manager(scheduler, clock, fire_delay_s=1.0)

    manager.handle_finished_track(make_track(0, last_seen=100.0, is_defect=True, defect_conf=0.9))
    clock[0] = 100.500
    manager.flush_expired(clock[0])  # finalized before the (delayed) fire executes
    assert len(records) == 1 and records[0].actual_fire_time is None

    _event_id, requested, completion_callback = scheduler.enqueued[0]
    completion_callback(
        RejectExecution(
            event_id=_event_id,
            queued_at=100.0,
            requested_fire_time=requested,
            trigger_on_time=requested,
            trigger_off_time=requested + 0.3,
        )
    )
    assert len(records) == 2
    assert records[-1].actual_fire_time is not None


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

def test_config_json_round_trip():
    config = replace(RuntimeConfig.defaults(), fire_delay_s=0.2, cameras=("2", "4"), imgsz=640)
    assert RuntimeConfig.from_json_dict(config.to_json_dict()) == config


def test_config_from_json_drops_unknown_legacy_keys():
    data = RuntimeConfig.defaults().to_json_dict()
    data["belt_speed_mm_per_s"] = 275.0  # v3 leftover that must be ignored
    data["anchor_axis"] = "x"
    config = RuntimeConfig.from_json_dict(data)
    assert config == RuntimeConfig.defaults()
    assert not hasattr(config, "belt_speed_mm_per_s")


def test_config_maps_legacy_global_cooldown_to_merge_window():
    data = RuntimeConfig.defaults().to_json_dict()
    del data["merge_window_ms"]
    data["global_cooldown_ms"] = 80.0  # a v4 settings file
    config = RuntimeConfig.from_json_dict(data)
    assert config.merge_window_ms == pytest.approx(80.0)

    # The new key wins when both are present.
    data["merge_window_ms"] = 120.0
    config = RuntimeConfig.from_json_dict(data)
    assert config.merge_window_ms == pytest.approx(120.0)


def test_validate_config_rejects_out_of_range_values():
    with pytest.raises(ValueError):
        validate_config(replace(RuntimeConfig.defaults(), reject_threshold=1.5))
    with pytest.raises(ValueError):
        validate_config(replace(RuntimeConfig.defaults(), min_fire_interval_ms=-1.0))
    with pytest.raises(ValueError):
        validate_config(replace(RuntimeConfig.defaults(), gpio_backend="esp32"))


# --------------------------------------------------------------------------- #
# Raspberry Pi GPIO driver (gpiozero faked)
# --------------------------------------------------------------------------- #

class _FakeDigitalOutputDevice:
    instances: list["_FakeDigitalOutputDevice"] = []

    def __init__(self, pin, active_high=True, initial_value=False):
        self.pin = pin
        self.active_high = active_high
        self.value = int(bool(initial_value))
        self.closed = False
        _FakeDigitalOutputDevice.instances.append(self)

    def on(self):
        self.value = 1

    def off(self):
        self.value = 0

    def close(self):
        self.closed = True


@pytest.fixture
def fake_gpiozero(monkeypatch):
    _FakeDigitalOutputDevice.instances = []
    module = types.ModuleType("gpiozero")
    module.DigitalOutputDevice = _FakeDigitalOutputDevice
    monkeypatch.setitem(sys.modules, "gpiozero", module)
    return module


def test_resolve_pin_spec_variants():
    assert resolve_pin_spec(17) == "GPIO17"
    assert resolve_pin_spec("17") == "GPIO17"
    assert resolve_pin_spec("GPIO17") == "GPIO17"
    assert resolve_pin_spec("BCM17") == "GPIO17"
    assert resolve_pin_spec("BOARD11") == "BOARD11"
    assert resolve_pin_spec("pin 11") == "BOARD11"
    with pytest.raises(ValueError):
        resolve_pin_spec("nonsense")
    with pytest.raises(ValueError):
        resolve_pin_spec("")


def test_rpi_pin_pulses_and_closes(fake_gpiozero):
    pin = rpi_gpio_output.RPiGPIOOutputPin(17)
    device = _FakeDigitalOutputDevice.instances[-1]
    assert device.pin == "GPIO17"
    assert device.active_high is True
    assert "gpiozero GPIO17" in pin.backend_name

    pin.on()
    assert pin.read() == 1
    pin.off()
    assert pin.read() == 0
    pin.close()
    assert device.closed is True and device.value == 0


def test_rpi_pin_active_low(fake_gpiozero):
    pin = rpi_gpio_output.RPiGPIOOutputPin("GPIO17", active_low=True)
    device = _FakeDigitalOutputDevice.instances[-1]
    assert device.active_high is False
    assert "active-low" in pin.backend_name
    pin.close()


def test_resolve_pin_factory_picks_backend(fake_gpiozero):
    simulate = replace(RuntimeConfig.defaults(), simulate_gpio=True)
    assert resolve_pin_factory(simulate) is NullGPIOOutputPin

    rpi = replace(RuntimeConfig.defaults(), gpio_backend="rpi")
    assert resolve_pin_factory(rpi) is rpi_gpio_output.RPiGPIOOutputPin


def test_v6_defaults_target_the_jetson():
    """v6 = v5 with Jetson GPIO defaults: gpio_output.py driver, BOARD pin 7."""

    from gpio_output import GPIOOutputPin

    defaults = RuntimeConfig.defaults()
    assert defaults.gpio_backend == "jetson"
    assert defaults.trigger_pin == 7  # GPIO09, Jetson Nano BOARD pin 7
    assert resolve_pin_factory(defaults) is GPIOOutputPin


# --------------------------------------------------------------------------- #
# Scheduler actually pulses the pin (real time, tiny pulse)
# --------------------------------------------------------------------------- #

def test_reject_scheduler_pulses_pin():
    events: list[str] = []

    class RecordingPin:
        backend_name = "recording"

        def __init__(self, pin):
            self.pin = pin

        def on(self):
            events.append("on")

        def off(self):
            events.append("off")

        def close(self):
            events.append("close")

    scheduler = RejectScheduler(
        trigger_pin=17,
        trigger_duration=0.001,
        trigger_min_gap=0.0,
        pin_factory=RecordingPin,
        log_fn=lambda *args, **kwargs: None,
    )
    done = threading.Event()
    scheduler.enqueue(1, time.monotonic(), completion_callback=lambda execution: done.set())
    assert done.wait(2.0)
    scheduler.close()

    assert events[:2] == ["on", "off"]
    assert events[-1] == "close"


def test_reject_scheduler_survives_a_failing_pulse():
    """A GPIO error during one pulse must not kill the scheduler thread: that
    would silently disable every future fire while the runtime keeps running."""

    on_calls: list[int] = []

    class FlakyPin:
        backend_name = "flaky"

        def __init__(self, pin):
            self.pin = pin

        def on(self):
            on_calls.append(len(on_calls))
            if len(on_calls) == 1:
                raise RuntimeError("channel torn down")

        def off(self):
            return None

        def close(self):
            return None

    logs: list[str] = []
    scheduler = RejectScheduler(
        trigger_pin=7,
        trigger_duration=0.001,
        trigger_min_gap=0.0,
        pin_factory=FlakyPin,
        log_fn=lambda message, *args, **kwargs: logs.append(str(message)),
    )
    done = threading.Event()
    now = time.monotonic()
    scheduler.enqueue(1, now)  # this pulse raises
    scheduler.enqueue(2, now, completion_callback=lambda execution: done.set())
    assert done.wait(2.0)  # the thread survived and served the second fire
    scheduler.close()

    assert len(on_calls) == 2
    assert any("[REJECT][ERROR]" in line for line in logs)


# --------------------------------------------------------------------------- #
# End-to-end: full run_detection wiring fires exactly once for a defect cap
# --------------------------------------------------------------------------- #

class _ScriptedFrame:
    shape = (100, 100, 3)

    def __init__(self, detections):
        self.detections = detections


class _ScriptedCamera:
    """Yields scripted detections for a while, then empty frames forever."""

    def __init__(self, scripted_detections):
        self._scripted = list(scripted_detections)
        self._index = 0
        self._lock = threading.Lock()

    def read(self):
        with self._lock:
            if self._index < len(self._scripted):
                detections = self._scripted[self._index]
                self._index += 1
            else:
                detections = []
        return True, _ScriptedFrame(detections)

    def isOpened(self):
        return True

    def release(self):
        return None


class _FakeInput:
    name = "images"
    shape = [1, 3, 100, 100]


class _FakeSession:
    def get_inputs(self):
        return [_FakeInput()]

    def run(self, _outputs, inputs):
        frame = next(iter(inputs.values()))
        return [frame.detections]


def test_run_detection_fires_once_for_defect_cap():
    from cap_line_v6.runtime import run_detection
    from cap_line_v6.types import RuntimeCallbacks

    defect_box = [10.0, 10.0, 50.0, 50.0, 0.90, 1]
    clean_box = [10.0, 10.0, 50.0, 50.0, 0.80, 0]
    cameras = [
        _ScriptedCamera([[defect_box]] * 5),  # camera 0 catches the dirt
        _ScriptedCamera([[clean_box]] * 5),  # camera 1 sees the same cap clean
    ]

    fires: list[float] = []

    class RecordingPin:
        backend_name = "recording"

        def __init__(self, pin):
            self.pin = pin

        def on(self):
            fires.append(time.monotonic())

        def off(self):
            return None

        def close(self):
            return None

    records: list[CapEventRecord] = []
    config = replace(
        RuntimeConfig.defaults(),
        cameras=("0", "1"),
        simulate_gpio=False,  # use the injected RecordingPin
        trigger_duration=0.001,
        track_timeout_ms=20.0,
        merge_window_ms=100.0,
        min_fire_interval_ms=250.0,
        fire_delay_s=0.0,
        live_preview_fps=0.0,
        no_display=True,
    )
    stop_event = threading.Event()
    worker = threading.Thread(
        target=run_detection,
        args=(config, RuntimeCallbacks(history_callback=records.append, log_fn=lambda *a, **k: None), stop_event),
        kwargs=dict(
            pin_factory=RecordingPin,
            camera_factory=lambda index, _source, _config: cameras[index],
            session_factory=lambda _model_path, _threads: _FakeSession(),
            preprocess_fn=lambda frame, _imgsz: (frame, {"frame_shape": frame.shape}),
            postprocess_fn=lambda output, _meta, conf_threshold: [
                box for box in output if float(box[4]) >= float(conf_threshold)
            ],
        ),
        daemon=True,
    )
    worker.start()
    time.sleep(0.4)
    stop_event.set()
    worker.join(timeout=3.0)

    assert not worker.is_alive()
    assert len(fires) == 1  # exactly one air pulse for the one physical cap
    rejects = [record for record in records if record.result == "reject"]
    # The same event may legitimately appear twice (re-emitted with the actual
    # fire time once the pulse lands after finalization) — but it must be ONE
    # physical cap event.
    assert {record.event_id for record in rejects} and len({record.event_id for record in rejects}) == 1
    assert 0 in rejects[-1].flagged_cameras  # camera 0 is the one that caught the dirt


def test_run_detection_manual_test_fire_pulses_the_runtime_pin():
    """The UI's Test Fire while running must go through the runtime's own
    scheduler/pin (a second GPIO handle on the same channel would tear the
    runtime's pin down on close)."""

    from cap_line_v6.runtime import run_detection
    from cap_line_v6.types import RuntimeCallbacks

    cameras = [_ScriptedCamera([]), _ScriptedCamera([])]  # no caps at all
    fires: list[float] = []

    class RecordingPin:
        backend_name = "recording"

        def __init__(self, pin):
            self.pin = pin

        def on(self):
            fires.append(time.monotonic())

        def off(self):
            return None

        def close(self):
            return None

    requested = threading.Event()

    def test_fire_poll() -> bool:
        if requested.is_set():
            return False
        requested.set()
        return True

    config = replace(
        RuntimeConfig.defaults(),
        cameras=("0", "1"),
        simulate_gpio=False,  # use the injected RecordingPin
        trigger_duration=0.001,
        live_preview_fps=0.0,
        no_display=True,
    )
    stop_event = threading.Event()
    worker = threading.Thread(
        target=run_detection,
        args=(config, RuntimeCallbacks(log_fn=lambda *a, **k: None, test_fire_poll=test_fire_poll), stop_event),
        kwargs=dict(
            pin_factory=RecordingPin,
            camera_factory=lambda index, _source, _config: cameras[index],
            session_factory=lambda _model_path, _threads: _FakeSession(),
            preprocess_fn=lambda frame, _imgsz: (frame, {"frame_shape": frame.shape}),
            postprocess_fn=lambda output, _meta, conf_threshold: list(output),
        ),
        daemon=True,
    )
    worker.start()
    time.sleep(0.3)
    stop_event.set()
    worker.join(timeout=3.0)

    assert not worker.is_alive()
    assert len(fires) == 1  # exactly one manual pulse, despite zero defect caps


# --------------------------------------------------------------------------- #
# UI controller: test-fire routing while running
# --------------------------------------------------------------------------- #

def test_controller_routes_test_fire_through_running_runtime():
    from cap_line_ui_v6 import DetectionAppController, HistoryRepository

    captured: dict[str, object] = {}

    def fake_runner(config, callbacks, stop_event):
        captured["callbacks"] = callbacks
        stop_event.wait(2.0)

    controller = DetectionAppController(
        HistoryRepository(":memory:"),
        detector_runner=fake_runner,
        config_factory=lambda: replace(RuntimeConfig.defaults(), simulate_gpio=True),
    )
    # Stopped: the UI must pulse a standalone pin instead.
    assert controller.request_test_fire() is False

    assert controller.start() is True
    for _ in range(400):
        if "callbacks" in captured:
            break
        time.sleep(0.005)
    callbacks = captured["callbacks"]
    worker = controller.worker_thread

    assert callbacks.test_fire_poll is not None
    assert callbacks.test_fire_poll() is False  # nothing requested yet
    assert controller.request_test_fire() is True
    assert callbacks.test_fire_poll() is True  # consumed exactly once...
    assert callbacks.test_fire_poll() is False  # ...then cleared

    controller.stop()
    worker.join(timeout=3.0)
    assert not worker.is_alive()


# --------------------------------------------------------------------------- #
# History repository: event ids restart every run and must not clobber history
# --------------------------------------------------------------------------- #

def test_history_repository_keeps_rows_across_runs():
    from cap_line_ui_v6 import HistoryRepository

    repository = HistoryRepository(":memory:")
    first_run = CapEventRecord(
        event_id=1,
        recorded_at="2026-07-12T10:00:00.000",
        result="reject",
        class_name="dirt_defect",
        confidence=0.9,
        cameras=[0],
        flagged_cameras=[0],
    )
    repository.upsert_record(first_run)
    # The same event re-emitted (fire completion) updates its row in place.
    repository.upsert_record(replace(first_run, actual_fire_time="2026-07-12T10:00:00.100"))
    rows = repository.fetch_history()
    assert len(rows) == 1
    assert rows[0]["actual_fire_time"] == "2026-07-12T10:00:00.100"

    # A new run restarts event ids at 1: it must add a row, not overwrite.
    repository.upsert_record(replace(first_run, recorded_at="2026-07-12T11:00:00.000", confidence=0.8))
    assert len(repository.fetch_history()) == 2

    # Pass caps are logged too: the history is one row per physical cap.
    repository.upsert_record(
        replace(
            first_run,
            event_id=2,
            recorded_at="2026-07-12T11:00:01.000",
            result="pass",
            class_name="undefected",
            confidence=0.95,
            flagged_cameras=[],
        )
    )
    rows = repository.fetch_history()
    assert len(rows) == 3
    assert rows[0]["result"] == "pass"


def test_history_repository_migrates_legacy_event_id_unique_schema(tmp_path):
    from cap_line_ui_v6 import HistoryRepository

    db_path = str(tmp_path / "history.sqlite3")
    connection = sqlite3.connect(db_path)
    connection.execute(
        """
        CREATE TABLE cap_line_history_v6 (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id INTEGER NOT NULL UNIQUE,
            recorded_at TEXT NOT NULL,
            result TEXT NOT NULL,
            class_name TEXT,
            confidence REAL,
            cameras_json TEXT NOT NULL,
            flagged_cameras_json TEXT NOT NULL,
            requested_fire_time TEXT,
            actual_fire_time TEXT,
            fire_suppressed INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    connection.execute(
        "INSERT INTO cap_line_history_v6 (event_id, recorded_at, result, class_name, confidence, "
        "cameras_json, flagged_cameras_json, fire_suppressed) "
        "VALUES (1, '2026-07-11T09:00:00.000', 'reject', 'dirt_defect', 0.9, '[0]', '[0]', 0)"
    )
    connection.commit()
    connection.close()

    repository = HistoryRepository(db_path)
    assert len(repository.fetch_history()) == 1  # legacy rows preserved

    # A fresh run's cap 1 must coexist with the legacy cap 1.
    repository.upsert_record(
        CapEventRecord(
            event_id=1,
            recorded_at="2026-07-12T10:00:00.000",
            result="reject",
            class_name="dirt_defect",
            confidence=0.8,
            cameras=[0],
            flagged_cameras=[0],
        )
    )
    assert len(repository.fetch_history()) == 2
