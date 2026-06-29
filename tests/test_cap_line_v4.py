"""Tests for the v4 cap-inspection runtime.

These exercise the logic that matters with injected fakes (scripted boxes, a fake
scheduler, and a controllable ``time_fn``) so everything is deterministic:

1. Tracker association + defect-wins (OR) decision.
2. Track finishes on timeout.
3. Fire scheduled at ``last_seen + fire_delay_s``.
4. Once-per-cap across cameras (exactly one fire).
5. Pass caps schedule nothing.
6. Sub-threshold detections are filtered out.
"""

from __future__ import annotations

import threading
import time
from dataclasses import replace
from itertools import count

import numpy as np
import pytest

from cap_line_v4.actuation import RejectExecution, RejectScheduler
from cap_line_v4.config import RuntimeConfig, validate_config
from cap_line_v4.decision import CapEventManager
from cap_line_v4.model import postprocess
from cap_line_v4.tracking import CameraTracker, Track
from cap_line_v4.types import CapEventRecord


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
    tracker = CameraTracker(0, track_iou=0.3, track_timeout_s=0.030)
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

    finished = tracker.collect_finished(0.064 + 0.030 + 0.001)
    assert len(finished) == 1 and finished[0].is_defect is True


# --------------------------------------------------------------------------- #
# 2. Track finishes on timeout
# --------------------------------------------------------------------------- #

def test_track_finishes_after_timeout():
    tracker = CameraTracker(0, track_iou=0.3, track_timeout_s=0.030)
    tracker.update([(10.0, 10.0, 50.0, 50.0, 0.90, 0)], 1.000)

    # Before the timeout the track is still active and not returned.
    assert tracker.collect_finished(1.000 + 0.020) == []
    assert len(tracker.active_tracks) == 1

    finished = tracker.collect_finished(1.000 + 0.031)
    assert len(finished) == 1
    assert tracker.active_tracks == ()


def test_new_detection_far_away_starts_a_second_track():
    tracker = CameraTracker(0, track_iou=0.3, track_timeout_s=0.030)
    tracker.update([(10.0, 10.0, 50.0, 50.0, 0.9, 0)], 0.0)
    tracker.update([(400.0, 400.0, 440.0, 440.0, 0.9, 0)], 0.016)  # no overlap, far -> new track
    assert len(tracker.active_tracks) == 2


# --------------------------------------------------------------------------- #
# 3. Fire timing
# --------------------------------------------------------------------------- #

def test_fire_scheduled_at_last_seen_plus_delay():
    scheduler = FakeScheduler()
    clock = [100.0]
    manager, _records = make_manager(scheduler, clock, fire_delay_s=0.25, global_cooldown_ms=50.0)

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
    manager, records = make_manager(scheduler, clock, fire_delay_s=0.0, global_cooldown_ms=50.0)

    manager.handle_finished_track(make_track(0, last_seen=100.0, is_defect=True, defect_conf=0.9))
    clock[0] = 100.020  # 20 ms later, well within the 50 ms cooldown
    manager.handle_finished_track(make_track(1, last_seen=100.020, is_defect=True, defect_conf=0.8))

    assert len(scheduler.enqueued) == 1  # exactly one fire for the one physical cap

    clock[0] = 100.200
    manager.flush_expired(clock[0])
    assert len(records) == 1
    assert records[0].result == "reject"
    assert records[0].cameras == [0, 1]
    assert records[0].flagged_cameras == [0, 1]
    assert manager.caps_seen == 1 and manager.rejects == 1


def test_clean_then_dirty_within_cooldown_fires_once_and_rejects():
    scheduler = FakeScheduler()
    clock = [100.0]
    manager, records = make_manager(scheduler, clock, fire_delay_s=0.0, global_cooldown_ms=50.0)

    # First camera sees it clean -> no fire yet.
    manager.handle_finished_track(make_track(0, last_seen=100.0, is_defect=False, undef_conf=0.7))
    assert scheduler.enqueued == []

    # Second camera (same cap) sees it dirty -> must fire now.
    clock[0] = 100.020
    manager.handle_finished_track(make_track(1, last_seen=100.020, is_defect=True, defect_conf=0.85))
    assert len(scheduler.enqueued) == 1
    assert scheduler.enqueued[0][1] == pytest.approx(100.020)  # fire keyed off the defect track

    clock[0] = 100.200
    manager.flush_expired(clock[0])
    assert len(records) == 1
    assert records[0].result == "reject"
    assert records[0].cameras == [0, 1]
    assert records[0].flagged_cameras == [1]


def test_two_separate_caps_beyond_cooldown_fire_twice():
    scheduler = FakeScheduler()
    clock = [100.0]
    manager, records = make_manager(scheduler, clock, fire_delay_s=0.0, global_cooldown_ms=50.0)

    manager.handle_finished_track(make_track(0, last_seen=100.0, is_defect=True, defect_conf=0.9))
    clock[0] = 100.200  # 200 ms later -> a different physical cap
    manager.handle_finished_track(make_track(0, last_seen=100.200, is_defect=True, defect_conf=0.9))

    assert len(scheduler.enqueued) == 2  # one per cap
    # Opening the second cap finalizes (logs) the first.
    assert len(records) == 1 and records[0].event_id == 1
    manager.finalize_all()
    assert len(records) == 2
    assert manager.caps_seen == 2 and manager.rejects == 2


# --------------------------------------------------------------------------- #
# 5. Pass caps don't fire
# --------------------------------------------------------------------------- #

def test_pass_cap_schedules_no_fire():
    scheduler = FakeScheduler()
    clock = [100.0]
    manager, records = make_manager(scheduler, clock, fire_delay_s=0.0, global_cooldown_ms=50.0)

    manager.handle_finished_track(make_track(0, last_seen=100.0, is_defect=False, undef_conf=0.7))
    assert scheduler.enqueued == []

    clock[0] = 100.200
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
    manager, records = make_manager(scheduler, clock, fire_delay_s=1.0, global_cooldown_ms=50.0)

    manager.handle_finished_track(make_track(0, last_seen=100.0, is_defect=True, defect_conf=0.9))
    clock[0] = 100.200
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


def test_config_from_json_drops_unknown_v3_keys():
    data = RuntimeConfig.defaults().to_json_dict()
    data["belt_speed_mm_per_s"] = 275.0  # v3 leftover that must be ignored
    data["anchor_axis"] = "x"
    config = RuntimeConfig.from_json_dict(data)
    assert config == RuntimeConfig.defaults()
    assert not hasattr(config, "belt_speed_mm_per_s")


def test_validate_config_rejects_out_of_range_threshold():
    with pytest.raises(ValueError):
        validate_config(replace(RuntimeConfig.defaults(), reject_threshold=1.5))


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
        trigger_pin=7,
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
    from cap_line_v4.runtime import run_detection
    from cap_line_v4.types import RuntimeCallbacks

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
        global_cooldown_ms=50.0,
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
    assert len(rejects) == 1
    assert 0 in rejects[0].flagged_cameras  # camera 0 is the one that caught the dirt
