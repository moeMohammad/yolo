"""Tests for the v6 cap-inspection runtime.

The v4/v5 suite ported to v6 plus regressions for the double-trigger and
high-confidence empty-belt false-fire failures:

- merging keyed to physical ``last_seen`` exit times (a late-*reported* second
  camera still merges instead of double-firing);
- the post-fire refractory (``min_fire_interval_ms``) suppressing anything that
  leaks past the merge window;
- the legacy ``global_cooldown_ms`` settings key mapping onto
  ``merge_window_ms``;
- multi-frame motion qualification and consecutive defect confirmation;
- presence-cycle idempotency, latest-frame timing, scheduler coalescing, and
  fail-safe shutdown;
- both GPIO backend selections (with a fake gpiozero module for the Pi path).
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
from cap_line_v6.model import deduplicate_boxes, postprocess
from cap_line_v6.runtime import CameraWorker, SharedRuntimeState, resolve_pin_factory
from cap_line_v6.tracking import CameraTracker, Track
from cap_line_v6.types import CapEventRecord, CapturedFrame

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
    presence_cycle_id: int | None = None,
) -> Track:
    track_id = next(_TRACK_IDS)
    return Track(
        track_id=track_id,
        camera_index=camera_index,
        first_seen=last_seen - 0.100 if first_seen is None else first_seen,
        last_seen=last_seen,
        frame_count=4,
        last_box=(0.0, 0.0, 10.0, 10.0, defect_conf if is_defect else undef_conf, 1 if is_defect else 0),
        is_defect=is_defect,
        best_defect_conf=defect_conf,
        best_undefected_conf=undef_conf,
        first_box=(-10.0, 0.0, 0.0, 10.0, defect_conf if is_defect else undef_conf, 1 if is_defect else 0),
        path_length_px=10.0,
        defect_frame_count=3 if is_defect else 0,
        undefected_frame_count=0 if is_defect else 4,
        consecutive_defect_frames=3 if is_defect else 0,
        max_consecutive_defect_frames=3 if is_defect else 0,
        min_defect_frames=3,
        presence_cycle_id=track_id if presence_cycle_id is None else presence_cycle_id,
        crossed_presence_line=True,
        line_negative_frames=2,
        line_positive_frames=2,
        largest_observation_gap_s=0.050,
        max_observation_gap_s=0.500,
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
# 1. Tracker association + temporal defect confirmation
# --------------------------------------------------------------------------- #

def test_tracker_ignores_two_false_dirt_frames_on_a_clean_moving_cap():
    tracker = CameraTracker(0, track_iou=0.3, track_timeout_s=0.150)
    observations = [
        (0.000, 10.0, 0),
        (0.016, 14.0, 0),
        (0.032, 18.0, 1),  # short 0.94 false-dirt burst
        (0.048, 22.0, 1),
        (0.064, 26.0, 0),
        (0.080, 30.0, 0),
    ]
    for timestamp, x1, class_id in observations:
        tracker.update(
            [(x1, 10.0, x1 + 40.0, 50.0, 0.94 if class_id else 0.90, class_id)],
            timestamp,
            (80, 100),
        )

    assert len(tracker.active_tracks) == 1  # all frames associated into one track
    track = tracker.active_tracks[0]
    assert track.frame_count == 6
    assert track.is_defect is False
    assert track.winning_class_id == 0
    assert track.max_consecutive_defect_frames == 2

    finished = tracker.collect_finished(0.080 + 0.150 + 0.001)
    assert len(finished) == 1 and finished[0].is_defect is False
    scheduler = FakeScheduler()
    clock = [1.0]
    manager, records = make_manager(scheduler, clock)
    manager.handle_finished_track(finished[0])
    manager.finalize_all()
    assert scheduler.enqueued == []
    assert records[0].result == "pass"


def test_tracker_requires_persistent_dirt_on_a_moving_cap():
    tracker = CameraTracker(0, track_iou=0.3, track_timeout_s=0.150, min_defect_frames=2)
    for index, class_id in enumerate((0, 1, 1, 1)):
        x1 = 10.0 + index * 5.0
        tracker.update([(x1, 10.0, x1 + 40.0, 50.0, 0.94, class_id)], index * 0.016, (80, 100))

    track = tracker.active_tracks[0]
    assert track.is_defect is True
    assert track.max_consecutive_defect_frames == 3


def test_empty_processed_frame_breaks_defect_confirmation_streak():
    tracker = CameraTracker(0, track_iou=0.3, track_timeout_s=1.0, min_defect_frames=2)
    tracker.update([(10.0, 10.0, 50.0, 50.0, 0.94, 1)], 0.000, (80, 100))
    tracker.update([], 0.016, (80, 100))
    tracker.update([(14.0, 10.0, 54.0, 50.0, 0.94, 1)], 0.032, (80, 100))

    track = tracker.active_tracks[0]
    assert track.is_defect is False
    assert track.consecutive_defect_frames == 1


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


def test_one_frame_high_confidence_phantom_never_reaches_actuation():
    scheduler = FakeScheduler()
    clock = [1.0]
    manager, records = make_manager(scheduler, clock)
    phantom = Track(
        track_id=999,
        camera_index=0,
        first_seen=0.0,
        last_seen=0.0,
        frame_count=1,
        last_box=(10.0, 10.0, 50.0, 50.0, 0.94, 1),
        is_defect=True,
        best_defect_conf=0.94,
        first_box=(10.0, 10.0, 50.0, 50.0, 0.94, 1),
    )

    manager.handle_finished_track(phantom)

    assert scheduler.enqueued == []
    assert records == []
    assert manager.filtered_tracks == 1


def test_static_jittering_high_confidence_phantom_never_fires():
    tracker = CameraTracker(0, track_iou=0.3, track_timeout_s=0.150, min_defect_frames=2)
    for index in range(10):
        jitter = float(index % 2)
        tracker.update(
            [(100.0 + jitter, 100.0, 140.0 + jitter, 140.0, 0.94, 1)],
            index * 0.016,
            (200, 200),
        )
    track = tracker.collect_finished(0.300)[0]
    assert track.frame_count == 10 and track.is_defect is True
    assert track.travel_ratio < RuntimeConfig.defaults().min_track_travel_ratio

    scheduler = FakeScheduler()
    clock = [1.0]
    manager, _records = make_manager(scheduler, clock)
    manager.handle_finished_track(track)
    assert scheduler.enqueued == []


def test_moving_persistent_defect_qualifies_and_fires_once():
    tracker = CameraTracker(0, track_iou=0.3, track_timeout_s=0.150, min_defect_frames=2)
    for index in range(5):
        x1 = 10.0 + index * 8.0
        tracker.update([(x1, 10.0, x1 + 40.0, 50.0, 0.94, 1)], index * 0.016, (100, 100))
    track = tracker.collect_finished(0.300)[0]

    scheduler = FakeScheduler()
    clock = [1.0]
    manager, _records = make_manager(scheduler, clock)
    manager.handle_finished_track(track)
    assert len(scheduler.enqueued) == 1


def test_short_four_frame_dirt_burst_does_not_fire_at_the_line_edge():
    """Four coherent 0.94 boxes are still insufficient with only one frame downstream."""

    tracker = CameraTracker(0, track_iou=0.3, track_timeout_s=0.150)
    for index, x1 in enumerate((22.0, 30.0, 38.0, 46.0)):
        tracker.update([(x1, 10.0, x1 + 20.0, 30.0, 0.94, 1)], index * 0.016, (100, 100))
    track = tracker.collect_finished(0.300)[0]

    assert track.frame_count == 4 and track.is_defect is True
    assert track.crossed_presence_line is True
    assert track.line_negative_frames == 3 and track.line_positive_frames == 1
    scheduler = FakeScheduler()
    manager, _records = make_manager(scheduler, [1.0])
    manager.handle_finished_track(track)
    assert scheduler.enqueued == []


def test_coherent_detection_that_never_crosses_inspection_line_does_not_fire():
    tracker = CameraTracker(0, track_iou=0.3, track_timeout_s=0.150, min_defect_frames=2)
    for index, x1 in enumerate((2.0, 8.0, 14.0, 20.0)):
        tracker.update([(x1, 10.0, x1 + 12.0, 22.0, 0.94, 1)], index * 0.016, (100, 100))
    track = tracker.collect_finished(0.300)[0]
    assert track.travel_ratio > 0.35
    assert track.crossed_presence_line is False

    scheduler = FakeScheduler()
    manager, _records = make_manager(scheduler, [1.0])
    manager.handle_finished_track(track)
    assert scheduler.enqueued == []


def test_detection_moving_across_the_wrong_axis_does_not_fire():
    """A box parked over the x line must also move with the x-axis belt."""

    tracker = CameraTracker(0, track_iou=0.3, track_timeout_s=0.150, min_defect_frames=2)
    for index, y1 in enumerate((0.0, 8.0, 16.0, 24.0)):
        tracker.update([(40.0, y1, 60.0, y1 + 20.0, 0.94, 1)], index * 0.016, (100, 100))
    track = tracker.collect_finished(0.300)[0]

    assert track.crossed_presence_line is False
    assert track.travel_ratio > 0.35
    assert track.motion_directionality == pytest.approx(0.0)
    scheduler = FakeScheduler()
    manager, _records = make_manager(scheduler, [1.0])
    manager.handle_finished_track(track)
    assert scheduler.enqueued == []


def test_detection_moving_against_the_configured_belt_direction_does_not_fire():
    tracker = CameraTracker(
        0,
        track_iou=0.3,
        track_timeout_s=0.150,
        min_defect_frames=2,
        presence_direction="positive",
    )
    for index, x1 in enumerate((46.0, 38.0, 30.0, 22.0)):
        tracker.update([(x1, 10.0, x1 + 20.0, 30.0, 0.94, 1)], index * 0.016, (100, 100))
    track = tracker.collect_finished(0.300)[0]

    assert track.crossed_presence_line is True
    assert track.travel_ratio > 0.35
    assert track.motion_directionality == pytest.approx(0.0)
    scheduler = FakeScheduler()
    manager, _records = make_manager(scheduler, [1.0])
    manager.handle_finished_track(track)
    assert scheduler.enqueued == []


def test_sparse_late_observations_cannot_build_a_confirmed_cap():
    tracker = CameraTracker(
        0,
        track_iou=0.3,
        track_timeout_s=0.150,
        min_defect_frames=2,
        max_track_gap_s=0.500,
    )
    finished = []
    # 200 ms sits in the old hole: beyond the 150 ms track timeout, but below
    # the separate 500 ms observation-gap guard. These must be fresh tracks.
    for timestamp, x1 in ((0.0, 10.0), (0.2, 20.0), (0.4, 30.0)):
        tracker.update([(x1, 10.0, x1 + 40.0, 50.0, 0.94, 1)], timestamp, (100, 100))
        finished.extend(tracker.collect_finished(timestamp))
    finished.extend(tracker.flush())

    assert len(finished) == 3
    assert all(track.frame_count == 1 for track in finished)
    assert all(track.is_defect is False for track in finished)
    assert all(
        not track.qualifies_as_cap(
            min_frames=3,
            min_travel_ratio=0.35,
            min_directionality=0.6,
            max_observation_gap=0.5,
            require_line_crossing=True,
        )
        for track in finished
    )


def test_default_timeout_keeps_a_cap_sampled_every_200ms_in_one_track():
    defaults = RuntimeConfig.defaults()
    tracker = CameraTracker(
        0,
        track_iou=0.3,
        track_timeout_s=defaults.track_timeout_ms / 1000.0,
        min_defect_frames=defaults.min_defect_frames,
        min_track_frames=defaults.min_track_frames,
        max_track_gap_s=defaults.max_track_gap_ms / 1000.0,
    )
    for timestamp, x1 in (
        (0.0, 10.0),
        (0.2, 20.0),
        (0.4, 30.0),
        (0.6, 40.0),
        (0.8, 50.0),
        (1.0, 60.0),
    ):
        tracker.update([(x1, 10.0, x1 + 20.0, 30.0, 0.94, 1)], timestamp, (100, 100))
    track = tracker.collect_finished(1.251)[0]

    assert track.frame_count == 6
    assert track.largest_observation_gap_s == pytest.approx(0.2)
    assert track.is_defect is True and track.crossed_presence_line is True
    scheduler = FakeScheduler()
    manager, _records = make_manager(scheduler, [1.0], max_track_gap_ms=defaults.max_track_gap_ms)
    manager.handle_finished_track(track)
    assert len(scheduler.enqueued) == 1


def test_tracker_presence_cycles_rearm_for_distinct_caps_300ms_apart():
    tracker = CameraTracker(
        0,
        track_iou=0.3,
        track_timeout_s=0.150,
        min_defect_frames=2,
        presence_clear_s=0.150,
    )
    for timestamp, x1 in ((0.00, 10.0), (0.02, 18.0), (0.04, 26.0), (0.06, 34.0), (0.08, 42.0)):
        tracker.update([(x1, 10.0, x1 + 40.0, 50.0, 0.94, 1)], timestamp, (100, 100))
    tracker.update([], 0.24, (100, 100))
    first = tracker.collect_finished(0.24)[0]

    for timestamp, x1 in ((0.30, 10.0), (0.32, 18.0), (0.34, 26.0), (0.36, 34.0), (0.38, 42.0)):
        tracker.update([(x1, 10.0, x1 + 40.0, 50.0, 0.94, 1)], timestamp, (100, 100))
    tracker.update([], 0.54, (100, 100))
    second = tracker.collect_finished(0.54)[0]

    assert first.presence_cycle_id != second.presence_cycle_id
    scheduler = FakeScheduler()
    clock = [100.0]
    manager, _records = make_manager(scheduler, clock)
    manager.handle_finished_track(first)
    clock[0] += 0.300
    manager.handle_finished_track(second)
    assert len(scheduler.enqueued) == 2


def test_overlapping_caps_in_different_image_bands_get_distinct_cycles():
    tracker = CameraTracker(
        0,
        track_iou=0.3,
        track_timeout_s=0.150,
        presence_clear_s=0.150,
    )
    frames = (
        (0.00, [(10.0, 10.0, 30.0, 30.0, 0.94, 1)]),
        (0.04, [(20.0, 10.0, 40.0, 30.0, 0.94, 1)]),
        (0.08, [(30.0, 10.0, 50.0, 30.0, 0.94, 1)]),
        (0.12, [(40.0, 10.0, 60.0, 30.0, 0.94, 1), (10.0, 110.0, 30.0, 130.0, 0.94, 1)]),
        (0.16, [(50.0, 10.0, 70.0, 30.0, 0.94, 1), (20.0, 110.0, 40.0, 130.0, 0.94, 1)]),
        (0.20, [(60.0, 10.0, 80.0, 30.0, 0.94, 1), (30.0, 110.0, 50.0, 130.0, 0.94, 1)]),
        (0.24, [(40.0, 110.0, 60.0, 130.0, 0.94, 1)]),
        (0.28, [(50.0, 110.0, 70.0, 130.0, 0.94, 1)]),
        (0.32, [(60.0, 110.0, 80.0, 130.0, 0.94, 1)]),
        (0.36, [(70.0, 110.0, 90.0, 130.0, 0.94, 1)]),
        (0.40, [(80.0, 110.0, 100.0, 130.0, 0.94, 1)]),
        (0.44, [(90.0, 110.0, 110.0, 130.0, 0.94, 1)]),
        (0.48, [(100.0, 110.0, 120.0, 130.0, 0.94, 1)]),
    )
    for timestamp, boxes in frames:
        tracker.update(boxes, timestamp, (100, 200))
    tracks = tracker.collect_finished(0.70)

    assert len(tracks) == 2
    assert tracks[0].presence_cycle_id != tracks[1].presence_cycle_id
    assert tracks[1].last_seen - tracks[0].last_seen > 0.250
    scheduler = FakeScheduler()
    manager, _records = make_manager(scheduler, [100.0])
    for track in tracks:
        manager.handle_finished_track(track)
    assert len(scheduler.enqueued) == 2


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


def test_same_camera_distinct_spatial_cycles_are_logged_as_distinct_caps():
    scheduler = FakeScheduler()
    clock = [100.0]
    manager, records = make_manager(scheduler, clock, fire_delay_s=0.0)

    manager.handle_finished_track(
        make_track(0, last_seen=100.0, is_defect=True, defect_conf=0.9, presence_cycle_id=1)
    )
    manager.handle_finished_track(
        make_track(0, last_seen=100.020, is_defect=True, defect_conf=0.9, presence_cycle_id=2)
    )

    assert len(records) == 1  # first cap finalized when the second opened
    manager.finalize_all()
    assert len(records) == 2 and manager.caps_seen == 2
    # Their reject windows overlap, so one physical 300 ms air pulse covers
    # both; the second command is terminally suppressed by the refractory.
    assert len(scheduler.enqueued) == 1 and manager.suppressed_fires == 1


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


def test_refractory_includes_exact_boundary():
    scheduler = FakeScheduler()
    clock = [100.0]
    manager, _records = make_manager(scheduler, clock, fire_delay_s=0.0)

    manager.handle_finished_track(make_track(0, last_seen=100.0, is_defect=True, defect_conf=0.9))
    clock[0] = 100.250
    manager.handle_finished_track(make_track(1, last_seen=100.250, is_defect=True, defect_conf=0.9))

    assert len(scheduler.enqueued) == 1
    assert manager.suppressed_fires == 1


def test_refractory_boundary_tolerates_monotonic_float_rounding():
    scheduler = FakeScheduler()
    base = 1023.8972425766768
    clock = [base]
    manager, _records = make_manager(scheduler, clock, fire_delay_s=0.0)

    manager.handle_finished_track(make_track(0, last_seen=base, is_defect=True, defect_conf=0.9))
    clock[0] = base + 0.250
    assert clock[0] - base > 0.250  # the regression depends on binary-float rounding
    manager.handle_finished_track(make_track(1, last_seen=clock[0], is_defect=True, defect_conf=0.9))

    assert len(scheduler.enqueued) == 1
    assert manager.suppressed_fires == 1


def test_merge_boundary_tolerates_monotonic_float_rounding():
    scheduler = FakeScheduler()
    base = 1_000_000.123456789
    clock = [base]
    manager, records = make_manager(scheduler, clock, fire_delay_s=0.0)

    manager.handle_finished_track(make_track(0, last_seen=base, is_defect=True, defect_conf=0.9))
    clock[0] = base + 0.150
    assert clock[0] - base > 0.150
    manager.handle_finished_track(make_track(1, last_seen=clock[0], is_defect=True, defect_conf=0.9))
    manager.finalize_all()

    assert len(scheduler.enqueued) == 1
    assert len(records) == 1 and records[0].cameras == [0, 1]


def test_refractory_suppression_is_terminal_for_later_merged_fragments():
    scheduler = FakeScheduler()
    clock = [100.0]
    manager, _records = make_manager(scheduler, clock, fire_delay_s=0.0)

    manager.handle_finished_track(make_track(0, last_seen=100.0, is_defect=True, defect_conf=0.9))
    # New event outside merge window but inside refractory: terminally suppressed.
    clock[0] = 100.200
    manager.handle_finished_track(
        make_track(1, last_seen=100.200, is_defect=True, defect_conf=0.9, presence_cycle_id=8)
    )
    # This fragment merges into the suppressed event after the original
    # refractory would otherwise have elapsed. It must never retry.
    clock[0] = 100.340
    manager.handle_finished_track(
        make_track(1, last_seen=100.340, is_defect=True, defect_conf=0.9, presence_cycle_id=8)
    )

    assert len(scheduler.enqueued) == 1
    assert manager.suppressed_fires == 1


def test_reentrant_dedup_logger_cannot_deadlock_decision_manager():
    scheduler = FakeScheduler()
    clock = [100.0]
    records: list[CapEventRecord] = []
    manager = None
    reentered = threading.Event()

    def reentrant_logger(message, *_args, **_kwargs):
        if "[DEDUP]" in str(message) and not reentered.is_set():
            reentered.set()
            manager.flush_expired(clock[0])

    manager = CapEventManager(
        replace(RuntimeConfig.defaults(), fire_delay_s=0.0),
        scheduler=scheduler,
        time_fn=lambda: clock[0],
        history_callback=records.append,
        log_fn=reentrant_logger,
    )
    manager.handle_finished_track(make_track(0, last_seen=100.0, is_defect=True, defect_conf=0.9))
    clock[0] = 100.200
    worker = threading.Thread(
        target=lambda: manager.handle_finished_track(
            make_track(1, last_seen=100.200, is_defect=True, defect_conf=0.9)
        ),
        daemon=True,
    )
    worker.start()
    worker.join(timeout=1.0)

    assert not worker.is_alive()
    assert reentered.is_set()
    assert len(scheduler.enqueued) == 1


def test_presence_cycle_is_an_idempotency_key_across_far_fragments():
    scheduler = FakeScheduler()
    clock = [100.0]
    manager, _records = make_manager(scheduler, clock, fire_delay_s=0.0)

    manager.handle_finished_track(
        make_track(0, last_seen=100.0, is_defect=True, defect_conf=0.9, presence_cycle_id=7)
    )
    clock[0] = 101.0  # well beyond both merge and refractory windows
    manager.handle_finished_track(
        make_track(0, last_seen=101.0, is_defect=True, defect_conf=0.94, presence_cycle_id=7)
    )

    assert len(scheduler.enqueued) == 1


def test_finalized_clean_cycle_cannot_rearm_as_a_late_dirty_fragment():
    scheduler = FakeScheduler()
    clock = [100.0]
    manager, records = make_manager(scheduler, clock, fire_delay_s=0.0)

    manager.handle_finished_track(
        make_track(0, last_seen=100.0, is_defect=False, undef_conf=0.95, presence_cycle_id=7)
    )
    clock[0] = 101.0
    manager.handle_finished_track(
        make_track(0, last_seen=101.0, is_defect=True, defect_conf=0.94, presence_cycle_id=7)
    )

    assert records[0].result == "pass"
    assert scheduler.enqueued == []
    assert manager.suppressed_fires == 1


def test_finalized_clean_cap_blocks_late_dirty_fragment_from_other_camera():
    scheduler = FakeScheduler()
    clock = [100.0]
    manager, records = make_manager(scheduler, clock, fire_delay_s=0.0)

    manager.handle_finished_track(
        make_track(0, last_seen=100.0, is_defect=False, undef_conf=0.95, presence_cycle_id=7)
    )
    clock[0] = 100.401
    manager.flush_expired(clock[0])
    assert records[0].result == "pass"

    # Camera 1 reports the same physical exit after finalization.  Its local
    # cycle id cannot match camera 0's, so timestamp tombstoning is required.
    manager.handle_finished_track(
        make_track(1, last_seen=100.020, is_defect=True, defect_conf=0.94, presence_cycle_id=9)
    )

    assert scheduler.enqueued == []
    assert manager.suppressed_fires == 1


def test_late_fragment_matches_finalized_cap_before_newer_open_cap():
    scheduler = FakeScheduler()
    clock = [100.0]
    manager, records = make_manager(scheduler, clock, fire_delay_s=0.0)

    manager.handle_finished_track(
        make_track(0, last_seen=100.0, is_defect=False, undef_conf=0.95, presence_cycle_id=1)
    )
    # A newer same-camera spatial cycle finalizes A and opens B.
    manager.handle_finished_track(
        make_track(0, last_seen=100.100, is_defect=False, undef_conf=0.95, presence_cycle_id=2)
    )
    assert len(records) == 1

    # Late camera-1 evidence for A must not be merged into open cap B or fire.
    manager.handle_finished_track(
        make_track(1, last_seen=100.020, is_defect=True, defect_conf=0.94, presence_cycle_id=3)
    )
    manager.finalize_all()

    assert scheduler.enqueued == []
    assert len(records) == 2
    assert records[1].result == "pass" and records[1].cameras == [0]


def test_closer_open_cap_wins_over_overlapping_finalized_tombstone():
    scheduler = FakeScheduler()
    clock = [100.0]
    manager, records = make_manager(scheduler, clock, fire_delay_s=0.0)

    manager.handle_finished_track(
        make_track(0, last_seen=100.0, is_defect=False, undef_conf=0.95, presence_cycle_id=1)
    )
    manager.handle_finished_track(
        make_track(0, last_seen=100.100, is_defect=False, undef_conf=0.95, presence_cycle_id=2)
    )
    # Camera 1's exit is 10 ms from open B but 90 ms from finalized A.
    manager.handle_finished_track(
        make_track(1, last_seen=100.090, is_defect=True, defect_conf=0.94, presence_cycle_id=3)
    )
    manager.finalize_all()

    assert len(scheduler.enqueued) == 1
    assert len(records) == 2
    assert records[1].result == "reject" and records[1].cameras == [0, 1]


def test_late_tombstone_fragment_records_camera_cycle_membership():
    scheduler = FakeScheduler()
    clock = [100.0]
    manager, _records = make_manager(scheduler, clock, fire_delay_s=0.0)

    manager.handle_finished_track(
        make_track(0, last_seen=100.0, is_defect=False, undef_conf=0.95, presence_cycle_id=1)
    )
    clock[0] = 100.401
    manager.flush_expired(clock[0])
    # Late camera-1 fragment belongs to A and teaches its tombstone camera 1's
    # local cycle. B is a different camera-1 spatial cycle and must escape it.
    manager.handle_finished_track(
        make_track(1, last_seen=100.020, is_defect=False, undef_conf=0.95, presence_cycle_id=2)
    )
    manager.handle_finished_track(
        make_track(1, last_seen=100.100, is_defect=True, defect_conf=0.94, presence_cycle_id=3)
    )

    assert len(scheduler.enqueued) == 1


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


def test_overlapping_same_frame_rows_are_deduplicated():
    boxes = deduplicate_boxes(
        [
            [10.0, 10.0, 50.0, 50.0, 0.92, 0],
            [11.0, 11.0, 51.0, 51.0, 0.94, 1],
            [80.0, 10.0, 120.0, 50.0, 0.88, 0],
        ],
        iou_threshold=0.65,
    )

    assert len(boxes) == 2
    assert boxes[0][4:] == [0.94, 1]


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


def test_legacy_150ms_timeout_without_gap_field_stays_valid():
    data = RuntimeConfig.defaults().to_json_dict()
    data["track_timeout_ms"] = 150.0
    del data["max_track_gap_ms"]

    config = RuntimeConfig.from_json_dict(data)

    assert config.track_timeout_ms == pytest.approx(150.0)
    assert config.max_track_gap_ms == pytest.approx(150.0)
    validate_config(config)


def test_validate_config_rejects_out_of_range_values():
    with pytest.raises(ValueError):
        validate_config(replace(RuntimeConfig.defaults(), reject_threshold=1.5))
    with pytest.raises(ValueError):
        validate_config(replace(RuntimeConfig.defaults(), min_fire_interval_ms=-1.0))
    with pytest.raises(ValueError):
        validate_config(replace(RuntimeConfig.defaults(), gpio_backend="esp32"))
    with pytest.raises(ValueError):
        validate_config(replace(RuntimeConfig.defaults(), track_timeout_ms=100.0, max_track_gap_ms=101.0))


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


def test_reject_scheduler_coalesces_a_burst_to_one_physical_pulse():
    on_calls: list[float] = []
    first_on = threading.Event()

    class RecordingPin:
        backend_name = "recording"

        def __init__(self, pin):
            self.pin = pin

        def on(self):
            on_calls.append(time.monotonic())
            first_on.set()

        def off(self):
            return None

        def close(self):
            return None

    scheduler = RejectScheduler(
        trigger_pin=7,
        trigger_duration=0.03,
        trigger_min_gap=0.0,
        max_queue_age=5.0,
        max_lateness=5.0,
        pin_factory=RecordingPin,
        log_fn=lambda *args, **kwargs: None,
    )
    due = time.monotonic()
    for event_id in range(10):
        scheduler.enqueue(event_id + 1, due)
    assert first_on.wait(1.0)
    time.sleep(0.08)
    scheduler.close()

    assert len(on_calls) == 1
    assert scheduler.coalesced_fires + scheduler.stale_drops == 9


def test_reject_scheduler_drops_newly_enqueued_but_old_target():
    events: list[str] = []

    class RecordingPin:
        backend_name = "recording"

        def __init__(self, pin):
            self.pin = pin

        def on(self):
            events.append("on")

        def off(self):
            return None

        def close(self):
            return None

    scheduler = RejectScheduler(
        trigger_pin=7,
        trigger_duration=0.001,
        trigger_min_gap=0.0,
        max_lateness=0.050,
        pin_factory=RecordingPin,
        log_fn=lambda *args, **kwargs: None,
    )
    scheduler.enqueue(1, time.monotonic() - 2.0)
    deadline = time.monotonic() + 1.0
    while scheduler.stale_drops == 0 and time.monotonic() < deadline:
        time.sleep(0.005)
    scheduler.close()

    assert events == []
    assert scheduler.stale_drops == 1


def test_reject_scheduler_close_cancels_pending_future_fire():
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
    scheduler.enqueue(1, time.monotonic() + 1.0)
    scheduler.close()

    assert "on" not in events
    assert scheduler.cancelled_on_close == 1
    assert events[-2:] == ["off", "close"]
    with pytest.raises(RuntimeError):
        scheduler.enqueue(2, time.monotonic())


def test_reject_scheduler_close_stays_bounded_when_logger_hangs():
    logger_entered = threading.Event()
    release_logger = threading.Event()

    class RecordingPin:
        backend_name = "recording"

        def __init__(self, pin):
            self.pin = pin

        def on(self):
            return None

        def off(self):
            return None

        def close(self):
            return None

    def blocking_logger(*_args, **_kwargs):
        logger_entered.set()
        release_logger.wait(2.0)

    scheduler = RejectScheduler(
        trigger_pin=7,
        trigger_duration=0.001,
        trigger_min_gap=0.0,
        max_lateness=0.001,
        pin_factory=RecordingPin,
        log_fn=blocking_logger,
    )
    scheduler._close_join_timeout_s = 0.05
    scheduler.enqueue(1, time.monotonic() - 1.0)
    assert logger_entered.wait(1.0)

    started = time.monotonic()
    assert scheduler.close() is False
    assert time.monotonic() - started < 0.250
    assert scheduler.fatal_error is not None

    release_logger.set()
    scheduler._thread.join(timeout=1.0)
    assert not scheduler._thread.is_alive()


def test_reject_scheduler_latches_an_unexpected_loop_crash_as_fatal():
    clock_calls = 0

    def crashing_clock():
        nonlocal clock_calls
        clock_calls += 1
        if clock_calls >= 2:
            raise RuntimeError("clock failed")
        return 100.0

    class RecordingPin:
        backend_name = "recording"

        def __init__(self, pin):
            self.pin = pin

        def on(self):
            raise AssertionError("crashed scheduler must not energize")

        def off(self):
            return None

        def close(self):
            return None

    scheduler = RejectScheduler(
        trigger_pin=7,
        trigger_duration=0.001,
        trigger_min_gap=0.0,
        pin_factory=RecordingPin,
        time_fn=crashing_clock,
        log_fn=lambda *_args, **_kwargs: None,
    )
    scheduler.enqueue(1, 100.0)

    scheduler._thread.join(timeout=1.0)
    assert not scheduler._thread.is_alive()
    assert scheduler.fatal_error == "reject scheduler crashed with RuntimeError"
    assert scheduler.close() is False


def test_reject_scheduler_marks_slow_gpio_activation_as_fatal_and_turns_off():
    events: list[str] = []
    callback_called = threading.Event()

    class SlowPin:
        backend_name = "slow"

        def __init__(self, pin):
            self.pin = pin

        def on(self):
            time.sleep(0.080)
            events.append("on")

        def off(self):
            events.append("off")

        def close(self):
            events.append("close")

    scheduler = RejectScheduler(
        trigger_pin=7,
        trigger_duration=0.050,
        trigger_min_gap=0.0,
        max_lateness=0.030,
        max_queue_age=1.0,
        pin_factory=SlowPin,
        log_fn=lambda *_args, **_kwargs: None,
    )
    scheduler.enqueue(1, time.monotonic(), completion_callback=lambda _execution: callback_called.set())
    scheduler._thread.join(timeout=1.0)

    assert not scheduler._thread.is_alive()
    assert events[:2] == ["on", "off"]
    assert callback_called.is_set() is False
    assert scheduler.stale_drops == 1
    assert scheduler.fatal_error is not None
    assert scheduler.close() is False


def test_reject_scheduler_surfaces_activation_that_finishes_after_close_request():
    entered_on = threading.Event()
    release_on = threading.Event()
    events: list[str] = []

    class PausedPin:
        backend_name = "paused"

        def __init__(self, pin):
            self.pin = pin

        def on(self):
            entered_on.set()
            release_on.wait(1.0)
            events.append("on")

        def off(self):
            events.append("off")

        def close(self):
            events.append("close")

    scheduler = RejectScheduler(
        trigger_pin=7,
        trigger_duration=0.050,
        trigger_min_gap=0.0,
        max_lateness=1.0,
        max_queue_age=1.0,
        pin_factory=PausedPin,
        log_fn=lambda *_args, **_kwargs: None,
    )
    scheduler.enqueue(1, time.monotonic())
    assert entered_on.wait(1.0)
    closer_result: list[bool] = []
    closer = threading.Thread(target=lambda: closer_result.append(scheduler.close()))
    closer.start()
    assert scheduler._close_requested.wait(1.0)
    release_on.set()
    closer.join(timeout=1.0)

    assert not closer.is_alive()
    assert closer_result == [False]
    assert events[0:2] == ["on", "off"]
    assert scheduler.fatal_error is not None


def test_reject_scheduler_shared_stop_event_rejects_new_work():
    events: list[str] = []
    stop_event = threading.Event()

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
        cancel_event=stop_event,
        log_fn=lambda *args, **kwargs: None,
    )
    stop_event.set()
    with pytest.raises(RuntimeError):
        scheduler.enqueue(1, time.monotonic())
    scheduler.close()

    assert "on" not in events


def test_reject_scheduler_close_interrupts_an_active_pulse():
    events: list[str] = []
    turned_on = threading.Event()

    class RecordingPin:
        backend_name = "recording"

        def __init__(self, pin):
            self.pin = pin

        def on(self):
            events.append("on")
            turned_on.set()

        def off(self):
            events.append("off")

        def close(self):
            events.append("close")

    scheduler = RejectScheduler(
        trigger_pin=7,
        trigger_duration=10.0,
        trigger_min_gap=0.0,
        pin_factory=RecordingPin,
        log_fn=lambda *args, **kwargs: None,
    )
    scheduler.enqueue(1, time.monotonic())
    assert turned_on.wait(1.0)
    started = time.monotonic()
    scheduler.close()

    assert time.monotonic() - started < 1.0
    assert events[0] == "on"
    assert "off" in events and events[-1] == "close"


def test_reject_scheduler_shared_stop_interrupts_an_active_pulse():
    stop_event = threading.Event()
    turned_on = threading.Event()
    turned_off = threading.Event()

    class RecordingPin:
        backend_name = "recording"

        def __init__(self, pin):
            self.pin = pin

        def on(self):
            turned_on.set()

        def off(self):
            turned_off.set()

        def close(self):
            return None

    scheduler = RejectScheduler(
        trigger_pin=7,
        trigger_duration=1.0,
        trigger_min_gap=0.0,
        pin_factory=RecordingPin,
        cancel_event=stop_event,
        log_fn=lambda *args, **kwargs: None,
    )
    scheduler.enqueue(1, time.monotonic())
    assert turned_on.wait(1.0)
    turned_off.clear()  # ignore constructor/fail-safe state from no-op fakes
    stop_event.set()

    assert turned_off.wait(0.200)
    scheduler.close()


def test_reject_scheduler_close_wins_after_due_job_is_popped_but_before_pin_on():
    events: list[str] = []
    popped = threading.Event()
    release_time = threading.Event()
    calls = [0]

    def controlled_time():
        calls[0] += 1
        # enqueue=call 1, heap due check=call 2, post-pop lateness=call 3
        if calls[0] == 3:
            popped.set()
            release_time.wait(1.0)
        return 100.0

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
        time_fn=controlled_time,
        log_fn=lambda *args, **kwargs: None,
    )
    scheduler.enqueue(1, 100.0)
    assert popped.wait(1.0)
    closer = threading.Thread(target=scheduler.close)
    closer.start()
    deadline = time.monotonic() + 1.0
    while not scheduler._closed and time.monotonic() < deadline:
        time.sleep(0.001)
    release_time.set()
    closer.join(timeout=1.0)

    assert not closer.is_alive()
    assert "on" not in events


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


class _ScriptThenFailureCamera(_ScriptedCamera):
    def read(self):
        with self._lock:
            if self._index >= len(self._scripted):
                return False, None
            detections = self._scripted[self._index]
            self._index += 1
        return True, _ScriptedFrame(detections)


class _FakeInput:
    name = "images"
    shape = [1, 3, 100, 100]


class _FakeSession:
    def get_inputs(self):
        return [_FakeInput()]

    def run(self, _outputs, inputs):
        frame = next(iter(inputs.values()))
        return [frame.detections]


def test_camera_worker_join_reports_a_thread_that_survived_timeout():
    class FakeThread:
        def __init__(self, alive):
            self.alive = alive
            self.join_calls: list[float | None] = []

        def join(self, timeout=None):
            self.join_calls.append(timeout)

        def is_alive(self):
            return self.alive

    worker = object.__new__(CameraWorker)
    worker._thread = FakeThread(False)
    worker._capture_thread = FakeThread(True)

    assert worker.join(timeout=0.010) is False
    worker._capture_thread.alive = False
    assert worker.join(timeout=0.010) is True


def test_camera_worker_entry_latches_unexpected_thread_crash_as_fatal():
    worker = object.__new__(CameraWorker)
    worker.camera_index = 1
    worker.stop_event = threading.Event()
    worker._fatal_lock = threading.Lock()
    worker.fatal_error = None

    def crash():
        raise TypeError("malformed detector row")

    worker._run = crash
    worker._worker_entry()

    assert worker.stop_event.is_set()
    assert worker.fatal_error == "camera 1 inference thread crashed with TypeError"


def test_camera_worker_uses_capture_time_for_track_timeout_after_slow_inference():
    stop_event = threading.Event()
    collected_at: list[float] = []
    calls: list[str] = []

    class SpyTracker:
        def update(self, boxes, timestamp, frame_size=None):
            calls.append("update")
            assert boxes and timestamp == pytest.approx(12.0)
            assert frame_size == (100, 100)

        def collect_finished(self, now):
            calls.append("collect")
            collected_at.append(float(now))
            stop_event.set()
            return []

    class Manager:
        def handle_finished_track(self, track):
            raise AssertionError("no track should finish on its just-processed frame")

    frame = _ScriptedFrame([[10.0, 10.0, 50.0, 50.0, 0.94, 1]])
    worker = CameraWorker(
        camera_index=0,
        camera=None,
        session=_FakeSession(),
        input_name="images",
        model_imgsz=100,
        reject_threshold=0.45,
        duplicate_iou_threshold=0.65,
        max_frame_age_s=1000.0,
        mirror_horizontal=False,
        tracker=SpyTracker(),
        manager=Manager(),
        shared=SharedRuntimeState(1),
        preprocess_fn=lambda value, _imgsz: (value, {}),
        postprocess_fn=lambda output, _meta, conf_threshold: output,
        stop_event=stop_event,
        time_fn=lambda: 99.0,  # simulates inference completing long after capture
        sleep_fn=lambda _seconds: None,
        log_fn=lambda *args, **kwargs: None,
    )
    worker._latest_capture = CapturedFrame(0, frame, 12.0, 1)

    worker._run()

    assert collected_at == [12.0]
    assert calls == ["collect", "update"]


def test_camera_worker_drops_an_inflight_result_that_became_stale():
    stop_event = threading.Event()
    logs: list[str] = []

    class NoUpdateTracker:
        def update(self, boxes, timestamp, frame_size=None):
            raise AssertionError("a stale inference result must not update tracking")

        def collect_finished(self, now):
            return []

    class Manager:
        def handle_finished_track(self, track):
            raise AssertionError("a stale inference result must not reach decisions")

    def log(message, *args, **kwargs):
        logs.append(str(message))
        if "[STALE]" in str(message):
            stop_event.set()

    frame = _ScriptedFrame([[10.0, 10.0, 50.0, 50.0, 0.94, 1]])
    worker = CameraWorker(
        camera_index=0,
        camera=None,
        session=_FakeSession(),
        input_name="images",
        model_imgsz=100,
        reject_threshold=0.45,
        duplicate_iou_threshold=0.65,
        max_frame_age_s=0.100,
        mirror_horizontal=False,
        tracker=NoUpdateTracker(),
        manager=Manager(),
        shared=SharedRuntimeState(1),
        preprocess_fn=lambda value, _imgsz: (value, {}),
        postprocess_fn=lambda output, _meta, conf_threshold: output,
        stop_event=stop_event,
        time_fn=lambda: 10.0,
        sleep_fn=lambda _seconds: None,
        log_fn=log,
    )
    worker._latest_capture = CapturedFrame(0, frame, 9.0, 1)

    worker._run()

    assert any("[STALE]" in line for line in logs)


def test_run_detection_fires_once_for_defect_cap():
    from cap_line_v6.runtime import run_detection
    from cap_line_v6.types import RuntimeCallbacks

    defect_sequence = [
        [[10.0 + offset, 10.0, 50.0 + offset, 50.0, 0.90, 1]]
        for offset in (0.0, 6.0, 12.0, 18.0, 24.0, 30.0)
    ]
    clean_sequence = [
        [[10.0 + offset, 10.0, 50.0 + offset, 50.0, 0.80, 0]]
        for offset in (0.0, 6.0, 12.0, 18.0, 24.0, 30.0)
    ]
    cameras = [
        _ScriptedCamera(defect_sequence),  # camera 0 catches the dirt
        _ScriptedCamera(clean_sequence),  # camera 1 sees the same cap clean
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
        max_track_gap_ms=20.0,
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


def test_runtime_stop_discards_valid_mid_track_instead_of_firing_on_shutdown():
    from cap_line_v6.runtime import run_detection
    from cap_line_v6.types import RuntimeCallbacks

    moving_defect_frames = []
    for offset in (0.0, 5.0, 10.0, 15.0, 20.0):
        moving_defect_frames.extend(
            [[[10.0 + offset, 10.0, 50.0 + offset, 50.0, 0.94, 1]]] * 4
        )
    cameras = [_ScriptThenFailureCamera(moving_defect_frames), _ScriptThenFailureCamera([])]
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

    config = replace(
        RuntimeConfig.defaults(),
        cameras=("0", "1"),
        track_timeout_ms=5000.0,
        trigger_duration=0.001,
        live_preview_fps=0.0,
        no_display=True,
    )
    stop_event = threading.Event()
    thread = threading.Thread(
        target=run_detection,
        args=(config, RuntimeCallbacks(log_fn=lambda *args, **kwargs: None), stop_event),
        kwargs=dict(
            pin_factory=RecordingPin,
            camera_factory=lambda index, _source, _config: cameras[index],
            session_factory=lambda _model_path, _threads: _FakeSession(),
            preprocess_fn=lambda frame, _imgsz: (frame, {}),
            postprocess_fn=lambda output, _meta, conf_threshold: output,
        ),
        daemon=True,
    )
    thread.start()
    time.sleep(0.10)
    stop_event.set()
    thread.join(timeout=3.0)

    assert not thread.is_alive()
    assert fires == []


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
