"""Tests for the v7 two-stage cap-inspection runtime.

Focus is the v7 delta over the (fully tested) v6 base:

- probability-based track voting: consecutive dirty frames AND trimmed-mean
  P(dirt), so isolated hallucinated frames can no longer condemn a clean cap;
- the crop/classify path (training-compatible crop geometry, plain RGB/255
  preprocessing, softmax-tolerant decode, dirt index 0);
- two-stage wiring through CameraWorker/run_detection with an injectable
  classify_fn;
- config defaults (Jetson pin 7, two-stage model names) and legacy settings
  migration (a v6 settings file must not smuggle its single-stage model in as
  the cap detector).

Decision/scheduler/actuation logic is byte-ported from v6 and keeps its
coverage there; representative cross-camera merge tests are re-run here
against the v7 Track shape.
"""

from __future__ import annotations

import threading
import time
from dataclasses import replace
from itertools import count

import numpy as np
import pytest

from cap_line_v7.config import RuntimeConfig, validate_config
from cap_line_v7.decision import CapEventManager
from cap_line_v7.model import (
    box_in_classify_band,
    classifier_postprocess,
    crop_cap_region,
    deduplicate_boxes,
    postprocess,
)
from cap_line_v7.runtime import CameraWorker, SharedRuntimeState, resolve_pin_factory
from cap_line_v7.tracking import CameraTracker, Track, trimmed_mean
from cap_line_v7.types import CapEventRecord, CapturedFrame
from gpio_output import GPIOOutputPin


# --------------------------------------------------------------------------- #
# Helpers / fakes
# --------------------------------------------------------------------------- #

_TRACK_IDS = count(1)


def make_track(
    camera_index: int,
    *,
    last_seen: float,
    dirt_probabilities: list[float],
    first_seen: float | None = None,
    presence_cycle_id: int | None = None,
    frame_dirt_threshold: float = 0.50,
    track_dirt_threshold: float = 0.45,
    min_defect_frames: int = 3,
) -> Track:
    """A qualified (moving, line-crossing) track with scripted P(dirt) history."""

    track_id = next(_TRACK_IDS)
    dirty = [p for p in dirt_probabilities if p >= frame_dirt_threshold]
    clean = [p for p in dirt_probabilities if p < frame_dirt_threshold]
    consecutive = 0
    max_consecutive = 0
    for probability in dirt_probabilities:
        consecutive = consecutive + 1 if probability >= frame_dirt_threshold else 0
        max_consecutive = max(max_consecutive, consecutive)
    return Track(
        track_id=track_id,
        camera_index=camera_index,
        first_seen=last_seen - 0.100 if first_seen is None else first_seen,
        last_seen=last_seen,
        frame_count=max(4, len(dirt_probabilities)),
        last_box=(0.0, 0.0, 10.0, 10.0, 0.9, 0),
        best_defect_conf=max(dirty, default=0.0),
        best_undefected_conf=max((1.0 - p for p in clean), default=0.0),
        first_box=(-10.0, 0.0, 0.0, 10.0, 0.9, 0),
        path_length_px=10.0,
        dirt_probabilities=list(dirt_probabilities),
        frame_dirt_threshold=frame_dirt_threshold,
        track_dirt_threshold=track_dirt_threshold,
        defect_frame_count=len(dirty),
        undefected_frame_count=len(clean),
        consecutive_defect_frames=consecutive,
        max_consecutive_defect_frames=max_consecutive,
        min_defect_frames=min_defect_frames,
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


def run_tracker(tracker: CameraTracker, observations, *, frame_size=(1000, 600)):
    """Feed (timestamp, x_offset, p_dirt) rows as one moving cap."""

    for timestamp, x_offset, p_dirt in observations:
        boxes = [(300.0 + x_offset, 100.0, 500.0 + x_offset, 300.0, 0.9, 0)]
        tracker.update(boxes, timestamp, frame_size, [p_dirt])
    return tracker.active_tracks


# --------------------------------------------------------------------------- #
# 1. trimmed_mean + Track probability voting
# --------------------------------------------------------------------------- #

def test_trimmed_mean_drops_extremes_for_long_inputs():
    values = [0.05] * 9 + [0.99]
    assert trimmed_mean(values) < 0.10  # the 0.99 outlier is trimmed
    assert trimmed_mean([]) == 0.0
    assert trimmed_mean([0.4]) == pytest.approx(0.4)


def test_two_isolated_hallucinated_frames_do_not_condemn_a_clean_cap():
    track = make_track(0, last_seen=1.0, dirt_probabilities=[0.05, 0.95, 0.04, 0.97, 0.06, 0.05])
    assert track.max_consecutive_defect_frames == 1
    assert track.is_defect is False


def test_sustained_dirt_with_high_mean_is_a_defect():
    track = make_track(0, last_seen=1.0, dirt_probabilities=[0.85, 0.92, 0.88, 0.90, 0.75])
    assert track.is_defect is True
    assert track.best_defect_conf == pytest.approx(0.92)


def test_consecutive_gate_alone_is_not_enough_when_track_mean_is_low():
    # Three marginally-dirty frames inside a long clean track: the consecutive
    # gate passes but the trimmed-mean gate holds the verdict at PASS.
    probabilities = [0.05] * 10 + [0.55, 0.56, 0.57] + [0.05] * 10
    track = make_track(0, last_seen=1.0, dirt_probabilities=probabilities)
    assert track.max_consecutive_defect_frames >= 3
    assert track.dirt_score < 0.45
    assert track.is_defect is False


def test_verdict_is_not_a_latch_more_clean_evidence_can_flip_it_back():
    track = make_track(0, last_seen=1.0, dirt_probabilities=[0.9, 0.9, 0.9])
    assert track.is_defect is True
    for probability in [0.02] * 30:
        track.observe((0.0, 0.0, 10.0, 10.0, 0.9, 0), track.last_seen + 0.016, p_dirt=probability)
    assert track.is_defect is False  # trimmed mean collapsed under clean evidence


def test_unclassified_observation_breaks_the_dirt_streak_and_adds_no_evidence():
    track = make_track(0, last_seen=1.0, dirt_probabilities=[0.9, 0.9])
    track.observe((0.0, 0.0, 10.0, 10.0, 0.9, 0), 1.016, p_dirt=None)
    assert track.consecutive_defect_frames == 0
    assert len(track.dirt_probabilities) == 2
    track.observe((0.0, 0.0, 10.0, 10.0, 0.9, 0), 1.032, p_dirt=0.9)
    assert track.max_consecutive_defect_frames == 2  # the streak restarted


# --------------------------------------------------------------------------- #
# 2. Tracker integration with per-box probabilities
# --------------------------------------------------------------------------- #

def test_tracker_threads_probabilities_into_one_track():
    tracker = CameraTracker(
        0, track_iou=0.3, track_timeout_s=0.150,
        frame_dirt_threshold=0.5, track_dirt_threshold=0.45,
    )
    tracks = run_tracker(
        tracker,
        [
            (0.000, 0.0, 0.10),
            (0.016, 6.0, 0.88),
            (0.032, 12.0, 0.91),
            (0.048, 18.0, 0.87),
            (0.064, 24.0, 0.09),
        ],
    )
    assert len(tracks) == 1
    track = tracks[0]
    assert track.dirt_probabilities == pytest.approx([0.10, 0.88, 0.91, 0.87, 0.09])
    assert track.max_consecutive_defect_frames == 3
    assert track.is_defect is True


def test_tracker_clean_cap_with_single_spike_stays_clean():
    tracker = CameraTracker(
        0, track_iou=0.3, track_timeout_s=0.150,
        frame_dirt_threshold=0.5, track_dirt_threshold=0.45,
    )
    observations = [(i * 0.016, i * 6.0, 0.05) for i in range(10)]
    observations[4] = (4 * 0.016, 24.0, 0.99)  # one hallucinated frame
    tracks = run_tracker(tracker, observations)
    assert len(tracks) == 1
    assert tracks[0].is_defect is False


def test_tracker_defaults_missing_probability_list_to_none():
    tracker = CameraTracker(0, track_iou=0.3, track_timeout_s=0.150)
    tracker.update([(0.0, 0.0, 10.0, 10.0, 0.9, 0)], 0.0, (100, 100))  # no dirt_probs arg
    (track,) = tracker.active_tracks
    assert track.dirt_probabilities == []
    assert track.is_defect is False


# --------------------------------------------------------------------------- #
# 3. Model helpers: crop, classifier decode, detector postprocess, dedup
# --------------------------------------------------------------------------- #

def test_crop_cap_region_is_square_and_contains_the_margin():
    frame = np.zeros((600, 960, 3), dtype=np.uint8)
    frame[100:300, 400:600] = 255  # a 200x200 white "cap"
    crop = crop_cap_region(frame, (400.0, 100.0, 600.0, 300.0), margin=0.10)
    assert crop is not None
    height, width = crop.shape[:2]
    assert height == width  # square-padded
    assert height == pytest.approx(220, abs=2)  # 200 px box + 10% margin
    assert crop.max() == 255


def test_crop_cap_region_rejects_degenerate_boxes():
    frame = np.zeros((600, 960, 3), dtype=np.uint8)
    assert crop_cap_region(frame, (10.0, 10.0, 12.0, 12.0)) is None


def test_classify_band_gates_edge_boxes():
    frame_size = (960, 600)
    center_box = (400.0, 100.0, 560.0, 300.0)   # center x = 480
    edge_box = (820.0, 100.0, 950.0, 300.0)     # center x = 885 (entry zone)
    assert box_in_classify_band(center_box, frame_size, axis="x", band_ratio=0.60) is True
    assert box_in_classify_band(edge_box, frame_size, axis="x", band_ratio=0.60) is False
    assert box_in_classify_band(edge_box, frame_size, axis="x", band_ratio=1.0) is True
    # y-axis belts gate on the vertical center instead
    assert box_in_classify_band((0.0, 500.0, 100.0, 590.0), frame_size, axis="y", band_ratio=0.60) is False


def test_classifier_postprocess_reads_dirt_from_index_zero():
    assert classifier_postprocess(np.asarray([[0.93, 0.07]])) == pytest.approx(0.93)


def test_classifier_postprocess_applies_softmax_to_raw_logits():
    p_dirt = classifier_postprocess(np.asarray([[4.0, -4.0]]))
    assert 0.999 <= p_dirt <= 1.0


def test_postprocess_normalizes_every_detection_to_class_cap():
    meta = {"scale": 1.0, "pad_left": 0, "pad_top": 0, "frame_shape": (600, 960, 3), "img_size": 640}
    output = np.asarray([
        [10.0, 10.0, 50.0, 50.0, 0.90, 1.0],  # legacy 2-class model dirt row
        [200.0, 10.0, 260.0, 50.0, 0.80, 0.0],
    ], dtype=np.float32)
    boxes = postprocess(output, meta, conf_threshold=0.25)
    assert len(boxes) == 2
    assert all(box[5] == 0 for box in boxes)


def test_deduplicate_boxes_preserves_extra_scripted_fields():
    boxes = [
        [10.0, 10.0, 50.0, 50.0, 0.90, 0, 0.88],
        [11.0, 11.0, 51.0, 51.0, 0.70, 0, 0.10],  # same cap, lower conf
    ]
    unique = deduplicate_boxes(boxes, iou_threshold=0.65)
    assert len(unique) == 1
    assert unique[0][6] == pytest.approx(0.88)


# --------------------------------------------------------------------------- #
# 4. Config: defaults, validation, legacy migration
# --------------------------------------------------------------------------- #

def test_v7_defaults_target_the_jetson_with_two_stage_models():
    defaults = RuntimeConfig.defaults()
    assert defaults.model == "cap_detector_640.onnx"
    assert defaults.classifier_model == "dirt_classifier_384.onnx"
    assert defaults.gpio_backend == "jetson"
    assert defaults.trigger_pin == 7
    assert resolve_pin_factory(defaults) is GPIOOutputPin
    validate_config(defaults)


def test_config_json_round_trip():
    config = RuntimeConfig.defaults()
    assert RuntimeConfig.from_json_dict(config.to_json_dict()) == config


def test_legacy_v6_settings_cannot_smuggle_a_single_stage_model_in():
    legacy = {
        "model": "dirtv7.onnx",
        "imgsz": 960,
        "reject_threshold": 0.45,
        "global_cooldown_ms": 90.0,
        "cameras": ["0", "3"],
    }
    config = RuntimeConfig.from_json_dict(legacy)
    assert config.model == "cap_detector_640.onnx"  # not dirtv7.onnx
    assert config.classifier_model == "dirt_classifier_384.onnx"
    assert config.imgsz is None
    assert config.detect_threshold == pytest.approx(0.25)  # legacy key dropped


def test_v7_settings_with_classifier_key_keep_their_model_choices():
    data = RuntimeConfig.defaults().to_json_dict()
    data["model"] = "cap_detector_retrained_640.onnx"
    data["classifier_model"] = "dirt_classifier_retrained_448.onnx"
    config = RuntimeConfig.from_json_dict(data)
    assert config.model == "cap_detector_retrained_640.onnx"
    assert config.classifier_model == "dirt_classifier_retrained_448.onnx"


def test_validate_config_rejects_out_of_range_thresholds():
    defaults = RuntimeConfig.defaults()
    for field_name, bad_value in (
        ("detect_threshold", 1.5),
        ("frame_dirt_threshold", -0.1),
        ("track_dirt_threshold", 2.0),
        ("crop_margin", 1.5),
        ("max_classified_boxes", 0),
        ("classifier_model", "  "),
    ):
        with pytest.raises(ValueError):
            validate_config(replace(defaults, **{field_name: bad_value}))


# --------------------------------------------------------------------------- #
# 5. Cross-camera decisions with v7 tracks
# --------------------------------------------------------------------------- #

def test_two_cameras_same_cap_fire_once():
    scheduler = FakeScheduler()
    clock = [10.0]
    manager, _records = make_manager(scheduler, clock)
    dirty = [0.9, 0.9, 0.9, 0.9]
    manager.handle_finished_track(make_track(0, last_seen=10.0, dirt_probabilities=dirty))
    clock[0] = 10.05
    manager.handle_finished_track(make_track(1, last_seen=10.03, dirt_probabilities=dirty))
    assert len(scheduler.enqueued) == 1


def test_one_sided_dirt_still_rejects_the_cap():
    scheduler = FakeScheduler()
    clock = [10.0]
    manager, records = make_manager(scheduler, clock)
    manager.handle_finished_track(make_track(0, last_seen=10.0, dirt_probabilities=[0.05, 0.04, 0.06, 0.05]))
    clock[0] = 10.05
    manager.handle_finished_track(make_track(1, last_seen=10.03, dirt_probabilities=[0.9, 0.92, 0.88, 0.91]))
    clock[0] = 11.0
    manager.flush_expired(clock[0])
    assert len(scheduler.enqueued) == 1
    assert records and records[-1].result == "reject"
    assert records[-1].flagged_cameras == [1]


def test_clean_cap_schedules_no_fire():
    scheduler = FakeScheduler()
    clock = [10.0]
    manager, records = make_manager(scheduler, clock)
    manager.handle_finished_track(make_track(0, last_seen=10.0, dirt_probabilities=[0.05, 0.1, 0.07, 0.05]))
    clock[0] = 11.0
    manager.flush_expired(clock[0])
    assert scheduler.enqueued == []
    assert records and records[-1].result == "pass"


# --------------------------------------------------------------------------- #
# 6. Two-stage CameraWorker / run_detection wiring
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


def _scripted_classify(_session, _input_name, _frame, box, **_kwargs):
    """Test classifier: P(dirt) rides along as the box's 7th field."""

    return float(box[6]) if len(box) > 6 else None


def test_camera_worker_classifies_and_feeds_probabilities_to_the_tracker():
    stop_event = threading.Event()
    received: list[tuple[tuple, tuple]] = []

    class SpyTracker:
        def update(self, boxes, timestamp, frame_size=None, dirt_probs=None):
            received.append((tuple(boxes), tuple(dirt_probs)))
            stop_event.set()

        def collect_finished(self, now):
            return []

    class Manager:
        def handle_finished_track(self, track):
            raise AssertionError("no track should finish here")

    frame = _ScriptedFrame([[10.0, 10.0, 50.0, 50.0, 0.90, 0, 0.87]])
    worker = CameraWorker(
        camera_index=0,
        camera=None,
        session=_FakeSession(),
        input_name="images",
        model_imgsz=100,
        classifier_session=_FakeSession(),
        classifier_input_name="images",
        classifier_imgsz=100,
        crop_margin=0.10,
        classify_band_ratio=1.0,
        presence_line_axis="x",
        frame_dirt_threshold=0.50,
        max_classified_boxes=2,
        detect_threshold=0.25,
        duplicate_iou_threshold=0.65,
        max_frame_age_s=1000.0,
        mirror_horizontal=False,
        tracker=SpyTracker(),
        manager=Manager(),
        shared=SharedRuntimeState(1),
        preprocess_fn=lambda value, _imgsz: (value, {}),
        postprocess_fn=lambda output, _meta, conf_threshold: output,
        classify_fn=_scripted_classify,
        stop_event=stop_event,
        time_fn=lambda: 12.0,
        sleep_fn=lambda _seconds: None,
        log_fn=lambda *args, **kwargs: None,
    )
    worker._latest_capture = CapturedFrame(0, frame, 12.0, 1)

    worker._run()

    assert len(received) == 1
    boxes, dirt_probs = received[0]
    assert dirt_probs == (0.87,)
    assert boxes[0][5] == 1  # display class reflects the dirty verdict
    assert boxes[0][4] == pytest.approx(0.87)


def test_run_detection_fires_once_for_a_cap_dirty_on_one_camera():
    from cap_line_v7.runtime import run_detection
    from cap_line_v7.types import RuntimeCallbacks

    # Caps travel right -> left on the real rig (negative x, the v7 default).
    dirty_sequence = [
        [[10.0 + offset, 10.0, 50.0 + offset, 50.0, 0.90, 0, 0.92]]
        for offset in (30.0, 24.0, 18.0, 12.0, 6.0, 0.0)
    ]
    clean_sequence = [
        [[10.0 + offset, 10.0, 50.0 + offset, 50.0, 0.90, 0, 0.05]]
        for offset in (30.0, 24.0, 18.0, 12.0, 6.0, 0.0)
    ]
    cameras = [_ScriptedCamera(dirty_sequence), _ScriptedCamera(clean_sequence)]

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
            classify_fn=_scripted_classify,
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
    assert rejects and len({record.event_id for record in rejects}) == 1
    assert 0 in rejects[-1].flagged_cameras  # camera 0 caught the dirt


def test_run_detection_passes_a_clean_cap_with_one_hallucinated_frame():
    from cap_line_v7.runtime import run_detection
    from cap_line_v7.types import RuntimeCallbacks

    probabilities = [0.05, 0.06, 0.97, 0.05, 0.04, 0.06]  # one spike, never 3 in a row
    spiky_sequence = [
        [[10.0 + offset, 10.0, 50.0 + offset, 50.0, 0.90, 0, probability]]
        for offset, probability in zip((30.0, 24.0, 18.0, 12.0, 6.0, 0.0), probabilities)
    ]
    cameras = [_ScriptedCamera(spiky_sequence), _ScriptedCamera([])]

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
        simulate_gpio=False,
        trigger_duration=0.001,
        track_timeout_ms=20.0,
        max_track_gap_ms=20.0,
        merge_window_ms=100.0,
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
            classify_fn=_scripted_classify,
        ),
        daemon=True,
    )
    worker.start()
    time.sleep(0.4)
    stop_event.set()
    worker.join(timeout=3.0)

    assert not worker.is_alive()
    assert fires == []  # v6 would have fired on the 0.97 frame; v7 must not
    passes = [record for record in records if record.result == "pass"]
    assert passes  # the cap was still seen and logged as a pass
