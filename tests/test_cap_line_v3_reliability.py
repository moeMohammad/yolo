from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
from dataclasses import replace
from pathlib import Path

import cap_line_ui_v3
import cap_line_v3.runtime as runtime
from cap_line_v3 import RuntimeCallbacks
from cap_line_v3.config import DEFAULT_MODEL, RuntimeConfig
from cap_line_v3.decision import (
    TrackedCap,
    TrackedCapManager,
    compute_requested_trigger_delay,
    decide_decision_ready,
    decide_tracked_cap,
)
from cap_line_v3.pairing import select_capture_batch
from cap_line_v3.preview import CameraPreviewView
from cap_line_v3.runtime import (
    LatestFrameCameraReader,
    LivePreviewPublisher,
    resolve_model_path,
    run_detection,
)
from cap_line_v3.types import CapturedFrame, DetectionPacket, FramePair, PairDropStats, TrackObservation
from cap_line_ui_v3 import ConfigSettingsStore, DEFAULT_SETTINGS_PATH


class ShapeFrame:
    shape = (100, 100, 3)

    def __init__(self, detections=None):
        self.detections = detections or []


class FakeInput:
    name = "images"
    shape = [1, 3, 100, 100]


class FakeSession:
    def get_inputs(self):
        return [FakeInput()]

    def run(self, _outputs, inputs):
        frame = next(iter(inputs.values()))
        return [frame.detections]


class FakeCamera:
    def __init__(self, frame):
        self.frame = frame
        self.read_count = 0

    def read(self):
        self.read_count += 1
        return True, self.frame

    def isOpened(self):
        return True

    def set(self, *_args):
        return True

    def get(self, prop_id):
        if int(prop_id) in (3, 4):
            return 100.0
        if int(prop_id) == 5:
            return 60.0
        return 0.0

    def release(self):
        return None


class FiniteCamera:
    def __init__(self, count):
        self.count = int(count)
        self.read_count = 0

    def read(self):
        if self.read_count >= self.count:
            time.sleep(0.002)
            return False, None
        self.read_count += 1
        return True, ShapeFrame()


class CapLineV3ReliabilityTests(unittest.TestCase):
    def test_default_v3_model_is_dirtv6(self):
        self.assertEqual(DEFAULT_MODEL, "dirtv6.onnx")
        self.assertEqual(RuntimeConfig.defaults().model, "dirtv6.onnx")
        resolved_path, _imgsz = resolve_model_path(RuntimeConfig.defaults().model)
        self.assertEqual(Path(resolved_path).name, "dirtv6.onnx")

    def test_ui_settings_use_tracked_defaults_file(self):
        settings_path = Path(DEFAULT_SETTINGS_PATH)
        self.assertTrue(settings_path.is_file(), msg=f"missing tracked defaults: {settings_path}")
        config = ConfigSettingsStore(settings_path).load()
        self.assertEqual(config.model, "dirtv6.onnx")

    def test_ui_settings_preserve_explicit_legacy_model(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            settings_path = Path(temp_dir) / "settings.json"
            settings_path.write_text('{"model": "dirtv5.onnx"}', encoding="utf-8")

            config = ConfigSettingsStore(settings_path).load()

        self.assertEqual(config.model, "dirtv5.onnx")

    def test_ui_settings_migrate_legacy_runtime_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            settings_path = temp_path / "cap_line_ui_v3_settings.json"
            legacy_path = temp_path / "data" / "cap_line_ui_v3_settings.json"
            settings_path.write_text(
                '{"model": "dirtv6.onnx", "reject_threshold": 0.45}',
                encoding="utf-8",
            )
            legacy_path.parent.mkdir(parents=True, exist_ok=True)
            legacy_path.write_text(
                '{"model": "dirtv5.onnx", "reject_threshold": 0.61, "cameras": ["1", "2"]}',
                encoding="utf-8",
            )

            config = ConfigSettingsStore(
                settings_path,
                legacy_path=legacy_path,
            ).load()
            saved = json.loads(settings_path.read_text(encoding="utf-8"))
            self.assertEqual(config.model, "dirtv6.onnx")
            self.assertEqual(config.reject_threshold, 0.61)
            self.assertEqual(tuple(config.cameras), ("1", "2"))
            self.assertEqual(saved["model"], "dirtv6.onnx")
            self.assertEqual(saved["reject_threshold"], 0.61)
            self.assertEqual(saved["cameras"], ["1", "2"])

    def test_ui_settings_migration_is_one_shot(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            settings_path = temp_path / "cap_line_ui_v3_settings.json"
            legacy_path = temp_path / "data" / "cap_line_ui_v3_settings.json"
            settings_path.write_text('{"model": "dirtv6.onnx"}', encoding="utf-8")
            legacy_path.parent.mkdir(parents=True, exist_ok=True)
            legacy_path.write_text('{"reject_threshold": 0.61}', encoding="utf-8")
            store = ConfigSettingsStore(settings_path, legacy_path=legacy_path)

            store.load()
            config = store.load()

        self.assertEqual(config.reject_threshold, 0.61)
        self.assertFalse(legacy_path.exists())

    def test_ui_settings_preserve_custom_model_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            settings_path = Path(temp_dir) / "settings.json"
            settings_path.write_text('{"model": "./custom/dirtv2.onnx"}', encoding="utf-8")

            config = ConfigSettingsStore(settings_path).load()

        self.assertEqual(config.model, "./custom/dirtv2.onnx")

    def test_prediction_text_formats_class_and_confidence(self):
        format_prediction_text = getattr(cap_line_ui_v3, "format_prediction_text", None)
        self.assertIsNotNone(format_prediction_text)
        self.assertEqual(format_prediction_text("dirt_defect", 0.9532), "dirt_defect 0.953")
        self.assertEqual(format_prediction_text("undefected", 0.8872), "undefected 0.887")
        self.assertEqual(format_prediction_text(None, None), "-")

    def test_pairing_uses_single_camera_fallback_after_wait(self):
        stats = PairDropStats()
        frame = CapturedFrame(0, ShapeFrame(), timestamp=1.0, sequence=1)

        waiting = select_capture_batch(
            ((frame,), ()),
            now=1.005,
            max_skew_ms=40.0,
            single_camera_wait_ms=20.0,
            stats=stats,
        )
        self.assertIsNone(waiting)

        batch = select_capture_batch(
            ((frame,), ()),
            now=1.025,
            max_skew_ms=40.0,
            single_camera_wait_ms=20.0,
            stats=stats,
        )
        self.assertIsNotNone(batch)
        self.assertTrue(batch.is_single_camera)
        self.assertEqual(batch.reason, "single_camera_wait")
        self.assertEqual(batch.missing_camera_indices, (1,))
        self.assertEqual(stats.missing_camera, 1)

    def test_buffered_reader_keeps_multiple_pending_frames(self):
        camera = FiniteCamera(4)
        reader = LatestFrameCameraReader(camera, 0, target_fps=None, capture_buffer_frames=3)
        reader.start()
        deadline = time.monotonic() + 1.0
        while reader.captured < 4 and time.monotonic() < deadline:
            time.sleep(0.005)
        reader.stop()

        pending = reader.pending_after(0)
        self.assertEqual([frame.sequence for frame in pending], [2, 3, 4])
        self.assertEqual(reader.overwritten, 1)

    def test_tracking_survives_configured_missing_frame(self):
        manager = TrackedCapManager(
            camera_count=2,
            merge_window_seconds=0.2,
            finalize_quiet_seconds=0.5,
            anchor_axis="x",
            anchor_line_ratio=0.75,
            track_iou=0.3,
            max_missing_frames=2,
        )
        frame_size = (100, 100)
        manager.update(
            [TrackObservation(0, (10, 40, 20, 60, 0.8, 1), 1.0, frame_size)],
            observed_camera_indices={0},
        )
        manager.update([], observed_camera_indices={0})
        manager.update(
            [TrackObservation(0, (40, 40, 50, 60, 0.9, 1), 1.033, frame_size)],
            observed_camera_indices={0},
        )

        open_caps = manager.open_caps()
        self.assertEqual(len(open_caps), 1)
        self.assertEqual(open_caps[0].camera_summaries[0].observation_count, 2)

    def test_predicted_actuation_can_trigger_without_exact_line_frame(self):
        config = replace(
            RuntimeConfig.defaults(),
            anchor_line_ratio=0.5,
            reject_threshold=0.45,
            actuation_prediction_horizon_ms=150.0,
            actuation_window_ms=250.0,
        )
        cap = TrackedCap(event_id=1, created_at=1.0, last_seen_at=1.0)
        frame_size = (100, 100)
        cap.add_observation(
            TrackObservation(0, (15, 40, 25, 60, 0.85, 1), 1.0, frame_size),
            anchor_axis="x",
            anchor_line_ratio=0.5,
        )
        cap.add_observation(
            TrackObservation(0, (30, 40, 40, 60, 0.90, 1), 1.1, frame_size),
            anchor_axis="x",
            anchor_line_ratio=0.5,
        )

        decision = decide_decision_ready(cap, config=config, decision_ready_time=1.101, camera_count=1)
        self.assertIsNotNone(decision)
        self.assertEqual(decision.result, "trigger")
        self.assertEqual(decision.decision_source, "predicted_actuation_threshold")
        self.assertAlmostEqual(cap.actuation_time or 0.0, 1.2, places=3)

    def test_deadline_fallback_waits_until_trigger_slack_is_low(self):
        config = replace(
            RuntimeConfig.defaults(),
            anchor_line_ratio=0.75,
            reject_threshold=0.45,
            merge_window_ms=2000.0,
            nozzle_distance_mm=100.0,
            belt_speed_mm_per_s=100.0,
            trigger_offset_s=0.0,
            latency_compensation_ms=0.0,
            decision_deadline_guard_ms=25.0,
        )
        cap = TrackedCap(event_id=1, created_at=10.0, last_seen_at=10.0)
        cap.add_observation(
            TrackObservation(0, (70, 40, 80, 60, 0.95, 1), 10.0, (100, 100), at_actuation_line=True),
            anchor_axis="x",
            anchor_line_ratio=0.75,
        )

        early = decide_decision_ready(cap, config=config, decision_ready_time=10.5, camera_count=2)
        self.assertIsNone(early)

        due_time = 10.0 + compute_requested_trigger_delay(config) - 0.010
        due = decide_decision_ready(cap, config=config, decision_ready_time=due_time, camera_count=2)
        self.assertIsNotNone(due)
        self.assertEqual(due.decision_source, "single_camera_deadline_fallback")

    def test_disagreeing_camera_votes_trigger_defect_regardless_of_confidence(self):
        config = replace(RuntimeConfig.defaults(), reject_threshold=0.80)
        cap = TrackedCap(event_id=1, created_at=1.0, last_seen_at=1.0)
        cap.add_observation(
            TrackObservation(0, (70, 40, 80, 60, 0.99, 0), 1.0, (100, 100), at_actuation_line=True),
            anchor_axis="x",
            anchor_line_ratio=0.75,
        )
        cap.add_observation(
            TrackObservation(1, (70, 40, 80, 60, 0.50, 1), 1.0, (100, 100), at_actuation_line=True),
            anchor_axis="x",
            anchor_line_ratio=0.75,
        )

        decision = decide_decision_ready(cap, config=config, decision_ready_time=1.0, camera_count=2)

        self.assertIsNotNone(decision)
        self.assertEqual(decision.result, "trigger")
        self.assertEqual(decision.decision_source, "camera_defect_vote")
        self.assertEqual(decision.final_class_name, "dirt_defect")
        self.assertAlmostEqual(decision.final_score or 0.0, 0.50)

    def test_finalized_dirty_cap_without_actuation_is_logged_as_missed(self):
        config = replace(RuntimeConfig.defaults(), reject_threshold=0.45)
        cap = TrackedCap(event_id=1, created_at=1.0, last_seen_at=1.0)
        cap.add_observation(
            TrackObservation(0, (10, 40, 20, 60, 0.9, 1), 1.0, (100, 100)),
            anchor_axis="x",
            anchor_line_ratio=0.75,
        )

        decision = decide_tracked_cap(cap, config=config, decision_time=2.0, camera_count=2)
        self.assertEqual(decision.result, "skip")
        self.assertEqual(decision.decision_source, "no_actuation_crossing")
        self.assertEqual(decision.review_reason, "missed_actuation")
        self.assertEqual(decision.final_class_name, "dirt_defect")

    def test_finalized_clean_cap_without_actuation_reports_undefected(self):
        config = replace(RuntimeConfig.defaults(), reject_threshold=0.45)
        cap = TrackedCap(event_id=1, created_at=1.0, last_seen_at=1.0)
        cap.add_observation(
            TrackObservation(0, (10, 40, 20, 60, 0.8872, 0), 1.0, (100, 100)),
            anchor_axis="x",
            anchor_line_ratio=0.75,
        )

        decision = decide_tracked_cap(cap, config=config, decision_time=2.0, camera_count=2)

        self.assertEqual(decision.result, "skip")
        self.assertEqual(decision.decision_source, "no_actuation_crossing")
        self.assertEqual(decision.final_class_name, "undefected")
        self.assertAlmostEqual(decision.final_score or 0.0, 0.8872)

    def test_decision_preview_views_use_actuation_and_other_camera_observations(self):
        build_views = getattr(runtime, "_build_decision_preview_views", None)
        self.assertIsNotNone(build_views)
        line_box = (70, 40, 80, 60, 0.95, 1)
        other_box = (20, 40, 30, 60, 0.91, 1)
        captured0 = CapturedFrame(0, ShapeFrame(), timestamp=1.0, sequence=5)
        captured1 = CapturedFrame(1, ShapeFrame(), timestamp=1.005, sequence=7)
        packet = DetectionPacket(
            FramePair((captured0, captured1), pair_timestamp=1.005, skew_ms=5.0),
            ((line_box,), (other_box,)),
            (2.0, 2.0),
        )
        cap = TrackedCap(event_id=1, created_at=1.0, last_seen_at=1.005)
        cap.add_observation(
            TrackObservation(0, line_box, 1.0, (100, 100), at_actuation_line=True, sequence=5),
            anchor_axis="x",
            anchor_line_ratio=0.75,
        )
        cap.add_observation(
            TrackObservation(1, other_box, 1.005, (100, 100), sequence=7),
            anchor_axis="x",
            anchor_line_ratio=0.75,
        )

        views = build_views(
            cap,
            (packet,),
            camera_count=2,
            anchor_axis="x",
            anchor_line_ratio=0.75,
        )

        self.assertIsNotNone(views[0])
        self.assertIsNotNone(views[1])
        self.assertIs(views[0].captured, captured0)
        self.assertEqual(views[0].boxes, (line_box,))
        self.assertIs(views[1].captured, captured1)
        self.assertEqual(views[1].boxes, (other_box,))

    def test_decision_snapshot_hold_survives_stale_live_preview(self):
        now = [10.0]
        line_box = (70, 40, 80, 60, 0.95, 1)
        snapshot_view = CameraPreviewView(
            CapturedFrame(0, ShapeFrame(), timestamp=9.9, sequence=4),
            (line_box,),
        )
        live_view = CameraPreviewView(
            CapturedFrame(0, ShapeFrame(), timestamp=10.2, sequence=5),
            (),
        )
        publisher = LivePreviewPublisher(
            [],
            lambda _preview: None,
            anchor_axis="x",
            anchor_line_ratio=0.75,
            preview_fps=30,
            overlay_target_fps=60,
            actuation_snapshot_hold_ms=900.0,
            time_fn=lambda: now[0],
        )
        self.assertTrue(hasattr(publisher, "update_decision_snapshot"))

        publisher.update_decision_snapshot((snapshot_view,))
        held = publisher._hold_actuation_snapshots((live_view,))

        self.assertIs(held[0], snapshot_view)

        now[0] = 10.95
        expired = publisher._hold_actuation_snapshots((live_view,))

        self.assertIs(expired[0], live_view)

    def test_fake_runtime_triggers_once_for_single_camera_deadline_fallback(self):
        dirty_box = [70, 40, 80, 60, 0.95, 1]
        cameras = [FakeCamera(ShapeFrame([dirty_box])), FakeCamera(ShapeFrame([]))]
        histories = []
        timings = []

        with tempfile.TemporaryDirectory() as temp_dir:
            config = replace(
                RuntimeConfig.defaults(),
                cameras=("0", "1"),
                simulate_gpio=True,
                trigger_duration=0.001,
                target_fps=60,
                merge_window_ms=2000.0,
                decision_deadline_guard_ms=5000.0,
                nozzle_distance_mm=0.001,
                belt_speed_mm_per_s=1000.0,
                trigger_offset_s=0.0,
                latency_compensation_ms=0.0,
                timing_log_dir=temp_dir,
                debug_dir=temp_dir,
                pictures_dir=temp_dir,
                session_log_dir=temp_dir,
            )

            run_detection(
                config,
                callbacks=RuntimeCallbacks(
                    history_callback=histories.append,
                    timing_log_callback=timings.append,
                    log_fn=lambda *_args, **_kwargs: None,
                ),
                stop_event=threading.Event(),
                camera_factory=lambda index, _source, _config: cameras[index],
                session_factory=lambda _model_path, _threads: FakeSession(),
                preprocess_fn=lambda frame, _imgsz: (frame, {"frame_shape": frame.shape}),
                postprocess_fn=lambda output, _meta, conf_threshold: [
                    box for box in output if float(box[4]) >= float(conf_threshold)
                ],
            )

        triggers = [record for record in histories if record.result == "trigger"]
        self.assertEqual(len(triggers), 1)
        self.assertEqual(triggers[0].decision_source, "single_camera_deadline_fallback")
        self.assertEqual(len(timings), 1)
        self.assertIsNotNone(timings[0].trigger_on_time)
        self.assertIsNotNone(timings[0].scheduler_late_ms)


if __name__ == "__main__":
    unittest.main()
