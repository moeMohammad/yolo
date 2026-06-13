from __future__ import annotations

import importlib.util
import concurrent.futures
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent


def load_module(module_name: str):
    module_path = REPO_ROOT / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module spec from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_tracked_cap(module, observations_by_camera):
    timestamps = [
        timestamp
        for entries in observations_by_camera.values()
        for _, _, timestamp in entries
    ]
    tracked_cap = module.TrackedCap(
        event_id=1,
        created_at=min(timestamps, default=1.0),
        last_seen_at=max(timestamps, default=1.0),
    )
    for camera_index, entries in observations_by_camera.items():
        summary = module.CameraObservationSummary()
        for class_id, score, timestamp in entries:
            summary.add(class_id, score, timestamp)
        if summary.observation_count > 0:
            tracked_cap.camera_summaries[camera_index] = summary
            tracked_cap.camera_indices.add(camera_index)
    if tracked_cap.camera_summaries:
        tracked_cap.anchor_time = min(
            summary.first_seen_at
            for summary in tracked_cap.camera_summaries.values()
            if summary.first_seen_at is not None
        )
        tracked_cap.anchor_camera_index = (
            0 if 0 in tracked_cap.camera_summaries else next(iter(tracked_cap.camera_summaries))
        )
        tracked_cap.actuation_time = tracked_cap.anchor_time
        tracked_cap.actuation_camera_index = tracked_cap.anchor_camera_index
        tracked_cap.actuation_camera_summaries = module.copy_camera_summaries(
            tracked_cap.camera_summaries
        )
    return tracked_cap


class CapLineRuntimeV2DecisionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = load_module("cap_line_runtime_v2")
        self.decision_kwargs = {
            "camera_count": 2,
            "timing_camera_index": 0,
            "decision_time": 3.0,
            "merge_window_seconds": 0.0,
            "nozzle_distance_mm": 430.0,
            "belt_speed_mm_per_s": 555.0,
            "trigger_offset_s": 0.245,
            "latency_compensation_ms": 50.0,
        }

    def test_parser_removes_old_policy_knobs(self) -> None:
        parser = self.module.build_arg_parser()
        option_strings = set(parser._option_string_actions)

        self.assertNotIn("--conf", option_strings)
        self.assertNotIn("--defect-min-score", option_strings)
        self.assertNotIn("--defect-margin", option_strings)
        self.assertNotIn("--single-camera-defect-score", option_strings)
        self.assertEqual(0.45, self.module.TRACKING_DETECTION_THRESHOLD)
        self.assertEqual(0.45, self.module.DEFECT_REJECT_THRESHOLD)

    def test_default_trigger_offset_starts_air_earlier(self) -> None:
        parser = self.module.build_arg_parser()
        args = parser.parse_args([])

        self.assertLess(self.module.DEFAULT_TRIGGER_OFFSET_S, 0.0)
        self.assertEqual(self.module.DEFAULT_TRIGGER_OFFSET_S, args.trigger_offset_s)
        self.assertEqual([960, 600], args.res)
        self.assertEqual(60, self.module.DEFAULT_CAMERA_FPS)
        self.assertEqual(self.module.DEFAULT_CAMERA_FPS, args.fps)
        self.assertEqual("YUYV", args.pixel_format)

    def test_set_camera_format_uses_configured_pixel_format(self) -> None:
        commands = []
        original_run = self.module.subprocess.run

        def fake_run(command, check=False, capture_output=False, text=False):
            commands.append((list(command), check))
            if "--get-fmt-video" in command:
                return types.SimpleNamespace(
                    stdout="Width/Height      : 960/600\nPixel Format      : 'YUYV' (YUYV 4:2:2)\n",
                    returncode=0,
                )
            if "--get-parm" in command:
                return types.SimpleNamespace(
                    stdout="Frames per second: 10.000\n",
                    returncode=0,
                )
            return types.SimpleNamespace(returncode=0)

        try:
            self.module.subprocess.run = fake_run
            self.module.set_camera_format(
                "/dev/video0",
                960,
                600,
                10,
                pixel_format="YUYV",
                log_fn=lambda *_args, **_kwargs: None,
            )
        finally:
            self.module.subprocess.run = original_run

        self.assertEqual(
            [
                "v4l2-ctl",
                "-d",
                "/dev/video0",
                "--set-fmt-video=width=960,height=600,pixelformat=YUYV",
            ],
            commands[0][0],
        )
        self.assertEqual(["v4l2-ctl", "-d", "/dev/video0", "--set-parm=10"], commands[1][0])

    def test_yuy2_pixel_format_is_normalized_to_yuyv(self) -> None:
        self.assertEqual("YUYV", self.module.normalize_camera_pixel_format("YUY2"))

    def test_validate_args_rejects_non_yuyv_pixel_format(self) -> None:
        parser = self.module.build_arg_parser()
        args = parser.parse_args(["--pixel-format", "MJPG"])
        with self.assertRaisesRegex(ValueError, "YUYV"):
            self.module.validate_args(args)

    def test_triggers_on_highest_defect_score_above_threshold(self) -> None:
        tracked_cap = build_tracked_cap(
            self.module,
            {
                0: [(0, 0.93, 1.0), (1, 0.81, 1.1)],
                1: [(1, 0.84, 1.2)],
            },
        )

        decision = self.module.decide_tracked_cap(tracked_cap, **self.decision_kwargs)

        self.assertEqual("trigger", decision.result)
        self.assertEqual("dirt_defect", decision.final_class_name)
        self.assertAlmostEqual(0.84, decision.final_score)
        self.assertEqual("highest_defect_threshold", decision.decision_source)
        self.assertEqual("undefected", decision.camera_votes[0].class_name)
        self.assertAlmostEqual(0.93, decision.camera_votes[0].score)
        self.assertEqual("dirt_defect", decision.camera_votes[1].class_name)
        self.assertAlmostEqual(0.84, decision.camera_votes[1].score)

    def test_triggers_when_highest_defect_score_equals_reject_threshold(self) -> None:
        tracked_cap = build_tracked_cap(
            self.module,
            {
                0: [(1, self.module.DEFECT_REJECT_THRESHOLD, 1.0)],
            },
        )

        decision = self.module.decide_tracked_cap(tracked_cap, **self.decision_kwargs)

        self.assertEqual("trigger", decision.result)
        self.assertAlmostEqual(self.module.DEFECT_REJECT_THRESHOLD, decision.final_score)
        self.assertEqual("highest_defect_threshold", decision.decision_source)

    def test_skips_when_defect_score_is_tracking_only_below_reject_threshold(self) -> None:
        tracked_cap = build_tracked_cap(
            self.module,
            {
                0: [(1, 0.44, 1.0), (0, 0.95, 1.1)],
                1: [(0, 0.91, 1.2)],
            },
        )

        decision = self.module.decide_tracked_cap(tracked_cap, **self.decision_kwargs)

        self.assertEqual("skip", decision.result)
        self.assertEqual("undefected", decision.final_class_name)
        self.assertAlmostEqual(0.44, decision.final_score)
        self.assertEqual("defect_below_threshold_at_actuation", decision.decision_source)

    def test_debug_frame_burst_keeps_before_actuation_and_after_frames(self) -> None:
        tracked_cap = self.module.TrackedCap(
            event_id=1,
            created_at=1.0,
            last_seen_at=1.3,
            actuation_time=1.2,
        )
        for index, timestamp in enumerate([1.0, 1.1, 1.2, 1.3, 1.4]):
            tracked_cap.append_debug_frame_snapshot(
                self.module.FrameSnapshot(
                    frame_index=index,
                    timestamp=timestamp,
                    raw_frames=[f"raw-{index}-0", f"raw-{index}-1"],
                    annotated_frames=[f"annot-{index}-0", f"annot-{index}-1"],
                    boxes_by_camera=[],
                    read_duration_ms=1.0,
                    frame_interval_ms=100.0 if index else None,
                    inference_ms_by_camera=[2.0, 3.0],
                    processing_duration_ms=6.0,
                )
            )

        burst = self.module.build_debug_frame_burst(
            tracked_cap,
            before_frames=2,
            after_frames=1,
        )

        self.assertEqual([1.1, 1.2, 1.3], [snapshot.timestamp for snapshot in burst])

    def test_debug_capture_writer_saves_timestamped_burst_artifacts(self) -> None:
        written_paths = []

        def fake_write_image(_frame, path: str) -> bool:
            written_paths.append(path)
            return True

        with tempfile.TemporaryDirectory() as temporary_dir:
            writer = self.module.DebugCaptureWriter(
                temporary_dir,
                pictures_dir=temporary_dir,
                write_image_fn=fake_write_image,
                log_fn=lambda *_args, **_kwargs: None,
            )
            try:
                task = self.module.DebugCaptureTask(
                    event_id=7,
                    recorded_at="2026-06-01T12:00:00.000+04:00",
                    result="skip",
                    review_reason="skip",
                    decision_source="defect_below_threshold_at_actuation",
                    final_class="undefected",
                    final_score=0.69,
                    score_summary="dirt_defect=0.690",
                    cam0_vote="undefected:0.950",
                    cam1_vote="none",
                    model_path="fake.onnx",
                    json_payload={"artifacts": {}},
                    frame_burst=[
                        self.module.DebugFrameSnapshot(
                            frame_index=3,
                            timestamp_monotonic=1.2,
                            timestamp_iso="2026-06-01T12:00:00.120+04:00",
                            offset_from_actuation_ms=0.0,
                            raw_frames=["raw0", "raw1"],
                            annotated_frames=["annot0", "annot1"],
                            boxes_by_camera=[],
                            read_duration_ms=1.0,
                            frame_interval_ms=66.7,
                            inference_ms_by_camera=[20.0, 21.0],
                            processing_duration_ms=45.0,
                        )
                    ],
                )

                writer.submit(task)
                writer.close()
            finally:
                if writer._thread.is_alive():
                    writer.close()

            self.assertTrue(
                any("_burst00_cam0_raw.jpg" in path for path in written_paths)
            )
            json_paths = list(Path(temporary_dir).glob("**/*_skip.json"))
            self.assertEqual(1, len(json_paths))
            self.assertIn(
                "frame_burst",
                json_paths[0].read_text(encoding="utf-8"),
            )

    def test_debug_capture_writer_does_not_write_training_pictures(self) -> None:
        written_paths = []

        def fake_write_image(_frame, path: str) -> bool:
            written_paths.append(path)
            return True

        with tempfile.TemporaryDirectory() as debug_dir:
            with tempfile.TemporaryDirectory() as pictures_dir:
                writer = self.module.DebugCaptureWriter(
                    debug_dir,
                    pictures_dir=pictures_dir,
                    write_image_fn=fake_write_image,
                    log_fn=lambda *_args, **_kwargs: None,
                )
                try:
                    writer.submit(
                        self.module.DebugCaptureTask(
                            event_id=8,
                            recorded_at="2026-06-01T12:00:00.000+04:00",
                            result="skip",
                            review_reason="missed_actuation",
                            decision_source="no_actuation_crossing",
                            final_class="dirt_defect",
                            final_score=0.91,
                            score_summary="dirt_defect=0.910",
                            cam0_vote="dirt_defect:0.910",
                            cam1_vote="none",
                            model_path="fake.onnx",
                            raw_frames=["raw0", "raw1"],
                            json_payload={"artifacts": {}},
                        )
                    )
                    writer.close()
                finally:
                    if writer._thread.is_alive():
                        writer.close()

                self.assertFalse(
                    any(Path(path).is_relative_to(Path(pictures_dir)) for path in written_paths)
                )
                self.assertTrue(any("_cam0_raw.jpg" in path for path in written_paths))

    def test_line_picture_task_writes_raw_frames_to_pictures_dir(self) -> None:
        written_paths = []

        def fake_write_image(_frame, path: str) -> bool:
            written_paths.append(path)
            return True

        with tempfile.TemporaryDirectory() as debug_dir:
            with tempfile.TemporaryDirectory() as pictures_dir:
                writer = self.module.DebugCaptureWriter(
                    debug_dir,
                    pictures_dir=pictures_dir,
                    write_image_fn=fake_write_image,
                    log_fn=lambda *_args, **_kwargs: None,
                )
                try:
                    writer.submit_line_picture(
                        self.module.LinePictureTask(
                            event_id=9,
                            recorded_at="2026-06-01T12:00:00.000+04:00",
                            result="skip",
                            final_class="undefected",
                            final_score=0.0,
                            decision_source="defect_below_threshold_at_actuation",
                            cam0_vote="undefected:0.960",
                            cam1_vote="undefected:0.940",
                            raw_frames=["line0", "line1"],
                            anchor_axis="x",
                            anchor_line_ratio=0.5,
                        )
                    )
                    writer.close()
                finally:
                    if writer._thread.is_alive():
                        writer.close()

                picture_root = Path(pictures_dir)
                self.assertTrue(
                    any(
                        Path(path).is_relative_to(picture_root)
                        and "_line_cam0.jpg" in path
                        for path in written_paths
                    )
                )
                self.assertTrue(
                    any(
                        Path(path).is_relative_to(picture_root)
                        and "_line_cam1.jpg" in path
                        for path in written_paths
                    )
                )
                self.assertFalse(
                    any(Path(path).is_relative_to(Path(debug_dir)) for path in written_paths)
                )
                manifest_paths = list(picture_root.glob("*.csv"))
                self.assertEqual(1, len(manifest_paths))
                manifest_text = manifest_paths[0].read_text(encoding="utf-8")
                self.assertIn("raw_cam0_path", manifest_text)
                self.assertIn("raw_cam1_path", manifest_text)
                self.assertIn("defect_below_threshold_at_actuation", manifest_text)

    def test_line_picture_capture_uses_actuation_frames_for_clean_cap(self) -> None:
        tracked_cap = build_tracked_cap(
            self.module,
            {
                0: [(0, 0.96, 1.0)],
                1: [(0, 0.94, 1.0)],
            },
        )
        tracked_cap.raw_frames_at_actuation = ["line0", "line1"]
        decision = self.module.decide_tracked_cap(tracked_cap, **self.decision_kwargs)

        class FakeWriter:
            def __init__(self):
                self.tasks = []

            def submit_line_picture(self, task):
                self.tasks.append(task)
                return len(self.tasks)

        writer = FakeWriter()
        args = self.module.parse_args([])

        self.module.submit_line_picture_capture(
            writer,
            tracked_cap,
            decision,
            clock=self.module.RuntimeClock(time_fn=lambda: 10.0),
            args=args,
        )

        self.assertEqual(1, len(writer.tasks))
        task = writer.tasks[0]
        self.assertEqual("skip", task.result)
        self.assertEqual("undefected", task.final_class)
        self.assertEqual(["line0", "line1"], task.raw_frames)

    def test_line_picture_capture_skips_caps_that_never_reach_line(self) -> None:
        tracked_cap = self.module.TrackedCap(
            event_id=10,
            created_at=1.0,
            last_seen_at=1.0,
        )
        tracked_cap.latest_raw_frames = ["latest0", "latest1"]
        decision = self.module.decide_tracked_cap(tracked_cap, **self.decision_kwargs)

        class FakeWriter:
            def __init__(self):
                self.tasks = []

            def submit_line_picture(self, task):
                self.tasks.append(task)
                return len(self.tasks)

        writer = FakeWriter()
        args = self.module.parse_args([])

        self.module.submit_line_picture_capture(
            writer,
            tracked_cap,
            decision,
            clock=self.module.RuntimeClock(time_fn=lambda: 10.0),
            args=args,
        )

        self.assertEqual([], writer.tasks)

    def test_line_picture_capture_falls_back_to_latest_complete_pair(self) -> None:
        tracked_cap = build_tracked_cap(
            self.module,
            {
                0: [(0, 0.96, 1.0)],
                1: [(0, 0.94, 1.0)],
            },
        )
        tracked_cap.raw_frames_at_actuation = ["line0"]
        tracked_cap.latest_raw_frames = ["latest0", "latest1"]
        decision = self.module.decide_tracked_cap(tracked_cap, **self.decision_kwargs)

        class FakeWriter:
            def __init__(self):
                self.tasks = []

            def submit_line_picture(self, task):
                self.tasks.append(task)
                return len(self.tasks)

        writer = FakeWriter()
        args = self.module.parse_args([])

        self.module.submit_line_picture_capture(
            writer,
            tracked_cap,
            decision,
            clock=self.module.RuntimeClock(time_fn=lambda: 10.0),
            args=args,
        )

        self.assertEqual(1, len(writer.tasks))
        self.assertEqual(["latest0", "latest1"], writer.tasks[0].raw_frames)

    def test_line_picture_capture_skips_incomplete_camera_pair(self) -> None:
        tracked_cap = build_tracked_cap(
            self.module,
            {
                0: [(0, 0.96, 1.0)],
                1: [(0, 0.94, 1.0)],
            },
        )
        tracked_cap.raw_frames_at_actuation = ["line0"]
        tracked_cap.latest_raw_frames = ["latest0"]
        decision = self.module.decide_tracked_cap(tracked_cap, **self.decision_kwargs)
        warnings = []

        class FakeWriter:
            def __init__(self):
                self.tasks = []
                self._log = lambda *args, **_kwargs: warnings.append(" ".join(map(str, args)))

            def submit_line_picture(self, task):
                self.tasks.append(task)
                return len(self.tasks)

        writer = FakeWriter()
        args = self.module.parse_args([])

        self.module.submit_line_picture_capture(
            writer,
            tracked_cap,
            decision,
            clock=self.module.RuntimeClock(time_fn=lambda: 10.0),
            args=args,
        )

        self.assertEqual([], writer.tasks)
        self.assertTrue(any("missing complete cam0/cam1" in warning for warning in warnings))

    def test_skips_when_only_clean_detections_are_present(self) -> None:
        tracked_cap = build_tracked_cap(
            self.module,
            {
                0: [(0, 0.92, 1.0)],
                1: [(0, 0.88, 1.1)],
            },
        )

        decision = self.module.decide_tracked_cap(tracked_cap, **self.decision_kwargs)

        self.assertEqual("skip", decision.result)
        self.assertEqual("undefected", decision.final_class_name)
        self.assertAlmostEqual(0.0, decision.final_score)
        self.assertEqual("defect_below_threshold_at_actuation", decision.decision_source)

    def test_dirty_before_clean_actuation_requests_debug_capture(self) -> None:
        tracked_cap = build_tracked_cap(
            self.module,
            {
                0: [(1, 0.92, 1.000), (0, 0.96, 1.100)],
            },
        )
        clean_actuation_summary = self.module.CameraObservationSummary()
        clean_actuation_summary.add(0, 0.96, 1.100)
        tracked_cap.actuation_camera_summaries = {0: clean_actuation_summary}

        decision = self.module.decide_tracked_cap(
            tracked_cap,
            camera_count=2,
            timing_camera_index=0,
            decision_time=1.200,
            merge_window_seconds=0.150,
            nozzle_distance_mm=100.0,
            belt_speed_mm_per_s=100.0,
            trigger_offset_s=0.0,
            latency_compensation_ms=0.0,
        )

        self.assertEqual("skip", decision.result)
        self.assertEqual("defect_below_threshold_at_actuation", decision.decision_source)
        self.assertEqual("dirty_before_clean_actuation", decision.review_reason)

    def test_pre_line_defect_is_ignored_when_line_frame_is_clean(self) -> None:
        manager = self.module.TrackedCapManager(
            merge_window_seconds=0.150,
            camera_count=2,
            timing_camera_index=0,
            anchor_axis="x",
            anchor_line_ratio=0.5,
            finalize_quiet_seconds=0.030,
        )

        manager.update(
            [
                self.module.TrackObservation(
                    camera_index=0,
                    track_id=1,
                    class_id=1,
                    box=[20.0, 10.0, 40.0, 30.0, 0.92, 1],
                    timestamp=1.000,
                    frame_size=(100, 40),
                )
            ],
            [],
        )
        tracked_cap = manager.open_caps()[0]
        self.assertIsNone(tracked_cap.actuation_time)

        manager.update(
            [
                self.module.TrackObservation(
                    camera_index=0,
                    track_id=1,
                    class_id=0,
                    box=[50.0, 10.0, 70.0, 30.0, 0.96, 0],
                    timestamp=2.000,
                    frame_size=(100, 40),
                )
            ],
            [],
        )

        decision = self.module.decide_tracked_cap(
            tracked_cap,
            camera_count=2,
            timing_camera_index=0,
            decision_time=2.100,
            merge_window_seconds=0.150,
            nozzle_distance_mm=100.0,
            belt_speed_mm_per_s=100.0,
            trigger_offset_s=0.0,
            latency_compensation_ms=0.0,
        )

        self.assertEqual("skip", decision.result)
        self.assertEqual("undefected", decision.final_class_name)
        self.assertAlmostEqual(0.0, decision.final_score)
        self.assertEqual("defect_below_threshold_at_actuation", decision.decision_source)
        self.assertEqual("undefected", decision.camera_votes[0].class_name)
        self.assertAlmostEqual(0.96, decision.camera_votes[0].score)

    def test_skips_with_no_observations(self) -> None:
        tracked_cap = build_tracked_cap(self.module, {})

        decision = self.module.decide_tracked_cap(tracked_cap, **self.decision_kwargs)

        self.assertEqual("skip", decision.result)
        self.assertIsNone(decision.final_class_name)
        self.assertIsNone(decision.final_score)
        self.assertEqual("no_observations", decision.decision_source)

    def test_defect_before_actuation_line_waits_until_box_spans_line(self) -> None:
        manager = self.module.TrackedCapManager(
            merge_window_seconds=0.150,
            camera_count=2,
            timing_camera_index=0,
            anchor_axis="x",
            anchor_line_ratio=0.5,
            finalize_quiet_seconds=0.030,
        )
        decision_kwargs = {
            "camera_count": 2,
            "timing_camera_index": 0,
            "decision_ready_time": 1.100,
            "merge_window_seconds": 0.150,
            "nozzle_distance_mm": 100.0,
            "belt_speed_mm_per_s": 100.0,
            "trigger_offset_s": 0.0,
            "latency_compensation_ms": 0.0,
        }

        manager.update(
            [
                self.module.TrackObservation(
                    camera_index=0,
                    track_id=1,
                    class_id=1,
                    box=[20.0, 10.0, 40.0, 30.0, 0.90, 1],
                    timestamp=1.000,
                    frame_size=(100, 40),
                )
            ],
            [],
        )
        tracked_cap = manager.open_caps()[0]

        self.assertIsNone(self.module.decide_decision_ready(tracked_cap, **decision_kwargs))

        manager.update(
            [
                self.module.TrackObservation(
                    camera_index=0,
                    track_id=1,
                    class_id=1,
                    box=[50.0, 10.0, 70.0, 30.0, 0.88, 1],
                    timestamp=2.000,
                    frame_size=(100, 40),
                )
            ],
            [],
        )
        decision_kwargs["decision_ready_time"] = 2.200

        decision = self.module.decide_decision_ready(tracked_cap, **decision_kwargs)

        self.assertIsNotNone(decision)
        self.assertEqual("trigger", decision.result)
        self.assertEqual("anchor_line", decision.anchor_source)
        self.assertAlmostEqual(2.000, decision.anchor_time)
        self.assertAlmostEqual(3.000, decision.requested_fire_time)

    def test_box_entirely_before_actuation_line_does_not_trigger(self) -> None:
        manager = self.module.TrackedCapManager(
            merge_window_seconds=0.150,
            camera_count=2,
            timing_camera_index=0,
            anchor_axis="x",
            anchor_line_ratio=0.5,
            finalize_quiet_seconds=0.030,
        )
        manager.update(
            [
                self.module.TrackObservation(
                    camera_index=0,
                    track_id=1,
                    class_id=1,
                    box=[20.0, 10.0, 45.0, 30.0, 0.90, 1],
                    timestamp=1.000,
                    frame_size=(100, 40),
                )
            ],
            [],
        )
        tracked_cap = manager.open_caps()[0]

        self.assertIsNone(tracked_cap.actuation_time)

    def test_box_entirely_after_actuation_line_does_not_trigger_on_first_sight(
        self,
    ) -> None:
        manager = self.module.TrackedCapManager(
            merge_window_seconds=0.150,
            camera_count=2,
            timing_camera_index=0,
            anchor_axis="x",
            anchor_line_ratio=0.5,
            finalize_quiet_seconds=0.030,
        )
        manager.update(
            [
                self.module.TrackObservation(
                    camera_index=0,
                    track_id=1,
                    class_id=1,
                    box=[55.0, 10.0, 75.0, 30.0, 0.90, 1],
                    timestamp=1.000,
                    frame_size=(100, 40),
                )
            ],
            [],
        )
        tracked_cap = manager.open_caps()[0]

        self.assertIsNone(tracked_cap.actuation_time)

    def test_defect_after_clean_actuation_crossing_is_skipped_as_late(self) -> None:
        manager = self.module.TrackedCapManager(
            merge_window_seconds=0.150,
            camera_count=2,
            timing_camera_index=0,
            anchor_axis="x",
            anchor_line_ratio=0.5,
            finalize_quiet_seconds=0.030,
        )

        manager.update(
            [
                self.module.TrackObservation(
                    camera_index=0,
                    track_id=1,
                    class_id=0,
                    box=[20.0, 10.0, 40.0, 30.0, 0.96, 0],
                    timestamp=1.000,
                    frame_size=(100, 40),
                )
            ],
            [],
        )
        manager.update(
            [
                self.module.TrackObservation(
                    camera_index=0,
                    track_id=1,
                    class_id=0,
                    box=[50.0, 10.0, 70.0, 30.0, 0.94, 0],
                    timestamp=2.000,
                    frame_size=(100, 40),
                )
            ],
            [],
        )
        tracked_cap = manager.open_caps()[0]
        self.assertIsNone(
            self.module.decide_decision_ready(
                tracked_cap,
                camera_count=2,
                timing_camera_index=0,
                decision_ready_time=2.050,
                merge_window_seconds=0.150,
                nozzle_distance_mm=100.0,
                belt_speed_mm_per_s=100.0,
                trigger_offset_s=0.0,
                latency_compensation_ms=0.0,
            )
        )

        manager.update(
            [
                self.module.TrackObservation(
                    camera_index=0,
                    track_id=1,
                    class_id=1,
                    box=[60.0, 10.0, 80.0, 30.0, 0.92, 1],
                    timestamp=2.100,
                    frame_size=(100, 40),
                )
            ],
            [],
        )

        decision = self.module.decide_tracked_cap(
            tracked_cap,
            camera_count=2,
            timing_camera_index=0,
            decision_time=2.200,
            merge_window_seconds=0.150,
            nozzle_distance_mm=100.0,
            belt_speed_mm_per_s=100.0,
            trigger_offset_s=0.0,
            latency_compensation_ms=0.0,
        )

        self.assertEqual("skip", decision.result)
        self.assertEqual("defect_below_threshold_at_actuation", decision.decision_source)
        self.assertEqual("undefected", decision.final_class_name)
        self.assertAlmostEqual(0.0, decision.final_score)

    def test_defect_that_never_crosses_actuation_line_is_skipped(self) -> None:
        manager = self.module.TrackedCapManager(
            merge_window_seconds=0.150,
            camera_count=2,
            timing_camera_index=0,
            anchor_axis="x",
            anchor_line_ratio=0.5,
            finalize_quiet_seconds=0.030,
        )
        manager.update(
            [
                self.module.TrackObservation(
                    camera_index=0,
                    track_id=1,
                    class_id=1,
                    box=[20.0, 10.0, 40.0, 30.0, 0.90, 1],
                    timestamp=1.000,
                    frame_size=(100, 40),
                )
            ],
            [],
        )
        tracked_cap = manager.open_caps()[0]

        decision = self.module.decide_tracked_cap(
            tracked_cap,
            camera_count=2,
            timing_camera_index=0,
            decision_time=1.200,
            merge_window_seconds=0.150,
            nozzle_distance_mm=100.0,
            belt_speed_mm_per_s=100.0,
            trigger_offset_s=0.0,
            latency_compensation_ms=0.0,
        )

        self.assertEqual("skip", decision.result)
        self.assertEqual("no_actuation_crossing", decision.decision_source)
        self.assertEqual("dirt_defect", decision.final_class_name)
        self.assertAlmostEqual(0.90, decision.final_score)
        self.assertEqual("missed_actuation", decision.review_reason)

    def test_no_actuation_clean_skip_does_not_request_debug_capture(self) -> None:
        manager = self.module.TrackedCapManager(
            merge_window_seconds=0.150,
            camera_count=2,
            timing_camera_index=0,
            anchor_axis="x",
            anchor_line_ratio=0.5,
            finalize_quiet_seconds=0.030,
        )
        manager.update(
            [
                self.module.TrackObservation(
                    camera_index=0,
                    track_id=1,
                    class_id=0,
                    box=[20.0, 10.0, 40.0, 30.0, 0.90, 0],
                    timestamp=1.000,
                    frame_size=(100, 40),
                )
            ],
            [],
        )
        tracked_cap = manager.open_caps()[0]

        decision = self.module.decide_tracked_cap(
            tracked_cap,
            camera_count=2,
            timing_camera_index=0,
            decision_time=1.200,
            merge_window_seconds=0.150,
            nozzle_distance_mm=100.0,
            belt_speed_mm_per_s=100.0,
            trigger_offset_s=0.0,
            latency_compensation_ms=0.0,
        )

        self.assertEqual("skip", decision.result)
        self.assertEqual("no_actuation_crossing", decision.decision_source)
        self.assertIsNone(decision.review_reason)

    def test_defect_first_seen_at_or_past_actuation_line_can_trigger(self) -> None:
        manager = self.module.TrackedCapManager(
            merge_window_seconds=0.150,
            camera_count=2,
            timing_camera_index=0,
            anchor_axis="x",
            anchor_line_ratio=0.5,
            finalize_quiet_seconds=0.030,
        )
        manager.update(
            [
                self.module.TrackObservation(
                    camera_index=0,
                    track_id=1,
                    class_id=1,
                    box=[50.0, 10.0, 70.0, 30.0, 0.90, 1],
                    timestamp=1.000,
                    frame_size=(100, 40),
                )
            ],
            [],
        )
        tracked_cap = manager.open_caps()[0]

        decision = self.module.decide_decision_ready(
            tracked_cap,
            camera_count=2,
            timing_camera_index=0,
            decision_ready_time=1.200,
            merge_window_seconds=0.150,
            nozzle_distance_mm=100.0,
            belt_speed_mm_per_s=100.0,
            trigger_offset_s=0.0,
            latency_compensation_ms=0.0,
        )

        self.assertIsNotNone(decision)
        self.assertEqual("trigger", decision.result)
        self.assertEqual("anchor_line", decision.anchor_source)
        self.assertAlmostEqual(1.000, decision.anchor_time)
        self.assertAlmostEqual(2.000, decision.requested_fire_time)

    def test_defect_seen_only_on_non_timing_camera_can_trigger_at_actuation_line(self) -> None:
        manager = self.module.TrackedCapManager(
            merge_window_seconds=0.150,
            camera_count=2,
            timing_camera_index=0,
            anchor_axis="x",
            anchor_line_ratio=0.5,
            finalize_quiet_seconds=0.030,
        )
        manager.update(
            [
                self.module.TrackObservation(
                    camera_index=1,
                    track_id=1,
                    class_id=1,
                    box=[50.0, 10.0, 70.0, 30.0, 0.91, 1],
                    timestamp=1.000,
                    frame_size=(100, 40),
                )
            ],
            [],
        )
        tracked_cap = manager.open_caps()[0]

        decision = self.module.decide_decision_ready(
            tracked_cap,
            camera_count=2,
            timing_camera_index=0,
            decision_ready_time=1.200,
            merge_window_seconds=0.150,
            nozzle_distance_mm=100.0,
            belt_speed_mm_per_s=100.0,
            trigger_offset_s=0.0,
            latency_compensation_ms=0.0,
        )

        self.assertIsNotNone(decision)
        self.assertEqual("trigger", decision.result)
        self.assertEqual("anchor_line", decision.anchor_source)
        self.assertAlmostEqual(1.000, decision.anchor_time)
        self.assertEqual("dirt_defect", decision.camera_votes[1].class_name)

    def test_same_frame_other_camera_defect_is_included_in_actuation_snapshot(self) -> None:
        manager = self.module.TrackedCapManager(
            merge_window_seconds=0.150,
            camera_count=2,
            timing_camera_index=0,
            anchor_axis="x",
            anchor_line_ratio=0.5,
            finalize_quiet_seconds=0.030,
        )
        manager.update(
            [
                self.module.TrackObservation(
                    camera_index=0,
                    track_id=1,
                    class_id=0,
                    box=[50.0, 10.0, 70.0, 30.0, 0.96, 0],
                    timestamp=1.000,
                    frame_size=(100, 40),
                ),
                self.module.TrackObservation(
                    camera_index=1,
                    track_id=1,
                    class_id=1,
                    box=[50.0, 10.0, 70.0, 30.0, 0.93, 1],
                    timestamp=1.000,
                    frame_size=(100, 40),
                ),
            ],
            [],
        )
        tracked_cap = manager.open_caps()[0]

        decision = self.module.decide_decision_ready(
            tracked_cap,
            camera_count=2,
            timing_camera_index=0,
            decision_ready_time=1.050,
            merge_window_seconds=0.150,
            nozzle_distance_mm=100.0,
            belt_speed_mm_per_s=100.0,
            trigger_offset_s=0.0,
            latency_compensation_ms=0.0,
        )

        self.assertIsNotNone(decision)
        self.assertEqual("trigger", decision.result)
        self.assertEqual("undefected", decision.camera_votes[0].class_name)
        self.assertEqual("dirt_defect", decision.camera_votes[1].class_name)
        self.assertAlmostEqual(0.93, decision.final_score)

    def test_line_frame_defect_wins_over_higher_confidence_clean_camera(self) -> None:
        manager = self.module.TrackedCapManager(
            merge_window_seconds=0.150,
            camera_count=2,
            timing_camera_index=0,
            anchor_axis="x",
            anchor_line_ratio=0.5,
            finalize_quiet_seconds=0.030,
        )
        manager.update(
            [
                self.module.TrackObservation(
                    camera_index=0,
                    track_id=1,
                    class_id=0,
                    box=[50.0, 10.0, 70.0, 30.0, 0.99, 0],
                    timestamp=1.000,
                    frame_size=(100, 40),
                ),
                self.module.TrackObservation(
                    camera_index=1,
                    track_id=1,
                    class_id=1,
                    box=[50.0, 10.0, 70.0, 30.0, 0.46, 1],
                    timestamp=1.000,
                    frame_size=(100, 40),
                ),
            ],
            [],
        )
        tracked_cap = manager.open_caps()[0]

        decision = self.module.decide_decision_ready(
            tracked_cap,
            camera_count=2,
            timing_camera_index=0,
            decision_ready_time=1.050,
            merge_window_seconds=0.150,
            nozzle_distance_mm=100.0,
            belt_speed_mm_per_s=100.0,
            trigger_offset_s=0.0,
            latency_compensation_ms=0.0,
        )

        self.assertIsNotNone(decision)
        self.assertEqual("trigger", decision.result)
        self.assertEqual("dirt_defect", decision.final_class_name)
        self.assertAlmostEqual(0.46, decision.final_score)
        self.assertEqual("undefected", decision.camera_votes[0].class_name)
        self.assertEqual("dirt_defect", decision.camera_votes[1].class_name)

    def test_defect_first_seen_past_actuation_line_keeps_first_anchor(
        self,
    ) -> None:
        manager = self.module.TrackedCapManager(
            merge_window_seconds=0.150,
            camera_count=2,
            timing_camera_index=0,
            anchor_axis="x",
            anchor_line_ratio=0.5,
            finalize_quiet_seconds=0.030,
        )
        decision_kwargs = {
            "camera_count": 2,
            "timing_camera_index": 0,
            "decision_ready_time": 1.200,
            "merge_window_seconds": 0.150,
            "nozzle_distance_mm": 100.0,
            "belt_speed_mm_per_s": 100.0,
            "trigger_offset_s": 0.0,
            "latency_compensation_ms": 0.0,
        }

        manager.update(
            [
                self.module.TrackObservation(
                    camera_index=0,
                    track_id=1,
                    class_id=1,
                    box=[50.0, 10.0, 70.0, 30.0, 0.90, 1],
                    timestamp=1.000,
                    frame_size=(100, 40),
                )
            ],
            [],
        )
        tracked_cap = manager.open_caps()[0]

        first_decision = self.module.decide_decision_ready(tracked_cap, **decision_kwargs)

        self.assertIsNotNone(first_decision)
        self.assertEqual("trigger", first_decision.result)
        self.assertEqual("anchor_line", first_decision.anchor_source)
        self.assertAlmostEqual(1.000, first_decision.anchor_time)
        self.assertAlmostEqual(2.000, first_decision.requested_fire_time)

        manager.update(
            [
                self.module.TrackObservation(
                    camera_index=0,
                    track_id=1,
                    class_id=1,
                    box=[60.0, 10.0, 80.0, 30.0, 0.88, 1],
                    timestamp=2.000,
                    frame_size=(100, 40),
                )
            ],
            [],
        )
        decision_kwargs["decision_ready_time"] = 2.200

        decision = self.module.decide_decision_ready(tracked_cap, **decision_kwargs)

        self.assertIsNotNone(decision)
        self.assertEqual("trigger", decision.result)
        self.assertEqual("anchor_line", decision.anchor_source)
        self.assertAlmostEqual(1.000, decision.anchor_time)
        self.assertAlmostEqual(2.000, decision.requested_fire_time)

    def test_late_camera_updates_actuation_snapshot_and_votes(self) -> None:
        manager = self.module.TrackedCapManager(
            merge_window_seconds=0.150,
            camera_count=2,
            timing_camera_index=0,
            anchor_axis="x",
            anchor_line_ratio=0.5,
            finalize_quiet_seconds=0.030,
        )
        manager.update(
            [
                self.module.TrackObservation(
                    camera_index=1,
                    track_id=1,
                    class_id=0,
                    box=[50.0, 10.0, 70.0, 30.0, 0.88, 0],
                    timestamp=1.000,
                    frame_size=(100, 40),
                )
            ],
            [],
        )
        tracked_cap = manager.open_caps()[0]
        self.assertIsNotNone(tracked_cap.actuation_time)

        manager.update(
            [
                self.module.TrackObservation(
                    camera_index=0,
                    track_id=1,
                    class_id=0,
                    box=[50.0, 10.0, 70.0, 30.0, 0.91, 0],
                    timestamp=1.040,
                    frame_size=(100, 40),
                )
            ],
            [],
        )

        decision = self.module.decide_tracked_cap(
            tracked_cap,
            camera_count=2,
            timing_camera_index=0,
            decision_time=1.200,
            merge_window_seconds=0.150,
            nozzle_distance_mm=100.0,
            belt_speed_mm_per_s=100.0,
            trigger_offset_s=0.0,
            latency_compensation_ms=0.0,
        )

        self.assertEqual("skip", decision.result)
        self.assertIsNotNone(decision.camera_votes[0].class_name)
        self.assertEqual("undefected", decision.camera_votes[0].class_name)
        self.assertEqual("undefected", decision.camera_votes[1].class_name)

    def test_staggered_same_cap_stays_single_event(self) -> None:
        manager = self.module.TrackedCapManager(
            merge_window_seconds=0.150,
            camera_count=2,
            timing_camera_index=0,
            anchor_axis="x",
            anchor_line_ratio=0.5,
            finalize_quiet_seconds=0.030,
        )
        manager.update(
            [
                self.module.TrackObservation(
                    camera_index=1,
                    track_id=1,
                    class_id=1,
                    box=[50.0, 10.0, 70.0, 30.0, 0.91, 1],
                    timestamp=1.000,
                    frame_size=(100, 40),
                )
            ],
            [],
        )
        manager.update(
            [
                self.module.TrackObservation(
                    camera_index=0,
                    track_id=1,
                    class_id=0,
                    box=[50.0, 10.0, 70.0, 30.0, 0.96, 0],
                    timestamp=1.040,
                    frame_size=(100, 40),
                )
            ],
            [],
        )

        self.assertEqual(1, len(manager.open_caps()))
        tracked_cap = manager.open_caps()[0]
        self.assertEqual({0, 1}, tracked_cap.camera_indices)

        decision = self.module.decide_decision_ready(
            tracked_cap,
            camera_count=2,
            timing_camera_index=0,
            decision_ready_time=1.200,
            merge_window_seconds=0.150,
            nozzle_distance_mm=100.0,
            belt_speed_mm_per_s=100.0,
            trigger_offset_s=0.0,
            latency_compensation_ms=0.0,
        )

        self.assertIsNotNone(decision)
        self.assertEqual("trigger", decision.result)
        self.assertIsNotNone(decision.camera_votes[0].class_name)
        self.assertEqual("dirt_defect", decision.camera_votes[1].class_name)

    def test_early_trigger_waits_for_merge_window(self) -> None:
        manager = self.module.TrackedCapManager(
            merge_window_seconds=0.150,
            camera_count=2,
            timing_camera_index=0,
            anchor_axis="x",
            anchor_line_ratio=0.5,
            finalize_quiet_seconds=0.030,
        )
        manager.update(
            [
                self.module.TrackObservation(
                    camera_index=1,
                    track_id=1,
                    class_id=1,
                    box=[50.0, 10.0, 70.0, 30.0, 0.91, 1],
                    timestamp=1.000,
                    frame_size=(100, 40),
                )
            ],
            [],
        )
        tracked_cap = manager.open_caps()[0]
        decision_kwargs = {
            "camera_count": 2,
            "timing_camera_index": 0,
            "decision_ready_time": 1.100,
            "merge_window_seconds": 0.150,
            "nozzle_distance_mm": 100.0,
            "belt_speed_mm_per_s": 100.0,
            "trigger_offset_s": 0.0,
            "latency_compensation_ms": 0.0,
        }

        self.assertIsNone(self.module.decide_decision_ready(tracked_cap, **decision_kwargs))

        decision_kwargs["decision_ready_time"] = 1.200
        decision = self.module.decide_decision_ready(tracked_cap, **decision_kwargs)

        self.assertIsNotNone(decision)
        self.assertEqual("trigger", decision.result)

    def test_cross_camera_prefers_cap_missing_camera(self) -> None:
        manager = self.module.TrackedCapManager(
            merge_window_seconds=0.150,
            camera_count=2,
            timing_camera_index=0,
            anchor_axis="x",
            anchor_line_ratio=0.5,
            finalize_quiet_seconds=0.030,
        )
        manager.update(
            [
                self.module.TrackObservation(
                    camera_index=0,
                    track_id=1,
                    class_id=0,
                    box=[10.0, 10.0, 20.0, 20.0, 0.90, 0],
                    timestamp=1.000,
                    frame_size=(100, 40),
                )
            ],
            [],
        )
        manager.update(
            [
                self.module.TrackObservation(
                    camera_index=0,
                    track_id=1,
                    class_id=0,
                    box=[20.0, 10.0, 30.0, 20.0, 0.90, 0],
                    timestamp=1.050,
                    frame_size=(100, 40),
                )
            ],
            [],
        )
        manager.update(
            [],
            [
                self.module.ClosedTrack(
                    camera_index=0,
                    track_id=1,
                    box=[20.0, 10.0, 30.0, 20.0, 0.90, 0],
                    last_seen_at=1.050,
                )
            ],
        )
        manager.update(
            [
                self.module.TrackObservation(
                    camera_index=1,
                    track_id=1,
                    class_id=0,
                    box=[20.0, 10.0, 40.0, 30.0, 0.85, 0],
                    timestamp=1.220,
                    frame_size=(100, 40),
                )
            ],
            [],
        )
        self.assertEqual(2, len(manager.open_caps()))

        manager.update(
            [
                self.module.TrackObservation(
                    camera_index=0,
                    track_id=2,
                    class_id=0,
                    box=[33.0, 10.0, 43.0, 20.0, 0.88, 0],
                    timestamp=1.250,
                    frame_size=(100, 40),
                )
            ],
            [],
        )
        cam0_cap = next(
            tracked_cap
            for tracked_cap in manager.open_caps()
            if 0 in tracked_cap.camera_indices and 1 not in tracked_cap.camera_indices
        )

        manager.update(
            [
                self.module.TrackObservation(
                    camera_index=1,
                    track_id=2,
                    class_id=0,
                    box=[35.0, 10.0, 45.0, 30.0, 0.86, 0],
                    timestamp=1.280,
                    frame_size=(100, 40),
                )
            ],
            [],
        )

        self.assertIn(1, cam0_cap.camera_indices)


class CapLineRuntimeV2TrackingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = load_module("cap_line_runtime_v2")

    def make_observation(
        self,
        *,
        camera_index: int,
        track_id: int,
        timestamp: float,
        box: list[float] | None = None,
    ):
        return self.module.TrackObservation(
            camera_index=camera_index,
            track_id=track_id,
            class_id=1,
            box=box or [10.0, 10.0, 20.0, 20.0, 0.82, 1],
            timestamp=timestamp,
            frame_size=(32, 32),
        )

    def test_parser_has_no_same_camera_delay_knob(self) -> None:
        parser = self.module.build_arg_parser()
        option_strings = set(parser._option_string_actions)

        self.assertNotIn("--same-camera-hold-ms", option_strings)

    def test_same_camera_reattach_uses_coordinate_trajectory_not_delay(self) -> None:
        manager = self.module.TrackedCapManager(
            merge_window_seconds=0.150,
            camera_count=2,
            timing_camera_index=0,
            anchor_axis="x",
            anchor_line_ratio=0.5,
            finalize_quiet_seconds=0.030,
        )

        manager.update(
            [
                self.make_observation(
                    camera_index=0,
                    track_id=1,
                    timestamp=1.000,
                    box=[10.0, 10.0, 20.0, 20.0, 0.82, 1],
                )
            ],
            [],
        )
        manager.update(
            [
                self.make_observation(
                    camera_index=0,
                    track_id=1,
                    timestamp=1.100,
                    box=[20.0, 10.0, 30.0, 20.0, 0.84, 1],
                )
            ],
            [],
        )
        manager.update(
            [],
            [
                self.module.ClosedTrack(
                    camera_index=0,
                    track_id=1,
                    box=[20.0, 10.0, 30.0, 20.0, 0.84, 1],
                    last_seen_at=1.100,
                )
            ],
        )

        reattached_touch = manager.update(
            [
                self.make_observation(
                    camera_index=0,
                    track_id=2,
                    timestamp=5.000,
                    box=[33.0, 10.0, 43.0, 20.0, 0.86, 1],
                )
            ],
            [],
        )

        self.assertEqual([1], [tracked_cap.event_id for tracked_cap in reattached_touch])

    def test_coordinate_matching_keeps_back_to_back_caps_separate(self) -> None:
        manager = self.module.TrackedCapManager(
            merge_window_seconds=0.150,
            camera_count=2,
            timing_camera_index=0,
            anchor_axis="x",
            anchor_line_ratio=0.5,
            finalize_quiet_seconds=0.030,
        )

        first_touch = manager.update(
            [self.make_observation(camera_index=0, track_id=1, timestamp=1.000)],
            [],
        )
        manager.update(
            [],
            [
                self.module.ClosedTrack(
                    camera_index=0,
                    track_id=1,
                    box=[10.0, 10.0, 20.0, 20.0, 0.82, 1],
                    last_seen_at=1.000,
                )
            ],
        )
        manager.update(
            [self.make_observation(camera_index=1, track_id=1, timestamp=1.050)],
            [],
        )

        second_touch = manager.update(
            [self.make_observation(camera_index=0, track_id=2, timestamp=1.250)],
            [],
        )

        self.assertEqual([1], [tracked_cap.event_id for tracked_cap in first_touch])
        self.assertEqual([2], [tracked_cap.event_id for tracked_cap in second_touch])

    def test_overlapping_duplicate_yolo_boxes_keep_one_box(self) -> None:
        boxes = [
            [10.0, 10.0, 30.0, 30.0, 0.82, 1],
            [11.0, 11.0, 31.0, 31.0, 0.81, 1],
            [70.0, 10.0, 90.0, 30.0, 0.83, 1],
        ]

        deduplicated = self.module.deduplicate_yolo_boxes(boxes)

        self.assertEqual(2, len(deduplicated))
        self.assertEqual(
            [[70.0, 10.0, 90.0, 30.0, 0.83, 1], [10.0, 10.0, 30.0, 30.0, 0.82, 1]],
            deduplicated,
        )

    def test_duplicate_clean_and_defect_boxes_preserve_defect_box(self) -> None:
        boxes = [
            [10.0, 10.0, 30.0, 30.0, 0.95, 0],
            [11.0, 11.0, 31.0, 31.0, 0.72, 1],
        ]

        deduplicated = self.module.deduplicate_yolo_boxes(boxes)

        self.assertEqual([[11.0, 11.0, 31.0, 31.0, 0.72, 1]], deduplicated)

    def test_camera_properties_warn_when_actual_resolution_differs(self) -> None:
        width_prop = self.module.CAP_PROP_FRAME_WIDTH
        height_prop = self.module.CAP_PROP_FRAME_HEIGHT
        fps_prop = self.module.CAP_PROP_FPS

        class FakeCamera:
            def get(self, prop_id):
                return {
                    width_prop: 960.0,
                    height_prop: 600.0,
                    fps_prop: 15.0,
                }[prop_id]

        logs = []
        properties = self.module.read_camera_properties(
            FakeCamera(),
            camera_index=0,
            source="0",
            requested_width=640,
            requested_height=480,
            requested_fps=15,
        )

        self.module.log_camera_properties(properties, log_fn=logs.append)

        self.assertEqual(960, properties.actual_width)
        self.assertTrue(any("[CAMERA] index=0" in message for message in logs))
        self.assertTrue(any("[CAMERA][WARN]" in message for message in logs))

    def test_parallel_inference_preserves_camera_order(self) -> None:
        module = self.module
        original_preprocess = module.preprocess
        original_postprocess = module.postprocess

        class FakeClock:
            def monotonic(self):
                return time.monotonic()

        class FakeSession:
            def __init__(self, camera_index):
                self.camera_index = camera_index

            def run(self, _outputs, _inputs):
                if self.camera_index == 0:
                    time.sleep(0.02)
                return [self.camera_index]

        try:
            module.preprocess = lambda frame, _imgsz: (frame, {"frame": frame})
            module.postprocess = lambda output, _meta: [[float(output), 0.0, 1.0, 1.0, 0.9, 1]]
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                results = module.infer_frame_pair(
                    ["frame0", "frame1"],
                    [FakeSession(0), FakeSession(1)],
                    ["images", "images"],
                    640,
                    FakeClock(),
                    executor,
                )
        finally:
            module.preprocess = original_preprocess
            module.postprocess = original_postprocess

        self.assertEqual([0, 1], [result.camera_index for result in results])
        self.assertEqual(0.0, results[0].boxes[0][0])
        self.assertEqual(1.0, results[1].boxes[0][0])

    def test_paired_inference_preserves_frame_pair_and_camera_order(self) -> None:
        module = self.module
        original_preprocess = module.preprocess
        original_postprocess = module.postprocess

        class FakeClock:
            def monotonic(self):
                return time.monotonic()

        class FakeSession:
            def __init__(self, camera_index):
                self.camera_index = camera_index

            def run(self, _outputs, _inputs):
                return [self.camera_index]

        frame_pair = module.build_frame_pair(
            [
                module.CapturedFrame(0, "frame0", 1.000, 4),
                module.CapturedFrame(1, "frame1", 1.020, 7),
            ]
        )

        try:
            module.preprocess = lambda frame, _imgsz: (frame, {"frame": frame})
            module.postprocess = lambda output, _meta: [[float(output), 0.0, 1.0, 1.0, 0.9, 1]]
            paired_result = module.infer_paired_frame(
                frame_pair,
                [FakeSession(0), FakeSession(1)],
                ["images", "images"],
                640,
                FakeClock(),
                None,
                serial=True,
            )
        finally:
            module.preprocess = original_preprocess
            module.postprocess = original_postprocess

        self.assertIs(frame_pair, paired_result.frame_pair)
        self.assertEqual([0, 1], [result.camera_index for result in paired_result.camera_results])
        self.assertEqual(0.0, paired_result.boxes_by_camera[0][0][0])
        self.assertEqual(1.0, paired_result.boxes_by_camera[1][0][0])

    def test_latest_frame_reader_continues_capture_while_caller_waits(self) -> None:
        module = self.module

        class FakeCamera:
            def __init__(self):
                self.read_count = 0
                self.released = False

            def read(self):
                self.read_count += 1
                time.sleep(0.001)
                return True, f"frame-{self.read_count}"

            def release(self):
                self.released = True

        fake_camera = FakeCamera()
        reader = module.LatestFrameCameraReader(
            fake_camera,
            camera_index=0,
            target_fps=240,
            time_fn=time.monotonic,
            sleep_fn=time.sleep,
        )
        reader.start()
        try:
            first = None
            deadline = time.monotonic() + 0.2
            while first is None and time.monotonic() < deadline:
                first = reader.latest()
                time.sleep(0.005)

            self.assertIsNotNone(first)
            time.sleep(0.05)
            latest = reader.latest()
        finally:
            reader.stop()

        self.assertIsNotNone(latest)
        self.assertGreater(latest.sequence, first.sequence + 1)
        self.assertGreater(fake_camera.read_count, 2)
        self.assertFalse(fake_camera.released)

    def test_fresh_frame_pair_requires_every_camera_to_advance(self) -> None:
        module = self.module

        self.assertTrue(module.is_fresh_frame_pair([1, 1], None))
        self.assertTrue(module.is_fresh_frame_pair([2, 2], [1, 1]))
        self.assertFalse(module.is_fresh_frame_pair([2, 1], [1, 1]))
        self.assertFalse(module.is_fresh_frame_pair([1, 2], [1, 1]))

    def test_select_synchronized_frame_pair_accepts_close_pair(self) -> None:
        module = self.module
        stats = module.PairingStats()
        frame_pair = module.select_synchronized_frame_pair(
            [
                module.CapturedFrame(0, "cam0", 1.000, 3, read_duration_ms=1.0),
                module.CapturedFrame(1, "cam1", 1.030, 5, read_duration_ms=2.0),
            ],
            None,
            max_skew_ms=40.0,
            pairing_stats=stats,
        )

        self.assertIsNotNone(frame_pair)
        self.assertEqual(["cam0", "cam1"], frame_pair.frames)
        self.assertEqual([3, 5], frame_pair.sequences)
        self.assertAlmostEqual(30.0, frame_pair.skew_ms)
        self.assertEqual(1, stats.accepted_pairs)

    def test_select_synchronized_frame_pair_rejects_stale_pair(self) -> None:
        module = self.module
        logs = []
        stats = module.PairingStats()
        frame_pair = module.select_synchronized_frame_pair(
            [
                module.CapturedFrame(0, "cam0", 1.000, 3),
                module.CapturedFrame(1, "cam1", 1.060, 5),
            ],
            None,
            max_skew_ms=40.0,
            pairing_stats=stats,
            log_fn=logs.append,
        )

        self.assertIsNone(frame_pair)
        self.assertEqual(1, stats.stale_pair_drops)
        self.assertTrue(any("dropped stale camera pair" in message for message in logs))

    def test_select_synchronized_frame_pair_rejects_one_camera_pair(self) -> None:
        module = self.module
        frame_pair = module.select_synchronized_frame_pair(
            [module.CapturedFrame(0, "cam0", 1.000, 3), None],
            None,
            max_skew_ms=40.0,
        )

        self.assertIsNone(frame_pair)


class CapLineRuntimeV2RunLoopTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = load_module("cap_line_runtime_v2")
        self.original_cv2 = sys.modules.get("cv2")
        self.original_onnxruntime = sys.modules.get("onnxruntime")
        sys.modules["cv2"] = types.SimpleNamespace(
            CAP_V4L2=0,
            WINDOW_NORMAL=0,
            namedWindow=lambda *args, **kwargs: None,
            destroyAllWindows=lambda: None,
            waitKey=lambda *_args, **_kwargs: -1,
        )
        sys.modules["onnxruntime"] = types.SimpleNamespace(
            InferenceSession=self._make_session_class()
        )

    def tearDown(self) -> None:
        if self.original_cv2 is None:
            sys.modules.pop("cv2", None)
        else:
            sys.modules["cv2"] = self.original_cv2
        if self.original_onnxruntime is None:
            sys.modules.pop("onnxruntime", None)
        else:
            sys.modules["onnxruntime"] = self.original_onnxruntime

    @staticmethod
    def _make_session_class():
        class FakeSession:
            def __init__(self, *_args, **_kwargs):
                self._inputs = [types.SimpleNamespace(name="images", shape=[1, 3, 640, 640])]

            def get_inputs(self):
                return self._inputs

            def run(self, *_args, **_kwargs):
                return [None]

        return FakeSession

    def test_clean_cap_at_line_submits_line_picture_task(self) -> None:
        module = self.module
        args = module.parse_args(
            [
                "--no-display",
                "--merge-window-ms",
                "0",
                "--finalize-quiet-ms",
                "0",
                "--max-missing-frames",
                "0",
                "--serial-inference",
            ]
        )
        args.simulate_gpio = True

        class FakeFrame:
            shape = (32, 32, 3)

            def __init__(self, label):
                self.label = label

            def copy(self):
                return self

        class FakeCamera:
            def __init__(self):
                self._frames = [FakeFrame("line"), FakeFrame("after"), FakeFrame("after")]
                self._index = 0

            def read(self):
                frame = self._frames[min(self._index, len(self._frames) - 1)]
                self._index += 1
                return True, frame

            def release(self):
                return None

        class FakeTimingLogger:
            def __init__(self, directory):
                self.directory = directory
                self.records = []

            def log(self, record):
                self.records.append(record)
                return "timing.csv"

        class FakeReviewWriter:
            instances = []

            def __init__(self, directory, **_kwargs):
                self.directory = directory
                self.submissions = []
                self.line_picture_tasks = []
                FakeReviewWriter.instances.append(self)

            def submit(self, task):
                self.submissions.append(task)
                return len(self.submissions)

            def submit_line_picture(self, task):
                self.line_picture_tasks.append(task)
                return len(self.line_picture_tasks)

            def close(self):
                return None

        box_schedule = iter(
            [
                [[12.0, 10.0, 20.0, 20.0, 0.96, 0]],
                [[12.0, 10.0, 20.0, 20.0, 0.94, 0]],
                [],
                [],
            ]
        )

        original_open_cam = module.open_cam
        original_set_camera_format = module.set_camera_format
        original_set_camera_controls = module.set_camera_controls
        original_timing_logger = module.TimingCsvLogger
        original_review_writer = module.ReviewCaptureWriter
        original_preprocess = module.preprocess
        original_postprocess = module.postprocess
        original_draw_boxes = module.draw_boxes
        original_draw_anchor_line = module.draw_anchor_line
        original_compose_preview = module.compose_preview
        original_resolve_model_path = module.resolve_model_path

        try:
            module.open_cam = lambda *_args, **_kwargs: FakeCamera()
            module.set_camera_format = lambda *_args, **_kwargs: None
            module.set_camera_controls = lambda *_args, **_kwargs: None
            module.TimingCsvLogger = FakeTimingLogger
            module.ReviewCaptureWriter = FakeReviewWriter
            module.preprocess = lambda frame, _imgsz: (frame, {"frame_shape": frame.shape})
            module.postprocess = lambda *_args, **_kwargs: next(box_schedule)
            module.draw_boxes = lambda frame, _boxes: frame
            module.draw_anchor_line = lambda frame, _axis, _ratio: frame
            module.compose_preview = lambda _frames, pad=6: None
            module.resolve_model_path = lambda _model: ("fake.onnx", None)

            stop_event = module.threading.Event()
            history_records = []

            def on_history(record):
                history_records.append(record)
                stop_event.set()

            module.run_detection(
                args,
                stop_event=stop_event,
                history_callback=on_history,
                log_fn=lambda *_args, **_kwargs: None,
            )
        finally:
            module.open_cam = original_open_cam
            module.set_camera_format = original_set_camera_format
            module.set_camera_controls = original_set_camera_controls
            module.TimingCsvLogger = original_timing_logger
            module.ReviewCaptureWriter = original_review_writer
            module.preprocess = original_preprocess
            module.postprocess = original_postprocess
            module.draw_boxes = original_draw_boxes
            module.draw_anchor_line = original_draw_anchor_line
            module.compose_preview = original_compose_preview
            module.resolve_model_path = original_resolve_model_path

        self.assertEqual(1, len(history_records))
        self.assertEqual("skip", history_records[0].result)
        self.assertEqual(1, len(FakeReviewWriter.instances))
        self.assertEqual(1, len(FakeReviewWriter.instances[0].line_picture_tasks))
        line_picture_task = FakeReviewWriter.instances[0].line_picture_tasks[0]
        self.assertEqual("skip", line_picture_task.result)
        self.assertEqual("undefected", line_picture_task.final_class)
        self.assertEqual(2, len(line_picture_task.raw_frames))

    def test_early_trigger_queues_only_once_for_one_cap(self) -> None:
        module = self.module
        args = module.parse_args(
            [
                "--no-display",
                "--merge-window-ms",
                "0",
                "--finalize-quiet-ms",
                "0",
                "--max-missing-frames",
                "0",
                "--serial-inference",
            ]
        )
        args.simulate_gpio = True

        class FakeFrame:
            shape = (32, 32, 3)

            def copy(self):
                return self

        class FakeCamera:
            def __init__(self):
                self._frames = [FakeFrame(), FakeFrame(), FakeFrame()]
                self._index = 0

            def read(self):
                frame = self._frames[min(self._index, len(self._frames) - 1)]
                self._index += 1
                return True, frame

            def release(self):
                return None

        class FakeTimingLogger:
            def __init__(self, directory):
                self.directory = directory
                self.records = []

            def log(self, record):
                self.records.append(record)
                return "timing.csv"

        class FakeReviewWriter:
            def __init__(self, directory, **_kwargs):
                self.directory = directory
                self.submissions = []

            def submit(self, task):
                self.submissions.append(task)
                return len(self.submissions)

            def close(self):
                return None

        class FakeScheduler:
            instances = []

            def __init__(self, *_args, **kwargs):
                self.time_fn = kwargs["time_fn"]
                self.enqueued = []
                FakeScheduler.instances.append(self)

            def enqueue(self, event_id, requested_fire_time, *, completion_callback=None):
                queued_at = float(self.time_fn())
                self.enqueued.append((event_id, requested_fire_time))
                if completion_callback is not None:
                    completion_callback(
                        module.RejectExecution(
                            event_id=event_id,
                            queued_at=queued_at,
                            requested_fire_time=requested_fire_time,
                            trigger_on_time=requested_fire_time,
                            trigger_off_time=requested_fire_time + 0.3,
                        )
                    )
                return module.RejectEnqueueResult(
                    queue_depth=len(self.enqueued),
                    queued_at=queued_at,
                    requested_fire_time=requested_fire_time,
                )

            def close(self):
                return None

        box_schedule = iter(
            [
                [[10.0, 10.0, 20.0, 20.0, 0.82, 1]],
                [],
                [[10.0, 10.0, 20.0, 20.0, 0.85, 1]],
                [],
                [],
                [],
            ]
        )

        original_open_cam = module.open_cam
        original_set_camera_controls = module.set_camera_controls
        original_timing_logger = module.TimingCsvLogger
        original_review_writer = module.ReviewCaptureWriter
        original_scheduler = module.RejectScheduler
        original_preprocess = module.preprocess
        original_postprocess = module.postprocess
        original_draw_boxes = module.draw_boxes
        original_draw_anchor_line = module.draw_anchor_line
        original_compose_preview = module.compose_preview
        original_resolve_model_path = module.resolve_model_path

        try:
            module.open_cam = lambda *_args, **_kwargs: FakeCamera()
            module.set_camera_controls = lambda *_args, **_kwargs: None
            module.TimingCsvLogger = FakeTimingLogger
            module.ReviewCaptureWriter = FakeReviewWriter
            module.RejectScheduler = FakeScheduler
            module.preprocess = lambda frame, _imgsz: (frame, {"frame_shape": frame.shape})
            module.postprocess = lambda *_args, **_kwargs: next(box_schedule)
            module.draw_boxes = lambda frame, _boxes: frame
            module.draw_anchor_line = lambda frame, _axis, _ratio: frame
            module.compose_preview = lambda _frames, pad=6: None
            module.resolve_model_path = lambda _model: ("fake.onnx", None)

            stop_event = module.threading.Event()
            history_records = []

            def on_history(record):
                history_records.append(record)
                stop_event.set()

            module.run_detection(
                args,
                stop_event=stop_event,
                history_callback=on_history,
                log_fn=lambda *_args, **_kwargs: None,
            )
        finally:
            module.open_cam = original_open_cam
            module.set_camera_controls = original_set_camera_controls
            module.TimingCsvLogger = original_timing_logger
            module.ReviewCaptureWriter = original_review_writer
            module.RejectScheduler = original_scheduler
            module.preprocess = original_preprocess
            module.postprocess = original_postprocess
            module.draw_boxes = original_draw_boxes
            module.draw_anchor_line = original_draw_anchor_line
            module.compose_preview = original_compose_preview
            module.resolve_model_path = original_resolve_model_path

        self.assertEqual(1, len(FakeScheduler.instances))
        self.assertEqual(1, len(FakeScheduler.instances[0].enqueued))
        self.assertEqual(1, len(history_records))
        self.assertEqual("trigger", history_records[0].result)


if __name__ == "__main__":
    unittest.main()
