from __future__ import annotations

import ast
import importlib.util
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


def iter_v3_source_files() -> list[Path]:
    return [
        REPO_ROOT / "cap_line_runtime_v3.py",
        *sorted((REPO_ROOT / "cap_line_v3").glob("*.py")),
    ]


class FakeFrame:
    def __init__(self, label: str, shape: tuple[int, int, int] = (64, 64, 3)):
        self.label = label
        self.shape = shape

    def copy(self):
        return FakeFrame(self.label, self.shape)


class CapLineRuntimeV3Tests(unittest.TestCase):
    def test_v3_runtime_does_not_import_v1_or_v2_runtime_modules(self) -> None:
        forbidden = {
            "cap_line_runtime",
            "cap_line_runtime_v2",
            "cap_line_runtime_v2_grey",
            "cap_line_ui",
            "cap_line_ui_v2",
            "cap_line_ui_v2_grey",
        }

        for path in iter_v3_source_files():
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported = {alias.name for alias in node.names}
                    self.assertTrue(
                        imported.isdisjoint(forbidden),
                        f"{path.name} imports {imported & forbidden}",
                    )
                elif isinstance(node, ast.ImportFrom) and node.module:
                    self.assertNotIn(node.module, forbidden, f"{path.name} imports {node.module}")

    def test_runtime_config_defaults_mirror_v2_rgb_operator_settings(self) -> None:
        module = load_module("cap_line_runtime_v3")

        config = module.RuntimeConfig.defaults()

        self.assertEqual("dirtv2.onnx", config.model)
        self.assertEqual(("0", "3"), config.cameras)
        self.assertEqual((960, 600), config.resolution)
        self.assertEqual(60, config.target_fps)
        self.assertEqual("YUYV", config.pixel_format)
        self.assertEqual(8, config.exposure)
        self.assertEqual(0.45, config.tracking_threshold)
        self.assertEqual(0.45, config.reject_threshold)
        self.assertEqual(40.0, config.pair_max_skew_ms)
        self.assertEqual(150.0, config.merge_window_ms)
        self.assertEqual(1, config.max_missing_frames)
        self.assertEqual(0.3, config.track_iou)
        self.assertTrue(config.debug_dir.replace("\\", "/").endswith("resources/debugging_v3"))
        self.assertTrue(config.pictures_dir.replace("\\", "/").endswith("resources/pictures_v3"))
        self.assertTrue(config.timing_log_dir.replace("\\", "/").endswith("data/timing_logs_v3"))

    def test_parser_exposes_v3_config_without_old_policy_knobs(self) -> None:
        module = load_module("cap_line_runtime_v3")
        parser = module.build_arg_parser()
        option_strings = set(parser._option_string_actions)

        self.assertIn("--tracking-threshold", option_strings)
        self.assertIn("--reject-threshold", option_strings)
        self.assertIn("--target-fps", option_strings)
        self.assertNotIn("--conf", option_strings)
        self.assertNotIn("--defect-min-score", option_strings)
        self.assertNotIn("--defect-margin", option_strings)
        self.assertNotIn("--single-camera-defect-score", option_strings)

        config = module.config_from_args(parser.parse_args(["--cams", "1", "4", "--target-fps", "120"]))
        self.assertEqual(("1", "4"), config.cameras)
        self.assertEqual(120, config.target_fps)

    def test_overlay_stale_timeout_uses_configured_target_fps(self) -> None:
        module = load_module("cap_line_runtime_v3")

        self.assertAlmostEqual(0.30, module.overlay_stale_timeout_s(10))
        self.assertAlmostEqual(0.10, module.overlay_stale_timeout_s(60))
        self.assertAlmostEqual(0.10, module.overlay_stale_timeout_s(120))
        self.assertAlmostEqual(0.35, module.overlay_stale_timeout_s(5))

    def test_preview_draws_current_actuation_box_without_prediction(self) -> None:
        module = load_module("cap_line_runtime_v3")
        previous = module.DetectionPacket(
            frame_pair=module.FramePair(
                frames=(module.CapturedFrame(0, FakeFrame("prev"), 1.00, 1),),
                pair_timestamp=1.00,
                skew_ms=0.0,
            ),
            boxes_by_camera=(((10.0, 10.0, 20.0, 20.0, 0.9, 1),),),
            inference_ms_by_camera=(1.0,),
        )
        current = module.DetectionPacket(
            frame_pair=module.FramePair(
                frames=(module.CapturedFrame(0, FakeFrame("current"), 1.10, 2),),
                pair_timestamp=1.10,
                skew_ms=0.0,
            ),
            boxes_by_camera=(((20.0, 10.0, 40.0, 20.0, 0.9, 1),),),
            inference_ms_by_camera=(1.0,),
        )
        live_frames = (module.CapturedFrame(0, FakeFrame("live"), 1.20, 3),)

        overlay = module.predict_preview_overlay(
            previous,
            current,
            live_frames,
            target_fps=60,
        )

        self.assertEqual(1, len(overlay))
        self.assertAlmostEqual(20.0, overlay[0][0][0])
        self.assertAlmostEqual(40.0, overlay[0][0][2])

    def test_preview_uses_detection_frame_for_actuation_box(self) -> None:
        module = load_module("cap_line_runtime_v3")
        packet = module.DetectionPacket(
            frame_pair=module.FramePair(
                frames=(module.CapturedFrame(0, FakeFrame("processed"), 1.00, 5),),
                pair_timestamp=1.00,
                skew_ms=0.0,
            ),
            boxes_by_camera=(((20.0, 10.0, 40.0, 20.0, 0.9, 1),),),
            inference_ms_by_camera=(1.0,),
        )
        live_frames = (module.CapturedFrame(0, FakeFrame("live"), 1.01, 5),)

        views = module.resolve_preview_views(None, packet, live_frames, target_fps=60)

        self.assertEqual("processed", views[0].captured.frame.label)
        self.assertEqual(packet.boxes_by_camera[0], views[0].boxes)

    def test_preview_hides_boxes_before_actuation_line(self) -> None:
        module = load_module("cap_line_runtime_v3")
        packet = module.DetectionPacket(
            frame_pair=module.FramePair(
                frames=(module.CapturedFrame(0, FakeFrame("processed"), 1.00, 1),),
                pair_timestamp=1.00,
                skew_ms=0.0,
            ),
            boxes_by_camera=(((5.0, 10.0, 15.0, 20.0, 0.9, 1),),),
            inference_ms_by_camera=(1.0,),
        )
        live_frames = (module.CapturedFrame(0, FakeFrame("live"), 1.01, 2),)

        overlay = module.predict_preview_overlay(None, packet, live_frames, target_fps=60)

        self.assertEqual(((),), overlay)

    def test_preview_hides_stale_actuation_overlay(self) -> None:
        module = load_module("cap_line_runtime_v3")
        packet = module.DetectionPacket(
            frame_pair=module.FramePair(
                frames=(module.CapturedFrame(0, FakeFrame("processed"), 1.00, 1),),
                pair_timestamp=1.00,
                skew_ms=0.0,
            ),
            boxes_by_camera=(((20.0, 10.0, 40.0, 20.0, 0.9, 1),),),
            inference_ms_by_camera=(1.0,),
        )
        live_frames = (module.CapturedFrame(0, FakeFrame("live"), 1.50, 2),)

        overlay = module.predict_preview_overlay(None, packet, live_frames, target_fps=60)

        self.assertEqual(((),), overlay)

    def test_preview_keeps_high_latency_actuation_detection_visible(self) -> None:
        module = load_module("cap_line_runtime_v3")
        packet = module.DetectionPacket(
            frame_pair=module.FramePair(
                frames=(module.CapturedFrame(0, FakeFrame("processed"), 1.00, 1),),
                pair_timestamp=1.00,
                skew_ms=0.0,
            ),
            boxes_by_camera=(((20.0, 10.0, 40.0, 20.0, 0.9, 1),),),
            inference_ms_by_camera=(450.0,),
        )
        live_frames = (module.CapturedFrame(0, FakeFrame("live"), 1.42, 2),)

        overlay = module.predict_preview_overlay(None, packet, live_frames, target_fps=60)
        views = module.resolve_preview_views(None, packet, live_frames, target_fps=60)

        self.assertEqual(packet.boxes_by_camera, overlay)
        self.assertEqual("processed", views[0].captured.frame.label)
        self.assertEqual(packet.boxes_by_camera[0], views[0].boxes)

    def test_preview_extends_timeout_for_slow_detection_cadence_without_shifting_box(self) -> None:
        module = load_module("cap_line_runtime_v3")
        previous = module.DetectionPacket(
            frame_pair=module.FramePair(
                frames=(module.CapturedFrame(0, FakeFrame("previous"), 1.00, 1),),
                pair_timestamp=1.00,
                skew_ms=0.0,
            ),
            boxes_by_camera=(((10.0, 10.0, 20.0, 20.0, 0.9, 1),),),
            inference_ms_by_camera=(1.0,),
        )
        current = module.DetectionPacket(
            frame_pair=module.FramePair(
                frames=(module.CapturedFrame(0, FakeFrame("processed"), 1.40, 2),),
                pair_timestamp=1.40,
                skew_ms=0.0,
            ),
            boxes_by_camera=(((30.0, 10.0, 40.0, 20.0, 0.9, 1),),),
            inference_ms_by_camera=(1.0,),
        )
        live_frames = (module.CapturedFrame(0, FakeFrame("live"), 1.70, 3),)

        overlay = module.predict_preview_overlay(previous, current, live_frames, target_fps=60)

        self.assertEqual(1, len(overlay[0]))
        self.assertAlmostEqual(30.0, overlay[0][0][0])
        self.assertAlmostEqual(40.0, overlay[0][0][2])

    def test_preview_does_not_project_large_jump_between_slow_detections(self) -> None:
        module = load_module("cap_line_runtime_v3")
        wide_frame = (64, 360, 3)
        previous = module.DetectionPacket(
            frame_pair=module.FramePair(
                frames=(module.CapturedFrame(0, FakeFrame("previous", wide_frame), 1.00, 1),),
                pair_timestamp=1.00,
                skew_ms=0.0,
            ),
            boxes_by_camera=(((10.0, 10.0, 30.0, 30.0, 0.9, 1),),),
            inference_ms_by_camera=(1.0,),
        )
        current = module.DetectionPacket(
            frame_pair=module.FramePair(
                frames=(module.CapturedFrame(0, FakeFrame("processed", wide_frame), 1.40, 2),),
                pair_timestamp=1.40,
                skew_ms=0.0,
            ),
            boxes_by_camera=(((170.0, 10.0, 190.0, 30.0, 0.9, 1),),),
            inference_ms_by_camera=(1.0,),
        )
        live_frames = (module.CapturedFrame(0, FakeFrame("live", wide_frame), 1.55, 3),)

        overlay = module.predict_preview_overlay(previous, current, live_frames, target_fps=60)

        self.assertEqual(1, len(overlay[0]))
        self.assertAlmostEqual(170.0, overlay[0][0][0])
        self.assertAlmostEqual(190.0, overlay[0][0][2])

    def test_preview_latency_compensation_does_not_shift_actuation_box(self) -> None:
        module = load_module("cap_line_runtime_v3")
        previous = module.DetectionPacket(
            frame_pair=module.FramePair(
                frames=(module.CapturedFrame(0, FakeFrame("previous"), 1.00, 1),),
                pair_timestamp=1.00,
                skew_ms=0.0,
            ),
            boxes_by_camera=(((10.0, 10.0, 20.0, 20.0, 0.9, 1),),),
            inference_ms_by_camera=(1.0,),
        )
        current = module.DetectionPacket(
            frame_pair=module.FramePair(
                frames=(module.CapturedFrame(0, FakeFrame("processed"), 1.10, 2),),
                pair_timestamp=1.10,
                skew_ms=0.0,
            ),
            boxes_by_camera=(((20.0, 10.0, 40.0, 20.0, 0.9, 1),),),
            inference_ms_by_camera=(1.0,),
        )
        live_frames = (module.CapturedFrame(0, FakeFrame("live"), 1.20, 3),)

        overlay = module.predict_preview_overlay(
            previous,
            current,
            live_frames,
            target_fps=60,
            preview_latency_compensation_ms=50.0,
        )

        self.assertAlmostEqual(20.0, overlay[0][0][0])
        self.assertAlmostEqual(40.0, overlay[0][0][2])

    def test_live_preview_publisher_draws_actuation_frame_without_prediction(self) -> None:
        module = load_module("cap_line_runtime_v3")

        class FakeReader:
            def latest(self):
                return module.CapturedFrame(
                    camera_index=0,
                    frame=FakeFrame("live"),
                    timestamp=1.20,
                    sequence=3,
                )

        drawn_boxes = []
        previews = []
        stop_event = module.threading.Event()

        def fake_draw_boxes(frame, boxes):
            drawn_boxes.append((frame.label, [[float(value) for value in box] for box in boxes]))
            return frame

        def capture_preview(preview):
            previews.append(preview)
            stop_event.set()

        publisher = module.LivePreviewPublisher(
            [FakeReader()],
            capture_preview,
            anchor_axis="x",
            anchor_line_ratio=0.5,
            preview_fps=60.0,
            overlay_target_fps=30.0,
            stop_event=stop_event,
            compose_preview_fn=lambda frames: frames,
            draw_boxes_fn=fake_draw_boxes,
            draw_anchor_line_fn=lambda frame, _axis, _ratio: frame,
            sleep_fn=lambda _seconds: None,
        )

        publisher.update_packet(
            module.DetectionPacket(
                frame_pair=module.FramePair(
                    frames=(module.CapturedFrame(0, FakeFrame("previous"), 1.00, 1),),
                    pair_timestamp=1.00,
                    skew_ms=0.0,
                ),
                boxes_by_camera=(((10.0, 10.0, 20.0, 20.0, 0.90, 1),),),
                inference_ms_by_camera=(1.0,),
            )
        )
        publisher.update_packet(
            module.DetectionPacket(
                frame_pair=module.FramePair(
                    frames=(module.CapturedFrame(0, FakeFrame("processed"), 1.10, 2),),
                    pair_timestamp=1.10,
                    skew_ms=0.0,
                ),
                boxes_by_camera=(((28.0, 10.0, 36.0, 20.0, 0.90, 1),),),
                inference_ms_by_camera=(1.0,),
            )
        )
        publisher._run()

        self.assertEqual(1, len(previews))
        self.assertEqual(1, len(drawn_boxes))
        self.assertEqual("processed", drawn_boxes[0][0])
        self.assertAlmostEqual(28.0, drawn_boxes[0][1][0][0])
        self.assertAlmostEqual(36.0, drawn_boxes[0][1][0][2])

    def test_latest_frame_reader_continues_capture_while_caller_waits(self) -> None:
        module = load_module("cap_line_runtime_v3")

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

    def test_synchronized_pair_requires_fresh_frames_and_respects_skew(self) -> None:
        module = load_module("cap_line_runtime_v3")

        accepted = module.select_synchronized_frame_pair(
            (
                module.CapturedFrame(0, "cam0", 1.000, 4),
                module.CapturedFrame(1, "cam1", 1.030, 8),
            ),
            last_sequences=(3, 7),
            max_skew_ms=40.0,
        )
        rejected_stale = module.select_synchronized_frame_pair(
            (
                module.CapturedFrame(0, "cam0", 1.000, 4),
                module.CapturedFrame(1, "cam1", 1.030, 8),
            ),
            last_sequences=(4, 8),
            max_skew_ms=40.0,
        )
        rejected_skew = module.select_synchronized_frame_pair(
            (
                module.CapturedFrame(0, "cam0", 1.000, 5),
                module.CapturedFrame(1, "cam1", 1.060, 9),
            ),
            last_sequences=(4, 8),
            max_skew_ms=40.0,
        )

        self.assertIsNotNone(accepted)
        self.assertEqual((4, 8), accepted.sequences)
        self.assertIsNone(rejected_stale)
        self.assertIsNone(rejected_skew)

    def test_postprocess_maps_model_boxes_back_to_original_frame(self) -> None:
        module = load_module("cap_line_runtime_v3")
        output = [[[320.0, 240.0, 420.0, 340.0, 0.91, 1.0]]]
        meta = {
            "scale": 0.5,
            "pad_left": 10,
            "pad_top": 20,
            "frame_shape": (600, 960, 3),
            "img_size": 640,
        }

        boxes = module.postprocess(output, meta, conf_threshold=0.45)

        self.assertEqual(1, len(boxes))
        self.assertAlmostEqual(620.0, boxes[0][0])
        self.assertAlmostEqual(440.0, boxes[0][1])
        self.assertAlmostEqual(820.0, boxes[0][2])
        self.assertAlmostEqual(599.0, boxes[0][3])
        self.assertAlmostEqual(0.91, boxes[0][4], places=5)
        self.assertEqual(1, boxes[0][5])

    def test_defect_at_reject_threshold_triggers_after_actuation_crossing(self) -> None:
        module = load_module("cap_line_runtime_v3")
        tracked_cap = module.TrackedCap(event_id=1, created_at=1.0, last_seen_at=1.0)
        tracked_cap.add_observation(
            module.TrackObservation(
                camera_index=0,
                box=(45.0, 10.0, 55.0, 20.0, 0.45, 1),
                timestamp=1.0,
                frame_size=(100, 40),
                at_actuation_line=True,
            )
        )

        decision = module.decide_decision_ready(
            tracked_cap,
            config=module.RuntimeConfig.defaults(),
            decision_ready_time=1.2,
        )

        self.assertIsNotNone(decision)
        self.assertEqual("trigger", decision.result)
        self.assertEqual("dirt_defect", decision.final_class_name)
        self.assertAlmostEqual(0.45, decision.final_score)

    def test_non_actuation_defect_does_not_classify_or_trigger(self) -> None:
        module = load_module("cap_line_runtime_v3")
        tracked_cap = module.TrackedCap(event_id=1, created_at=1.0, last_seen_at=1.0)
        tracked_cap.add_observation(
            module.TrackObservation(
                camera_index=0,
                box=(5.0, 10.0, 15.0, 20.0, 0.99, 1),
                timestamp=1.0,
                frame_size=(100, 40),
                at_actuation_line=False,
            )
        )

        ready = module.decide_decision_ready(
            tracked_cap,
            config=module.RuntimeConfig.defaults(),
            decision_ready_time=1.2,
        )
        decision = module.decide_tracked_cap(
            tracked_cap,
            config=module.RuntimeConfig.defaults(),
            decision_time=1.3,
        )

        self.assertIsNone(ready)
        self.assertEqual("skip", decision.result)
        self.assertIsNone(decision.final_class_name)
        self.assertIsNone(decision.final_score)
        self.assertEqual("no_actuation_crossing", decision.decision_source)
        self.assertIsNone(decision.review_reason)

    def test_dirty_before_line_but_clean_at_actuation_is_clean_skip(self) -> None:
        module = load_module("cap_line_runtime_v3")
        tracked_cap = module.TrackedCap(event_id=1, created_at=1.0, last_seen_at=1.0)
        tracked_cap.add_observation(
            module.TrackObservation(
                camera_index=0,
                box=(5.0, 10.0, 15.0, 20.0, 0.99, 1),
                timestamp=1.0,
                frame_size=(100, 40),
                at_actuation_line=False,
            )
        )
        tracked_cap.add_observation(
            module.TrackObservation(
                camera_index=0,
                box=(45.0, 10.0, 55.0, 20.0, 0.96, 0),
                timestamp=1.1,
                frame_size=(100, 40),
                at_actuation_line=True,
            )
        )

        ready = module.decide_decision_ready(
            tracked_cap,
            config=module.RuntimeConfig.defaults(),
            decision_ready_time=1.2,
        )
        decision = module.decide_tracked_cap(
            tracked_cap,
            config=module.RuntimeConfig.defaults(),
            decision_time=1.3,
        )

        self.assertIsNone(ready)
        self.assertEqual("skip", decision.result)
        self.assertEqual("undefected", decision.final_class_name)
        self.assertAlmostEqual(0.96, decision.final_score)
        self.assertEqual("below_reject_threshold", decision.decision_source)

    def test_fake_runtime_loop_emits_preview_and_performance_for_target_fps(self) -> None:
        module = load_module("cap_line_runtime_v3")

        class FakeFrame:
            shape = (32, 32, 3)

            def __init__(self, label):
                self.label = label

            def copy(self):
                return FakeFrame(self.label)

        class FakeCamera:
            def __init__(self, camera_index):
                self.camera_index = camera_index
                self.read_count = 0

            def read(self):
                self.read_count += 1
                return True, FakeFrame(f"cam{self.camera_index}-{self.read_count}")

            def get(self, _property_id):
                return 0

            def release(self):
                return None

        class FakeSession:
            def get_inputs(self):
                return [types.SimpleNamespace(name="images", shape=[1, 3, 640, 640])]

            def run(self, *_args, **_kwargs):
                time.sleep(0.05)
                return [None]

        previews = []
        performances = []
        stop_event = module.threading.Event()

        config = module.RuntimeConfig.defaults()
        config = module.replace(
            config,
            target_fps=120,
            live_preview_fps=120.0,
            serial_inference=True,
            simulate_gpio=True,
            no_display=True,
        )
        callbacks = module.RuntimeCallbacks(
            preview_callback=previews.append,
            performance_callback=performances.append,
            log_fn=lambda *_args, **_kwargs: None,
        )

        module.run_detection(
            config,
            callbacks,
            stop_event=stop_event,
            camera_factory=lambda camera_index, _source, _config: FakeCamera(camera_index),
            session_factory=lambda _model_path, _threads: FakeSession(),
            preprocess_fn=lambda frame, _imgsz: (frame, {"frame_shape": frame.shape}),
            postprocess_fn=lambda *_args, **_kwargs: [],
            compose_preview_fn=lambda _frames: "preview",
            draw_boxes_fn=lambda frame, _boxes: frame,
            draw_anchor_line_fn=lambda frame, _axis, _ratio: frame,
            time_fn=time.monotonic,
            sleep_fn=time.sleep,
        )

        self.assertGreaterEqual(len(previews), 1)
        self.assertTrue(any(snapshot.target_fps == 120 for snapshot in performances))
        self.assertTrue(any(snapshot.preview_fps > snapshot.processed_fps for snapshot in performances))

    def test_runtime_fails_fast_when_camera_does_not_open(self) -> None:
        module = load_module("cap_line_runtime_v3")

        class FakeClosedCamera:
            def __init__(self):
                self.released = False

            def isOpened(self):
                return False

            def release(self):
                self.released = True

        class FakeOpenCamera:
            def __init__(self):
                self.released = False

            def isOpened(self):
                return True

            def release(self):
                self.released = True

        cameras = [FakeClosedCamera(), FakeOpenCamera()]
        config = module.replace(
            module.RuntimeConfig.defaults(),
            cameras=("bad0", "bad1"),
            simulate_gpio=True,
            no_display=True,
        )

        with self.assertRaisesRegex(RuntimeError, "camera 0.*bad0"):
            module.run_detection(
                config,
                module.RuntimeCallbacks(log_fn=lambda *_args, **_kwargs: None),
                camera_factory=lambda camera_index, _source, _config: cameras[camera_index],
                session_factory=lambda *_args, **_kwargs: None,
            )

        self.assertTrue(cameras[0].released)
        self.assertTrue(cameras[1].released)

    def test_trigger_runtime_writes_v3_debug_artifact(self) -> None:
        module = load_module("cap_line_runtime_v3")

        class FakeFrame:
            shape = (32, 32, 3)

            def copy(self):
                return self

        class FakeCamera:
            def read(self):
                return True, FakeFrame()

            def release(self):
                return None

        class FakeSession:
            def get_inputs(self):
                return [types.SimpleNamespace(name="images", shape=[1, 3, 640, 640])]

            def run(self, *_args, **_kwargs):
                return [None]

        with tempfile.TemporaryDirectory() as tmpdir:
            config = module.replace(
                module.RuntimeConfig.defaults(),
                debug_dir=str(Path(tmpdir) / "debugging_v3"),
                pictures_dir=str(Path(tmpdir) / "pictures_v3"),
                timing_log_dir=str(Path(tmpdir) / "timing_logs_v3"),
                simulate_gpio=True,
                no_display=True,
                merge_window_ms=0.0,
            )
            callbacks = module.RuntimeCallbacks(log_fn=lambda *_args, **_kwargs: None)
            stop_event = module.threading.Event()

            module.run_detection(
                config,
                callbacks,
                stop_event=stop_event,
                camera_factory=lambda camera_index, _source, _config: FakeCamera(),
                session_factory=lambda _model_path, _threads: FakeSession(),
                preprocess_fn=lambda frame, _imgsz: (frame, {"frame_shape": frame.shape}),
                postprocess_fn=lambda *_args, **_kwargs: [[12.0, 0.0, 20.0, 31.0, 0.9, 1.0]],
                compose_preview_fn=lambda _frames: "preview",
                time_fn=time.monotonic,
                sleep_fn=time.sleep,
            )

            debug_json = list((Path(tmpdir) / "debugging_v3").glob("event_*.json"))
            timing_csv = list((Path(tmpdir) / "timing_logs_v3").glob("timing_*.csv"))

        self.assertEqual(1, len(debug_json))
        self.assertEqual(1, len(timing_csv))


if __name__ == "__main__":
    unittest.main()
