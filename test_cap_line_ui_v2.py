from __future__ import annotations

import importlib.util
import sys
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


class CapLineUiV2Tests(unittest.TestCase):
    def test_gui_args_come_from_v2_runtime_defaults(self) -> None:
        runtime_module = load_module("cap_line_runtime_v2")
        ui_module = load_module("cap_line_ui_v2")

        args = ui_module.create_gui_args()

        self.assertEqual(runtime_module.DEFAULT_MODEL, args.model)
        self.assertEqual(runtime_module.DEFAULT_TIMING_LOG_DIR, args.timing_log_dir)
        self.assertEqual(runtime_module.DEFAULT_REVIEW_DIR, args.review_dir)
        self.assertEqual(runtime_module.DEFAULT_DEBUG_DIR, args.debug_dir)
        self.assertEqual(runtime_module.DEFAULT_PICTURES_DIR, args.pictures_dir)
        self.assertEqual(runtime_module.DEFAULT_DEBUG_DIR, runtime_module.DEFAULT_REVIEW_DIR)
        self.assertEqual(runtime_module.TRACKING_DETECTION_THRESHOLD, args.global_threshold)
        self.assertEqual(runtime_module.TRACKING_DETECTION_THRESHOLD, args.tracking_threshold)
        self.assertEqual(runtime_module.DEFECT_REJECT_THRESHOLD, args.reject_threshold)
        self.assertEqual(runtime_module.DEFAULT_PAIR_MAX_SKEW_MS, args.pair_max_skew_ms)
        self.assertEqual(
            runtime_module.DEFAULT_SAVE_QUEUE_WARNING_THRESHOLD,
            args.save_queue_warning_threshold,
        )
        self.assertEqual(runtime_module.DEFAULT_TRIGGER_PIN, args.trigger_pin)

    def test_resources_dirs_live_under_resources_tree(self) -> None:
        runtime_module = load_module("cap_line_runtime_v2")
        self.assertTrue(
            runtime_module.DEFAULT_DEBUG_DIR.replace("\\", "/").endswith(
                "resources/debugging"
            )
        )
        self.assertTrue(
            runtime_module.DEFAULT_PICTURES_DIR.replace("\\", "/").endswith(
                "resources/pictures"
            )
        )

    def test_pictures_dir_is_exposed_in_calibration_labels(self) -> None:
        ui_module = load_module("cap_line_ui_v2")
        self.assertIn("Debug Dir", ui_module.CALIBRATION_FIELD_LABELS)
        self.assertIn("Pictures Dir", ui_module.CALIBRATION_FIELD_LABELS)
        self.assertNotIn("Review Capture Dir", ui_module.CALIBRATION_FIELD_LABELS)

    def test_fps_label_clarifies_camera_target(self) -> None:
        ui_module = load_module("cap_line_ui_v2")

        self.assertIn("Camera Target FPS", ui_module.CALIBRATION_FIELD_LABELS)
        self.assertNotIn("FPS", ui_module.CALIBRATION_FIELD_LABELS)

    def test_threshold_controls_are_exposed(self) -> None:
        ui_module = load_module("cap_line_ui_v2")

        self.assertIn("Tracking Threshold", ui_module.CALIBRATION_FIELD_LABELS)
        self.assertIn("Reject Threshold", ui_module.CALIBRATION_FIELD_LABELS)
        self.assertIn("Pair Max Skew ms", ui_module.CALIBRATION_FIELD_LABELS)
        self.assertNotIn("Detection Threshold", ui_module.CALIBRATION_FIELD_LABELS)

    def test_trigger_pin_label_clarifies_jetson_gpio_board_numbering(self) -> None:
        ui_module = load_module("cap_line_ui_v2")

        self.assertIn(
            "Trigger GPIO09 (Jetson BOARD pin 7)",
            ui_module.CALIBRATION_FIELD_LABELS,
        )
        self.assertNotIn("Trigger Pin", ui_module.CALIBRATION_FIELD_LABELS)

    def test_removed_policy_controls_are_absent_from_v2_ui(self) -> None:
        ui_module = load_module("cap_line_ui_v2")

        self.assertNotIn("Confidence", ui_module.CALIBRATION_FIELD_LABELS)
        self.assertNotIn("Defect Min Score", ui_module.CALIBRATION_FIELD_LABELS)
        self.assertNotIn("Defect Margin", ui_module.CALIBRATION_FIELD_LABELS)
        self.assertNotIn("Single-Camera Defect Score", ui_module.CALIBRATION_FIELD_LABELS)
        self.assertNotIn("Same-Camera Hold ms", ui_module.CALIBRATION_FIELD_LABELS)
        self.assertNotIn("Anchor Axis", ui_module.CALIBRATION_FIELD_LABELS)
        self.assertNotIn("Anchor Line Ratio", ui_module.CALIBRATION_FIELD_LABELS)
        self.assertIn("Actuation Axis", ui_module.CALIBRATION_FIELD_LABELS)
        self.assertIn("Actuation Line Ratio", ui_module.CALIBRATION_FIELD_LABELS)

    def test_threshold_label_notes_configurable_thresholds(self) -> None:
        ui_module = load_module("cap_line_ui_v2")

        self.assertEqual(
            "Tracking and reject thresholds are configurable",
            ui_module.DETECTION_RULE_LABEL,
        )


if __name__ == "__main__":
    unittest.main()
