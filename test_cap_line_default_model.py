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


class CapLineDefaultModelTests(unittest.TestCase):
    def test_runtime_defaults_to_dirtv2(self) -> None:
        module = load_module("cap_line_runtime")

        self.assertEqual("dirtv2.onnx", module.DEFAULT_MODEL)
        self.assertEqual("dirtv2.onnx", module.parse_args([]).model)

    def test_runtime_resolves_default_to_model_directory(self) -> None:
        module = load_module("cap_line_runtime")

        model_path, preset_imgsz = module.resolve_model_path(None)

        self.assertEqual(str(REPO_ROOT / "model" / "dirtv2.onnx"), model_path)
        self.assertIsNone(preset_imgsz)

    def test_ui_gui_args_inherit_runtime_default_model(self) -> None:
        runtime_module = load_module("cap_line_runtime")
        ui_module = load_module("cap_line_ui")

        args = ui_module.create_gui_args()

        self.assertEqual("dirtv2.onnx", runtime_module.DEFAULT_MODEL)
        self.assertEqual(runtime_module.DEFAULT_MODEL, args.model)

    def test_runtime_trigger_pin_defaults_to_gpio09(self) -> None:
        module = load_module("cap_line_runtime")
        parser = module.build_arg_parser()

        action = parser._option_string_actions["--trigger-pin"]

        self.assertEqual(7, module.DEFAULT_TRIGGER_PIN)
        self.assertEqual(7, parser.parse_args([]).trigger_pin)
        self.assertIn("GPIO09", action.help)

    def test_ui_trigger_pin_label_mentions_board_pin(self) -> None:
        ui_module = load_module("cap_line_ui")

        self.assertEqual("Trigger GPIO09 (BOARD pin 7)", ui_module.TRIGGER_PIN_LABEL)


if __name__ == "__main__":
    unittest.main()
