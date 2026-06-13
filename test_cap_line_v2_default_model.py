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


class CapLineV2DefaultModelTests(unittest.TestCase):
    def test_runtime_v2_defaults_to_dirtv3(self) -> None:
        module = load_module("cap_line_runtime_v2")

        self.assertEqual("dirtv3.onnx", module.DEFAULT_MODEL)
        self.assertEqual("dirtv3.onnx", module.parse_args([]).model)

    def test_runtime_v2_resolves_default_to_model_directory(self) -> None:
        module = load_module("cap_line_runtime_v2")

        model_path, preset_imgsz = module.resolve_model_path(None)

        self.assertEqual(str(REPO_ROOT / "model" / "dirtv3.onnx"), model_path)
        self.assertIsNone(preset_imgsz)

    def test_ui_v2_gui_args_inherit_runtime_default_model(self) -> None:
        runtime_module = load_module("cap_line_runtime_v2")
        ui_module = load_module("cap_line_ui_v2")

        args = ui_module.create_gui_args()

        self.assertEqual("dirtv3.onnx", runtime_module.DEFAULT_MODEL)
        self.assertEqual(runtime_module.DEFAULT_MODEL, args.model)


if __name__ == "__main__":
    unittest.main()
