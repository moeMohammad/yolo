from __future__ import annotations

import ast
import importlib.util
import sys
import tempfile
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


def iter_ui_source_files() -> list[Path]:
    return [
        REPO_ROOT / "cap_line_ui_v3.py",
        *sorted((REPO_ROOT / "cap_line_v3").glob("*.py")),
    ]


class CapLineUiV3Tests(unittest.TestCase):
    def test_v3_ui_does_not_import_v1_or_v2_ui_or_runtime_modules(self) -> None:
        forbidden = {
            "cap_line_runtime",
            "cap_line_runtime_v2",
            "cap_line_runtime_v2_grey",
            "cap_line_ui",
            "cap_line_ui_v2",
            "cap_line_ui_v2_grey",
        }

        for path in iter_ui_source_files():
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

    def test_gui_default_config_matches_runtime_defaults(self) -> None:
        runtime_module = load_module("cap_line_runtime_v3")
        ui_module = load_module("cap_line_ui_v3")

        config = ui_module.create_gui_config()
        defaults = runtime_module.RuntimeConfig.defaults()

        self.assertEqual(defaults, config)
        self.assertEqual(("0", "3"), config.cameras)
        self.assertEqual(60, config.target_fps)
        self.assertTrue(config.simulate_gpio if sys.platform.startswith("win") else True)

    def test_config_tab_labels_expose_operator_runtime_parameters(self) -> None:
        ui_module = load_module("cap_line_ui_v3")

        labels = set(ui_module.CONFIG_FIELD_LABELS)

        for expected in {
            "Model",
            "Camera 0",
            "Camera 1",
            "Width",
            "Height",
            "Camera Target FPS",
            "Exposure",
            "Tracking Threshold",
            "Reject Threshold",
            "Pair Max Skew ms",
            "Trigger GPIO09 (BOARD pin 7)",
            "Nozzle Distance mm",
            "Belt Speed mm/s",
            "Trigger Offset s",
            "Latency Compensation ms",
            "Preview Lead ms",
            "Timing Log Dir",
            "Debug Dir",
            "Pictures Dir",
        }:
            self.assertIn(expected, labels)
        self.assertNotIn("Calibration", labels)
        self.assertNotIn("Confidence", labels)

    def test_settings_store_persists_last_used_config(self) -> None:
        runtime_module = load_module("cap_line_runtime_v3")
        ui_module = load_module("cap_line_ui_v3")

        with tempfile.TemporaryDirectory() as tmpdir:
            store = ui_module.ConfigSettingsStore(Path(tmpdir) / "settings.json")
            config = runtime_module.replace(
                runtime_module.RuntimeConfig.defaults(),
                cameras=("2", "5"),
                target_fps=120,
                exposure=12,
                tracking_threshold=0.51,
                preview_latency_compensation_ms=85.0,
            )

            store.save(config)
            loaded = store.load()

        self.assertEqual(("2", "5"), loaded.cameras)
        self.assertEqual(120, loaded.target_fps)
        self.assertEqual(12, loaded.exposure)
        self.assertAlmostEqual(0.51, loaded.tracking_threshold)
        self.assertAlmostEqual(85.0, loaded.preview_latency_compensation_ms)

    def test_controller_passes_latest_config_to_runner(self) -> None:
        runtime_module = load_module("cap_line_runtime_v3")
        ui_module = load_module("cap_line_ui_v3")

        captured = []
        config = runtime_module.replace(
            runtime_module.RuntimeConfig.defaults(),
            cameras=("7", "8"),
            target_fps=30,
            no_display=True,
            simulate_gpio=True,
        )

        def fake_runner(run_config, callbacks, *, stop_event=None, pin_factory=None):
            captured.append((run_config, callbacks, stop_event, pin_factory))

        controller = ui_module.DetectionAppController(
            repository=ui_module.HistoryRepository(":memory:"),
            detector_runner=fake_runner,
            config_factory=lambda: config,
        )

        self.assertTrue(controller.start())
        controller.worker_thread.join(timeout=1.0)
        changes = controller.drain_messages()

        self.assertEqual(config, captured[0][0])
        self.assertEqual("Stopped", controller.status_text)
        self.assertTrue(changes["stopped"])


if __name__ == "__main__":
    unittest.main()
