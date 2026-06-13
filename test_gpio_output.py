from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


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


class FakeJetsonGPIO(types.ModuleType):
    BOARD = "BOARD"
    OUT = "OUT"
    LOW = 0
    HIGH = 1

    def __init__(self):
        super().__init__("Jetson.GPIO")
        self.calls = []

    def setwarnings(self, enabled):
        self.calls.append(("setwarnings", enabled))

    def setmode(self, mode):
        self.calls.append(("setmode", mode))

    def setup(self, pin, direction, initial=None):
        self.calls.append(("setup", pin, direction, initial))

    def output(self, pin, state):
        self.calls.append(("output", pin, state))

    def cleanup(self, pin):
        self.calls.append(("cleanup", pin))


def build_fake_jetson_modules(fake_gpio):
    jetson = types.ModuleType("Jetson")
    jetson.__path__ = []
    jetson.GPIO = fake_gpio
    return {
        "Jetson": jetson,
        "Jetson.GPIO": fake_gpio,
    }


class GPIOOutputPinTests(unittest.TestCase):
    def test_output_pin_uses_jetson_gpio_board_backend_for_gpio09(self) -> None:
        module = load_module("gpio_output")
        fake_gpio = FakeJetsonGPIO()

        with patch.dict(sys.modules, build_fake_jetson_modules(fake_gpio)):
            pin = module.GPIOOutputPin("GPIO09")
            pin.on()
            pin.off()
            pin.close()

        self.assertEqual(7, pin.pin)
        self.assertEqual("Jetson.GPIO BOARD", pin.backend_name)
        self.assertEqual(
            [
                ("setwarnings", False),
                ("setmode", "BOARD"),
                ("setup", 7, "OUT", 0),
                ("output", 7, 0),
                ("output", 7, 1),
                ("output", 7, 0),
                ("output", 7, 0),
                ("cleanup", 7),
            ],
            fake_gpio.calls,
        )

    def test_missing_jetson_gpio_reports_install_hint(self) -> None:
        module = load_module("gpio_output")

        with patch.dict(sys.modules, {"Jetson": None, "Jetson.GPIO": None}):
            with self.assertRaisesRegex(RuntimeError, "Jetson.GPIO"):
                module.GPIOOutputPin("GPIO09")

    def test_manual_gpio_defaults_to_gpio09_board_pin(self) -> None:
        module = load_module("manual_gpio")
        parser = module.build_parser()

        action = parser._option_string_actions["--pin"]

        self.assertEqual(7, module.DEFAULT_TRIGGER_PIN)
        self.assertEqual(7, parser.parse_args([]).pin)
        self.assertIn("GPIO09", action.help)

    def test_reject_scheduler_exposes_pin_backend_name(self) -> None:
        module = load_module("cap_line_runtime")

        class FakePin:
            backend_name = "Jetson.GPIO BOARD"

            def __init__(self, pin):
                self.pin = pin

            def on(self):
                return None

            def off(self):
                return None

            def close(self):
                return None

        scheduler = module.RejectScheduler(
            trigger_pin=7,
            trigger_duration=0.01,
            pin_factory=FakePin,
            log_fn=lambda *_args, **_kwargs: None,
        )
        try:
            self.assertEqual(7, scheduler.trigger_pin)
            self.assertEqual("Jetson.GPIO BOARD", scheduler.backend_name)
        finally:
            scheduler.close()


if __name__ == "__main__":
    unittest.main()
