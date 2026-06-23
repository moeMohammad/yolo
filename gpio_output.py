from __future__ import annotations

import os


GPIO09 = 7  # Jetson Nano J41 physical BOARD pin 7.
DEFAULT_TRIGGER_PIN = GPIO09

_GPIO_PERMISSIONS_HINT = (
    "Configure Jetson GPIO access once, then log out and back in:\n"
    "  sudo groupadd -f -r gpio\n"
    "  sudo usermod -aG gpio $USER\n"
    "  sudo cp $(python3 -c \"import Jetson.GPIO as m, os; "
    "print(os.path.join(os.path.dirname(m.__file__), '99-gpio.rules'))\") "
    "/etc/udev/rules.d/\n"
    "  sudo udevadm control --reload-rules && sudo udevadm trigger\n"
    "If it still fails immediately, also run:\n"
    "  for c in /dev/gpiochip*; do sudo chown root:gpio $c; sudo chmod 660 $c; done"
)


def _env_flag(name: str) -> bool:
    value = os.environ.get(name, "").strip().lower()
    return value in {"1", "true", "yes", "y", "on"}


def _compact_pin_name(pin_text: str) -> str:
    return pin_text.strip().upper().replace("_", "").replace("-", "").replace(" ", "")


def _number_after_prefix(compact_pin: str, prefixes: tuple[str, ...]) -> int | None:
    for prefix in prefixes:
        if compact_pin.startswith(prefix):
            suffix = compact_pin[len(prefix) :]
            if suffix.isdigit():
                return int(suffix)
    return None


def _is_gpio09_alias(pin_text: str) -> bool:
    compact_name = _compact_pin_name(pin_text)
    if compact_name in {"GPIO09", "GPIO9"}:
        return True
    suffix = _number_after_prefix(compact_name, ("GPIO",))
    return suffix == 9


def _positive_int(value: int, label: str) -> int:
    if value < 0:
        raise ValueError(f"{label} cannot be negative")
    return value


class GPIOOutputPin:
    def __init__(self, pin=DEFAULT_TRIGGER_PIN, *, active_low: bool | None = None):
        self.requested_pin = pin
        self.pin = pin
        self._gpio = None
        self.active_low = _env_flag("GPIO_OUTPUT_ACTIVE_LOW") if active_low is None else bool(active_low)

        try:
            import Jetson.GPIO as GPIO
        except Exception as exc:
            raise RuntimeError(
                f"Could not import Jetson.GPIO for GPIO pin {pin}. "
                "Install Jetson.GPIO on the Jetson Nano and make sure the "
                "runtime user has GPIO permissions."
            ) from exc

        mode, channel, mode_name = self._resolve_channel(GPIO, pin)
        self._active_state = GPIO.LOW if self.active_low else GPIO.HIGH
        self._inactive_state = GPIO.HIGH if self.active_low else GPIO.LOW

        try:
            GPIO.setwarnings(True)
            GPIO.setmode(mode)
            GPIO.setup(channel, GPIO.OUT, initial=self._inactive_state)
            GPIO.output(channel, self._inactive_state)
        except Exception as exc:
            raise RuntimeError(
                f"Could not initialize Jetson.GPIO {mode_name} pin {channel}. "
                "Check that the selected Jetson Nano header pin is configured for "
                "GPIO output and that pinmux/permissions are set correctly.\n\n"
                f"{_GPIO_PERMISSIONS_HINT}"
            ) from exc

        self.pin = channel
        self.mode_name = mode_name
        self._gpio = GPIO
        polarity = "active-low" if self.active_low else "active-high"
        self.backend_name = f"Jetson.GPIO {mode_name} {polarity}"

    @staticmethod
    def _resolve_channel(GPIO, pin):
        if isinstance(pin, int):
            return GPIO.BOARD, _positive_int(pin, "Jetson BOARD pin"), "BOARD"

        pin_text = str(pin).strip()
        if not pin_text:
            raise ValueError("GPIO pin cannot be empty")

        compact_pin = _compact_pin_name(pin_text)
        if compact_pin.isdigit():
            return GPIO.BOARD, _positive_int(int(compact_pin), "Jetson BOARD pin"), "BOARD"
        if _is_gpio09_alias(pin_text):
            return GPIO.BOARD, GPIO09, "BOARD"

        board = _number_after_prefix(compact_pin, ("BOARDPIN", "PHYSICAL", "BOARD", "PIN"))
        if board is not None:
            return GPIO.BOARD, _positive_int(board, "Jetson BOARD pin"), "BOARD"

        bcm = _number_after_prefix(compact_pin, ("BCM",))
        if bcm is not None:
            return GPIO.BCM, _positive_int(bcm, "Jetson BCM pin"), "BCM"

        raise ValueError(
            f"Unsupported Jetson GPIO pin {pin!r}. Use GPIO09, a BOARD pin "
            "number such as 7, BOARD7, or a BCM pin such as BCM4."
        )

    def on(self) -> None:
        self._gpio.output(self.pin, self._active_state)

    def off(self) -> None:
        self._gpio.output(self.pin, self._inactive_state)

    def read(self) -> int:
        return int(self._gpio.input(self.pin))

    def read_label(self) -> str:
        value = self.read()
        if value == int(self._gpio.HIGH):
            return "HIGH"
        if value == int(self._gpio.LOW):
            return "LOW"
        return str(value)

    def close(self) -> None:
        if self._gpio is not None:
            try:
                self._gpio.output(self.pin, self._inactive_state)
            finally:
                self._gpio.cleanup(self.pin)
