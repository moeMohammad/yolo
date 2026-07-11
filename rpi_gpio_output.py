from __future__ import annotations

import os


RPI_TRIGGER_PIN = 17  # BCM GPIO17 = Raspberry Pi physical header pin 11.
DEFAULT_TRIGGER_PIN = RPI_TRIGGER_PIN

_GPIOZERO_INSTALL_HINT = (
    "Install gpiozero on the Raspberry Pi (it drives the Pi 5 via lgpio):\n"
    "  sudo apt install python3-gpiozero python3-lgpio\n"
    "or inside a virtualenv:\n"
    "  pip install gpiozero lgpio"
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


def resolve_pin_spec(pin) -> str:
    """Normalize a pin value into a gpiozero pin spec string.

    An int or digit string is a BCM number (``17`` -> ``"GPIO17"``); GPIO/BCM
    prefixes are BCM numbers too; BOARD/PIN/PHYSICAL prefixes address the
    physical header (``"BOARD11"``).
    """

    if isinstance(pin, int):
        if pin < 0:
            raise ValueError("BCM GPIO number cannot be negative")
        return f"GPIO{pin}"

    pin_text = str(pin).strip()
    if not pin_text:
        raise ValueError("GPIO pin cannot be empty")

    compact_pin = _compact_pin_name(pin_text)
    if compact_pin.isdigit():
        return f"GPIO{int(compact_pin)}"

    bcm = _number_after_prefix(compact_pin, ("GPIO", "BCM"))
    if bcm is not None:
        return f"GPIO{bcm}"

    board = _number_after_prefix(compact_pin, ("BOARDPIN", "PHYSICAL", "BOARD", "PIN"))
    if board is not None:
        return f"BOARD{board}"

    raise ValueError(
        f"Unsupported Raspberry Pi GPIO pin {pin!r}. Use a BCM number such as "
        "17, GPIO17, BCM17, or a physical header pin such as BOARD11."
    )


class RPiGPIOOutputPin:
    """Raspberry Pi solenoid output driven by gpiozero (lgpio backend on Pi 5).

    Same interface as the Jetson ``gpio_output.GPIOOutputPin`` so the two are
    interchangeable pin factories for the reject scheduler.
    """

    def __init__(self, pin=DEFAULT_TRIGGER_PIN, *, active_low: bool | None = None):
        self.requested_pin = pin
        self.pin = resolve_pin_spec(pin)
        self._device = None
        self.active_low = _env_flag("GPIO_OUTPUT_ACTIVE_LOW") if active_low is None else bool(active_low)

        try:
            from gpiozero import DigitalOutputDevice
        except Exception as exc:
            raise RuntimeError(
                f"Could not import gpiozero for GPIO pin {self.pin}.\n\n{_GPIOZERO_INSTALL_HINT}"
            ) from exc

        try:
            self._device = DigitalOutputDevice(
                self.pin,
                active_high=not self.active_low,
                initial_value=False,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Could not initialize gpiozero output on {self.pin}. Check that "
                "the pin is a valid free GPIO and that the user may access "
                "/dev/gpiochip* (add the user to the `gpio` group if not)."
            ) from exc

        polarity = "active-low" if self.active_low else "active-high"
        self.backend_name = f"gpiozero {self.pin} {polarity}"

    def on(self) -> None:
        self._device.on()

    def off(self) -> None:
        self._device.off()

    def read(self) -> int:
        return int(self._device.value)

    def read_label(self) -> str:
        return "ACTIVE" if self.read() else "INACTIVE"

    def close(self) -> None:
        if self._device is not None:
            try:
                self._device.off()
            finally:
                self._device.close()
                self._device = None
