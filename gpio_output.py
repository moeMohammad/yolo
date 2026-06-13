from __future__ import annotations


GPIO09 = 7  # physical pin 7 on 40-pin header (BOARD numbering)
DEFAULT_TRIGGER_PIN = GPIO09


def _is_gpio09_alias(pin_text: str) -> bool:
    compact_name = pin_text.strip().upper().replace("_", "-")
    if compact_name in {"GPIO09", "GPIO-09", "GPIO-9"}:
        return True
    if compact_name.startswith("GPIO-"):
        suffix = compact_name[5:]
    elif compact_name.startswith("GPIO"):
        suffix = compact_name[4:]
    else:
        return False
    return suffix.isdigit() and int(suffix) == 9


class GPIOOutputPin:
    def __init__(self, pin=DEFAULT_TRIGGER_PIN):
        self.requested_pin = pin
        self.pin = pin
        self._gpio = None

        try:
            import Jetson.GPIO as GPIO
        except Exception as exc:
            raise RuntimeError(
                f"Could not import Jetson.GPIO for GPIO pin {pin}. "
                "Install Jetson.GPIO on the Jetson device and make sure the "
                "runtime user has GPIO permissions."
            ) from exc

        mode, channel, mode_name = self._resolve_channel(GPIO, pin)
        try:
            GPIO.setwarnings(False)
            GPIO.setmode(mode)
            GPIO.setup(channel, GPIO.OUT, initial=GPIO.LOW)
            GPIO.output(channel, GPIO.LOW)
        except Exception as exc:
            raise RuntimeError(
                f"Could not initialize Jetson.GPIO {mode_name} pin {channel}. "
                "Check that the selected Jetson header pin is configured for GPIO "
                "output and that pinmux/permissions are set correctly."
            ) from exc

        self.pin = channel
        self._gpio = GPIO
        self.backend_name = f"Jetson.GPIO {mode_name}"

    @staticmethod
    def _resolve_channel(GPIO, pin):
        if isinstance(pin, int):
            return GPIO.BOARD, pin, "BOARD"

        pin_text = str(pin).strip()
        if not pin_text:
            raise ValueError("GPIO pin cannot be empty")
        if pin_text.isdigit():
            return GPIO.BOARD, int(pin_text), "BOARD"
        if _is_gpio09_alias(pin_text):
            return GPIO.BOARD, GPIO09, "BOARD"

        raise ValueError(
            f"Unsupported GPIO pin {pin!r}. Use a BOARD pin number or GPIO09."
        )

    def on(self) -> None:
        self._gpio.output(self.pin, self._gpio.HIGH)

    def off(self) -> None:
        self._gpio.output(self.pin, self._gpio.LOW)

    def close(self) -> None:
        if self._gpio is not None:
            try:
                self._gpio.output(self.pin, self._gpio.LOW)
            finally:
                self._gpio.cleanup(self.pin)
