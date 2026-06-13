from __future__ import annotations


DEFAULT_TRIGGER_PIN = "GPIO-09"


def _normalize_cvm_gpio_name(pin_name: str) -> str | None:
    compact_name = pin_name.strip().upper().replace("_", "-")
    if compact_name.startswith("GPIO-"):
        suffix = compact_name[5:]
    elif compact_name.startswith("GPIO"):
        suffix = compact_name[4:]
    else:
        return None

    if not suffix.isdigit():
        return None
    return f"GPIO{int(suffix):02d}"


class GPIOOutputPin:
    def __init__(self, pin):
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
            return GPIO.BCM, pin, "BCM"

        pin_text = str(pin).strip()
        if not pin_text:
            raise ValueError("GPIO pin cannot be empty")
        if pin_text.isdigit():
            return GPIO.BCM, int(pin_text), "BCM"

        cvm_name = _normalize_cvm_gpio_name(pin_text)
        if cvm_name is not None:
            return GPIO.CVM, cvm_name, "CVM"

        channel = pin_text.upper()
        if channel.startswith("GP"):
            return GPIO.TEGRA_SOC, channel, "TEGRA_SOC"
        return GPIO.CVM, channel, "CVM"

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
