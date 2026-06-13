from __future__ import annotations


class GPIOOutputPin:
    def __init__(self, pin: int):
        self.pin = int(pin)
        self.backend_name = ""
        self._device = None
        self._gpio = None

        gpiozero_error = None
        try:
            from gpiozero import OutputDevice
            from gpiozero.pins.lgpio import LGPIOFactory

            self._device = OutputDevice(
                self.pin,
                active_high=True,
                initial_value=False,
                pin_factory=LGPIOFactory(chip=0),
            )
            self.backend_name = "gpiozero+lgpio"
            return
        except Exception as exc:
            gpiozero_error = exc

        rpi_gpio_error = None
        try:
            import RPi.GPIO as GPIO

            GPIO.setwarnings(False)
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.pin, GPIO.OUT, initial=GPIO.LOW)
            GPIO.output(self.pin, GPIO.LOW)
            self._gpio = GPIO
            self.backend_name = "RPi.GPIO"
            return
        except Exception as exc:
            rpi_gpio_error = exc

        raise RuntimeError(
            f"Could not initialize GPIO pin {self.pin}. "
            "Tried gpiozero+lgpio and RPi.GPIO. "
            f"gpiozero+lgpio error: {gpiozero_error}. "
            f"RPi.GPIO error: {rpi_gpio_error}. "
            "If this is a Raspberry Pi 5 or a Bookworm/Trixie setup, "
            "install python3-rpi-lgpio or python3-gpiozero plus python3-lgpio. "
            "In a virtual environment, install rpi-lgpio or gpiozero plus lgpio."
        )

    def on(self) -> None:
        if self._device is not None:
            self._device.on()
            return
        self._gpio.output(self.pin, self._gpio.HIGH)

    def off(self) -> None:
        if self._device is not None:
            self._device.off()
            return
        self._gpio.output(self.pin, self._gpio.LOW)

    def close(self) -> None:
        if self._device is not None:
            try:
                self._device.off()
            finally:
                self._device.close()
            return

        if self._gpio is not None:
            try:
                self._gpio.output(self.pin, self._gpio.LOW)
            finally:
                self._gpio.cleanup(self.pin)
