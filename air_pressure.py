import atexit
import time

import Jetson.GPIO as GPIO

from gpio_output import DEFAULT_TRIGGER_PIN

TRIGGER_PIN = DEFAULT_TRIGGER_PIN
TRIGGER_CHANNEL = TRIGGER_PIN.upper().replace("-", "")

trigger_delay = 2      # delay before trigger (seconds)
trigger_duration = 1   # ON time (seconds)

GPIO.setwarnings(False)
GPIO.setmode(GPIO.CVM)
GPIO.setup(TRIGGER_CHANNEL, GPIO.OUT, initial=GPIO.LOW)


def _turn_off() -> None:
    GPIO.output(TRIGGER_CHANNEL, GPIO.LOW)


def _cleanup() -> None:
    try:
        _turn_off()
    finally:
        GPIO.cleanup(TRIGGER_CHANNEL)


atexit.register(_cleanup)

start_time = 0.0
trigger_on = False
trigger_start = 0.0

try:
    while True:
        current_time = time.monotonic()

        # Replace this with your real condition
        condition_met = True

        if condition_met:
            # Start timing once
            if start_time == 0.0:
                start_time = current_time

            # After delay → turn ON
            if (current_time - start_time >= trigger_delay) and not trigger_on:
                GPIO.output(TRIGGER_CHANNEL, GPIO.HIGH)
                trigger_on = True
                trigger_start = current_time
                print("ON")

        # Handle turning OFF after duration
        if trigger_on and (current_time - trigger_start >= trigger_duration):
            _turn_off()
            trigger_on = False
            start_time = 0.0
            print("OFF")

        # Your other code can run here freely
except KeyboardInterrupt:
    print("\nStopping.")
