import Jetson.GPIO as GPIO
import time

from gpio_output import GPIO09

GPIO.setmode(GPIO.BOARD)

GPIO.setup(GPIO09, GPIO.OUT, initial=GPIO.LOW)

try:
    test_detection = True

    if test_detection:
        GPIO.output(GPIO09, GPIO.HIGH)
        time.sleep(1)
        GPIO.output(GPIO09, GPIO.LOW)

finally:
    GPIO.output(GPIO09, GPIO.LOW)
    GPIO.cleanup()
