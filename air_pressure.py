import time

from gpio_output import GPIO09, GPIOOutputPin


def main() -> None:
    pin = GPIOOutputPin(GPIO09)
    try:
        pin.on()
        time.sleep(1)
        pin.off()
    finally:
        pin.close()


if __name__ == "__main__":
    main()
