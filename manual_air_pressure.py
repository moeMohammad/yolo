#!/usr/bin/env python3
"""
Manual air-pressure GPIO test.

Press Space to toggle GPIO09 on/off. Press q or Ctrl+C to turn it off and exit.
"""

from __future__ import annotations

import argparse
import contextlib
import select
import sys
import termios
import tty
from collections.abc import Iterator

from gpio_output import GPIO09, GPIOOutputPin


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Manually toggle the air-pressure GPIO output from the keyboard."
    )
    parser.add_argument(
        "--pin",
        default=GPIO09,
        help=f"Jetson.GPIO BOARD pin to control (default: {GPIO09}, GPIO09)",
    )
    return parser


@contextlib.contextmanager
def single_key_input() -> Iterator[None]:
    if not sys.stdin.isatty():
        raise RuntimeError("Manual air-pressure toggle requires an interactive terminal.")

    fd = sys.stdin.fileno()
    previous_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, previous_settings)


def read_key() -> str:
    select.select([sys.stdin], [], [])
    return sys.stdin.read(1)


def main() -> None:
    args = build_parser().parse_args()
    pin = GPIOOutputPin(args.pin)
    is_on = False

    print(f"Using Jetson.GPIO pin {args.pin} via {pin.backend_name}")
    print("Press SPACE to toggle air pressure. Press q to quit.")

    try:
        with single_key_input():
            while True:
                key = read_key().lower()
                if key == "q":
                    break
                if key != " ":
                    continue

                if is_on:
                    pin.off()
                    is_on = False
                    print("\rAir pressure OFF", flush=True)
                else:
                    pin.on()
                    is_on = True
                    print("\rAir pressure ON ", flush=True)
    except KeyboardInterrupt:
        print()
    finally:
        pin.off()
        pin.close()
        print("Air pressure OFF")


if __name__ == "__main__":
    main()
