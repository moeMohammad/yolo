#!/usr/bin/env python3
"""
Manual Jetson.GPIO controller for the cap-line trigger pin.

Examples:
  python3 manual_gpio.py
  python3 manual_gpio.py on
  python3 manual_gpio.py off
  python3 manual_gpio.py pulse --duration 0.5
"""

from __future__ import annotations

import argparse
import shlex
import time

from gpio_output import DEFAULT_TRIGGER_PIN, GPIOOutputPin


DEFAULT_PULSE_DURATION = 1.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Manually control the Jetson.GPIO output pin used by the cap-line runtime."
    )
    parser.add_argument(
        "--pin",
        default=DEFAULT_TRIGGER_PIN,
        help=(
            f"Jetson.GPIO BOARD pin to control "
            f"(default: {DEFAULT_TRIGGER_PIN}, GPIO09)"
        ),
    )

    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("on", help="turn the pin on and hold it until Ctrl+C")
    subparsers.add_parser("off", help="turn the pin off and exit")

    pulse_parser = subparsers.add_parser(
        "pulse",
        help="turn the pin on briefly, then back off",
    )
    pulse_parser.add_argument(
        "--duration",
        type=float,
        default=DEFAULT_PULSE_DURATION,
        help=f"seconds to keep the pin on (default: {DEFAULT_PULSE_DURATION})",
    )

    subparsers.add_parser(
        "interactive",
        help="start an interactive prompt (used by default)",
    )
    return parser


def pulse(pin: GPIOOutputPin, duration: float) -> None:
    if duration <= 0:
        raise ValueError("pulse duration must be greater than 0")

    pin.on()
    print(f"Pin ON for {duration:.3f}s")
    try:
        time.sleep(duration)
    finally:
        pin.off()
        print("Pin OFF")


def hold_on(pin: GPIOOutputPin) -> None:
    pin.on()
    print("Pin ON")
    print("Press Ctrl+C to turn it off and exit.")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nStopping manual hold.")


def run_interactive(pin: GPIOOutputPin, pin_number: int) -> None:
    state = "OFF"
    print(f"Interactive Jetson.GPIO control for pin {pin_number}")
    print("Commands: on, off, pulse [seconds], status, quit")

    while True:
        try:
            raw = input("gpio> ").strip()
        except EOFError:
            print()
            break
        except KeyboardInterrupt:
            print()
            break

        if not raw:
            continue

        parts = shlex.split(raw)
        command = parts[0].lower()

        if command in {"quit", "exit"}:
            break
        if command == "status":
            print(f"Pin state: {state}")
            continue
        if command == "on":
            pin.on()
            state = "ON"
            print("Pin ON")
            continue
        if command == "off":
            pin.off()
            state = "OFF"
            print("Pin OFF")
            continue
        if command == "pulse":
            duration = DEFAULT_PULSE_DURATION
            if len(parts) > 2:
                print("Usage: pulse [seconds]")
                continue
            if len(parts) == 2:
                try:
                    duration = float(parts[1])
                except ValueError:
                    print("Pulse duration must be a number.")
                    continue
            try:
                pulse(pin, duration)
            except ValueError as exc:
                print(exc)
                continue
            state = "OFF"
            continue

        print("Unknown command. Use: on, off, pulse [seconds], status, quit")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    command = args.command or "interactive"

    pin = GPIOOutputPin(args.pin)
    print(f"Using Jetson.GPIO pin {args.pin} via {pin.backend_name}")
    print("Do not run this while the cap-line runtime controls the same pin.")

    try:
        if command == "interactive":
            run_interactive(pin, args.pin)
            return
        if command == "on":
            hold_on(pin)
            return
        if command == "off":
            pin.off()
            print("Pin OFF")
            return
        if command == "pulse":
            pulse(pin, args.duration)
            return
        parser.error(f"Unknown command: {command}")
    finally:
        pin.close()


if __name__ == "__main__":
    main()
