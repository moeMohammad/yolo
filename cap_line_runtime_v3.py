#!/usr/bin/env python3
"""Standalone V3 cap-line runtime entrypoint."""

from __future__ import annotations

import threading
from dataclasses import replace

from cap_line_v3 import *  # noqa: F401,F403
from cap_line_v3.config import config_from_args, parse_args
from cap_line_v3.runtime import run_detection
from cap_line_v3.types import RuntimeCallbacks


def main() -> None:
    config = config_from_args(parse_args())
    run_detection(config, RuntimeCallbacks(), stop_event=threading.Event())


if __name__ == "__main__":
    main()
