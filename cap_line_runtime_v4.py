#!/usr/bin/env python3
"""Standalone v4 cap-inspection runtime entry point (headless).

Parse args -> build config -> run the detection loop until Ctrl-C. Per-cap
decisions are printed as they happen via a small history callback.
"""

from __future__ import annotations

import sys
import threading

from cap_line_v4.config import config_from_args, parse_args
from cap_line_v4.runtime import run_detection
from cap_line_v4.types import CapEventRecord, RuntimeCallbacks


def _print_cap_event(record: CapEventRecord) -> None:
    cameras = ",".join(str(index) for index in record.flagged_cameras) or "-"
    confidence = "-" if record.confidence is None else f"{record.confidence:.3f}"
    print(
        f"[CAP {record.event_id}] {record.result.upper():6} "
        f"class={record.class_name} conf={confidence} flagged_by={cameras} "
        f"fire={record.requested_fire_time or '-'}"
    )


def main() -> None:
    config = config_from_args(parse_args())
    stop_event = threading.Event()
    callbacks = RuntimeCallbacks(history_callback=_print_cap_event, log_fn=print)
    try:
        run_detection(config, callbacks, stop_event)
    except KeyboardInterrupt:
        stop_event.set()
        print("\nStopping v4 runtime...")
    except RuntimeError as exc:
        # e.g. cameras could not be opened: report cleanly instead of a traceback.
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
