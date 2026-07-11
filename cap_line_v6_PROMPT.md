# Build the "v6" cap-inspection runtime + operator UI

## Role & objective

v6 is **v5 for the Jetson Nano rig**. Read `cap_line_v5_PROMPT.md` first —
v6 is byte-for-byte the same design (dirtv7 model, the double-trigger fix with
`merge_window_ms` keyed to physical cap-exit times plus the
`min_fire_interval_ms` post-fire refractory, the final per-cap verdict banner
in the UI) with exactly one change: **GPIO defaults to the Jetson driver**
instead of the Raspberry Pi one.

There is no new GPIO code in v6. Both drivers already exist and both stay
selectable at runtime via the `gpio_backend` config value:

- `"jetson"` (the v6 default) → `GPIOOutputPin` from the untouched
  `gpio_output.py` (Jetson.GPIO, BOARD numbering).
- `"rpi"` → `RPiGPIOOutputPin` from `rpi_gpio_output.py` (gpiozero).
- `simulate_gpio: true` overrides both with `NullGPIOOutputPin`.

## Config deltas (everything else identical to v5)

| Key | v5 | v6 |
|---|---|---|
| `gpio_backend` | `"rpi"` | `"jetson"` |
| `trigger_pin` | `17` (BCM GPIO17) | `7` (GPIO09, Jetson Nano BOARD pin 7) |
| `db_path` | `...history_v5.sqlite3` | `data/cap_line_history_v6.sqlite3` |

The UI trigger-pin field is relabeled "Trigger Pin (Jetson BOARD, e.g. 7 =
GPIO09)"; settings persist to `cap_line_ui_v6_settings.json`; sqlite history
uses table `cap_line_history_v6`.

## Deliverables (new files only; never modify v1–v5 files)

- `cap_line_v6_PROMPT.md` — this document.
- `cap_line_v6/` — Python package (same module layout as `cap_line_v5/`).
- `cap_line_runtime_v6.py` — headless entry point.
- `cap_line_ui_v6.py` — PyQt6 operator UI.
- `cap_line_ui_v6_settings.json` — persisted UI settings.
- `tests/test_cap_line_v6.py` — the v5 suite ported to v6, plus a test that
  the defaults resolve to the Jetson driver (`gpio_backend == "jetson"`,
  `trigger_pin == 7`, `resolve_pin_factory(defaults) is GPIOOutputPin`).

## Acceptance criteria

- `python -m pytest tests/test_cap_line_v6.py` passes (run from the repo root).
- `python cap_line_runtime_v6.py --simulate-gpio` runs (or fails with a clear
  camera message) and prints one line per physical cap.
- `python cap_line_ui_v6.py` shows the live preview and the verdict banner.
- On the Jetson: real fires go through Jetson.GPIO on BOARD pin 7, and one
  defective cap produces exactly one pulse.
- `gpio_output.py`, `rpi_gpio_output.py`, and every v1–v5 file byte-identical
  to before.
