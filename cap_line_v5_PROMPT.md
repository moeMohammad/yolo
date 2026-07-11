# Build the "v5" cap-inspection runtime + operator UI

## Role & objective

v5 is an incremental, surgical upgrade of the v4 system described in
`cap_line_v4_PROMPT.md` (read that first — everything not listed below is
**unchanged** from v4). v4's architecture is correct: two independent
per-camera capture+inference loops, per-camera tracking with a defect-wins (OR)
decision, a shared `CapEventManager` for cross-camera de-dup, a
`RejectScheduler` heap for delayed fires, PyQt6 UI, sqlite history.

v5 exists to fix four things observed on the real rig:

1. **The double-trigger bug** — one defective cap sometimes gets **two air
   pulses back-to-back**.
2. **New model** — default to `dirtv7.onnx` (same end2end YOLO26 export format
   as dirtv6: 640×640 input, `[1, 300, 6]` pre-NMS'd output, same two classes,
   so the v4 model I/O code works unchanged).
3. **New target hardware** — the rig now runs on a **Raspberry Pi 5**, not a
   Jetson Nano. GPIO must go through **gpiozero** (which uses the lgpio backend
   on the Pi 5). The solenoid is wired to **BCM GPIO17 (physical pin 11)**.
   **Do not delete or modify `gpio_output.py`** (the Jetson driver used by
   v1–v4) — add a new, parallel Raspberry Pi driver.
4. **Operator visibility** — the UI must show the **final per-cap
   classification**, i.e. the merged cross-camera verdict that the trigger
   decision is actually based on, not just the per-frame boxes.

Deliverables (new files only; never modify v1–v4 files):

- `cap_line_v5_PROMPT.md` — this document.
- `rpi_gpio_output.py` — Raspberry Pi (gpiozero) solenoid driver, repo root,
  next to the untouched `gpio_output.py`.
- `cap_line_v5/` — Python package (same module layout as `cap_line_v4/`).
- `cap_line_runtime_v5.py` — headless entry point.
- `cap_line_ui_v5.py` — PyQt6 operator UI.
- `cap_line_ui_v5_settings.json` — persisted UI settings.
- `tests/test_cap_line_v5.py` — pytest suite.

## 1. The double-trigger fix (the core of v5)

### Why v4 double-fires

v4's de-dup merges a finished track into the open cap event only if the merge
happens within `global_cooldown_ms` (50 ms) of the moment the **first finish
was detected** (*processing* time). But the two cameras' finishes are detected
by two independent threads whose inference cadence on a Pi 5 CPU is slow
(~50–100 ms per frame) and jittery. When camera B's finish lands >50 ms after
camera A's, v4 opens a **second cap event for the same physical cap and fires
again**. `track_timeout_ms` = 30 ms makes it worse: it is shorter than one
processed-frame interval, so a single missed detection "finishes" a track while
the cap is still in view.

### The v5 rule (implement exactly this, in `cap_line_v5/decision.py`)

Three layers, all timed off the **physical** timestamp `track.last_seen`
(the capture time of the last frame the cap was seen in), never off the time
the finish happened to be *detected*:

1. **Merge window keyed to cap-exit times.** A finished track belongs to the
   open cap event iff `abs(track.last_seen − event.last_seen) ≤
   merge_window_ms` (default **150**). Because both cameras watch the same cap
   at the same time, their `last_seen` values are physically close no matter
   how late either finish is *reported*. A genuinely new cap has a `last_seen`
   far outside the window → finalize the old event, open a new one (as v4 did).
   `event.last_seen` is the max across merged tracks; out-of-order arrivals
   (camera B reporting an *earlier* exit) must still merge — hence `abs()`.

2. **Post-fire refractory (the hard once-per-cap guarantee).** The manager
   remembers the `last_seen` reference of the most recent **scheduled fire**.
   A new fire is **suppressed** (not delayed) if its reference satisfies
   `ref − last_fire_ref < min_fire_interval_ms` (default **250**). This
   catches anything that leaks past the merge window (extreme thread stall,
   track fragmentation). Suppression is logged (`[DEDUP]`), the event is still
   recorded as a reject, but no second pulse happens. The default is safe
   because a single pulse is already `trigger_duration` = 0.3 s long — two caps
   the air system could distinguish can't reach the nozzle 250 ms apart.

3. **Saner track timeout.** `track_timeout_ms` default raised **30 → 150**.
   Rule of thumb printed in the docs/UI tooltip: set it above ~2× your worst
   processed-frame interval (watch the UI's processed FPS), and below the gap
   between consecutive caps at one camera.

Event finalization (logging) still happens on the coordinator flush: close the
open event once `now` is past `event.last_seen + merge_window + track_timeout`
(all finishes that could merge must have been reported by then). A
pathologically late finish after finalization at worst logs a second row — the
refractory guarantees it can never re-fire.

Config: `global_cooldown_ms` is **renamed** to `merge_window_ms`;
`min_fire_interval_ms` is new. Loading a v4 settings JSON must keep working:
map a legacy `global_cooldown_ms` key onto `merge_window_ms` when the new key
is absent.

Fire timing is unchanged from v4: `requested_fire_time = track.last_seen +
fire_delay_s`, scheduled on the `RejectScheduler` heap.

## 2. Raspberry Pi GPIO (`rpi_gpio_output.py`)

A new self-contained driver mirroring the shape of `gpio_output.py`
(`on()` / `off()` / `read()` / `read_label()` / `close()`, `backend_name`,
lazy hardware import inside `__init__` so importing the module is harmless on
a dev laptop):

- `RPI_TRIGGER_PIN = 17` (BCM GPIO17, physical pin 11) as the default.
- Class `RPiGPIOOutputPin(pin=RPI_TRIGGER_PIN, *, active_low=None)` built on
  `gpiozero.DigitalOutputDevice(pin_spec, active_high=not active_low,
  initial_value=False)`.
- Pin spec parsing: an `int` or digit string is a **BCM** number (`17` →
  `"GPIO17"`); `"GPIO17"` / `"BCM17"` → BCM 17; `"BOARD11"` / `"PIN11"` /
  `"PHYSICAL11"` → gpiozero board spec `"BOARD11"`. Reject anything else with
  a clear `ValueError`.
- `active_low` falls back to the `GPIO_OUTPUT_ACTIVE_LOW` env flag, same as
  the Jetson driver.
- If `gpiozero` can't be imported, raise a `RuntimeError` telling the operator
  to `sudo apt install python3-gpiozero python3-lgpio` (or `pip install
  gpiozero lgpio`).
- `close()` drives the pin inactive, then closes the device.

The runtime grows a `gpio_backend` config value: `"rpi"` (default) or
`"jetson"`. `simulate_gpio: true` still short-circuits both to
`NullGPIOOutputPin`. The Jetson path simply imports the old `GPIOOutputPin`
from `gpio_output.py` — kept working, never edited.

## 3. UI: final per-cap verdict (`cap_line_ui_v5.py`)

Same UI as v4 plus a **verdict banner** at the top of the Live tab: a large
colored label showing the most recent *finalized cap event* — exactly the
merged record the trigger decision used, delivered via `history_callback`:

- Reject: red background — `CAP <event_id>: REJECT — dirt_defect 0.87 (cam 0,1)`
- Pass: green background — `CAP <event_id>: PASS — undefected 0.93`

It updates only from per-cap history records (never from raw per-frame boxes),
so what the operator sees is literally what fired (or didn't fire) the nozzle.
Config tab gains `merge_window_ms`, `min_fire_interval_ms`, and a
`gpio_backend` selector; the trigger-pin field is relabeled
"Trigger Pin (BCM GPIO, e.g. 17)". "Test Fire" respects `gpio_backend` +
`simulate_gpio`. History goes to `data/cap_line_history_v5.sqlite3`
(table `cap_line_history_v5`).

## 4. Config deltas (everything else identical to the v4 table)

| Key | v4 | v5 |
|---|---|---|
| `model` | `dirtv6.onnx` | `dirtv7.onnx` |
| `global_cooldown_ms` | `50` | **renamed** `merge_window_ms`, default `150`, keyed to `last_seen` |
| `min_fire_interval_ms` | — | new, default `250` (post-fire refractory, suppresses) |
| `track_timeout_ms` | `30` | `150` |
| `trigger_pin` | `7` (Jetson BOARD) | `17` (BCM GPIO17 = physical pin 11) |
| `gpio_backend` | — | new: `"rpi"` (default) \| `"jetson"` |
| `db_path` | `...history_v4.sqlite3` | `data/cap_line_history_v5.sqlite3` |

## 5. Tests (`tests/test_cap_line_v5.py`)

Port the v4 suite to v5 names/defaults, then add the regression tests that
encode the double-trigger fix:

1. **Late second camera does not double-fire**: camera 0's defective track
   finishes; camera 1's finish for the same cap is *reported* 200 ms later
   (processing lag) but with `last_seen` only 20 ms apart → merges, exactly
   one fire, one history row.
2. **Refractory suppression**: two defective finishes whose `last_seen` are
   200 ms apart (outside the 150 ms merge window, inside the 250 ms
   refractory) → two events, **one** fire.
3. **Distinct caps still both fire**: `last_seen` 400 ms apart → two fires.
4. **Legacy settings key**: a JSON dict containing `global_cooldown_ms` and no
   `merge_window_ms` loads with that value as `merge_window_ms`.
5. **RPi pin driver**: with a fake `gpiozero` module injected into
   `sys.modules`, `RPiGPIOOutputPin` parses `17`, `"GPIO17"`, `"BOARD11"`,
   honors `active_low`, pulses on/off, and rejects garbage pin specs.

## Acceptance criteria

- `python -m pytest tests/test_cap_line_v5.py` passes (run from the repo root).
- `python cap_line_runtime_v5.py --simulate-gpio` runs (or fails with a clear
  camera message) and prints one line per physical cap.
- `python cap_line_ui_v5.py` shows the live preview **and the verdict banner**.
- On the Pi 5: real fires go through gpiozero on BCM GPIO17, and one defective
  cap produces **exactly one** pulse even when the two cameras' track finishes
  are reported far apart.
- `gpio_output.py` and every v1–v4 file byte-identical to before.
