# Build the "v6" cap-inspection runtime + operator UI

## Role & objective

v6 began as **v5 for the Jetson Nano rig**. Read `cap_line_v5_PROMPT.md` first —
the original v6 was the same design (dirtv7 model, the double-trigger fix with
`merge_window_ms` keyed to physical cap-exit times plus the
`min_fire_interval_ms` post-fire refractory, the final per-cap verdict banner
in the UI) with exactly one change: **GPIO defaults to the Jetson driver**
instead of the Raspberry Pi one.

## Empty-belt / repeated-pulse safety hardening

The production rig later showed that `dirtv7.onnx` can emit class-1 boxes above
0.9 confidence with no cap visible. The old maximum-sensitivity rule treated
one such row as a physical cap and could enqueue many pulses. v6 therefore has
additional fail-safe behavior that is intentionally not backported to v1-v5:

- Overlapping model rows are class-agnostically deduplicated before tracking.
- A track must span at least four processed frames, move coherently along the
  configured conveyor axis/direction, and move its center across the inspection
  line before it is accepted as a physical cap. A box merely overlapping the
  line is insufficient; at least two observations are required on each side of
  the line. Place the previewed gate where real caps have room on both sides.
  Confidence never bypasses this presence gate.
- Dirt must appear in at least three consecutive frames; one or two dirty glitches among
  otherwise clean observations does not reject the cap.
- Camera-local presence-cycle ids cluster near-simultaneous crossings only in
  the same perpendicular image band, then are consumed when firing or
  suppressing. Fragments cannot re-arm, while two concurrently visible caps in
  different bands remain distinct.
- Recently finalized cap decisions are retained as bounded timestamp
  tombstones. A fragment is matched to the closest eligible physical-exit time
  across the current and finalized caps, so a late camera cannot re-arm an old
  clean decision or contaminate the following cap. The post-fire refractory
  includes its exact boundary.
- Capture continuously drains each camera into a latest-frame slot so slow
  inference cannot walk through buffered old cap frames.
- Track timeout uses capture timestamps, not post-inference wall time.
- The default track timeout/maximum observation gap is 250 ms, so a cap sampled
  at a 200 ms processed-frame interval remains one track instead of fragmenting.
- In-flight inference older than 500 ms is dropped. The scheduler rejects fire
  targets over 500 ms late, coalesces targets already covered by the previous
  pulse, drops runnable backlog older than 250 ms, and cancels pending work on
  shutdown. Shutdown is bounded even if an injected logger or GPIO driver
  hangs; an incomplete actuator shutdown is surfaced as a fatal runtime error
  instead of being reported as a normal stop. Unexpected camera-thread crashes
  and camera workers that survive their shutdown deadline are surfaced the
  same way.
- Stopping discards active partial tracks; shutdown never creates a new fire.

The corresponding defaults are exposed in the v6 CLI/UI:
`duplicate_iou_threshold`, `min_track_frames`, `min_track_travel_ratio`,
`min_track_directionality`, `min_defect_frames`, `presence_line_axis`,
`presence_line_ratio`, `presence_direction`, `max_track_gap_ms`, `presence_clear_ms`,
`max_frame_age_ms`, `trigger_max_queue_age_ms`, and
`trigger_max_lateness_ms`.

There is no new GPIO driver in v6. Both hardware drivers already exist and stay
selectable at runtime via the `gpio_backend` config value:

- `"jetson"` (the v6 default) → `GPIOOutputPin` from the untouched
  `gpio_output.py` (Jetson.GPIO, BOARD numbering).
- `"rpi"` → `RPiGPIOOutputPin` from `rpi_gpio_output.py` (gpiozero).
- `simulate_gpio: true` overrides both with `NullGPIOOutputPin`.

## Original hardware config deltas

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
