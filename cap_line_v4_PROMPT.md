# Build a fresh "v4" cap-inspection runtime + operator UI

## Role & objective

You are implementing a **brand-new, clean v4** of an industrial cap-inspection
system from scratch. The existing v1–v3 code in this repository is **heavy with
"production line" geometry** — anchor lines, belt-speed-in-mm/s, nozzle-distance
prediction, snapshot holds, frame-pair skew synchronization, prediction
horizons, etc. **v4 must throw all of that away.** Do **not** import from, copy,
or depend on `cap_line_v3/`, `cap_line_runtime*.py`, or `cap_line_ui_v3.py`. v4
is a fresh, self-contained implementation that is dramatically simpler.

Build only two user-facing entry points plus their support package:

- `cap_line_runtime_v4.py` — headless runtime entry point.
- `cap_line_ui_v4.py` — PyQt6 operator UI that drives the runtime.
- `cap_line_v4/` — Python package holding all v4 logic.
- `cap_line_ui_v4_settings.json` — persisted UI settings (slim config only).

## The physical setup (read carefully — this drives every design choice)

- A conveyor carries plastic **caps** past an inspection station, one at a time.
  **At most one cap is in any camera's field of view at a time** (caps are
  spaced out). Design for this common case but don't crash if two briefly
  overlap.
- **Two USB cameras** (V4L2) both look at the **same cap from two different
  angles** (e.g. two sides / top) so dirt visible to one camera but hidden from
  the other is still caught. Because the angles differ, **a box in camera A
  cannot be matched to a box in camera B by pixel position** — they are
  different viewpoints of the same physical object.
- A single **ONNX YOLO model** classifies each detected cap as one of two
  classes (see `classes.txt`): `undefected` (id 0) or `dirt_defect` (id 1).
- An **air nozzle is mounted downstream** of the cameras. When a cap is judged
  defective, we pulse a solenoid (via GPIO) to blow that cap off the belt.
- Because the nozzle is downstream, firing must be **delayed** by a tunable
  amount after the cap is seen, so the air fires when the cap has physically
  reached the nozzle.

## The core problem v4 must solve

A cap is a **video stream object**: each camera sees it across many consecutive
frames, and both cameras see it at roughly the same time. We must
**fire the air exactly once per physical cap**, never zero times for a defect
and never twice.

### Required algorithm (implement this precisely)

**1. Per-camera tracking.** For each camera, independently:
   - Each frame: run the model, keep detections whose confidence ≥
     `reject_threshold`.
   - Associate each detection to an existing **track** for that camera by box
     overlap (IoU ≥ `track_iou`) / nearest-centroid (greedy). Create a new
     track for any unmatched detection.
   - A track accumulates state across frames: `first_seen`, `last_seen`,
     `frame_count`, and `is_defect`.
   - **Decision rule = pure single-frame "defect wins" (OR).** If *any* frame in
     the track is classified `dirt_defect` (conf ≥ `reject_threshold`), the track
     is defective. (No N-frame voting, no majority — a single defect frame is
     enough. This is intentional; maximum sensitivity.)
   - A track is considered **finished ("the cap left view")** when it has gone
     unmatched for `track_timeout_ms` (i.e. no detection associated to it for
     that long).

**2. Firing timing.** When a defective track **finishes**, schedule a fire at:

   ```
   requested_fire_time = track.last_seen_time + fire_delay_s
   ```

   `fire_delay_s` is the single tunable that replaces ALL of v3's
   belt-speed / nozzle-distance / prediction-horizon math. It is hand-tuned on
   the rig. The **delay countdown starts from when the cap leaves view**
   (`last_seen`), which is the most consistent reference point. Use the existing
   `RejectScheduler` pattern (heap keyed by `requested_fire_time`) so fires
   happen on time on a dedicated thread.

**3. Cross-camera de-duplication (the once-per-cap guarantee).** Both cameras
   see the same cap and both tracks finish at nearly the same time. To avoid
   double-firing:
   - Maintain a **single global "cap event" manager** shared by both cameras.
   - When a finished track wants to schedule a fire, check a **global cooldown**:
     if a fire was already scheduled within the last `global_cooldown_ms`, treat
     this finished track as **the same physical cap** and do **not** schedule a
     second fire (just merge it into the existing cap event for logging/decision
     — remember defect-wins still applies, so if the first camera saw it clean
     and the second saw it dirty, the second still triggers a fire **only if the
     first didn't already fire**; design the merge so the cap is rejected if
     *either* camera flagged it).
   - `global_cooldown_ms` should be large enough to cover the small spread
     between the two cameras' exit times plus jitter, but smaller than the gap to
     the next cap. **This is a small, fast conveyor — caps move quickly and are
     close together, so this window MUST be very short (default `50` ms).** It
     must never bridge two adjacent caps. Make it a tunable config value.
   - The same cooldown also prevents a single camera's slightly-late frames from
     producing a second fire.

**4. Logging.** Record **one row per physical cap** (after cross-camera merge):
   timestamp, result (`reject` / `pass`), winning class, confidence, which
   camera(s) flagged it, and the scheduled/actual fire times. Pass caps may be
   logged or counted-only — keep it simple.

> Net effect: undefected caps are ignored; each defective physical cap produces
> exactly one air pulse, delayed by `fire_delay_s` from when it left the
> cameras' view, regardless of how many frames or which cameras saw it.

## What to KEEP from the existing repo (reuse, don't reinvent)

- **`gpio_output.py`** — use `GPIOOutputPin` / `GPIO09` (Jetson.GPIO BOARD pin 7)
  exactly as-is for the air solenoid. Provide a **`NullGPIOOutputPin`
  simulation** fallback (copy the tiny class from `cap_line_v3/actuation.py`)
  selected by a `simulate_gpio` flag, so the system runs on a dev laptop with no
  Jetson hardware.
- **The `RejectScheduler` actuation pattern** from `cap_line_v3/actuation.py` —
  copy it into `cap_line_v4/actuation.py` (self-contained; do not import from
  v3). It already supports a `requested_fire_time` heap, `trigger_duration`
  pulse, `trigger_min_gap`, injectable `time_fn`/`sleep_fn` for tests, and a
  completion callback. This is exactly what step 2 needs.
- **Model I/O code** — the ONNX inference path (letterbox resize, BGR→RGB,
  CHW/normalize preprocess, `onnxruntime.InferenceSession` with CUDA→CPU
  provider fallback, and the YOLO output decode/NMS) in `cap_line_v3/runtime.py`
  is correct. **Copy the relevant functions** into `cap_line_v4/` (e.g.
  `model.py`) and strip them down — do not import from v3. Keep
  `imgsz` auto-detection from the model input shape, with an optional override.
- **PyQt6** for the UI, and **sqlite3** for history logging (same libraries v3
  uses).
- **Target hardware:** Jetson Nano + Jetson.GPIO + two V4L2 USB cameras, with
  the simulate/no-GPIO fallback for development.

## What to REMOVE / NOT build (explicit non-goals)

Delete every concept below — none of it exists in v4:

- Anchor lines / `anchor_axis` / `anchor_line_ratio` / any "line crossing" logic.
- `belt_speed_mm_per_s`, `nozzle_distance_mm`, and all mm-based geometry.
- `actuation_prediction_horizon_ms`, prediction/extrapolation of cap position.
- Snapshots / `actuation_snapshot_hold_ms` / debug burst frames / saved
  picture pipelines.
- Frame-pair synchronization: `pair_max_skew_ms`, `merge_window_ms`,
  `select_synchronized_frame_pair`, `select_capture_batch`,
  `single_camera_wait_ms`, `max_missing_frames`, `CaptureBatch`/`FramePair`
  pairing machinery. **Each camera is processed independently** in v4 — no
  cross-camera frame pairing at the capture layer; the only cross-camera logic
  is the de-dup in step 3.
- `latency_compensation_ms`, `preview_latency_compensation_ms`,
  `decision_deadline_guard_ms`, `finalize_quiet_ms`, `trigger_offset_s`,
  `timing_camera`, separate `tracking_threshold` vs `reject_threshold`.

## Configuration (the entire slim v4 config)

Define a `RuntimeConfig` dataclass in `cap_line_v4/config.py` with
`parse_args()` (argparse) and `config_from_args()`, plus JSON load/save for the
UI settings file. Keep it to roughly this set:

| Key | Meaning | Sensible default |
|---|---|---|
| `model` | ONNX model path | `dirtv6.onnx` |
| `cameras` | list of 2 camera indices/paths | `["0", "3"]` |
| `mirror_cameras` | per-camera horizontal flip | `[false, true]` |
| `resolution` | `[width, height]` | `[960, 600]` |
| `target_fps` | requested camera FPS | `60` |
| `exposure` | V4L2 exposure | `8` |
| `pixel_format` | V4L2 fourcc | `"YUYV"` |
| `imgsz` | model input override (null = auto) | `null` |
| `onnx_intra_op_threads` | ORT threads | `max(1, cpu//2)` |
| `reject_threshold` | min confidence for a valid detection / defect | `0.45` |
| `track_iou` | min IoU to associate a detection to a track | `0.3` |
| `track_timeout_ms` | no-match time before a track is "finished" | `30` |
| `fire_delay_s` | delay from cap-leaves-view to air fire | `0.0` (tune on rig) |
| `global_cooldown_ms` | once-per-cap / cross-camera suppression window | `50` |
| `trigger_pin` | Jetson BOARD pin for solenoid | `7` |
| `trigger_duration` | air pulse length (s) | `0.3` |
| `trigger_min_gap` | min gap between pulses (s) | `0.0` |
| `simulate_gpio` | use NullGPIOOutputPin instead of real GPIO | `false` |
| `live_preview_fps` | UI preview throttle | `30` |
| `db_path` | sqlite history path | `data/cap_line_history_v4.sqlite3` |
| `no_display` | run headless (no preview callback) | `false` |

Drop everything from `cap_line_ui_v3_settings.json` not in this table.

## Runtime contract (so the UI can drive it)

In `cap_line_v4` / `cap_line_runtime_v4.py`, expose the same shape of API v3 used
so the UI wiring is familiar:

```python
@dataclass(frozen=True)
class RuntimeCallbacks:
    preview_callback: Callable[[object], None] | None = None   # composite BGR frame for display
    history_callback: Callable[[CapEventRecord], None] | None = None
    performance_callback: Callable[[PerfSnapshot], None] | None = None
    log_fn: Callable[..., None] = print

def run_detection(config: RuntimeConfig,
                  callbacks: RuntimeCallbacks,
                  stop_event: threading.Event) -> None: ...
```

- One **capture+inference loop per camera** (threads), each feeding its own
  tracker; a shared **`CapEventManager`** owns cross-camera de-dup + the
  `RejectScheduler`.
- `preview_callback` receives a composite (e.g. side-by-side) BGR image with
  overlay boxes drawn: **green for `undefected`, red for `dirt_defect`**, plus a
  small label with class + confidence. No line/anchor overlays.
- `performance_callback` reports per-camera capture FPS, processed FPS,
  inference ms, and current GPIO backend name.
- `run_detection` must shut down cleanly on `stop_event`: stop threads, close
  cameras, close the scheduler/GPIO pin.

`cap_line_runtime_v4.py` itself is just: parse args → build config → call
`run_detection(config, RuntimeCallbacks(), threading.Event())`.

## Operator UI (`cap_line_ui_v4.py`, PyQt6)

Mirror the structure of the v3 UI but with the slim feature set. Include:

- **Live dual-camera preview** (the composite frame from `preview_callback`)
  with overlay boxes.
- **Start / Stop** buttons that launch/stop `run_detection` on a worker thread
  with a `stop_event`.
- **Status bar:** GPIO backend (real vs simulation), per-camera FPS, inference
  ms, running/stopped.
- **Counters:** total caps seen, total rejects (fires), this session.
- **Settings panel** editing exactly the config table above (no line/snapshot
  controls), persisted to `cap_line_ui_v4_settings.json` (load on launch, save on
  change/close). Reuse v3's settings load/save approach.
- **Manual "Test fire"** button that pulses the solenoid once (respects
  `simulate_gpio`) so the operator can verify the air line.
- **Recent rejects table** populated from `history_callback` / the sqlite DB
  (one row per physical cap): time, result, class, confidence, camera(s),
  fire time.

## Tests

Add `tests/test_cap_line_v4.py` (pytest) covering the logic that matters, using
injected fakes (fake frames, a fake ONNX session returning scripted boxes, and
an injected `time_fn`/`sleep_fn` into the scheduler so timing is deterministic):

1. **Tracker association + OR decision:** a sequence of frames for one camera
   where a cap appears, is `undefected` for several frames then `dirt_defect`
   once → track finishes as defective.
2. **Track finish on timeout:** track ends after `track_timeout_ms` of no match.
3. **Fire timing:** a finished defective track schedules a fire at
   `last_seen + fire_delay_s` (assert the scheduler's `requested_fire_time`).
4. **Once-per-cap across cameras:** both cameras report the same defective cap
   finishing within `global_cooldown_ms` → **exactly one** fire is scheduled.
5. **Pass caps don't fire:** an all-`undefected` track schedules nothing.
6. **Threshold filtering:** detections below `reject_threshold` are ignored.

## Deliverables & acceptance criteria

- New files only: `cap_line_runtime_v4.py`, `cap_line_ui_v4.py`,
  `cap_line_v4/` package (`__init__.py`, `config.py`, `model.py`, `tracking.py`,
  `decision.py` (the `CapEventManager`), `actuation.py`, `runtime.py`,
  `types.py`), `cap_line_ui_v4_settings.json`, `tests/test_cap_line_v4.py`. Do
  not modify or break existing v1–v3 files (except you may reuse
  `gpio_output.py` unchanged).
- `python cap_line_runtime_v4.py --simulate-gpio` runs against the configured
  cameras (or fails gracefully with a clear message if absent) and prints
  per-cap decisions.
- `python cap_line_ui_v4.py` launches the PyQt6 UI, shows live previews, and
  Start/Stop works.
- `pytest tests/test_cap_line_v4.py` passes.
- The code contains **no** anchor-line, belt-speed, nozzle-distance, snapshot,
  prediction-horizon, or frame-pairing logic.
- Keep the code small, readable, and well-commented where the de-dup/timing
  logic is subtle. Match the existing repo's Python style
  (`from __future__ import annotations`, dataclasses, type hints).

## Summary of the locked design decisions

- Two cameras = **same cap, two angles**; combine with **defect-wins (OR)**.
- **Pure single-frame** defect decision (no N-frame voting).
- **Per-camera position tracking** assigns one track per cap; a track finishes
  when the cap leaves view (`track_timeout_ms`).
- **Cross-camera de-dup via a global cooldown** guarantees one fire per cap.
  The conveyor is **small and fast** — all timing windows must be **very short**
  (`global_cooldown_ms` ≈ 50 ms, `track_timeout_ms` ≈ 30 ms) so they never bridge
  two adjacent caps; treat these as the most safety-critical tunables.
- Nozzle is **downstream** → fire at **`last_seen + fire_delay_s`**, the delay
  countdown starting **when the cap leaves view**.
- Reuse **`gpio_output.py`**, the **`RejectScheduler`** pattern, the ONNX model
  I/O, **PyQt6**, and **sqlite** logging; target the same **Jetson Nano**
  hardware with a **simulate** fallback.
- One cap in view at a time (design for it; degrade gracefully if not).
