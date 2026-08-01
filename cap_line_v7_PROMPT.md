# Build the "v7" cap-inspection runtime + operator UI

## Role & objective

v7 began as **v6 with the two-stage model pipeline and probability voting**. Read
`cap_line_v6_PROMPT.md` first — v6's rig-safety concepts carry over
(physical track qualification, presence-cycle idempotency,
merge window keyed to physical exit times, post-fire refractory, latest-frame
capture, stale-safe reject scheduler, Jetson GPIO defaults on BOARD pin 7).

What changes is *how a frame becomes dirt evidence*:

1. **Stage 1 — cap detector** (`cap_detector_640.onnx`, YOLO26n end-to-end,
   single class `cap`, 640 px). Finds and tracks caps; knows nothing about
   dirt. Replaces the 2-class `dirtv7.onnx`.
2. **Stage 2 — dirt classifier** (`dirt_classifier_384.onnx`, YOLO26n-cls,
   384 px). Each tracked cap box is cropped from the ORIGINAL frame with a 10%
   margin, square-padded to gray (114), resized, and scored: output index 0 is
   `P(dirt_defect)` (softmax baked into the export; plain RGB/255 input, no
   ImageNet normalization). The crop geometry must stay byte-compatible with
   `detectx/script/build_two_stage_dataset.py` — the thresholds were
   calibrated on it.
3. **Per-track probability vote** replaces v6's per-frame class latch. A track
   is a defect only when BOTH:
   - at least `min_defect_frames` (2) **consecutive classified** observations scored
     `P(dirt) >= frame_dirt_threshold` (0.50) — the v6 hallucination guard —
   - AND the **trimmed-mean** `P(dirt)` over the whole track reaches
     `track_dirt_threshold` (0.45).
   The verdict is not a latch: enough later clean evidence pulls the trimmed
   mean back down. Cross-camera OR fusion in `decision.py` is unchanged —
   dirt visible from one side still rejects the cap (~8% of labeled pairs are
   one-sided).

Threshold provenance began with the cap-level sweep in
`detectx/exported_models/two_stage_v1_*/training_summary.json`
(`cap_level.operating_point` / `sweep`). Re-run that notebook's Section 6 and
update `frame_dirt_threshold` / `track_dirt_threshold` whenever either model
is retrained — thresholds are model-specific. The v2 classifier's live dirty
recall has not been established, so the inherited values are defaults awaiting
per-cap ground-truth qualification, not a certified operating point.

## Video replay revision (Jul 19, 2026; corrected Aug 1)

Validating on real conveyor recordings (`recorded_videos/`, 30 fps true speed
after the `-itsscale 2` remux) changed three things:

1. **Rig geometry defaults.** Both *saved MP4 views* travel right → left, so
   offline replay uses `mirror_cameras=(False, False)` and
   `presence_direction="negative"`. These files may already contain the
   recorder's camera-2 mirror; they do not prove both raw live orientations.
   The old positive direction does filter camera-0 tracks, but replay still
   produced events from the other view, so direction alone did not explain a
   live zero-cap result. Live mirror/direction calibration remains mandatory.
2. **Classify band** (`classify_band_ratio`, default 0.75): only frames whose
   box center sits in the central 75% of the frame along the belt axis are
   classified. Entry/exit perspectives are out-of-domain for the classifier
   and scored spurious P(dirt); edge frames now contribute no evidence.
3. **v2 classifier.** `dirt_classifier_384.onnx` is retrained
   (`detectx/script/retrain_classifier_v2.py`, exported under
   `exported_models/two_stage_v2_*`) with ~2.3k clean video crops added to
   train and ~3.5k (held-out recording) to val; v1 kept as
   `dirt_classifier_384_v1_backup.onnx`. Full-pipeline replay
   (`validate_v7_on_videos.py`): clean recordings **0/353 false rejects**
   (v1: 235/353), but only **25/257** dirt-folder events were rejected. The
   dirt folder has known label noise, yet the other 232 events were never
   individually labeled; true dirty recall is therefore unknown, not 90.3%.
   Tooling: `harvest_video_transits.py` (crops + per-transit
   P(dirt)), `sweep_video_thresholds.py` (threshold calibration).

## Live-path correction (Aug 1, 2026)

The July replay used ideal virtual timestamps and processed every selected
frame synchronously. It did not reproduce live latest-frame dropping, provider
selection, stale deadlines, UI work, camera failures, or valve timing. A
stride replay of a 50-cap recording showed the original gates falling from
49 events at 7.5 processed FPS to 30 at 6 FPS and **zero at 5 FPS**.

Current v7 therefore adds:

- settings schema migration, including stale local JSON values that survive a
  Git pull;
- 30 FPS camera requests, two-frame/one-per-gate-side cap qualification, wider
  sparse-observation gaps, and explicit insufficient-classifier evidence;
- fail-closed `uninspected` decisions (configurable, two inspected cameras by
  default), instead of silently converting no evidence into a clean pass;
- raw detector-box/confidence counters, processed FPS, stale/filter/unknown
  counters, selected providers, negotiated camera format, visible runtime
  warnings, and fatal sustained camera-read failures;
- interpolated presence-gate timestamps, so the requested fire deadline is
  tied to conveyor position rather than load-dependent track completion;
- terminal actuator outcomes (`fired`, `stale`, `coalesced`, `gpio_failed`, or
  `cancelled`) with requested and actual timestamps kept distinct; and
- atomic settings writes and in-place SQLite schema migration in the UI.

At 7.5 effective FPS (stride 4), the final 0.75 classify band found all 353
clean-folder events with zero classifier dirt rejects, one one-camera-only
fail-closed unknown, 352 valid passes, and four filtered fragments. The old
0.60 band produced 15 unknowns on the same cached predictions. A focused
112-event dirt-folder sample produced 12 classifier rejects, one unknown, and
99 valid passes; those folder labels are not per-cap ground truth. These are
sparse-rate regression checks, not a production accuracy certificate. See
`JETSON_ORIN_RUNBOOK.md` for live commissioning and timing calibration.

## Config deltas (everything else identical to v6)

| Key | v6 | v7 |
|---|---|---|
| `model` | `"dirtv7.onnx"` (2-class) | `"cap_detector_640.onnx"` (1-class cap) |
| `classifier_model` | — | `"dirt_classifier_384.onnx"` |
| `classifier_imgsz` | — | `null` (auto from ONNX input, 384) |
| `reject_threshold` (0.45) | detection conf filter | **removed**, split into: |
| `detect_threshold` | — | `0.25` (stage-1 cap confidence) |
| `frame_dirt_threshold` | — | `0.50` (per-frame P(dirt)) |
| `track_dirt_threshold` | — | `0.45` (trimmed-mean P(dirt) per track) |
| `crop_margin` | — | `0.10` (must match training crops) |
| `classify_band_ratio` | — | `0.75` (current v2 classifier) |
| `max_classified_boxes` | — | `2` (stage-2 budget per frame) |
| `track_timeout_ms` | `250` | `350` |
| `max_track_gap_ms` | `250` | `300` |
| `min_track_frames` | `4` | `2` (sparse-rate live gate) |
| `min_line_side_frames` | — | `1` per side of presence gate |
| `min_classified_frames` | — | `2`; less is `uninspected` |
| `required_inspected_cameras` | — | `2` |
| `reject_uninspected` | — | `true` (fail closed) |
| `camera_read_timeout_s` | — | `2.0` |
| `simulate_gpio` | `false` | `true` (fresh/migrated installs start disarmed) |
| `db_path` | `...history_v6.sqlite3` | `data/cap_line_history_v7.sqlite3` |

Settings migration: a settings dict **without** `classifier_model` predates
the two-stage pipeline, so its `model`/`imgsz` values are ignored (a 2-class
model silently running as the cap detector would break inspection). Unknown
legacy keys (`reject_threshold`, `global_cooldown_ms`, ...) are dropped as in
v6. `postprocess` still accepts class ids 0 and 1 but normalizes both to
class 0, so even an un-migrated legacy model yields cap boxes rather than
dropping the dirty ones.

## Deliverables

- `cap_line_v7_PROMPT.md` — this document.
- `cap_line_v7/` — Python package (same module layout as `cap_line_v6/`).
- `cap_line_runtime_v7.py` — headless entry point.
- `cap_line_ui_v7.py` — PyQt6 operator UI (two-stage model fields + the three
  thresholds in the Config tab; sqlite table `cap_line_history_v7`; settings
  persist to `cap_line_ui_v7_settings.json`).
- `cap_detector_640.onnx`, `dirt_classifier_384.onnx` — repo-root model files
  copied from `detectx/exported_models/two_stage_v1_20260715_195231/`.
- `JETSON_ORIN_RUNBOOK.md` — deployment, diagnostics, model qualification and
  gate-to-nozzle timing procedure.
- `tests/test_cap_line_v7.py` — probability-voting, crop/classify, config
  migration, cross-camera decision, and two-stage runtime wiring tests, plus
  the defaults test (`gpio_backend == "jetson"`, `trigger_pin == 7`, and
  simulation selected until explicitly commissioned).

## Acceptance criteria

- `python -m pytest tests/test_cap_line_v7.py` passes (run from the repo
  root), and `tests/test_cap_line_v6.py` still passes untouched.
- `python cap_line_runtime_v7.py --simulate-gpio` runs (or fails with a clear
  camera message) and prints one line per physical cap.
- `python cap_line_ui_v7.py` shows the live preview and the verdict banner;
  preview boxes show the classifier verdict (red `dirt_defect:P(dirt)` /
  green `undefected:P(clean)` / amber `cap/unclassified`).
- Real-model sanity: both ONNX files load through `create_onnx_session` and
  expose the documented detector/classifier tensor interfaces. This interface
  check is not an accuracy or recall claim.
- A cap dirty on one camera and clean on the other produces exactly one air
  pulse; a clean cap with a single hallucinated 0.97 frame produces none
  (both covered by tests).
- Requested fire targets are never presented as actual GPIO activations, and
  incomplete two-camera inspection never silently passes as clean.
