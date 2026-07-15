# Build the "v7" cap-inspection runtime + operator UI

## Role & objective

v7 is **v6 with the two-stage model pipeline and probability voting**. Read
`cap_line_v6_PROMPT.md` first — all of v6's rig-safety machinery carries over
byte-compatibly (physical track qualification, presence-cycle idempotency,
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
   - at least `min_defect_frames` (3) **consecutive** observations scored
     `P(dirt) >= frame_dirt_threshold` (0.50) — the v6 hallucination guard —
   - AND the **trimmed-mean** `P(dirt)` over the whole track reaches
     `track_dirt_threshold` (0.45).
   The verdict is not a latch: enough later clean evidence pulls the trimmed
   mean back down. Cross-camera OR fusion in `decision.py` is unchanged —
   dirt visible from one side still rejects the cap (~8% of labeled pairs are
   one-sided).

Threshold provenance: the cap-level sweep in
`detectx/exported_models/two_stage_v1_*/training_summary.json`
(`cap_level.operating_point` / `sweep`). Re-run that notebook's Section 6 and
update `frame_dirt_threshold` / `track_dirt_threshold` whenever either model
is retrained — thresholds are model-specific.

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
| `max_classified_boxes` | — | `2` (stage-2 budget per frame) |
| `db_path` | `...history_v6.sqlite3` | `data/cap_line_history_v7.sqlite3` |

Settings migration: a settings dict **without** `classifier_model` predates
the two-stage pipeline, so its `model`/`imgsz` values are ignored (a 2-class
model silently running as the cap detector would break inspection). Unknown
legacy keys (`reject_threshold`, `global_cooldown_ms`, ...) are dropped as in
v6. `postprocess` still accepts class ids 0 and 1 but normalizes both to
class 0, so even an un-migrated legacy model yields cap boxes rather than
dropping the dirty ones.

## Deliverables (new files only; never modify v1–v6 files)

- `cap_line_v7_PROMPT.md` — this document.
- `cap_line_v7/` — Python package (same module layout as `cap_line_v6/`).
- `cap_line_runtime_v7.py` — headless entry point.
- `cap_line_ui_v7.py` — PyQt6 operator UI (two-stage model fields + the three
  thresholds in the Config tab; sqlite table `cap_line_history_v7`; settings
  persist to `cap_line_ui_v7_settings.json`).
- `cap_detector_640.onnx`, `dirt_classifier_384.onnx` — repo-root model files
  copied from `detectx/exported_models/two_stage_v1_20260715_195231/`.
- `tests/test_cap_line_v7.py` — probability-voting, crop/classify, config
  migration, cross-camera decision, and two-stage runtime wiring tests, plus
  the defaults test (`gpio_backend == "jetson"`, `trigger_pin == 7`,
  `resolve_pin_factory(defaults) is GPIOOutputPin`).

## Acceptance criteria

- `python -m pytest tests/test_cap_line_v7.py` passes (run from the repo
  root), and `tests/test_cap_line_v6.py` still passes untouched.
- `python cap_line_runtime_v7.py --simulate-gpio` runs (or fails with a clear
  camera message) and prints one line per physical cap.
- `python cap_line_ui_v7.py` shows the live preview and the verdict banner;
  preview boxes show the classifier verdict (red `dirt_defect:P(dirt)` /
  green `undefected:P(clean)`).
- Real-model sanity: both ONNX files load through `create_onnx_session`;
  held-out dirty frames score `P(dirt)` near 1.0 and clean frames near 0.0
  through `classify_dirt_probability` (verified at build time, ~30 ms/frame
  CPU for the full two-stage pass).
- A cap dirty on one camera and clean on the other produces exactly one air
  pulse; a clean cap with a single hallucinated 0.97 frame produces none
  (both covered by tests).
- `gpio_output.py`, `rpi_gpio_output.py`, and every v1–v6 file byte-identical
  to before.
