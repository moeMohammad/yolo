# Cap Line Inspection

This repository contains several generations of the conveyor cap-inspection
system. The current live system is **v7**, a two-stage pipeline that detects a
cap, classifies its crop as `undefected` or `dirt_defect`, tracks the cap across
both cameras, and schedules the reject nozzle.

Use only these entry points for the current system:

- `python3 cap_line_ui_v7.py` — operator UI. Its settings are stored locally in
  `cap_line_ui_v7_settings.json`.
- `python3 cap_line_runtime_v7.py` — headless runtime. It uses command-line
  options and does not load the UI settings file.

The required v7 models are `cap_detector_640.onnx` and
`dirt_classifier_384.onnx` in the repository root. Files named v1–v6 and
`dirtv7.onnx` are legacy and must not be used as the v7 live entry point or
v7 model pair.

Before running on the conveyor, follow the
[Jetson Orin operator runbook](JETSON_ORIN_RUNBOOK.md). It covers exact
deployment verification, stable camera mapping, settings migration, live
direction calibration, provider/FPS diagnosis, model limitations, and safe
camera-gate-to-nozzle timing calibration. Fresh and migrated v7 settings keep
GPIO simulated until that commissioning is completed.

The implementation contract and model provenance are documented in
[`cap_line_v7_PROMPT.md`](cap_line_v7_PROMPT.md). Offline video tools such as
`validate_v7_on_videos.py` are validation utilities; they do not deploy or
start the live Jetson runtime.
