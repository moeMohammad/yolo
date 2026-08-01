# Jetson Orin v7 Operator Runbook

This runbook is for commissioning and operating the two-camera v7 cap line on
a Jetson Orin. Complete the checks in order. Keep the air valve disabled or
use `simulate_gpio` until the camera identity, tracking direction, model
performance, inspection results, and fire delay have all been verified.

## What Is and Is Not Proven

The current model pair is:

- `cap_detector_640.onnx`: single-class cap detector.
- `dirt_classifier_384.onnx`: crop classifier; output index 0 is
  `dirt_defect`.

Offline replay found zero false rejects on 353 known-clean recorded caps after
the v2 classifier retrain. That result is useful, but it is not a production
accuracy guarantee. The recordings under the `dirt` folder did not have
reliable per-cap ground truth: only 25 of 257 folder-labelled events were
flagged, and only a sample of the events that passed was visually reviewed.
The true dirty-cap miss rate remains unknown.

Also treat tipped or upside-down caps, new dirt types, new lighting, and new
cap batches as out of domain until they are tested. The default fail-closed
policy rejects an event that lacks enough classifier evidence from both
cameras as `uninspected`; this prevents a silent clean pass, but it does not
prove dirt recall.

## 1. Verify the Deployed Release

Run all commands from the repository root using the same virtual environment
that will run the service or UI:

```bash
cd /path/to/yolo
source /path/to/venv/bin/activate
```

Stop if the Jetson has uncommitted changes. Do not erase them to make an
update succeed:

```bash
git status --short
git branch --show-current
git rev-parse HEAD
git remote -v
```

Update only when the tree is understood and clean:

```bash
git fetch origin
git pull --ff-only origin main
git rev-parse HEAD
```

Compare the exact commit SHA with the validated workstation or release record;
matching the branch name is not enough. Then verify that both model artifacts
exist and compare their hashes with the same release source:

```bash
ls -lh cap_detector_640.onnx dirt_classifier_384.onnx
sha256sum cap_detector_640.onnx dirt_classifier_384.onnx
```

Verify imports and run the v7 tests:

```bash
python3 - <<'PY'
import cv2
import numpy
import onnxruntime
from PyQt6 import QtCore

print("OpenCV:", cv2.__version__)
print("NumPy:", numpy.__version__)
print("ONNX Runtime:", onnxruntime.__version__)
print("PyQt6: available")
PY

python3 -m pytest tests/test_cap_line_v7.py -q
```

If the UI is not required on a headless installation, PyQt6 may be omitted;
all other imports and the headless runtime are still required. Use an NVIDIA
or JetPack-compatible ONNX Runtime build for acceleration rather than blindly
installing a generic wheel over a working Jetson environment.

## 2. Use the Correct v7 Entry Point

The live entry points are different in an important way:

- `python3 cap_line_ui_v7.py` loads and saves
  `cap_line_ui_v7_settings.json` in the repository root.
- `python3 cap_line_runtime_v7.py` uses v7 defaults plus its command-line
  arguments. It does **not** load the UI JSON.

Do not launch `cap_line_ui_v6.py`, `cap_line_runtime_v6.py`, or an old desktop
or service command. Check any autostart unit explicitly:

```bash
systemctl --user cat cap-line.service 2>/dev/null || true
grep -R "cap_line_.*v[1-6]" \
  ~/.config/autostart ~/.config/systemd/user /etc/systemd/system 2>/dev/null || true
```

The first commissioning run should be from an interactive terminal so startup
provider, camera, filter, and actuator messages remain visible.

## 3. Make Camera Identity Stable

The shipped numeric defaults are camera nodes `0` and `2`, because that was
the observed rig mapping. `/dev/video0` and `/dev/video2` are not stable
physical identities: they can change after a reboot, unplug, USB reset, or
kernel update. One physical camera can also expose more than one video node.

Inventory the devices:

```bash
v4l2-ctl --list-devices
ls -l /dev/v4l/by-id /dev/v4l/by-path 2>/dev/null
```

Prefer `/dev/v4l/by-id/...` when each camera exposes a unique serial number.
For otherwise identical cameras, use `/dev/v4l/by-path/...` and keep each
camera on its assigned USB port. Confirm that the chosen link is the capture
stream supporting the configured 960×600 YUYV mode:

```bash
v4l2-ctl -d /dev/v4l/by-id/CHOSEN_CAMERA --list-formats-ext
```

Cover one lens at a time in the v7 preview and record which path is the first
and second panel. Store the stable paths in the UI's Camera 0 and Camera 1
fields, or pass them explicitly to headless v7:

```bash
export CAP_CAM0=/dev/v4l/by-id/CAMERA_A_VIDEO_INDEX0
export CAP_CAM1=/dev/v4l/by-id/CAMERA_B_VIDEO_INDEX0

python3 cap_line_runtime_v7.py \
  --cams "$CAP_CAM0" "$CAP_CAM1" \
  --simulate-gpio --no-display
```

If by-id links are unavailable, substitute the verified by-path links. Do not
assume that a node is correct merely because OpenCV can open it.

## 4. Inspect and Migrate UI Settings

`cap_line_ui_v7_settings.json` is deployment-local and ignored by Git, so it
survives `git pull`. Current v7 uses a settings schema version and migrates
known old shipped values, including the old `0,3` camera pair, the old shared
`positive` direction, and old throughput defaults. It writes the normalized
settings back atomically and shows a `[SETTINGS]` message when migration
occurs.

Migration deliberately cannot discover physical camera identity or orientation.
In particular, it preserves per-camera mirror choices. Inspect the effective
configuration before Start:

```bash
python3 - <<'PY'
import json
from pathlib import Path
from cap_line_v7.config import RuntimeConfig

path = Path("cap_line_ui_v7_settings.json")
raw = json.loads(path.read_text()) if path.exists() else {}
config = RuntimeConfig.from_json_dict(raw) if raw else RuntimeConfig.defaults()
data = config.to_json_dict()
for key in (
    "settings_schema_version",
    "model",
    "classifier_model",
    "cameras",
    "mirror_cameras",
    "target_fps",
    "classify_band_ratio",
    "track_timeout_ms",
    "max_track_gap_ms",
    "presence_direction",
    "required_inspected_cameras",
    "reject_uninspected",
    "fire_delay_s",
    "simulate_gpio",
):
    print(f"{key}: {data[key]}")
PY
```

Expected model names are the two v7 ONNX files. Camera sources should be the
stable paths established above, target FPS should normally be 30, and real
commissioning must begin with simulated GPIO. Do not copy or rename the v6
settings file as a v7 settings file.

If the provenance of an existing v7 JSON is unknown, archive it rather than
deleting it, then let the UI create a fresh file:

```bash
if [ -f cap_line_ui_v7_settings.json ]; then
  mv cap_line_ui_v7_settings.json \
    "cap_line_ui_v7_settings.json.bak.$(date +%Y%m%d-%H%M%S)"
fi
```

## 5. Calibrate Mirroring and Belt Direction Live

Do not infer raw camera direction from the saved recordings. The recording
script may already have mirrored camera 2, so apparently matching motion in
two MP4 files does not prove the two raw devices have matching orientation.

The runtime has one shared `presence_direction`. Therefore, after each
camera's mirror setting is applied, caps must move in the same processed
direction in both preview panels.

1. Disconnect or disable air, enable Simulate GPIO, and stop the conveyor.
2. In Config, start with both mirror options off and set Belt Direction to
   `either` for diagnosis only.
3. Start v7 and pass one cap through slowly. Observe its processed left/right
   movement separately in both panels.
4. If the panels show opposite directions, stop v7 and enable mirroring for
   exactly one camera. Restart and repeat until processed motion agrees.
5. Set the shared direction to `negative` for right-to-left processed motion
   or `positive` for left-to-right processed motion. Do not leave `either` in
   production; it weakens the wrong-way-motion safety gate.
6. Pass at least ten single, well-spaced caps. `Caps Seen` should increment
   once per physical cap, `Filtered Tracks` should not increase for every
   transit, and `Unknown Inspections` should remain zero.

The current fresh-install defaults are cameras `0,2`, mirrors `false,false`,
direction `negative`, and simulated GPIO enabled. Schema migration also
re-disarms old settings. The geometry values are a starting point for this
rig, not a substitute for the live calibration above.

If boxes follow the cap but `Caps Seen` remains zero, investigate the presence
gate, direction, processed FPS, and `[FILTER]` messages. If there are no boxes,
investigate camera input, detector model/provider, exposure, and focus first.

## 6. Verify Providers, FPS, and Latency With Air Disabled

Check what ONNX Runtime can offer before starting v7:

```bash
python3 - <<'PY'
import onnxruntime as ort
print("Available providers:", ort.get_available_providers())
PY
```

The runtime requests TensorRT, then CUDA, then CPU from the providers actually
available. Its startup line and the UI's Model Providers field report the
selected first provider for every detector and classifier session. A provider
name alone is not proof of sufficient throughput; use the live counters.

With Simulate GPIO enabled, run for several minutes and feed representative
caps. Interpret the UI fields as follows:

- **Capture FPS:** both cameras should be near their negotiated rate, normally
  about 30 FPS. A low or oscillating value points to USB bandwidth, power,
  permissions, pixel format, or camera-driver trouble.
- **Processed FPS:** should be high enough to observe every cap on both sides
  of the presence line and classify it repeatedly. At 30 FPS, 33 ms is one
  frame period; sustained inference above that means a worker cannot process
  every captured frame. Real-footage stride replay degraded sharply below
  7.5 processed FPS per camera, so treat 7.5 as a minimum operational floor
  and do not arm air while the UI reports `unsafe_low_fps`.
- **Throughput Status:** becomes `qualified` only after warm-up when both
  cameras meet that replay-derived floor. It checks observation rate only; it
  does not certify model accuracy, camera geometry, or nozzle timing.
- **Inference ms:** compare both cameras and watch for thermal degradation.
  A large camera-to-camera difference suggests mapping, format, or device
  trouble rather than a threshold problem.
- **Stale Results:** should remain zero. A rising count means inference is
  finishing on frames too old to use safely.
- **Filtered Tracks:** repeated increments while valid caps pass indicate
  direction, gate crossing, travel, gaps, or insufficient processed FPS.
- **Unknown Inspections:** should remain zero. With fail-closed inspection, an
  unknown event is rejected as `uninspected`; it is not evidence of dirt.
- **Runtime Log:** treat `[CAMERA][FATAL]` and `[REJECT][FATAL]` as stop-line
  failures. Investigate `[CAMERA][WARN]`, `[FILTER]`, and `[REJECT][STALE]`
  rather than tuning them away.

For a terminal-only performance readout, use this diagnostic and stop it with
Ctrl-C:

```bash
CAP_CAM0="$CAP_CAM0" CAP_CAM1="$CAP_CAM1" python3 - <<'PY'
import os
import threading
from dataclasses import replace

from cap_line_v7.config import RuntimeConfig
from cap_line_v7.runtime import run_detection
from cap_line_v7.types import RuntimeCallbacks

config = replace(
    RuntimeConfig.defaults(),
    cameras=(os.environ["CAP_CAM0"], os.environ["CAP_CAM1"]),
    simulate_gpio=True,
    no_display=True,
    live_preview_fps=0.0,
)
stop = threading.Event()

def report(snapshot):
    print(
        "capture=", snapshot.capture_fps_by_camera,
        "processed=", snapshot.processed_fps_by_camera,
        "inference_ms=", snapshot.inference_ms_by_camera,
        "stale=", snapshot.stale_results_by_camera,
        "detector=", snapshot.detector_providers,
        "classifier=", snapshot.classifier_providers,
        flush=True,
    )

try:
    run_detection(
        config,
        RuntimeCallbacks(performance_callback=report, log_fn=print),
        stop,
    )
except KeyboardInterrupt:
    stop.set()
PY
```

If the models use CPU and cannot keep up, fix the JetPack-compatible ONNX
Runtime/provider installation before changing model thresholds or track
qualification. Check power and thermal state with `tegrastats`. If inference
is healthy but only the UI feels slow, reduce Live Preview FPS to 10 or 5 and
compare again; headless v7 avoids preview composition entirely.

## 7. Establish Honest Model Recall

Do not enable unattended rejection based only on folder names or aggregate
offline claims. Create a per-cap qualification run:

1. Give every physical cap an ID and record its ground truth before it enters
   the conveyor.
2. Include verified-clean caps and verified-dirty caps with small, large,
   light, dark, one-sided, and varied-position defects.
3. Run one cap at a time through both camera views. Record the cap-log result,
   confidence, inspection status, cameras, and air status.
4. Repeat across representative belt speed, lighting, focus, cap batches, and
   both camera positions.
5. Report clean false-reject rate and dirty miss rate per physical cap. Keep a
   separate count of `uninspected` fail-closed rejects.
6. Review every miss and false reject visually before changing
   `frame_dirt_threshold` or `track_dirt_threshold`.

Thresholds are model-specific. Retraining either ONNX model requires a new
held-out, per-cap threshold sweep. Lowering a threshold may improve recall at
the cost of ejecting clean caps; it cannot recover dirt that is invisible due
to blur, glare, focus, or lighting.

## 8. Calibrate Presence-Gate-to-Nozzle Fire Delay

`fire_delay_s` is not an inference delay and is not measured from when a track
finishes. For a confirmed reject, v7 uses the interpolated time at which the
cap center crossed the configured presence line, then requests GPIO ON at:

```text
requested fire time = presence-gate crossing time + fire_delay_s
```

The default `fire_delay_s=0.0` is uncalibrated and is normally wrong when the
nozzle is downstream. Fresh and migrated configurations therefore default to
simulated GPIO. Real GPIO is refused unless the configured delay exceeds the
minimum software decision horizon (currently 0.50 s for fail-closed unknowns:
350 ms track timeout + 150 ms merge window). This prevents a mathematically
guaranteed stale command; it does not prove that the delay is long enough for
the actual cap trajectory. Calibrate only after camera mapping, mirroring,
direction, presence-line ratio, belt speed, FPS, and model performance are
stable. Changing any of those can invalidate the result.

### Verify the Orin header before enabling pressure

The default project channel is physical BOARD pin 7. Confirm that this is the
pin actually wired to the valve-driver input; the old project name `GPIO09` is
only a compatibility alias and is not a portable Orin signal name. NVIDIA's
Jetson.GPIO tool accepts BOARD pin numbers and provides a pinmux lookup for
Orin devices:

```bash
jetson-gpio-pinmux-lookup 7
python3 - <<'PY'
import Jetson.GPIO as GPIO
print(GPIO.JETSON_INFO)
GPIO.setmode(GPIO.BOARD)
print("BOARD 7 function:", GPIO.gpio_function(7))
GPIO.cleanup()
PY
```

On Orin Nano, Jetson.GPIO warns when a selected header pin is not correctly
pinmuxed as GPIO/output. If initialization reports that warning or Test Fire
does not switch the driver input, configure the 40-pin header with Jetson-IO,
save, reboot, and recheck; do not bypass the warning. See NVIDIA's
[Jetson.GPIO documentation](https://github.com/NVIDIA/jetson-gpio) and
[expansion-header configuration guide](https://docs.nvidia.com/jetson/archives/r38.4/DeveloperGuide/HR/ConfiguringTheJetsonExpansionHeaders.html).

The Jetson pin must control a correctly rated valve driver/relay interface; it
must not power the solenoid directly. Verify the electrical interface with air
disabled before using the UI's Test Fire.

### Calculate a safe starting value

1. Mark the physical conveyor point corresponding to the configured presence
   line, normally the center line at ratio 0.50. Confirm it in both processed
   previews. If the two projected lines correspond to materially different
   belt positions, correct the camera geometry before continuing; one shared
   delay cannot compensate for a different reference depending on which
   camera sees the dirt.
2. Measure belt travel from that physical gate point to the center of the air
   jet. Do not measure from the camera housing or edge of frame.
3. Measure gate-to-nozzle travel time directly over at least 20 passes at the
   production belt speed. Video with a trustworthy frame rate is preferable
   to a handheld stopwatch. Use the median and record the range.
4. Measure or conservatively estimate the delay from GPIO ON to useful air at
   the nozzle, including valve opening, hose fill, and pressure build-up.
5. Choose where in the pulse the cap should meet useful air. A practical
   starting calculation is:

```text
fire_delay_s = median gate-to-nozzle travel time
               - measured pneumatic response time
               - desired early-air lead
```

Use zero early lead initially if pneumatic response was measured at the point
of useful air. Never enter a negative delay; a negative result means the gate
or nozzle geometry cannot be serviced in time and must be moved.

A fixed delay is valid only for a stable, bounded belt speed. If the observed
travel-time range is wider than the useful air-pulse window, lock the conveyor
to one qualified speed or implement speed-aware timing; do not hide the
variation by averaging it into one unreliable delay.

### Tune empirically

1. Guard the nozzle and clear people and loose material from the area. Use low
   safe pressure and a sacrificial, verified-reject cap.
2. First run with Simulate GPIO and confirm exactly one reject event per cap,
   zero unknown inspections, stable processed FPS, and no `[REJECT][STALE]`.
3. Enter the calculated delay in **Fire Delay from Presence Gate Crossing s**.
   Keep the existing trigger duration unchanged while calibrating delay.
4. Disable simulation and use Test Fire once to verify controlled valve
   operation. While detection is running, Test Fire is routed through the
   runtime scheduler; do not open a second GPIO process on the same pin.
5. Send one verified-reject cap at a time. If the air pulse is before the cap,
   increase `fire_delay_s`; if it is after the cap, decrease it. Adjust in
   50 ms steps until the pulse overlaps the cap, then in 10 ms steps to center
   the useful air window.
6. Repeat at the slowest and fastest allowed belt speeds and at least 20 times
   at nominal speed. Record requested fire, actual fire, outcome, pressure,
   speed, and final setting.
7. Run well-spaced verified-clean caps and adjacent-cap spacing tests to prove
   the 0.3 s pulse does not eject a neighbor.

The Cap Log distinguishes Requested Fire, Actual Fire, and Air Status. A
requested timestamp is not proof that GPIO activated. `fired` with an Actual
Fire timestamp confirms a normally completed distinct activation. An Actual
Fire timestamp paired with `stale_after_on`, `cancelled_after_on`, or
`gpio_failed` means GPIO physically went ON but the pulse was unsafe or failed;
stop and investigate it. `coalesced` means the scheduler considered the target
covered by the preceding pulse, not that a new pulse occurred; verify spacing
and the physical outcome. `stale`,
`gpio_failed`, `cancelled`, or a missing Actual Fire after the target has
settled must be treated as a failed actuation, not a successful reject.

Do not increase Trigger Max Lateness or queue age merely to suppress stale
warnings. A late pulse can eject the following clean cap. If the decision is
not ready before the required command time, reduce processing latency or move
the presence gate farther upstream.

## 9. Production Release Checklist

Do not enable unattended line operation until all items are true:

- Exact code commit and both model hashes match the approved release.
- The service or shortcut launches a v7 entry point.
- Both stable camera paths survive reboot and map to the intended panels.
- Live mirroring is calibrated and both processed views share the configured
  non-`either` direction.
- Provider, capture FPS, processed FPS, inference latency, stale results,
  filtered tracks, and unknown inspections are acceptable under load.
- A per-cap ground-truth run establishes both clean false-reject and dirty miss
  rates for the current lighting, focus, belt, and caps.
- Fire delay is calibrated from the physical presence gate to the nozzle and
  verified at the allowed belt-speed limits.
- Test Fire works, actual reject pulses are recorded as `fired`, and no stale,
  failed, or unexplained duplicate pulses remain.
- Operators know that Stop is a safety boundary: pending actuator jobs are
  cancelled rather than fired during shutdown.

After any camera move, USB-port change, lens/focus adjustment, lighting
change, belt-speed change, nozzle move, pressure change, model replacement, or
JetPack/ONNX Runtime upgrade, repeat the affected commissioning sections.
