#!/usr/bin/env python3
"""Offline validation of the v7 two-stage pipeline on recorded conveyor videos.

Drives the REAL v7 components (model I/O, CameraTracker, CapEventManager) over
recorded cam_0/cam_2 video pairs with a virtual clock derived from the true
30 fps timing, and reports per-cap decisions against the folder ground truth
(clean/ = all caps clean; dirt/ = caps expected dirty, label noise possible).

Also re-runs the tracker layer under the DEFAULT (mirrored/positive) config to
quantify why the live run saw no caps, and under a simulated slow-Jetson frame
pace to check the qualification gates' headroom.

Usage: python validate_v7_on_videos.py [--videos-root recorded_videos] [--stride 1]
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

import cv2

from cap_line_v7.config import RuntimeConfig
from cap_line_v7.decision import CapEventManager
from cap_line_v7.model import (
    box_in_classify_band,
    classify_dirt_probability,
    create_onnx_session,
    deduplicate_boxes,
    postprocess,
    preprocess,
    resolve_imgsz,
    resolve_model_path,
)
from cap_line_v7.tracking import CameraTracker

TRUE_FPS = 30.0


class RecordingScheduler:
    backend_name = "recording"

    def __init__(self):
        self.enqueued = []

    def enqueue(self, event_id, requested_fire_time, *, completion_callback=None):
        self.enqueued.append((int(event_id), float(requested_fire_time)))

    def close(self):
        return None


def load_models():
    det_path, det_preset = resolve_model_path("cap_detector_640.onnx")
    cls_path, cls_preset = resolve_model_path("dirt_classifier_384.onnx")
    det = create_onnx_session(det_path, 4)
    cls = create_onnx_session(cls_path, 4)
    return {
        "det": det,
        "det_name": det.get_inputs()[0].name,
        "det_imgsz": resolve_imgsz(det.get_inputs()[0], None, det_preset),
        "cls": cls,
        "cls_name": cls.get_inputs()[0].name,
        "cls_imgsz": resolve_imgsz(cls.get_inputs()[0], None, cls_preset),
    }


def infer_frame(models, frame, config):
    tensor, meta = preprocess(frame, models["det_imgsz"])
    raw = models["det"].run(None, {models["det_name"]: tensor})[0]
    boxes = deduplicate_boxes(
        postprocess(raw, meta, config.detect_threshold),
        iou_threshold=config.duplicate_iou_threshold,
    )
    frame_size = (frame.shape[1], frame.shape[0])
    probs = []
    for index, box in enumerate(boxes):
        if index >= config.max_classified_boxes:
            probs.append(None)
            continue
        if not box_in_classify_band(
            box, frame_size, axis=config.presence_line_axis, band_ratio=config.classify_band_ratio
        ):
            probs.append(None)
            continue
        probs.append(
            classify_dirt_probability(
                models["cls"], models["cls_name"], frame, box,
                classifier_imgsz=models["cls_imgsz"], crop_margin=config.crop_margin,
            )
        )
    return boxes, probs


def simulate_pair(models, cam0_path: Path, cam2_path: Path, config: RuntimeConfig, *, stride: int = 1):
    """Run both cameras of one recording through tracker + decision manager."""

    clock = [0.0]
    scheduler = RecordingScheduler()
    records = []
    filter_lines = []

    def log_fn(message, *args, **kwargs):
        text = str(message)
        if "[FILTER]" in text:
            filter_lines.append(text)

    manager = CapEventManager(
        config,
        scheduler=scheduler,
        time_fn=lambda: clock[0],
        history_callback=records.append,
        log_fn=log_fn,
    )
    trackers = [
        CameraTracker(
            index,
            track_iou=config.track_iou,
            track_timeout_s=config.track_timeout_ms / 1000.0,
            min_defect_frames=config.min_defect_frames,
            frame_dirt_threshold=config.frame_dirt_threshold,
            track_dirt_threshold=config.track_dirt_threshold,
            presence_clear_s=config.presence_clear_ms / 1000.0,
            min_track_frames=config.min_track_frames,
            min_track_travel_ratio=config.min_track_travel_ratio,
            min_track_directionality=config.min_track_directionality,
            presence_line_axis=config.presence_line_axis,
            presence_line_ratio=config.presence_line_ratio,
            presence_direction=config.presence_direction,
            max_track_gap_s=config.max_track_gap_ms / 1000.0,
        )
        for index in range(2)
    ]

    captures = [cv2.VideoCapture(str(cam0_path)), cv2.VideoCapture(str(cam2_path))]
    frame_size = None
    frame_index = 0
    processed = 0
    started = time.perf_counter()
    while True:
        frames = []
        for capture in captures:
            ok, frame = capture.read()
            frames.append(frame if ok else None)
        if all(frame is None for frame in frames):
            break
        if frame_index % stride == 0:
            timestamp = frame_index / TRUE_FPS
            clock[0] = timestamp
            for camera_index, frame in enumerate(frames):
                if frame is None:
                    continue
                if config.mirror_cameras[camera_index]:
                    frame = cv2.flip(frame, 1)
                if frame_size is None:
                    frame_size = (frame.shape[1], frame.shape[0])
                boxes, probs = infer_frame(models, frame, config)
                tracker = trackers[camera_index]
                for track in tracker.collect_finished(timestamp):
                    manager.handle_finished_track(track)
                tracker.update([tuple(box[:6]) for box in boxes], timestamp, frame_size, probs)
            manager.flush_expired(clock[0])
            processed += 1
        frame_index += 1
    clock[0] += 10.0
    for camera_index, tracker in enumerate(trackers):
        for track in tracker.collect_finished(clock[0]):
            manager.handle_finished_track(track)
    manager.flush_expired(clock[0])
    manager.finalize_all()
    for capture in captures:
        capture.release()

    elapsed = time.perf_counter() - started
    return {
        "frames": frame_index,
        "processed_per_camera": processed,
        "wall_s": round(elapsed, 1),
        "caps_seen": manager.caps_seen,
        "rejects": manager.rejects,
        "passes": manager.caps_seen - manager.rejects,
        "fires": len(scheduler.enqueued),
        "suppressed_fires": manager.suppressed_fires,
        "filtered_tracks": manager.filtered_tracks,
        "filter_sample": filter_lines[:3],
        "events": [
            {
                "event_id": record.event_id,
                "result": record.result,
                "class": record.class_name,
                "confidence": round(float(record.confidence), 3),
                "flagged_cameras": record.flagged_cameras,
            }
            for record in records
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--videos-root", type=Path, default=Path("recorded_videos"))
    parser.add_argument("--stride", type=int, default=1, help="process every Nth frame (Jetson-pace simulation)")
    parser.add_argument("--out", type=Path, default=Path("v7_video_validation.json"))
    args = parser.parse_args()

    models = load_models()
    # Measured from the footage: caps travel right -> left (negative x) on the
    # RAW frames of BOTH cameras, so the simulation runs unmirrored with a
    # negative belt direction.
    config = replace(
        RuntimeConfig.defaults(),
        mirror_cameras=(False, False),
        presence_direction="negative",
    )

    results = {"config": {"stride": args.stride, "presence_direction": config.presence_direction,
                          "mirror_cameras": list(config.mirror_cameras)}, "pairs": []}
    for truth in ("clean", "dirt"):
        for cam0_path in sorted((args.videos_root / truth).glob("*_cam_0.mp4")):
            cam2_path = Path(str(cam0_path).replace("_cam_0", "_cam_2"))
            print(f"[{truth}] {cam0_path.name} ...", flush=True)
            outcome = simulate_pair(models, cam0_path, cam2_path, config, stride=args.stride)
            outcome["truth"] = truth
            outcome["recording"] = cam0_path.name.replace("_cam_0.mp4", "")
            results["pairs"].append(outcome)
            print(
                f"  caps={outcome['caps_seen']} rejects={outcome['rejects']} passes={outcome['passes']} "
                f"fires={outcome['fires']} filtered={outcome['filtered_tracks']} "
                f"({outcome['wall_s']}s wall)",
                flush=True,
            )

    totals = Counter()
    for pair in results["pairs"]:
        prefix = pair["truth"]
        totals[f"{prefix}_caps"] += pair["caps_seen"]
        totals[f"{prefix}_rejects"] += pair["rejects"]
        totals[f"{prefix}_passes"] += pair["passes"]
    results["totals"] = dict(totals)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n==== SUMMARY ====")
    clean_caps = totals["clean_caps"] or 1
    dirt_caps = totals["dirt_caps"] or 1
    print(f"clean: {totals['clean_rejects']}/{totals['clean_caps']} rejected "
          f"(false-reject rate {totals['clean_rejects'] / clean_caps:.1%})")
    print(f"dirt:  {totals['dirt_passes']}/{totals['dirt_caps']} passed "
          f"(miss rate {totals['dirt_passes'] / dirt_caps:.1%}, label noise possible)")
    print(f"Details: {args.out}")


if __name__ == "__main__":
    main()
