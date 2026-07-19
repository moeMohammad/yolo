#!/usr/bin/env python3
"""Harvest per-cap transits from recorded conveyor videos.

For every video: run the v7 detector on each frame, group consecutive
detections into transits (one physical cap crossing the view), score each
frame's cap crop with the dirt classifier, and write:

- ``<out>/transits.json`` — per transit: video, camera, frame span, per-frame
  P(dirt) list (threshold calibration + error analysis input);
- ``<out>/crops/<truth>/<recording>_<camera>/t<NNN>_f<FFFF>_p<PPP>.jpg`` —
  the classifier-geometry crops (retraining + visual inspection input).

Crops are saved every ``--crop-stride`` detected frames to bound volume.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2

from cap_line_v7.config import RuntimeConfig
from cap_line_v7.model import (
    classify_dirt_probability,
    crop_cap_region,
    create_onnx_session,
    deduplicate_boxes,
    postprocess,
    preprocess,
    resolve_imgsz,
    resolve_model_path,
)

TRUE_FPS = 30.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--videos-root", type=Path, default=Path("recorded_videos"))
    parser.add_argument("--out", type=Path, default=Path("video_harvest"))
    parser.add_argument("--crop-stride", type=int, default=2, help="save every Nth detected frame's crop")
    parser.add_argument("--gap-frames", type=int, default=5, help="undetected frames that end a transit")
    args = parser.parse_args()

    config = RuntimeConfig.defaults()
    det_path, det_preset = resolve_model_path("cap_detector_640.onnx")
    cls_path, cls_preset = resolve_model_path("dirt_classifier_384.onnx")
    det = create_onnx_session(det_path, 4)
    cls = create_onnx_session(cls_path, 4)
    det_name = det.get_inputs()[0].name
    cls_name = cls.get_inputs()[0].name
    det_imgsz = resolve_imgsz(det.get_inputs()[0], None, det_preset)
    cls_imgsz = resolve_imgsz(cls.get_inputs()[0], None, cls_preset)

    transits = []
    for truth in ("clean", "dirt"):
        for video_path in sorted((args.videos_root / truth).glob("*.mp4")):
            recording = video_path.stem.replace("_recording", "")
            camera = "cam0" if video_path.stem.endswith("cam_0") else "cam2"
            crop_dir = args.out / "crops" / truth / f"{recording}"
            crop_dir.mkdir(parents=True, exist_ok=True)

            capture = cv2.VideoCapture(str(video_path))
            current = None
            transit_index = 0
            frame_index = -1
            started = time.perf_counter()

            def finish(transit):
                if transit is not None and len(transit["p_dirt"]) >= 2:
                    transits.append(transit)

            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                frame_index += 1
                tensor, meta = preprocess(frame, det_imgsz)
                boxes = deduplicate_boxes(
                    postprocess(det.run(None, {det_name: tensor})[0], meta, config.detect_threshold),
                    iou_threshold=config.duplicate_iou_threshold,
                )
                if not boxes:
                    if current is not None and frame_index - current["last_frame"] > args.gap_frames:
                        finish(current)
                        current = None
                    continue
                box = boxes[0]
                p_dirt = classify_dirt_probability(
                    cls, cls_name, frame, box, classifier_imgsz=cls_imgsz, crop_margin=config.crop_margin
                )
                if p_dirt is None:
                    continue
                if current is None or frame_index - current["last_frame"] > args.gap_frames:
                    finish(current)
                    transit_index += 1
                    current = {
                        "truth": truth,
                        "recording": recording,
                        "camera": camera,
                        "transit": transit_index,
                        "first_frame": frame_index,
                        "last_frame": frame_index,
                        "p_dirt": [],
                        "det_conf": [],
                    }
                current["last_frame"] = frame_index
                current["p_dirt"].append(round(float(p_dirt), 4))
                current["det_conf"].append(round(float(box[4]), 3))
                if len(current["p_dirt"]) % args.crop_stride == 1:
                    crop = crop_cap_region(frame, box, margin=config.crop_margin)
                    if crop is not None:
                        crop_name = f"t{transit_index:03d}_f{frame_index:05d}_p{int(round(p_dirt * 100)):03d}.jpg"
                        cv2.imwrite(str(crop_dir / crop_name), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            finish(current)
            capture.release()
            elapsed = time.perf_counter() - started
            count = sum(1 for t in transits if t["recording"] == recording and t["camera"] == camera)
            print(f"[{truth}] {video_path.name}: {count} transits ({elapsed:.0f}s)", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "transits.json").write_text(json.dumps(transits, indent=1), encoding="utf-8")
    print(f"\n{len(transits)} transits -> {args.out / 'transits.json'}")


if __name__ == "__main__":
    main()
