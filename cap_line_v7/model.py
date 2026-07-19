"""ONNX model I/O for the v7 two-stage pipeline.

Stage 1 keeps the v6 detector path (model-path resolution, ONNX session with
CUDA->CPU provider fallback, letterbox preprocess, end-to-end YOLO decode) but
the detector is now a single-class *cap* model. Stage 2 adds the crop
classifier: ``crop_cap_region`` reproduces the training-crop geometry
(margin + square pad) exactly, ``classifier_preprocess`` is plain RGB/255 at
the classifier input size (no ImageNet normalization — verified against the
exported model), and ``classifier_postprocess`` returns ``P(dirt)`` from
output index ``CLASSIFIER_DIRT_INDEX``, applying softmax only when the export
did not already bake it in.

Heavy deps (numpy / cv2 / onnxruntime) are imported lazily inside the
functions so importing this module is cheap and so the pure-Python pieces stay
testable without those packages.
"""

from __future__ import annotations

import os
from pathlib import Path

from .config import CLASSIFIER_DIRT_INDEX, DEFAULT_MODEL


# v7 lives at the repo root next to the .onnx files; also look in an optional model/ dir.
MODEL_SEARCH_DIRS = (
    Path(__file__).resolve().parent.parent,
    Path(__file__).resolve().parent.parent / "model",
)


def infer_model_imgsz_from_name(path: str) -> int | None:
    """Pull a square input size out of a filename like ``best_640.onnx``."""

    name = Path(path).name
    for token in name.replace("-", "_").split("_"):
        if token.isdigit():
            value = int(token)
            if 128 <= value <= 4096:
                return value
    return None


def resolve_model_path(model: str) -> tuple[str, int | None]:
    requested = str(model or DEFAULT_MODEL)
    path = Path(os.path.expanduser(requested))
    if path.is_absolute() or path.parent != Path("."):
        candidates = [path, Path(__file__).resolve().parent.parent / path]
    else:
        candidates = [directory / path for directory in MODEL_SEARCH_DIRS]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate), infer_model_imgsz_from_name(str(candidate))
    return str(candidates[0]), infer_model_imgsz_from_name(str(candidates[0]))


def create_onnx_session(model_path: str, intra_op_threads: int):
    import onnxruntime as ort

    options = ort.SessionOptions() if hasattr(ort, "SessionOptions") else None
    if options is not None:
        options.intra_op_num_threads = max(1, int(intra_op_threads))
        options.inter_op_num_threads = 1
    available = list(getattr(ort, "get_available_providers", lambda: [])())
    providers = [
        provider
        for provider in ("TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider")
        if provider in available
    ] or ["CPUExecutionProvider"]
    if options is None:
        return ort.InferenceSession(model_path, providers=providers)
    return ort.InferenceSession(model_path, sess_options=options, providers=providers)


def resolve_imgsz(input_meta, override: int | None, preset: int | None) -> int:
    """Auto-detect the square model input size, with an optional override."""

    if override:
        return int(override)
    shape = list(getattr(input_meta, "shape", []) or [])
    for value in reversed(shape):
        if isinstance(value, int) and value > 0:
            return int(value)
    return int(preset or 640)


def letterbox_resize(image_bgr, new_shape: tuple[int, int] = (640, 640), color=(114, 114, 114)):
    import cv2

    original_height, original_width = image_bgr.shape[:2]
    scale = min(new_shape[0] / original_height, new_shape[1] / original_width)
    resized_width = int(round(original_width * scale))
    resized_height = int(round(original_height * scale))
    resized = cv2.resize(image_bgr, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    pad_width = new_shape[1] - resized_width
    pad_height = new_shape[0] - resized_height
    pad_left = int(round(pad_width / 2 - 0.1))
    pad_right = int(round(pad_width / 2 + 0.1))
    pad_top = int(round(pad_height / 2 - 0.1))
    pad_bottom = int(round(pad_height / 2 + 0.1))
    padded = cv2.copyMakeBorder(
        resized, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=color
    )
    return padded, float(scale), (pad_left, pad_top)


def preprocess(frame, model_imgsz: int):
    import cv2
    import numpy as np

    image, resize_scale, padding = letterbox_resize(frame, new_shape=(int(model_imgsz), int(model_imgsz)))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = image.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    return tensor, {
        "scale": float(resize_scale),
        "pad_left": int(padding[0]),
        "pad_top": int(padding[1]),
        "frame_shape": frame.shape,
        "img_size": int(model_imgsz),
    }


def _box_iou(box_a, box_b) -> float:
    ax1, ay1, ax2, ay2 = (float(value) for value in box_a[:4])
    bx1, by1, bx2, by2 = (float(value) for value in box_b[:4])
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = inter_w * inter_h
    if intersection <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return 0.0 if union <= 0.0 else intersection / union


def deduplicate_boxes(boxes, *, iou_threshold: float = 0.65):
    """Collapse overlapping end-to-end output rows into one observation.

    The YOLO26 end-to-end export emits a 300-row output with ``nms: False``.
    Letting overlapping rows through creates parallel tracks for one visible
    object. Group class-agnostically, retain the highest-confidence row per
    group, and preserve any extra per-row fields beyond the standard six (test
    doubles use them to script classifier scores).
    """

    groups: list[list[list[float]]] = []
    for raw_box in boxes:
        candidate = (
            [float(value) for value in raw_box[:5]]
            + [int(raw_box[5])]
            + [float(value) for value in raw_box[6:]]
        )
        for group in groups:
            if any(_box_iou(candidate, existing) >= float(iou_threshold) for existing in group):
                group.append(candidate)
                break
        else:
            groups.append([candidate])
    selected = [max(group, key=lambda box: float(box[4])) for group in groups]
    selected.sort(key=lambda box: float(box[4]), reverse=True)
    return selected


def postprocess(output, preprocess_meta, conf_threshold: float):
    """Decode the detector output into pixel-space cap boxes.

    This is where the ``detect_threshold`` filter lives: detections below the
    threshold are dropped here and never reach the tracker. The v7 detector is
    single-class, so every surviving row is normalized to class 0 (``cap``);
    class ids 0 and 1 are both accepted so a legacy 2-class model pointed at
    v7 still yields cap boxes instead of silently dropping the dirty ones.
    """

    import numpy as np

    detections = np.asarray(output, dtype=np.float32)
    if detections.ndim == 3 and detections.shape[0] == 1:
        detections = detections[0]
    if detections.ndim != 2:
        return []
    if detections.shape[1] != 6 and detections.shape[0] == 6:
        detections = detections.T
    if detections.shape[1] != 6:
        return []

    scale = float(preprocess_meta["scale"])
    pad_left = float(preprocess_meta["pad_left"])
    pad_top = float(preprocess_meta["pad_top"])
    frame_h, frame_w = preprocess_meta["frame_shape"][:2]
    img_size = int(preprocess_meta["img_size"])

    boxes = []
    for detection in detections:
        x1, y1, x2, y2, score, class_id_value = detection[:6]
        score = float(score)
        if score < float(conf_threshold):
            continue
        coords = np.asarray([x1, y1, x2, y2], dtype=np.float32)
        if float(np.max(np.abs(coords))) <= 1.5:  # normalized output -> scale up to letterbox px
            coords[[0, 2]] *= img_size
            coords[[1, 3]] *= img_size
        x1, y1, x2, y2 = coords.tolist()
        x1 = (x1 - pad_left) / scale
        y1 = (y1 - pad_top) / scale
        x2 = (x2 - pad_left) / scale
        y2 = (y2 - pad_top) / scale
        x1 = max(0.0, min(float(frame_w) - 1.0, x1))
        y1 = max(0.0, min(float(frame_h) - 1.0, y1))
        x2 = max(0.0, min(float(frame_w) - 1.0, x2))
        y2 = max(0.0, min(float(frame_h) - 1.0, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        class_id = int(round(float(class_id_value)))
        if class_id not in (0, 1):
            continue
        boxes.append([x1, y1, x2, y2, score, 0])
    boxes.sort(key=lambda box: float(box[4]), reverse=True)
    return boxes


# --------------------------------------------------------------------------- #
# Stage 2: crop classifier.
# --------------------------------------------------------------------------- #

def box_in_classify_band(box, frame_size, *, axis: str = "x", band_ratio: float = 0.60) -> bool:
    """Whether the box center sits in the central band of the frame.

    The classifier was trained on center-frame captures; near the frame edges
    the entry/exit perspective is out-of-domain and produces spurious high
    P(dirt). Frames outside the band are not classified (no dirt evidence).
    """

    width, height = frame_size
    dimension = float(height if str(axis).lower() == "y" else width)
    x1, y1, x2, y2 = (float(value) for value in box[:4])
    center = (y1 + y2) / 2.0 if str(axis).lower() == "y" else (x1 + x2) / 2.0
    half_band = dimension * float(band_ratio) / 2.0
    return abs(center - dimension / 2.0) <= half_band


def crop_cap_region(frame, box, *, margin: float = 0.10, pad_color=(114, 114, 114)):
    """Crop the cap box (+margin) from the original frame and square-pad it.

    Must stay byte-compatible with ``build_two_stage_dataset.py`` /
    Section 6 of the training notebook: the classifier was trained and
    threshold-calibrated on exactly this geometry.
    """

    import cv2

    frame_h, frame_w = frame.shape[:2]
    x1, y1, x2, y2 = (float(value) for value in box[:4])
    margin_x = (x2 - x1) * float(margin) / 2.0
    margin_y = (y2 - y1) * float(margin) / 2.0
    x1 = max(0, int(round(x1 - margin_x)))
    y1 = max(0, int(round(y1 - margin_y)))
    x2 = min(frame_w, int(round(x2 + margin_x)))
    y2 = min(frame_h, int(round(y2 + margin_y)))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    crop = frame[y1:y2, x1:x2]
    height, width = crop.shape[:2]
    side = max(height, width)
    top = (side - height) // 2
    left = (side - width) // 2
    return cv2.copyMakeBorder(
        crop, top, side - height - top, left, side - width - left,
        cv2.BORDER_CONSTANT, value=pad_color,
    )


def classifier_preprocess(crop_bgr, classifier_imgsz: int):
    """Square crop -> (1, 3, s, s) float32 RGB in [0, 1] (no ImageNet norm)."""

    import cv2
    import numpy as np

    size = int(classifier_imgsz)
    resized = cv2.resize(crop_bgr, (size, size), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return rgb.transpose(2, 0, 1)[None].astype(np.float32) / 255.0


def classifier_postprocess(output, *, dirt_index: int = CLASSIFIER_DIRT_INDEX) -> float:
    """Extract ``P(dirt)`` from the classifier output.

    The current export bakes softmax in; apply it defensively only when the
    scores do not already behave like a probability distribution.
    """

    import numpy as np

    scores = np.asarray(output, dtype=np.float32).reshape(-1)
    if scores.size < 2:
        raise ValueError(f"Classifier output has {scores.size} values; expected at least 2")
    if not (abs(float(scores.sum()) - 1.0) <= 1e-3 and float(scores.min()) >= 0.0):
        exp = np.exp(scores - float(scores.max()))
        scores = exp / float(exp.sum())
    return float(scores[int(dirt_index)])


def classify_dirt_probability(
    session,
    input_name: str,
    frame,
    box,
    *,
    classifier_imgsz: int,
    crop_margin: float = 0.10,
) -> float | None:
    """Run the crop classifier for one detected cap box; None if uncroppable."""

    crop = crop_cap_region(frame, box, margin=crop_margin)
    if crop is None:
        return None
    tensor = classifier_preprocess(crop, classifier_imgsz)
    output = session.run(None, {input_name: tensor})[0]
    return classifier_postprocess(output)
