"""ONNX YOLO model I/O for v4.

The inference path (model-path resolution, ONNX session with CUDA->CPU provider
fallback, letterbox preprocess, and the YOLO output decode) is copied and
stripped down from the v3 runtime. Heavy deps (numpy / cv2 / onnxruntime) are
imported lazily inside the functions so importing this module is cheap and so the
pure-Python pieces stay testable without those packages.
"""

from __future__ import annotations

import os
from pathlib import Path

from .config import DEFAULT_MODEL


# v4 lives at the repo root next to the .onnx files; also look in an optional model/ dir.
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


def postprocess(output, preprocess_meta, conf_threshold: float):
    """Decode the model output into pixel-space boxes, filtering by confidence.

    This is where the ``reject_threshold`` filter lives: detections below the
    threshold are dropped here and never reach the tracker.
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
        boxes.append([x1, y1, x2, y2, score, class_id])
    boxes.sort(key=lambda box: float(box[4]), reverse=True)
    return boxes
