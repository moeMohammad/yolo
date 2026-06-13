from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent


def load_module(module_name: str):
    module_path = REPO_ROOT / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module spec from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_fake_cv2():
    def resize(image, size, interpolation=None):
        width, height = size
        if image.ndim == 3:
            return np.zeros((height, width, image.shape[2]), dtype=image.dtype)
        return np.zeros((height, width), dtype=image.dtype)

    def copy_make_border(image, top, bottom, left, right, border_type, value=(0, 0, 0)):
        pad_value = value[0] if isinstance(value, tuple) else value
        pad_config = (
            ((top, bottom), (left, right), (0, 0))
            if image.ndim == 3
            else ((top, bottom), (left, right))
        )
        return np.pad(image, pad_config, mode="constant", constant_values=pad_value)

    def cvt_color(image, code):
        if code == fake_cv2.COLOR_BGR2GRAY:
            return image.mean(axis=2).astype(image.dtype)
        raise ValueError(f"Unsupported cvtColor code: {code}")

    fake_cv2 = SimpleNamespace(
        COLOR_BGR2GRAY=1,
        INTER_LINEAR=2,
        BORDER_CONSTANT=3,
        resize=resize,
        copyMakeBorder=copy_make_border,
        cvtColor=cvt_color,
    )
    return fake_cv2


class CapLineV2GreyTests(unittest.TestCase):
    def test_grey_runtime_defaults_to_grey_model_and_picture_dir(self) -> None:
        module = load_module("cap_line_runtime_v2_grey")

        self.assertEqual("dirtv3_grey.onnx", module.DEFAULT_MODEL)
        self.assertEqual("dirtv3_grey.onnx", module.parse_args([]).model)
        self.assertTrue(
            module.DEFAULT_PICTURES_DIR.replace("\\", "/").endswith(
                "resources/pictures_grey"
            )
        )

    def test_grey_ui_args_inherit_grey_runtime_defaults(self) -> None:
        runtime_module = load_module("cap_line_runtime_v2_grey")
        ui_module = load_module("cap_line_ui_v2_grey")

        args = ui_module.create_gui_args()

        self.assertEqual(runtime_module.DEFAULT_MODEL, args.model)
        self.assertEqual(runtime_module.DEFAULT_PICTURES_DIR, args.pictures_dir)

    def test_requested_cape_ui_filename_is_available(self) -> None:
        ui_module = load_module("cape_line_ui_v2_grey")

        self.assertEqual("dirtv3_grey.onnx", ui_module.DEFAULT_MODEL)

    def test_grey_preprocess_feeds_three_equal_channels(self) -> None:
        module = load_module("cap_line_runtime_v2_grey")
        frame = np.array(
            [
                [[0, 30, 60], [90, 120, 150]],
                [[180, 210, 240], [30, 60, 90]],
            ],
            dtype=np.uint8,
        )

        with patch.dict(sys.modules, {"cv2": make_fake_cv2()}):
            tensor, meta = module.preprocess(frame, img_size=8)

        self.assertEqual((1, 3, 8, 8), tensor.shape)
        self.assertEqual(np.float32, tensor.dtype)
        self.assertTrue(np.array_equal(tensor[0, 0], tensor[0, 1]))
        self.assertTrue(np.array_equal(tensor[0, 1], tensor[0, 2]))
        self.assertEqual(frame.shape, meta["frame_shape"])


if __name__ == "__main__":
    unittest.main()
