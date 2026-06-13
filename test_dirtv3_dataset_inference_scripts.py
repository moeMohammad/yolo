from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent
SCRIPT_DIR = REPO_ROOT / "script"


def load_script_module(module_name: str):
    module_path = SCRIPT_DIR / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module spec from {module_path}")

    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_fake_cv2():
    def cvt_color(image, code):
        if code == fake_cv2.COLOR_BGR2GRAY:
            return image.mean(axis=2).astype(image.dtype)
        if code == fake_cv2.COLOR_GRAY2BGR:
            return np.repeat(image[:, :, np.newaxis], 3, axis=2)
        raise ValueError(f"Unsupported cvtColor code: {code}")

    fake_cv2 = SimpleNamespace(
        COLOR_BGR2GRAY=1,
        COLOR_GRAY2BGR=2,
        cvtColor=cvt_color,
    )
    return fake_cv2


class DirtV3DatasetInferenceScriptTests(unittest.TestCase):
    def test_rgb_script_defaults_to_raw_dataset_dirtv3_and_rgb_result_dir(self) -> None:
        module = load_script_module("infer_raw_dataset_dirtv3_rgb")

        args = module.parse_args([])

        self.assertEqual(REPO_ROOT / "raw_dataset", args.input_dir)
        self.assertEqual(REPO_ROOT / "model" / "dirtv3.onnx", args.model)
        self.assertEqual(REPO_ROOT / "rgb_model_result", args.output_dir)
        self.assertFalse(args.grayscale)

    def test_grey_script_defaults_to_raw_dataset_grey_model_and_grey_result_dir(self) -> None:
        module = load_script_module("infer_raw_dataset_dirtv3_grey")

        args = module.parse_args([])

        self.assertEqual(REPO_ROOT / "raw_dataset", args.input_dir)
        self.assertEqual(REPO_ROOT / "model" / "dirtv3_grey.onnx", args.model)
        self.assertEqual(REPO_ROOT / "grey_model_result", args.output_dir)
        self.assertTrue(args.grayscale)

    def test_grayscale_preparation_feeds_three_equal_channels(self) -> None:
        module = load_script_module("dirtv3_dataset_inference")
        image = np.array(
            [
                [[0, 30, 60], [90, 120, 150]],
                [[180, 210, 240], [30, 60, 90]],
            ],
            dtype=np.uint8,
        )

        prepared = module.prepare_image_for_model(
            image,
            grayscale=True,
            cv2_module=make_fake_cv2(),
        )

        self.assertEqual(image.shape, prepared.shape)
        self.assertTrue(np.array_equal(prepared[:, :, 0], prepared[:, :, 1]))
        self.assertTrue(np.array_equal(prepared[:, :, 1], prepared[:, :, 2]))


if __name__ == "__main__":
    unittest.main()
