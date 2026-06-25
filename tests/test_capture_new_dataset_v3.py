from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cap_line_v3.config import RuntimeConfig


class CaptureNewDatasetV3Tests(unittest.TestCase):
    def test_cli_defaults_match_v3_camera_settings(self):
        import capture_new_dataset_v3 as capture

        args = capture.parse_args([])
        defaults = RuntimeConfig.defaults()

        self.assertEqual(args.cams, list(defaults.cameras))
        self.assertEqual(args.res, list(defaults.resolution))
        self.assertEqual(args.target_fps, defaults.target_fps)
        self.assertEqual(args.pixel_format, defaults.pixel_format)
        self.assertEqual(args.exposure, defaults.exposure)
        self.assertEqual(
            (args.mirror_camera_0, args.mirror_camera_1),
            defaults.mirror_cameras,
        )
        self.assertEqual(Path(args.output_dir).name, "new_dataset")

    def test_save_frames_writes_paired_camera_images(self):
        import capture_new_dataset_v3 as capture

        class FakeCV2:
            IMWRITE_JPEG_QUALITY = 1

            def __init__(self):
                self.writes = []

            def imwrite(self, path, frame, params):
                self.writes.append((Path(path).name, frame, params))
                return True

        fake_cv2 = FakeCV2()
        with tempfile.TemporaryDirectory() as temp_dir:
            saved = capture.save_frames(
                ["left-frame", "right-frame"],
                [0, 3],
                Path(temp_dir),
                "sample",
                12,
                90,
                timestamp="20260625_120102_333",
                cv2_module=fake_cv2,
            )

        self.assertEqual(
            [path.name for path in saved],
            [
                "20260625_120102_333_sample_000012_cam_0.jpg",
                "20260625_120102_333_sample_000012_cam_3.jpg",
            ],
        )
        self.assertEqual(
            fake_cv2.writes,
            [
                (
                    "20260625_120102_333_sample_000012_cam_0.jpg",
                    "left-frame",
                    [1, 90],
                ),
                (
                    "20260625_120102_333_sample_000012_cam_3.jpg",
                    "right-frame",
                    [1, 90],
                ),
            ],
        )


if __name__ == "__main__":
    unittest.main()
