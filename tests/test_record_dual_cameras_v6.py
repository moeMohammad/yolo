from __future__ import annotations

from pathlib import Path

from cap_line_v6.config import RuntimeConfig

import record_dual_cameras_v6 as recorder


def test_cli_defaults_match_v6_camera_settings_with_requested_indices():
    args = recorder.parse_args([])
    defaults = RuntimeConfig.defaults()

    assert args.cams == ["0", "2"]
    assert args.res == list(defaults.resolution)
    assert args.target_fps == defaults.target_fps
    assert args.pixel_format == defaults.pixel_format
    assert args.exposure == defaults.exposure
    assert (args.mirror_camera_0, args.mirror_camera_1) == defaults.mirror_cameras


def test_output_paths_use_one_shared_session_timestamp():
    paths = recorder.build_output_paths(
        Path("videos"),
        "line_test",
        [0, "/dev/video2"],
        timestamp="20260715_101112_345",
    )

    assert paths == [
        Path("videos/20260715_101112_345_line_test_cam_0.mp4"),
        Path("videos/20260715_101112_345_line_test_cam_dev_video2.mp4"),
    ]


def test_create_video_writers_releases_all_writers_if_one_fails():
    class FakeWriter:
        def __init__(self, opened):
            self.opened = opened
            self.released = False

        def isOpened(self):
            return self.opened

        def release(self):
            self.released = True

    class FakeCV2:
        def __init__(self):
            self.writers = []

        @staticmethod
        def VideoWriter_fourcc(*_codec):
            return 1234

        def VideoWriter(self, *_args):
            writer = FakeWriter(opened=not self.writers)
            self.writers.append(writer)
            return writer

    fake_cv2 = FakeCV2()

    try:
        recorder.create_video_writers(
            [Path("camera0.mp4"), Path("camera2.mp4")],
            (960, 600),
            60,
            "mp4v",
            cv2_module=fake_cv2,
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected the second writer to fail")

    assert len(fake_cv2.writers) == 2
    assert all(writer.released for writer in fake_cv2.writers)
