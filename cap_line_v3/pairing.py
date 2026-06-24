from __future__ import annotations

from .types import CaptureBatch, CapturedFrame, FramePair, PairDropStats


def select_synchronized_frame_pair(
    latest_frames: tuple[CapturedFrame | None, ...],
    last_sequences: tuple[int, ...] | None,
    *,
    max_skew_ms: float,
) -> FramePair | None:
    if not latest_frames or any(frame is None for frame in latest_frames):
        return None
    frames = tuple(frame for frame in latest_frames if frame is not None)
    sequences = tuple(int(frame.sequence) for frame in frames)
    if last_sequences is not None:
        if len(sequences) == len(last_sequences) and any(
            sequence <= previous for sequence, previous in zip(sequences, last_sequences)
        ):
            return None
    timestamps = tuple(float(frame.timestamp) for frame in frames)
    skew_ms = (max(timestamps) - min(timestamps)) * 1000.0
    if skew_ms > float(max_skew_ms):
        return None
    return FramePair(frames=frames, pair_timestamp=max(timestamps), skew_ms=skew_ms)


def default_single_camera_wait_ms(target_fps: int | float, pair_max_skew_ms: float) -> float:
    frame_interval_ms = 1000.0 / max(1.0, float(target_fps))
    return min(float(pair_max_skew_ms), 2.0 * frame_interval_ms)


def select_capture_batch(
    pending_frames_by_camera: tuple[tuple[CapturedFrame, ...], ...],
    *,
    now: float,
    max_skew_ms: float,
    single_camera_wait_ms: float,
    stats: PairDropStats | None = None,
) -> CaptureBatch | None:
    camera_count = len(pending_frames_by_camera)
    if camera_count <= 0:
        return None

    first_pending = tuple(frames[0] if frames else None for frames in pending_frames_by_camera)
    present = tuple(frame for frame in first_pending if frame is not None)
    if not present:
        return None

    missing = tuple(index for index, frame in enumerate(first_pending) if frame is None)
    if not missing:
        timestamps = tuple(float(frame.timestamp) for frame in present)
        skew_ms = (max(timestamps) - min(timestamps)) * 1000.0
        if skew_ms <= float(max_skew_ms):
            return CaptureBatch(
                frames_by_camera=first_pending,
                batch_timestamp=max(timestamps),
                skew_ms=skew_ms,
                reason="paired",
                missing_camera_indices=(),
            )
        skew_blocked_pair = True
    else:
        skew_blocked_pair = False

    oldest = min(present, key=lambda frame: float(frame.timestamp))
    oldest_age_ms = (float(now) - float(oldest.timestamp)) * 1000.0
    if oldest_age_ms < float(single_camera_wait_ms):
        return None

    if stats is not None:
        stats.missing_camera += 1
        if skew_blocked_pair:
            stats.skew += 1
    frames_by_camera = tuple(
        oldest if index == int(oldest.camera_index) else None
        for index in range(camera_count)
    )
    missing_single = tuple(index for index in range(camera_count) if index != int(oldest.camera_index))
    return CaptureBatch(
        frames_by_camera=frames_by_camera,
        batch_timestamp=float(oldest.timestamp),
        skew_ms=None,
        reason="single_camera_wait",
        missing_camera_indices=missing_single,
    )
