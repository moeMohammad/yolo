from __future__ import annotations

from .types import CapturedFrame, FramePair


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
