#!/usr/bin/env python3
"""Sweep v7 voting parameters over harvested video transits.

Reads ``video_harvest/transits.json``, replays the v7 track vote per transit
(central-band filter approximated by transit position, consecutive-dirty gate,
trimmed-mean gate), merges cam_0/cam_2 transits of the same physical cap by
frame overlap, and prints the cap-level FRR/miss table across parameter
combinations. Ground truth: clean/ recordings are all clean; dirt/ recordings
are treated as all dirty (label noise possible, so the miss floor may be > 0).
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

from cap_line_v7.tracking import trimmed_mean


def band_slice(p_dirt: list[float], band_ratio: float) -> list[float]:
    n = len(p_dirt)
    keep = max(1, int(round(n * band_ratio)))
    start = (n - keep) // 2
    return p_dirt[start : start + keep]


def track_is_dirty(p_dirt: list[float], *, frame_thr: float, track_thr: float, k: int) -> bool:
    if not p_dirt:
        return False
    consecutive = best = 0
    for probability in p_dirt:
        consecutive = consecutive + 1 if probability >= frame_thr else 0
        best = max(best, consecutive)
    dirty_frames = sum(1 for probability in p_dirt if probability >= frame_thr)
    return best >= k and dirty_frames >= k and trimmed_mean(p_dirt) >= track_thr


def merge_caps(transits):
    """Group cam_0/cam_2 transits of one recording by overlapping frame spans."""

    caps = []
    by_recording = {}
    for transit in transits:
        by_recording.setdefault((transit["truth"], transit["recording"].rsplit("_cam", 1)[0]), []).append(transit)
    for (truth, _rec), items in by_recording.items():
        cam0 = sorted((t for t in items if t["camera"] == "cam0"), key=lambda t: t["first_frame"])
        cam2 = sorted((t for t in items if t["camera"] == "cam2"), key=lambda t: t["first_frame"])
        used = set()
        for a in cam0:
            partner = None
            for j, b in enumerate(cam2):
                if j in used:
                    continue
                if a["first_frame"] <= b["last_frame"] and b["first_frame"] <= a["last_frame"]:
                    partner = b
                    used.add(j)
                    break
            caps.append({"truth": truth, "tracks": [a] + ([partner] if partner else [])})
        for j, b in enumerate(cam2):
            if j not in used:
                caps.append({"truth": truth, "tracks": [b]})
    return caps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--transits", type=Path, default=Path("video_harvest/transits.json"))
    parser.add_argument("--band", type=float, default=0.60)
    parser.add_argument("--top", type=int, default=12)
    args = parser.parse_args()

    transits = json.loads(args.transits.read_text(encoding="utf-8"))
    caps = merge_caps(transits)
    n_clean = sum(1 for cap in caps if cap["truth"] == "clean")
    n_dirt = len(caps) - n_clean
    print(f"{len(caps)} merged caps ({n_clean} clean / {n_dirt} dirty), band={args.band}\n")

    rows = []
    for frame_thr, track_thr, k in itertools.product(
        (0.5, 0.6, 0.7, 0.8, 0.9), (0.3, 0.45, 0.6, 0.75, 0.85), (2, 3, 4)
    ):
        false_rejects = misses = 0
        for cap in caps:
            dirty = any(
                track_is_dirty(
                    band_slice(track["p_dirt"], args.band),
                    frame_thr=frame_thr, track_thr=track_thr, k=k,
                )
                for track in cap["tracks"]
            )
            if cap["truth"] == "clean" and dirty:
                false_rejects += 1
            if cap["truth"] == "dirt" and not dirty:
                misses += 1
        rows.append({
            "frame_thr": frame_thr, "track_thr": track_thr, "k": k,
            "frr": false_rejects / max(1, n_clean), "miss": misses / max(1, n_dirt),
            "false_rejects": false_rejects, "misses": misses,
        })

    rows.sort(key=lambda row: (row["frr"] + row["miss"],))
    print(f"{'frame':>6} {'track':>6} {'k':>2} | {'FRR':>12} | {'miss':>12}")
    for row in rows[: args.top]:
        print(
            f"{row['frame_thr']:>6.2f} {row['track_thr']:>6.2f} {row['k']:>2} | "
            f"{row['false_rejects']:>4}/{n_clean} {row['frr']:>6.1%} | "
            f"{row['misses']:>4}/{n_dirt} {row['miss']:>6.1%}"
        )


if __name__ == "__main__":
    main()
