#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


BIN_COUNT = 10


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    with np.load(data_dir / "train.npz", allow_pickle=False) as archive:
        actions = archive["actions"].astype(np.float32)
        trajectory_ids = archive["trajectory_id"].astype(np.int64)
        is_hold = archive["is_hold"].astype(bool)
    envelope = np.full((BIN_COUNT + 1, 2, 3), np.nan, dtype=np.float32)
    for trajectory_id in np.unique(trajectory_ids):
        indices = np.flatnonzero(trajectory_ids == trajectory_id)
        raw = actions[indices]
        non_hold = int(np.count_nonzero(~is_hold[indices]))
        denominator = max(non_hold - 1, 1)
        phase = np.minimum(np.arange(len(indices)) / denominator, 1.0)
        bins = np.minimum((phase * BIN_COUNT).astype(int), BIN_COUNT)
        for b in range(BIN_COUNT + 1):
            mask = bins == b
            if not mask.any():
                continue
            low = np.percentile(raw[mask, :3], 3, axis=0)
            high = np.percentile(raw[mask, :3], 97, axis=0)
            for dim in range(3):
                old_low, old_high = envelope[b, 0, dim], envelope[b, 1, dim]
                envelope[b, 0, dim] = low[dim] if np.isnan(old_low) else min(old_low, low[dim])
                envelope[b, 1, dim] = high[dim] if np.isnan(old_high) else max(old_high, high[dim])
    with np.load(data_dir / "normalization.npz", allow_pickle=False) as archive:
        fields = {name: archive[name] for name in archive.files}
    fields["action_phase_position_envelope"] = envelope
    np.savez(data_dir / "normalization.npz", **fields)
    print("POSITION_ENVELOPE_READY")
    print(f"output={data_dir / 'normalization.npz'}")


if __name__ == "__main__":
    main()
