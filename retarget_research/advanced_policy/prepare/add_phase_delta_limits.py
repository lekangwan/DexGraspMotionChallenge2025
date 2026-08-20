#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


BIN_COUNT = 10
MIN_LIMIT = 1e-4


def compute_phase_limits(actions, trajectory_ids, is_hold):
    actions = np.asarray(actions, dtype=np.float32)
    trajectory_ids = np.asarray(trajectory_ids, dtype=np.int64)
    is_hold = np.asarray(is_hold, dtype=bool)
    limits = np.full((BIN_COUNT + 1, actions.shape[1]), MIN_LIMIT, dtype=np.float32)
    for trajectory_id in np.unique(trajectory_ids):
        indices = np.flatnonzero(trajectory_ids == trajectory_id)
        raw = actions[indices]
        deltas = np.empty_like(raw)
        deltas[0] = raw[0] - (2.0 * raw[0] - raw[1])
        deltas[1:] = raw[1:] - raw[:-1]
        non_hold = int(np.count_nonzero(~is_hold[indices]))
        denominator = max(non_hold - 1, 1)
        phase = np.minimum(np.arange(len(indices)) / denominator, 1.0)
        bins = np.minimum((phase * BIN_COUNT).astype(int), BIN_COUNT)
        for b in range(BIN_COUNT + 1):
            mask = bins == b
            if not mask.any():
                continue
            p95 = np.percentile(np.abs(deltas[mask]), 95, axis=0)
            limits[b] = np.maximum(limits[b], p95.astype(np.float32))
    return limits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    with np.load(data_dir / "train.npz", allow_pickle=False) as archive:
        actions = archive["actions"].astype(np.float32)
        trajectory_ids = archive["trajectory_id"].astype(np.int64)
        is_hold = archive["is_hold"].astype(bool)
    limits = compute_phase_limits(actions, trajectory_ids, is_hold)
    with np.load(data_dir / "normalization.npz", allow_pickle=False) as archive:
        fields = {name: archive[name] for name in archive.files}
    fields["action_phase_delta_limits"] = limits
    np.savez(data_dir / "normalization.npz", **fields)
    print("PHASE_DELTA_LIMITS_READY")
    print(f"output={data_dir / 'normalization.npz'}")


if __name__ == "__main__":
    main()
