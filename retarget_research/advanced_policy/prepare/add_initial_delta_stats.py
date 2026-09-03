#!/usr/bin/env python3
"""从train轨迹计算“相对本条初始张开命令”的动作统计。"""

import argparse
from pathlib import Path

import numpy as np


def add_stats(data_dir):
    data_dir = Path(data_dir)
    with np.load(data_dir / "train.npz", allow_pickle=False) as archive:
        actions = archive["actions"].astype(np.float32)
        trajectory_ids = archive["trajectory_id"].astype(np.int64)
    deltas = np.empty_like(actions)
    for trajectory_id in np.unique(trajectory_ids):
        indices = np.flatnonzero(trajectory_ids == trajectory_id)
        initial = actions[indices[0]].copy()
        initial[6:] = 0.0
        deltas[indices] = actions[indices] - initial
    path = data_dir / "normalization.npz"
    with np.load(path, allow_pickle=False) as archive:
        values = {name: archive[name] for name in archive.files}
    values["initial_delta_mean"] = deltas.mean(axis=0).astype(np.float32)
    values["initial_delta_std"] = np.maximum(deltas.std(axis=0), 1e-6).astype(np.float32)
    np.savez_compressed(path, **values)
    print(f"INITIAL_DELTA_STATS={data_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", nargs="+", type=Path)
    args = parser.parse_args()
    for data_dir in args.data_dir:
        add_stats(data_dir)


if __name__ == "__main__":
    main()
