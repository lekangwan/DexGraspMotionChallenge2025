#!/usr/bin/env python3
"""把成功train轨迹整理成只查询初始手腕的kNN整段轨迹策略。"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--finger-k", type=int)
    parser.add_argument("--features", choices=("wrist", "wrist_shape"), default="wrist")
    args = parser.parse_args()
    with np.load(args.data_dir / "train.npz", allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    with np.load(args.data_dir / "normalization.npz", allow_pickle=False) as archive:
        mean = archive["observation_mean"].astype(np.float32)
        std = archive["observation_std"].astype(np.float32)
    mappings = json.loads((args.data_dir / "mappings.json").read_text(encoding="utf-8"))
    initial_observations, action_deltas = [], []
    lengths = set()
    for trajectory_id in np.unique(data["trajectory_id"]):
        indices = np.flatnonzero(data["trajectory_id"] == trajectory_id)
        actions = data["actions"][indices].astype(np.float32)
        initial = actions[0].copy()
        initial[6:] = 0.0
        initial_observations.append(
            (data["observations"][indices[0]].astype(np.float32) - mean) / std
        )
        action_deltas.append(actions - initial)
        lengths.add(len(indices))
    if len(lengths) != 1:
        raise ValueError(f"train轨迹长度不一致: {sorted(lengths)}")
    observation_dim = int(initial_observations[0].shape[0])
    hand_dof_dim = (observation_dim - 32) // 2
    feature_indices = np.arange(6, dtype=np.int64)
    if args.features == "wrist_shape":
        shape_start = 2 * hand_dof_dim + 13
        feature_indices = np.concatenate([
            feature_indices, np.arange(shape_start, shape_start + 14, dtype=np.int64)
        ])
    payload = {
        "config": {
            "model_type": "trajectory_retrieval",
            "retrieval_k": int(args.k),
            "finger_retrieval_k": int(args.finger_k or args.k),
            "feature_rule": args.features,
        },
        "dimensions": {
            "observation_dim": observation_dim,
            "action_dim": int(action_deltas[0].shape[1]),
            "category_count": len(mappings["category_to_id"]),
        },
        "retrieval_initial_observations": np.stack(initial_observations),
        "retrieval_action_deltas": np.stack(action_deltas),
        "retrieval_feature_indices": feature_indices,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(f"TRAJECTORY_RETRIEVAL={args.output} trajectories={len(action_deltas)} k={args.k} features={args.features}")


if __name__ == "__main__":
    main()
