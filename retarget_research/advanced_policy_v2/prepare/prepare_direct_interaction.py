#!/usr/bin/env python3
"""为无PCA直接策略计算逐步手—物交互特征。"""

import argparse
from pathlib import Path
import sys

import numpy as np

MODULE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE))

from interaction import (  # noqa: E402
    TargetHandGeometry, interaction_features, moving_object_points,
    policy_pose_from_observations,
)


def load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def build_features(hand, data, geometry, hand_model):
    """输入物理观测和初始点云，输出与每个物理步对齐的75维交互特征。"""
    rows = {int(value): row for row, value in enumerate(geometry["trajectory_id"])}
    result = np.empty((len(data["actions"]), 75), dtype=np.float32)
    for trajectory_id in np.unique(data["trajectory_id"]):
        indices = np.flatnonzero(data["trajectory_id"] == trajectory_id)
        row = rows[int(trajectory_id)]
        observations = data["observations"][indices]
        poses = policy_pose_from_observations(hand, observations)
        hand_points = hand_model.points(poses)
        object_points = moving_object_points(
            geometry["object_points"][row], geometry["initial_command"][row],
            observations,
        )
        result[indices] = interaction_features(
            hand_points, object_points, poses[:, 3:6]
        )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hand", choices=["linker", "xhand", "wuji"], required=True)
    args = parser.parse_args()
    data_dir = MODULE / f"data/final/{args.hand}"
    hand_model = TargetHandGeometry(args.hand)
    features = {}
    for split in ("train", "valid"):
        data = load_npz(data_dir / f"{split}.npz")
        geometry = load_npz(data_dir / f"geometry_{split}.npz")
        values = build_features(args.hand, data, geometry, hand_model)
        features[split] = values
        np.savez_compressed(
            data_dir / f"direct_interaction_{split}.npz",
            interaction=values,
            trajectory_id=data["trajectory_id"].astype(np.int64),
        )
    mean = features["train"].mean(axis=0).astype(np.float32)
    std = np.maximum(features["train"].std(axis=0), 1e-4).astype(np.float32)
    np.savez_compressed(
        data_dir / "direct_interaction_normalization.npz",
        interaction_mean=mean, interaction_std=std,
    )
    print(
        f"{args.hand}: train={features['train'].shape} "
        f"valid={features['valid'].shape} finite="
        f"{np.isfinite(features['train']).all() and np.isfinite(features['valid']).all()}"
    )


if __name__ == "__main__":
    main()
