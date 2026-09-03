#!/usr/bin/env python3
"""为整轨迹PCA预计算episode初始时刻的15点手—物几何关系。"""

import argparse
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation


MODULE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE))
from interaction import TargetHandGeometry, interaction_features  # noqa: E402


def build_split(hand_model, data_dir, split):
    """输入几何sidecar，输出与trajectory_id一一对应的75维初始交互。"""
    with np.load(data_dir / f"geometry_{split}.npz", allow_pickle=False) as archive:
        trajectory_id = archive["trajectory_id"].copy()
        commands = archive["initial_command"].astype(np.float32)
        clouds = archive["object_points"].astype(np.float32)
    wrist_rotation = Rotation.from_euler("xyz", commands[:, 3:6]).as_matrix().astype(np.float32)
    world_clouds = (
        np.einsum("npj,nij->npi", clouds, wrist_rotation)
        + commands[:, None, :3]
    )
    hand_points = hand_model.points(commands)
    interaction = interaction_features(
        hand_points, world_clouds, commands[:, 3:6]
    )
    np.savez_compressed(
        data_dir / f"initial_interaction_{split}.npz",
        trajectory_id=trajectory_id,
        interaction=interaction.astype(np.float32),
    )
    return interaction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hand", choices=("linker", "xhand", "wuji"), required=True)
    args = parser.parse_args()
    data_dir = MODULE / f"data/final/{args.hand}"
    hand_model = TargetHandGeometry(args.hand)
    train = build_split(hand_model, data_dir, "train")
    valid = build_split(hand_model, data_dir, "valid")
    mean = train.mean(axis=0).astype(np.float32)
    std = np.maximum(train.std(axis=0), 1e-5).astype(np.float32)
    np.savez_compressed(
        data_dir / "initial_interaction_normalization.npz",
        mean=mean, std=std,
    )
    print(args.hand, train.shape, valid.shape)


if __name__ == "__main__":
    main()
