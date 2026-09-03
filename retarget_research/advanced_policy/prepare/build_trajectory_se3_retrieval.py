#!/usr/bin/env python3
"""把训练轨迹转换为相对初始手腕的SE(3)局部运动库。"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--finger-k", type=int)
    parser.add_argument("--translation-frame", choices=("local", "world"), default="local")
    args = parser.parse_args()
    with np.load(args.data_dir / "train.npz", allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    with np.load(args.data_dir / "normalization.npz", allow_pickle=False) as archive:
        mean = archive["observation_mean"].astype(np.float32)
        std = archive["observation_std"].astype(np.float32)
    mappings = json.loads((args.data_dir / "mappings.json").read_text(encoding="utf-8"))
    initial_observations, translations, rotvecs, fingers = [], [], [], []
    for trajectory_id in np.unique(data["trajectory_id"]):
        indices = np.flatnonzero(data["trajectory_id"] == trajectory_id)
        actions = data["actions"][indices].astype(np.float32)
        rotations = Rotation.from_euler("xyz", actions[:, 3:6])
        initial_rotation = rotations[0]
        initial_observations.append(
            (data["observations"][indices[0]].astype(np.float32) - mean) / std
        )
        translation = actions[:, :3] - actions[0, :3]
        if args.translation_frame == "local":
            translation = initial_rotation.inv().apply(translation)
        translations.append(translation)
        rotvecs.append((initial_rotation.inv() * rotations).as_rotvec())
        fingers.append(actions[:, 6:])
    payload = {
        "config": {
            "model_type": "trajectory_se3_retrieval",
            "retrieval_k": args.k,
            "finger_retrieval_k": int(args.finger_k or args.k),
            "translation_frame": args.translation_frame,
        },
        "dimensions": {
            "observation_dim": len(mean),
            "action_dim": data["actions"].shape[1],
            "category_count": len(mappings["category_to_id"]),
        },
        "retrieval_initial_observations": np.stack(initial_observations).astype(np.float32),
        "retrieval_local_translation": np.stack(translations).astype(np.float32),
        "retrieval_relative_rotvec": np.stack(rotvecs).astype(np.float32),
        "retrieval_finger_actions": np.stack(fingers).astype(np.float32),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(
        f"TRAJECTORY_SE3_RETRIEVAL={args.output} trajectories={len(translations)} "
        f"k={args.k} translation_frame={args.translation_frame}"
    )


if __name__ == "__main__":
    main()
