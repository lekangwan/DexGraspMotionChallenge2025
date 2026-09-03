#!/usr/bin/env python3
"""以PCA输出为名义轨迹，构造动态手—物交互残差监督数据。"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

MODULE = Path(__file__).resolve().parents[1]
PROJECT = MODULE.parents[1]
sys.path.insert(0, str(MODULE))
sys.path.insert(0, str(PROJECT))
from interaction import (  # noqa: E402
    TargetHandGeometry, interaction_features, moving_object_points,
)
from models import build_pca_model  # noqa: E402


BASE_RANK = {"linker": 32, "xhand": 16, "wuji": 16}


def load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def residual_limit(action_dim):
    values = np.full(action_dim, 0.50, dtype=np.float32)
    values[:3] = 0.05
    values[3:6] = 0.25
    return values


def build_split(hand, split, data_dir, normalization, checkpoint, model, geometry_model,
                quality_by_key, object_by_id):
    data = load_npz(data_dir / f"{split}.npz")
    geometry = load_npz(data_dir / f"geometry_{split}.npz")
    geometry_row = {int(value): row for row, value in enumerate(geometry["trajectory_id"])}
    limit = residual_limit(data["actions"].shape[1])
    output = {key: [] for key in (
        "current_task", "nominal_delta", "interaction", "phase",
        "residual_target", "quality_weight", "trajectory_id",
    )}
    saturation = []
    for trajectory_id in np.unique(data["trajectory_id"]):
        indices = np.flatnonzero(data["trajectory_id"] == trajectory_id)
        row = geometry_row[int(trajectory_id)]
        observations = data["observations"][indices]
        initial = geometry["initial_command"][row].astype(np.float32)
        task = ((observations[0] - normalization["observation_mean"])
                / normalization["observation_std"])[-32:]
        command = ((initial - normalization["initial_command_mean"])
                   / normalization["initial_command_std"])
        cloud = ((geometry["object_points"][row] - normalization["point_mean"])
                 / normalization["point_std"])
        with torch.no_grad():
            normalized_coefficient = model(
                torch.from_numpy(task[None]), torch.from_numpy(command[None]),
                torch.from_numpy(cloud[None]),
            )[0].cpu().numpy()
        coefficient = (
            normalized_coefficient * checkpoint["coefficient_std"]
            + checkpoint["coefficient_mean"]
        )
        normalized_sequence = (
            checkpoint["pca_mean"] + coefficient @ checkpoint["pca_components"]
        ).reshape(checkpoint["sequence_shape"])
        nominal_delta = (
            normalized_sequence * normalization["initial_delta_std"]
            + normalization["initial_delta_mean"]
        ).astype(np.float32)
        nominal = initial + nominal_delta
        hand_points = geometry_model.points(nominal)
        object_points = moving_object_points(
            geometry["object_points"][row], initial, observations
        )
        interaction = interaction_features(
            hand_points, object_points, nominal[:, 3:6]
        )
        current_task = (
            (observations - normalization["observation_mean"])
            / normalization["observation_std"]
        )[:, -32:].astype(np.float32)
        target_raw = data["actions"][indices] - nominal
        saturation.append(np.mean(np.abs(target_raw) > limit[None]))
        target = np.clip(target_raw / limit[None], -1.0, 1.0).astype(np.float32)
        motion_steps = max(int(np.count_nonzero(~data["is_hold"][indices])), 2)
        phase = np.minimum(
            np.arange(len(indices), dtype=np.float32) / float(motion_steps - 1), 1.0
        )[:, None]
        object_name = object_by_id[int(data["object_id"][indices[0]])]
        source_index = int(data["source_trajectory_index"][indices[0]])
        weight = float(quality_by_key.get((object_name, source_index), 0.25))
        values = {
            "current_task": current_task,
            "nominal_delta": normalized_sequence.astype(np.float32),
            "interaction": interaction,
            "phase": phase,
            "residual_target": target,
            "quality_weight": np.full(len(indices), weight, np.float32),
            "trajectory_id": np.full(len(indices), int(trajectory_id), np.int64),
        }
        for key, value in values.items():
            output[key].append(value)
    return {key: np.concatenate(value) for key, value in output.items()}, float(np.mean(saturation))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hand", choices=["linker", "xhand", "wuji"], required=True)
    args = parser.parse_args()
    data_dir = MODULE / f"data/final/{args.hand}"
    rank = BASE_RANK[args.hand]
    checkpoint_path = MODULE / f"runs/candidates_v1/{args.hand}/geometry_pca{rank}/best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = build_pca_model(
        checkpoint["config"], checkpoint["dimensions"]["task_observation_dim"],
        checkpoint["dimensions"]["action_dim"],
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    with np.load(data_dir / "geometry_normalization.npz", allow_pickle=False) as archive:
        normalization = {key: archive[key].astype(np.float32) for key in archive.files}
    mappings = json.loads((data_dir / "mappings.json").read_text(encoding="utf-8"))
    object_by_id = {int(value): key for key, value in mappings["object_to_id"].items()}
    audit = json.loads(
        (MODULE / "data/final/EXPERT_LEARNABILITY_AUDIT.json").read_text(encoding="utf-8")
    )
    quality = {
        (row["object_name"], int(row["source_trajectory_index"])): float(row["learnability_score"])
        for row in audit["hands"][args.hand]["results"]
    }
    geometry_model = TargetHandGeometry(args.hand)
    summary = {"hand": args.hand, "base_checkpoint": str(checkpoint_path.resolve()), "splits": {}}
    for split in ("train", "valid"):
        arrays, saturation = build_split(
            args.hand, split, data_dir, normalization, checkpoint, model,
            geometry_model, quality, object_by_id,
        )
        np.savez_compressed(data_dir / f"interaction_{split}.npz", **arrays)
        summary["splits"][split] = {
            "steps": int(len(arrays["phase"])),
            "trajectories": int(len(np.unique(arrays["trajectory_id"]))),
            "target_saturation_fraction": saturation,
        }
    (data_dir / "interaction_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
