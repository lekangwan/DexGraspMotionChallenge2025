#!/usr/bin/env python3
"""训练初始几何条件的整轨迹PCA自主策略。"""

import argparse
import csv
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from models import build_pca_model  # noqa: E402


def load_trajectories(
    data_dir, split, normalization, use_initial_interaction=False,
    interaction_normalization=None,
):
    """把逐步数据整理为每条轨迹一个初始条件和一条动作序列。"""
    with np.load(data_dir / f"{split}.npz", allow_pickle=False) as archive:
        data = {name: archive[name].copy() for name in archive.files}
    with np.load(data_dir / f"geometry_{split}.npz", allow_pickle=False) as archive:
        geometry = {name: archive[name].copy() for name in archive.files}
    geometry_row = {int(value): i for i, value in enumerate(geometry["trajectory_id"])}
    interaction_row = {}
    interaction_values = None
    if use_initial_interaction:
        with np.load(
            data_dir / f"initial_interaction_{split}.npz", allow_pickle=False
        ) as archive:
            interaction_row = {
                int(value): row for row, value in enumerate(archive["trajectory_id"])
            }
            interaction_values = archive["interaction"].astype(np.float32)
    task_observations, commands, points, interactions, sequences = [], [], [], [], []
    for trajectory_id in np.unique(data["trajectory_id"]):
        indices = np.flatnonzero(data["trajectory_id"] == trajectory_id)
        row = geometry_row[int(trajectory_id)]
        initial = geometry["initial_command"][row].astype(np.float32)
        task_observations.append(
            ((data["observations"][indices[0]] - normalization["observation_mean"])
             / normalization["observation_std"])[-32:]
        )
        commands.append(
            (initial - normalization["initial_command_mean"])
            / normalization["initial_command_std"]
        )
        points.append(
            (geometry["object_points"][row] - normalization["point_mean"])
            / normalization["point_std"]
        )
        if use_initial_interaction:
            value = interaction_values[interaction_row[int(trajectory_id)]]
            interactions.append(
                (value - interaction_normalization["mean"])
                / interaction_normalization["std"]
            )
        else:
            interactions.append(np.empty(0, dtype=np.float32))
        sequences.append(
            ((data["actions"][indices] - initial - normalization["initial_delta_mean"])
             / normalization["initial_delta_std"]).reshape(-1)
        )
    return tuple(np.asarray(value, dtype=np.float32) for value in (
        task_observations, commands, points, interactions, sequences,
    ))


def coefficient_loss(prediction, target, coefficient_std):
    """按PCA分量真实方差加权，使损失等价于关注动作序列重建。"""
    error = (prediction - target) * coefficient_std
    return torch.mean(error.square()) / torch.mean(coefficient_std.square())


def evaluate(model, loader, device, coefficient_std):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for task, command, points, interaction, target in loader:
            interaction = interaction.to(device) if interaction.shape[1] else None
            prediction = model(
                task.to(device), command.to(device), points.to(device), interaction
            )
            loss = coefficient_loss(prediction, target.to(device), coefficient_std)
            total += float(loss) * len(task)
    return total / len(loader.dataset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    seed = int(config.get("seed", 20260901))
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device(args.device or config.get("device", "cuda"))
    data_dir = Path(config["data_dir"])
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(data_dir / "geometry_normalization.npz", allow_pickle=False) as archive:
        normalization = {name: archive[name].astype(np.float32) for name in archive.files}
    use_initial_interaction = int(config.get("interaction_dim", 0)) > 0
    interaction_normalization = None
    if use_initial_interaction:
        with np.load(
            data_dir / "initial_interaction_normalization.npz", allow_pickle=False
        ) as archive:
            interaction_normalization = {
                name: archive[name].astype(np.float32) for name in archive.files
            }
    train = load_trajectories(
        data_dir, "train", normalization, use_initial_interaction,
        interaction_normalization,
    )
    valid = load_trajectories(
        data_dir, "valid", normalization, use_initial_interaction,
        interaction_normalization,
    )
    mean = train[4].mean(axis=0)
    _, _, right = np.linalg.svd(train[4] - mean, full_matrices=False)
    rank = min(int(config["pca_rank"]), len(right))
    components = right[:rank].astype(np.float32)
    train_coeff = (train[4] - mean) @ components.T
    valid_coeff = (valid[4] - mean) @ components.T
    coeff_mean = train_coeff.mean(axis=0).astype(np.float32)
    coeff_std = np.maximum(train_coeff.std(axis=0), 1e-5).astype(np.float32)
    train_target = ((train_coeff - coeff_mean) / coeff_std).astype(np.float32)
    valid_target = ((valid_coeff - coeff_mean) / coeff_std).astype(np.float32)
    train_set = TensorDataset(*(torch.from_numpy(x) for x in (*train[:4], train_target)))
    valid_set = TensorDataset(*(torch.from_numpy(x) for x in (*valid[:4], valid_target)))
    train_loader = DataLoader(train_set, batch_size=min(int(config.get("batch_size", 64)), len(train_set)), shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=len(valid_set), shuffle=False)
    model = build_pca_model(config, train[0].shape[1], train[1].shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.get("learning_rate", 3e-4)),
                                  weight_decay=float(config.get("weight_decay", 1e-3)))
    coefficient_std_tensor = torch.from_numpy(coeff_std).to(device)
    best, best_state, stale, history = float("inf"), None, 0, []
    for epoch in range(1, int(config.get("epochs", 300)) + 1):
        model.train(); total = 0.0
        for task, command, cloud, interaction, target in train_loader:
            cloud = cloud.to(device)
            jitter = float(config.get("point_jitter_std", 0.0))
            if jitter: cloud = cloud + jitter * torch.randn_like(cloud)
            interaction = interaction.to(device) if interaction.shape[1] else None
            prediction = model(task.to(device), command.to(device), cloud, interaction)
            loss = coefficient_loss(prediction, target.to(device), coefficient_std_tensor)
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            total += float(loss) * len(task)
        train_loss = total / len(train_set)
        valid_loss = evaluate(model, valid_loader, device, coefficient_std_tensor)
        history.append((epoch, train_loss, valid_loss))
        if valid_loss < best - 1e-5:
            best = valid_loss; stale = 0
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        else:
            stale += 1
        if epoch == 1 or epoch % 20 == 0:
            print(f"epoch={epoch:03d} train={train_loss:.6f} valid={valid_loss:.6f}", flush=True)
        if stale >= int(config.get("early_stopping_patience", 35)): break
    payload = {
        "schema": "geometry_trajectory_pca_policy_v1", "config": config,
        "dimensions": {"task_observation_dim": train[0].shape[1], "observation_dim": len(normalization["observation_mean"]),
                       "action_dim": train[1].shape[1], "point_count": train[2].shape[1],
                       "interaction_dim": train[3].shape[1]},
        "model_state": best_state, "pca_mean": mean.astype(np.float32),
        "pca_components": components, "coefficient_mean": coeff_mean,
        "coefficient_std": coeff_std, "sequence_shape": [train[4].shape[1] // train[1].shape[1], train[1].shape[1]],
        "best_valid_loss": best,
    }
    if use_initial_interaction:
        payload["interaction_mean"] = interaction_normalization["mean"]
        payload["interaction_std"] = interaction_normalization["std"]
    torch.save(payload, output_dir / "best.pt")
    (output_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (output_dir / "metrics.csv").open("w", newline="") as stream:
        writer = csv.writer(stream); writer.writerow(["epoch", "train_loss", "valid_loss"]); writer.writerows(history)
    (output_dir / "training_summary.json").write_text(json.dumps({
        "schema": "geometry_trajectory_pca_training_v1", "rank": rank,
        "best_valid_loss": best, "last_epoch": history[-1][0],
        "checkpoint_contains_training_trajectories": False,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"PCA_POLICY={output_dir / 'best.pt'} valid={best:.6f}")


if __name__ == "__main__":
    main()
