#!/usr/bin/env python3
"""训练“预抓取—抓稳—运输结束”三关键状态自主策略。"""

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
from models import build_keypose_model  # noqa: E402


def load_split(data_dir, split, normalization, interaction_normalization):
    """整理每条轨迹的初始条件、三个关键状态和真实关键时刻。"""
    with np.load(data_dir / f"{split}.npz", allow_pickle=False) as archive:
        data = {name: archive[name].copy() for name in archive.files}
    with np.load(data_dir / f"geometry_{split}.npz", allow_pickle=False) as archive:
        geometry = {name: archive[name].copy() for name in archive.files}
    with np.load(
        data_dir / f"initial_interaction_{split}.npz", allow_pickle=False
    ) as archive:
        interaction = {name: archive[name].copy() for name in archive.files}
    geometry_row = {int(value): row for row, value in enumerate(geometry["trajectory_id"])}
    interaction_row = {int(value): row for row, value in enumerate(interaction["trajectory_id"])}
    tasks, commands, clouds, relations, targets = [], [], [], [], []
    pre_frames, grasp_frames = [], []
    for trajectory_id in np.unique(data["trajectory_id"]):
        indices = np.flatnonzero(data["trajectory_id"] == trajectory_id)
        actions = data["actions"][indices].astype(np.float32)
        observations = data["observations"][indices].astype(np.float32)
        row = geometry_row[int(trajectory_id)]
        initial = geometry["initial_command"][row].astype(np.float32)
        contact = observations[:, -1] > 0.0
        relative_z = observations[:, -3]
        contact_indices = np.flatnonzero(contact)
        first_contact = int(contact_indices[0]) if len(contact_indices) else 85
        lift_indices = np.flatnonzero(contact & (relative_z >= 0.03))
        first_lift = int(lift_indices[0]) if len(lift_indices) else 117
        pre_frame = max(first_contact - 12, 0)
        grasp_frame = max(first_contact, first_lift - 1)
        final_wrist = np.median(actions[-30:, :6], axis=0)
        target = np.concatenate([
            actions[pre_frame, :6] - initial[:6],
            actions[grasp_frame] - initial,
            final_wrist - initial[:6],
        ]).astype(np.float32)
        tasks.append(
            ((observations[0] - normalization["observation_mean"])
             / normalization["observation_std"])[-32:]
        )
        commands.append(
            (initial - normalization["initial_command_mean"])
            / normalization["initial_command_std"]
        )
        clouds.append(
            (geometry["object_points"][row] - normalization["point_mean"])
            / normalization["point_std"]
        )
        value = interaction["interaction"][interaction_row[int(trajectory_id)]]
        relations.append(
            (value - interaction_normalization["mean"])
            / interaction_normalization["std"]
        )
        targets.append(target)
        pre_frames.append(pre_frame)
        grasp_frames.append(grasp_frame)
    arrays = tuple(np.asarray(value, dtype=np.float32) for value in (
        tasks, commands, clouds, relations, targets,
    ))
    return arrays, np.asarray(pre_frames), np.asarray(grasp_frames)


def validation_loss(model, loader, device):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for task, command, cloud, relation, target in loader:
            prediction = model(
                task.to(device), command.to(device), cloud.to(device),
                relation.to(device),
            )
            total += float(nn.functional.smooth_l1_loss(
                prediction, target.to(device), beta=0.5
            )) * len(task)
    return total / len(loader.dataset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    seed = int(config.get("seed", 20260902))
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device(args.device or config.get("device", "cuda"))
    data_dir = Path(config["data_dir"])
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(data_dir / "geometry_normalization.npz", allow_pickle=False) as archive:
        normalization = {name: archive[name].astype(np.float32) for name in archive.files}
    with np.load(
        data_dir / "initial_interaction_normalization.npz", allow_pickle=False
    ) as archive:
        interaction_normalization = {
            name: archive[name].astype(np.float32) for name in archive.files
        }
    train, train_pre, train_grasp = load_split(
        data_dir, "train", normalization, interaction_normalization
    )
    valid, _, _ = load_split(
        data_dir, "valid", normalization, interaction_normalization
    )
    target_mean = train[4].mean(axis=0).astype(np.float32)
    target_std = np.maximum(train[4].std(axis=0), 1e-5).astype(np.float32)
    train_target = ((train[4] - target_mean) / target_std).astype(np.float32)
    valid_target = ((valid[4] - target_mean) / target_std).astype(np.float32)
    train_set = TensorDataset(*(
        torch.from_numpy(value) for value in (*train[:4], train_target)
    ))
    valid_set = TensorDataset(*(
        torch.from_numpy(value) for value in (*valid[:4], valid_target)
    ))
    train_loader = DataLoader(
        train_set, batch_size=min(int(config.get("batch_size", 64)), len(train_set)),
        shuffle=True,
    )
    valid_loader = DataLoader(valid_set, batch_size=len(valid_set), shuffle=False)
    model = build_keypose_model(
        config, train[0].shape[1], train[1].shape[1], train[4].shape[1]
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config.get("learning_rate", 3e-4)),
        weight_decay=float(config.get("weight_decay", 1e-3)),
    )
    best, best_state, stale, history = float("inf"), None, 0, []
    for epoch in range(1, int(config.get("epochs", 250)) + 1):
        model.train(); total = 0.0
        for task, command, cloud, relation, target in train_loader:
            prediction = model(
                task.to(device), command.to(device), cloud.to(device),
                relation.to(device),
            )
            loss = nn.functional.smooth_l1_loss(
                prediction, target.to(device), beta=0.5
            )
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss) * len(task)
        train_loss = total / len(train_set)
        valid_loss = validation_loss(model, valid_loader, device)
        history.append((epoch, train_loss, valid_loss))
        if valid_loss < best - 1e-5:
            best, stale = valid_loss, 0
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        else:
            stale += 1
        if epoch == 1 or epoch % 20 == 0:
            print(
                f"epoch={epoch:03d} train={train_loss:.6f} valid={valid_loss:.6f}",
                flush=True,
            )
        if stale >= int(config.get("early_stopping_patience", 35)):
            break
    payload = {
        "schema": "geometry_keypose_lift_policy_v1",
        "config": config,
        "dimensions": {
            "task_observation_dim": train[0].shape[1],
            "observation_dim": len(normalization["observation_mean"]),
            "action_dim": train[1].shape[1],
            "point_count": train[2].shape[1],
            "interaction_dim": train[3].shape[1],
            "keypose_dim": train[4].shape[1],
        },
        "model_state": best_state,
        "target_mean": target_mean,
        "target_std": target_std,
        "interaction_mean": interaction_normalization["mean"],
        "interaction_std": interaction_normalization["std"],
        "pregrasp_frame": int(round(np.median(train_pre))),
        "grasp_frame": int(round(np.median(train_grasp))),
        "lift_end_frame": 209,
        "sequence_length": 240,
        "best_valid_loss": best,
    }
    torch.save(payload, output_dir / "best.pt")
    with (output_dir / "metrics.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["epoch", "train_loss", "valid_loss"])
        writer.writerows(history)
    (output_dir / "training_summary.json").write_text(
        json.dumps({
            "schema": "geometry_keypose_lift_training_v1",
            "best_valid_loss": best,
            "last_epoch": history[-1][0],
            "pregrasp_frame": payload["pregrasp_frame"],
            "grasp_frame": payload["grasp_frame"],
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"KEYPOSE_POLICY={output_dir / 'best.pt'} valid={best:.6f}")


if __name__ == "__main__":
    main()
