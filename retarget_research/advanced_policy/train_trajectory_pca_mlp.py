#!/usr/bin/env python3
"""训练“初始观测→整段轨迹PCA系数”的纯参数化自主策略。"""

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

try:
    from .models import make_mlp
except ImportError:
    from models import make_mlp


def trajectories(path, normalization):
    with np.load(path, allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    features, sequences = [], []
    for trajectory_id in np.unique(data["trajectory_id"]):
        indices = np.flatnonzero(data["trajectory_id"] == trajectory_id)
        actions = data["actions"][indices].astype(np.float32)
        initial = actions[0].copy()
        initial[6:] = 0.0
        features.append(
            (data["observations"][indices[0]] - normalization["observation_mean"])
            / normalization["observation_std"]
        )
        sequences.append(
            ((actions - initial - normalization["initial_delta_mean"])
             / normalization["initial_delta_std"]).reshape(-1)
        )
    return np.stack(features).astype(np.float32), np.stack(sequences).astype(np.float32)


def evaluate(model, features, targets, device):
    model.eval()
    with torch.no_grad():
        prediction = model(torch.from_numpy(features).to(device))
        target = torch.from_numpy(targets).to(device)
        return float(nn.functional.mse_loss(prediction, target))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    with np.load(args.data_dir / "normalization.npz", allow_pickle=False) as archive:
        normalization = {name: archive[name].astype(np.float32) for name in archive.files}
    train_x, train_y = trajectories(args.data_dir / "train.npz", normalization)
    valid_x, valid_y = trajectories(args.data_dir / "valid.npz", normalization)
    pca_mean = train_y.mean(0)
    _, _, right = np.linalg.svd(train_y - pca_mean, full_matrices=False)
    rank = min(args.rank, len(right))
    components = right[:rank].astype(np.float32)
    train_coefficients = (train_y - pca_mean) @ components.T
    valid_coefficients = (valid_y - pca_mean) @ components.T
    coefficient_mean = train_coefficients.mean(0).astype(np.float32)
    coefficient_std = np.maximum(train_coefficients.std(0), 1e-6).astype(np.float32)
    train_targets = ((train_coefficients - coefficient_mean) / coefficient_std).astype(np.float32)
    valid_targets = ((valid_coefficients - coefficient_mean) / coefficient_std).astype(np.float32)
    hidden_dims = [512, 512, 384]
    model = make_mlp(train_x.shape[1], rank, hidden_dims, 0.02).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_targets)),
        batch_size=min(64, len(train_x)), shuffle=True,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history, best_loss, best_state, stale = [], float("inf"), None, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for features, targets in loader:
            prediction = model(features.to(device))
            loss = nn.functional.mse_loss(prediction, targets.to(device))
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss) * len(features)
        train_loss = total / len(train_x)
        valid_loss = evaluate(model, valid_x, valid_targets, device)
        history.append((epoch, train_loss, valid_loss))
        if valid_loss < best_loss - 1e-6:
            best_loss = valid_loss
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % 25 == 0:
            print(f"epoch={epoch} train={train_loss:.6f} valid={valid_loss:.6f}", flush=True)
        if stale >= 40:
            break
    mappings = json.loads((args.data_dir / "mappings.json").read_text(encoding="utf-8"))
    payload = {
        "config": {
            "model_type": "trajectory_pca_mlp", "pca_rank": rank,
            "hidden_dims": hidden_dims, "dropout": 0.02,
        },
        "dimensions": {
            "observation_dim": train_x.shape[1],
            "action_dim": train_y.shape[1] // 240,
            "category_count": len(mappings["category_to_id"]),
        },
        "model_state": best_state,
        "pca_mean": pca_mean.astype(np.float32),
        "pca_components": components,
        "coefficient_mean": coefficient_mean,
        "coefficient_std": coefficient_std,
        "sequence_shape": [240, train_y.shape[1] // 240],
    }
    torch.save(payload, args.output_dir / "best.pt")
    with (args.output_dir / "history.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["epoch", "train_loss", "valid_loss"])
        writer.writerows(history)
    (args.output_dir / "summary.json").write_text(json.dumps({
        "status": "complete", "best_valid_coefficient_mse": best_loss,
        "epochs_completed": len(history), "rank": rank,
        "checkpoint_contains_training_trajectories": False,
    }, indent=2) + "\n")
    print(f"TRAJECTORY_PCA_MLP={args.output_dir / 'best.pt'} valid={best_loss:.6f}")


if __name__ == "__main__":
    main()
