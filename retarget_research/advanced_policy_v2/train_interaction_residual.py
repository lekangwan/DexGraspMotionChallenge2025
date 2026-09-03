#!/usr/bin/env python3
"""训练PCA名义轨迹上的动态手—物交互残差策略。"""

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
from models import build_interaction_residual  # noqa: E402


def load(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def dataset(arrays):
    return TensorDataset(*(
        torch.from_numpy(arrays[key]).float()
        for key in (
            "current_task", "nominal_delta", "interaction", "phase",
            "residual_target", "quality_weight",
        )
    ))


def run_epoch(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total = mae = count = 0.0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for task, nominal, interaction, phase, target, quality in loader:
            task = task.to(device); nominal = nominal.to(device)
            interaction = interaction.to(device); phase = phase.to(device)
            target = target.to(device); quality = quality.to(device)
            prediction = model(task, nominal, interaction, phase)
            per_sample = nn.functional.smooth_l1_loss(
                prediction, target, beta=0.10, reduction="none"
            ).mean(dim=-1)
            weight = 0.5 + quality
            loss = torch.sum(per_sample * weight) / torch.sum(weight)
            if training:
                optimizer.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            batch = len(task)
            total += float(loss) * batch
            mae += float(torch.mean(torch.abs(prediction - target))) * batch
            count += batch
    return total / count, mae / count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    seed = int(config.get("seed", 20260901))
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device(args.device or config.get("device", "cuda"))
    data_dir = Path(config["data_dir"])
    output_dir = args.output_dir or Path(config["output_dir"])
    train_arrays = load(data_dir / "interaction_train.npz")
    valid_arrays = load(data_dir / "interaction_valid.npz")
    train = dataset(train_arrays); valid = dataset(valid_arrays)
    batch_size = int(config.get("batch_size", 512))
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid, batch_size=batch_size * 2, shuffle=False, num_workers=0)
    action_dim = train_arrays["nominal_delta"].shape[1]
    model = build_interaction_residual(
        config, train_arrays["current_task"].shape[1], action_dim,
        train_arrays["interaction"].shape[1],
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config.get("learning_rate", 3e-4)),
        weight_decay=float(config.get("weight_decay", 1e-3)),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    best = float("inf"); best_state = None; stale = 0; history = []
    for epoch in range(1, int(args.epochs or config.get("epochs", 120)) + 1):
        train_loss, train_mae = run_epoch(model, train_loader, device, optimizer)
        valid_loss, valid_mae = run_epoch(model, valid_loader, device)
        history.append((epoch, train_loss, train_mae, valid_loss, valid_mae))
        if valid_loss < best - 1e-5:
            best = valid_loss; stale = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"epoch={epoch:03d} train={train_loss:.5f} "
                f"valid={valid_loss:.5f} valid_mae={valid_mae:.4f}", flush=True
            )
        if stale >= int(config.get("early_stopping_patience", 20)):
            break
    payload = {
        "schema": "geometry_pca_interaction_residual_v1",
        "config": config,
        "dimensions": {
            "task_dim": int(train_arrays["current_task"].shape[1]),
            "interaction_dim": int(train_arrays["interaction"].shape[1]),
            "action_dim": int(action_dim),
        },
        "model_state": best_state,
        "base_checkpoint": str(Path(config["base_checkpoint"]).resolve()),
        "residual_limit": np.asarray(config["residual_limit"], dtype=np.float32),
        "best_valid_loss": float(best),
    }
    torch.save(payload, output_dir / "best.pt")
    with (output_dir / "metrics.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["epoch", "train_loss", "train_mae", "valid_loss", "valid_mae"])
        writer.writerows(history)
    (output_dir / "training_summary.json").write_text(json.dumps({
        "best_valid_loss": best, "last_epoch": history[-1][0],
        "best_or_later_valid_mae": min(row[4] for row in history),
        "base_checkpoint": payload["base_checkpoint"],
    }, indent=2) + "\n", encoding="utf-8")
    print(f"INTERACTION_RESIDUAL={output_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
