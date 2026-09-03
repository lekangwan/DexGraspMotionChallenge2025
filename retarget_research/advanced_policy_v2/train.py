#!/usr/bin/env python3
"""训练几何条件的单步或动作块自主策略。"""

import argparse
import csv
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

MODULE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(MODULE_ROOT))

from dataset import GeometryPolicyDataset  # noqa: E402
from models import build_model  # noqa: E402


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def chunk_loss(prediction, target, decay=0.9, beta=0.5):
    """对近期动作赋更高权重的逐元素Huber动作块损失。"""
    if prediction.shape != target.shape:
        raise ValueError(f"预测与目标尺寸不一致: {prediction.shape} vs {target.shape}")
    horizon = prediction.shape[1]
    weights = torch.pow(
        torch.as_tensor(float(decay), device=prediction.device, dtype=prediction.dtype),
        torch.arange(horizon, device=prediction.device, dtype=prediction.dtype),
    )
    weights = weights / weights.mean()
    element = nn.functional.smooth_l1_loss(
        prediction, target, beta=float(beta), reduction="none"
    )
    return (element * weights.view(1, -1, 1)).mean(), element.mean()


def run_epoch(model, loader, device, optimizer, config):
    training = optimizer is not None
    model.train(training)
    total_loss = total_mae = 0.0
    samples = batches = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch_index, batch in enumerate(loader):
            maximum = config.get("max_train_batches" if training else "max_valid_batches")
            if maximum is not None and batch_index >= int(maximum):
                break
            batch = {name: value.to(device) for name, value in batch.items()}
            if training:
                optimizer.zero_grad(set_to_none=True)
                jitter = float(config.get("point_jitter_std", 0.0))
                if jitter > 0:
                    batch["object_points"] = batch["object_points"] + jitter * torch.randn_like(batch["object_points"])
            prediction = model(
                batch["initial_observation"], batch["initial_command"],
                batch["object_points"], batch["observation_history"],
                batch["previous_delta_history"], batch["phase"],
            )
            loss, _ = chunk_loss(
                prediction, batch["action_chunk"],
                config.get("chunk_loss_decay", 0.9), config.get("huber_beta", 0.5),
            )
            mae = nn.functional.l1_loss(prediction, batch["action_chunk"])
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), float(config.get("gradient_clip", 1.0)))
                optimizer.step()
            batch_size = len(batch["phase"])
            total_loss += float(loss.item()) * batch_size
            total_mae += float(mae.item()) * batch_size
            samples += batch_size
            batches += 1
    if samples == 0:
        raise ValueError("没有处理任何样本")
    return {"loss": total_loss / samples, "mae": total_mae / samples, "samples": samples, "batches": batches}


def save_metrics(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    seed = int(config.get("seed", 20260831))
    set_seed(seed)
    device_name = args.device or config.get("device", "cuda")
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("配置要求CUDA，但当前环境未发现CUDA")
    device = torch.device(device_name)
    data_dir = Path(config["data_dir"]).expanduser().resolve()
    output_dir = Path(config["output_dir"]).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    train_set = GeometryPolicyDataset(
        data_dir, "train", config["model_type"],
        config.get("history", 3), config.get("action_horizon", 8),
    )
    valid_set = GeometryPolicyDataset(
        data_dir, "valid", config["model_type"],
        config.get("history", 3), config.get("action_horizon", 8),
    )
    loader_args = {
        "batch_size": int(config.get("batch_size", 512)),
        "num_workers": int(config.get("num_workers", 4)),
        "pin_memory": device.type == "cuda",
    }
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_set, shuffle=True, generator=generator, **loader_args)
    valid_loader = DataLoader(valid_set, shuffle=False, **loader_args)
    model = build_model(config, train_set.observation_dim, train_set.action_dim).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config.get("learning_rate", 3e-4)),
        weight_decay=float(config.get("weight_decay", 1e-4)),
    )
    epochs = int(config.get("epochs", 120))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=1e-6)
    best = float("inf")
    stale = 0
    rows = []
    started = time.perf_counter()
    dimensions = {
        "observation_dim": train_set.observation_dim,
        "action_dim": train_set.action_dim,
        "point_count": train_set.point_count,
    }
    for epoch in range(1, epochs + 1):
        train = run_epoch(model, train_loader, device, optimizer, config)
        valid = run_epoch(model, valid_loader, device, None, config)
        scheduler.step()
        row = {
            "epoch": epoch, "train_loss": train["loss"], "valid_loss": valid["loss"],
            "train_mae": train["mae"], "valid_mae": valid["mae"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "elapsed_seconds": time.perf_counter() - started,
        }
        rows.append(row)
        save_metrics(output_dir / "metrics.csv", rows)
        payload = {
            "schema": "geometry_action_chunk_policy_v1",
            "model_state": model.state_dict(),
            "config": config,
            "dimensions": dimensions,
            "epoch": epoch,
            "best_valid_loss": min(best, valid["loss"]),
        }
        torch.save(payload, output_dir / "last.pt")
        if valid["loss"] < best - float(config.get("minimum_improvement", 1e-5)):
            best = valid["loss"]
            stale = 0
            torch.save(payload, output_dir / "best.pt")
        else:
            stale += 1
        print(
            f"epoch={epoch:03d} train={train['loss']:.6f} valid={valid['loss']:.6f} "
            f"best={best:.6f}", flush=True
        )
        if stale >= int(config.get("early_stopping_patience", 20)):
            break
    summary = {
        "schema": "geometry_action_chunk_training_v1",
        "model_type": config["model_type"],
        "best_valid_loss": best,
        "last_epoch": rows[-1]["epoch"],
        "dimensions": dimensions,
        "wall_time_seconds": time.perf_counter() - started,
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

