#!/usr/bin/env python3
"""训练无PCA的动态手物Temporal3直接动作策略。"""

import argparse
import csv
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, WeightedRandomSampler

from .direct_dataset import DirectInteractionDataset
from .models import build_direct_interaction
from .train import chunk_loss


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loader(dataset, batch_size, workers, training, seed):
    if training:
        categories = dataset.data["category_id"].astype(np.int64)
        counts = np.bincount(categories)
        weights = 1.0 / counts[categories]
        sampler = WeightedRandomSampler(
            torch.from_numpy(weights).double(), len(weights), replacement=True,
            generator=torch.Generator().manual_seed(seed),
        )
    else:
        sampler = None
    return DataLoader(
        dataset, batch_size=batch_size, sampler=sampler, shuffle=False,
        num_workers=workers, pin_memory=True,
    )


def run_epoch(model, loader, device, optimizer, config):
    training = optimizer is not None
    model.train(training)
    total_loss = total_mae = 0.0
    samples = 0
    with torch.enable_grad() if training else torch.no_grad():
        for batch in loader:
            batch = {name: value.to(device) for name, value in batch.items()}
            if training:
                optimizer.zero_grad(set_to_none=True)
                noise = float(config.get("state_noise_std", 0.0))
                if noise:
                    batch["observation_history"] += noise * torch.randn_like(
                        batch["observation_history"]
                    )
                    batch["interaction_history"] += noise * torch.randn_like(
                        batch["interaction_history"]
                    )
            prediction = model(
                batch["initial_observation"], batch["initial_command"],
                batch["object_points"], batch["observation_history"],
                batch["interaction_history"], batch["previous_delta_history"],
                batch["phase"],
            )
            loss, _ = chunk_loss(
                prediction, batch["action_chunk"],
                config.get("chunk_loss_decay", 0.9),
                config.get("huber_beta", 0.5),
            )
            mae = torch.nn.functional.l1_loss(prediction, batch["action_chunk"])
            if training:
                loss.backward()
                clip_grad_norm_(model.parameters(), float(config.get("gradient_clip", 1.0)))
                optimizer.step()
            count = len(batch["phase"])
            total_loss += float(loss) * count
            total_mae += float(mae) * count
            samples += count
    return total_loss / samples, total_mae / samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    seed = int(config.get("seed", 20260902))
    set_seed(seed)
    device = torch.device(args.device)
    data_dir = Path(config["data_dir"])
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    train_set = DirectInteractionDataset(
        data_dir, "train", config.get("history", 3),
        config.get("action_horizon", 1),
    )
    valid_set = DirectInteractionDataset(
        data_dir, "valid", config.get("history", 3),
        config.get("action_horizon", 1),
    )
    train_loader = make_loader(
        train_set, config.get("batch_size", 512), config.get("num_workers", 4),
        True, seed,
    )
    valid_loader = make_loader(
        valid_set, config.get("batch_size", 512), config.get("num_workers", 4),
        False, seed,
    )
    model = build_direct_interaction(
        config, train_set.observation_dim, train_set.action_dim,
        train_set.interaction_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config.get("learning_rate", 3e-4)),
        weight_decay=float(config.get("weight_decay", 1e-4)),
    )
    epochs = int(config.get("epochs", 100))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, epochs, eta_min=1e-6
    )
    best = float("inf")
    stale = 0
    rows = []
    started = time.perf_counter()
    dimensions = {
        "observation_dim": train_set.observation_dim,
        "action_dim": train_set.action_dim,
        "interaction_dim": train_set.interaction_dim,
        "point_count": train_set.point_count,
    }
    for epoch in range(1, epochs + 1):
        train_loss, train_mae = run_epoch(model, train_loader, device, optimizer, config)
        valid_loss, valid_mae = run_epoch(model, valid_loader, device, None, config)
        scheduler.step()
        rows.append({
            "epoch": epoch, "train_loss": train_loss, "valid_loss": valid_loss,
            "train_mae": train_mae, "valid_mae": valid_mae,
            "learning_rate": optimizer.param_groups[0]["lr"],
        })
        with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        payload = {
            "schema": "direct_interaction_temporal_policy_v1",
            "config": config, "dimensions": dimensions,
            "model_state": model.state_dict(), "epoch": epoch,
            "best_valid_loss": min(best, valid_loss),
            "interaction_mean": train_set.interaction_mean,
            "interaction_std": train_set.interaction_std,
        }
        torch.save(payload, output_dir / "last.pt")
        if valid_loss < best - float(config.get("minimum_improvement", 1e-5)):
            best = valid_loss
            stale = 0
            torch.save(payload, output_dir / "best.pt")
        else:
            stale += 1
        print(
            f"epoch={epoch:03d} train={train_loss:.6f} valid={valid_loss:.6f} "
            f"best={best:.6f}", flush=True,
        )
        if stale >= int(config.get("early_stopping_patience", 15)):
            break
    (output_dir / "training_summary.json").write_text(json.dumps({
        "schema": "direct_interaction_temporal_training_v1",
        "best_valid_loss": best, "last_epoch": rows[-1]["epoch"],
        "wall_time_seconds": time.perf_counter() - started,
        "dimensions": dimensions,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
