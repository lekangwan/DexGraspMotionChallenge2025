#!/usr/bin/env python3
"""训练PCA多轨迹生成器和利用全部成功/失败标签的质量判别器。"""

import argparse
import copy
import csv
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score
from torch import nn

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from models import build_pca_mixture  # noqa: E402


def load(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def batches(size, batch_size, shuffle, generator):
    indices = np.arange(size)
    if shuffle:
        generator.shuffle(indices)
    for start in range(0, size, batch_size):
        yield indices[start:start + batch_size]


def compute_losses(model, data, labels, indices, device, coefficient_scale, pos_weight):
    task = torch.from_numpy(data["task_observation"][indices]).to(device)
    command = torch.from_numpy(data["initial_command"][indices]).to(device)
    points = torch.from_numpy(data["object_points"][indices]).to(device)
    target = torch.from_numpy(data["pca_coefficient"][indices]).to(device)
    success = torch.from_numpy(data["success"][indices].astype(np.float32)).to(device)
    condition, logits, candidates = model.generate(task, command, points)
    quality_logits = model.score(condition, target)
    critic = nn.functional.binary_cross_entropy_with_logits(
        quality_logits, success, pos_weight=pos_weight,
    )
    positive = success.bool()
    if positive.any():
        mode = torch.from_numpy(labels[indices][positive.cpu().numpy()]).long().to(device)
        selected = candidates[positive, mode]
        error = (selected - target[positive]) * coefficient_scale
        sample = error.square().mean(dim=-1)
        weight = torch.from_numpy(data["quality_weight"][indices][positive.cpu().numpy()]).to(device)
        regression = (sample * (0.5 + weight)).sum() / (0.5 + weight).sum()
        routing = nn.functional.cross_entropy(logits[positive], mode)
    else:
        regression = routing = critic.new_zeros(())
    total = regression + 0.2 * routing + 0.5 * critic
    return total, regression, routing, critic, quality_logits.detach(), success.detach()


def run_epoch(model, data, labels, device, coefficient_scale, pos_weight, optimizer, batch_size, seed):
    training = optimizer is not None
    model.train(training)
    generator = np.random.default_rng(seed)
    totals = np.zeros(4, dtype=np.float64); count = 0
    quality, truth = [], []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for indices in batches(len(data["success"]), batch_size, training, generator):
            values = compute_losses(model, data, labels, indices, device, coefficient_scale, pos_weight)
            if training:
                optimizer.zero_grad(); values[0].backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            totals += np.asarray([float(value) for value in values[:4]]) * len(indices)
            count += len(indices)
            quality.append(values[4].cpu().numpy()); truth.append(values[5].cpu().numpy())
    auc = roc_auc_score(np.concatenate(truth), np.concatenate(quality))
    return [*(totals / count), auc]


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
    data_dir = Path(config["data_dir"]); output_dir = args.output_dir or Path(config["output_dir"])
    train = load(data_dir / "mixture_train.npz"); valid = load(data_dir / "mixture_valid.npz")
    positive_coefficients = train["pca_coefficient"][train["success"]]
    kmeans = KMeans(n_clusters=int(config.get("mode_count", 4)), n_init=20, random_state=seed)
    kmeans.fit(positive_coefficients)
    labels = {}
    for name, data in (("train", train), ("valid", valid)):
        labels[name] = kmeans.predict(data["pca_coefficient"]).astype(np.int64)
    model = build_pca_mixture(
        config, train["task_observation"].shape[1], train["initial_command"].shape[1]
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config.get("learning_rate", 3e-4)),
        weight_decay=float(config.get("weight_decay", 1e-3)),
    )
    pca = torch.load(config["pca_checkpoint"], map_location="cpu")
    coefficient_scale = torch.from_numpy(np.asarray(pca["coefficient_std"], np.float32)).to(device)
    coefficient_scale = coefficient_scale / torch.sqrt(torch.mean(coefficient_scale.square()))
    positives = int(train["success"].sum()); negatives = len(train["success"]) - positives
    pos_weight = torch.tensor(negatives / max(positives, 1), dtype=torch.float32, device=device)
    output_dir.mkdir(parents=True, exist_ok=True)
    history, best, best_state, stale = [], float("inf"), None, 0
    epochs = args.epochs or int(config.get("epochs", 250))
    for epoch in range(1, epochs + 1):
        train_values = run_epoch(
            model, train, labels["train"], device, coefficient_scale, pos_weight,
            optimizer, int(config.get("batch_size", 64)), seed + epoch,
        )
        valid_values = run_epoch(
            model, valid, labels["valid"], device, coefficient_scale, pos_weight,
            None, len(valid["success"]), seed,
        )
        row = [epoch, *train_values, *valid_values]
        history.append(row)
        valid_objective = valid_values[0]
        if valid_objective < best - 1e-4:
            best = valid_objective; stale = 0
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0:
            print(f"epoch={epoch:03d} train={train_values[0]:.4f} valid={valid_values[0]:.4f} auc={valid_values[-1]:.3f}", flush=True)
        if stale >= int(config.get("early_stopping_patience", 30)): break
    base_payload = {
        "schema": "geometry_pca_mixture_policy_v1", "config": config,
        "dimensions": {"task_observation_dim": train["task_observation"].shape[1],
                       "observation_dim": int(pca["dimensions"]["observation_dim"]),
                       "action_dim": train["initial_command"].shape[1],
                       "point_count": train["object_points"].shape[1]},
        "model_state": best_state, "pca_mean": pca["pca_mean"],
        "pca_components": pca["pca_components"], "coefficient_mean": pca["coefficient_mean"],
        "coefficient_std": pca["coefficient_std"], "sequence_shape": pca["sequence_shape"],
        "cluster_centers": kmeans.cluster_centers_.astype(np.float32), "best_valid_loss": best,
    }
    for selection in ("gate", "critic"):
        payload = copy.deepcopy(base_payload)
        payload["config"] = dict(config, selection=selection, model_type=f"geometry_mixture_{selection}")
        directory = output_dir.parent / f"geometry_mixture_{selection}"
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(payload, directory / "best.pt")
        (directory / "config.json").write_text(json.dumps(payload["config"], indent=2) + "\n")
    fields = ["epoch", "train_total", "train_regression", "train_routing", "train_critic", "train_auc",
              "valid_total", "valid_regression", "valid_routing", "valid_critic", "valid_auc"]
    with (output_dir / "metrics.csv").open("w", newline="") as stream:
        writer = csv.writer(stream); writer.writerow(fields); writer.writerows(history)
    summary = {"best_valid_loss": best, "last_epoch": history[-1][0],
               "valid_auc_at_best_or_later": max(row[-1] for row in history),
               "mode_count": int(config.get("mode_count", 4))}
    (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"MIXTURE={output_dir} best={best:.4f}")


if __name__ == "__main__":
    main()
