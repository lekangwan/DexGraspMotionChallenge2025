#!/usr/bin/env python3
"""训练纯参数化的条件PCA潜空间Diffusion轨迹生成策略。"""

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
from models import (  # noqa: E402
    build_pca_latent_diffusion, build_pca_model,
    sample_pca_latent_diffusion,
)
from train_geometry_pca import load_trajectories  # noqa: E402


def coefficient_errors(candidates, target, coefficient_std):
    """把标准化PCA系数误差按各主成分的真实尺度还原。"""
    difference = (candidates - target[:, None]) * coefficient_std
    return difference.square().mean(dim=-1) / coefficient_std.square().mean()


def diffusion_schedule(step_count, device):
    betas = torch.linspace(1e-4, 0.02, int(step_count), device=device)
    return torch.cumprod(1.0 - betas, dim=0)


@torch.no_grad()
def evaluate(model, base_model, tensors, alpha_bars, coefficient_std, config, device):
    model.eval()
    task, command, points, target = (value.to(device) for value in tensors)
    condition = model.encode(task, command, points)
    baseline = base_model(task, command, points, None)
    regression = model.regression(condition)
    generated = sample_pca_latent_diffusion(
        model, condition, alpha_bars,
        config.get("candidate_count", 8), config.get("sample_seed", 20260902),
    )
    candidates = torch.cat([
        baseline[:, None], regression[:, None], generated,
    ], dim=1)
    scores = model.score(condition, candidates)
    margin = float(config.get("selection_margin", 0.1))
    generated_best = scores[:, 1:].max(dim=1)
    use_generated = generated_best.values > scores[:, 0] + margin
    selected_index = torch.where(
        use_generated, generated_best.indices + 1,
        torch.zeros_like(generated_best.indices),
    )
    errors = coefficient_errors(candidates, target, coefficient_std)
    selected = errors.gather(1, selected_index[:, None]).mean()
    return {
        "base_recomputed_mse": float(errors[:, 0].mean()),
        "regression_mse": float(errors[:, 1].mean()),
        "selected_mse": float(selected),
        "oracle_mse": float(errors.min(dim=1).values.mean()),
        "alternative_selection_fraction": float(use_generated.float().mean()),
        "oracle_alternative_fraction": float((errors.argmin(dim=1) > 0).float().mean()),
        "candidate_diversity": float(torch.sqrt(
            (generated[:, :, None] - generated[:, None, :]).square().mean(dim=-1)
            + 1e-12
        ).mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device")
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    seed = int(config.get("seed", 20260902))
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device(args.device or config.get("device", "cuda"))
    data_dir = Path(config["data_dir"])
    output_dir = args.output_dir or Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    base = torch.load(config["base_checkpoint"], map_location="cpu")
    with np.load(data_dir / "geometry_normalization.npz", allow_pickle=False) as archive:
        normalization = {name: archive[name].astype(np.float32) for name in archive.files}
    train = load_trajectories(data_dir, "train", normalization)
    valid = load_trajectories(data_dir, "valid", normalization)
    pca_mean = np.asarray(base["pca_mean"], dtype=np.float32)
    components = np.asarray(base["pca_components"], dtype=np.float32)
    coefficient_mean = np.asarray(base["coefficient_mean"], dtype=np.float32)
    coefficient_std = np.asarray(base["coefficient_std"], dtype=np.float32)

    def targets(sequences):
        coefficients = (sequences - pca_mean) @ components.T
        return ((coefficients - coefficient_mean) / coefficient_std).astype(np.float32)

    train_target, valid_target = targets(train[4]), targets(valid[4])
    train_set = TensorDataset(*(torch.from_numpy(value) for value in (
        train[0], train[1], train[2], train_target,
    )))
    valid_tensors = tuple(torch.from_numpy(value) for value in (
        valid[0], valid[1], valid[2], valid_target,
    ))
    loader = DataLoader(
        train_set, batch_size=min(int(config.get("batch_size", 64)), len(train_set)),
        shuffle=True, drop_last=False,
    )
    model = build_pca_latent_diffusion(
        config, train[0].shape[1], train[1].shape[1]
    ).to(device)
    base_model = build_pca_model(
        base["config"], base["dimensions"]["task_observation_dim"],
        base["dimensions"]["action_dim"],
    )
    base_model.load_state_dict(base["model_state"])
    base_model.to(device).eval()
    for parameter in base_model.parameters():
        parameter.requires_grad_(False)
    model.points.load_state_dict(base_model.points.state_dict())
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config.get("learning_rate", 2e-4)),
        weight_decay=float(config.get("weight_decay", 1e-3)),
    )
    alpha_bars = diffusion_schedule(config.get("diffusion_steps", 50), device)
    coefficient_std_tensor = torch.from_numpy(coefficient_std).to(device)
    score_loss = nn.BCEWithLogitsLoss()
    best, best_state, stale, history = float("inf"), None, 0, []
    evaluation_interval = int(config.get("evaluation_interval", 5))
    epoch_count = args.max_epochs or int(config.get("epochs", 400))
    for epoch in range(1, epoch_count + 1):
        model.train(); sums = np.zeros(4, dtype=np.float64)
        for task, command, points, target in loader:
            task, command, points, target = (
                value.to(device) for value in (task, command, points, target)
            )
            jitter = float(config.get("point_jitter_std", 0.0))
            if jitter:
                points = points + jitter * torch.randn_like(points)
            condition = model.encode(task, command, points)
            indices = torch.randint(len(alpha_bars), (len(target),), device=device)
            alpha = alpha_bars[indices, None]
            noise = torch.randn_like(target)
            noisy = torch.sqrt(alpha) * target + torch.sqrt(1.0 - alpha) * noise
            time = indices[:, None].float() / float(max(len(alpha_bars) - 1, 1))
            diffusion = (model.predict_noise(condition, noisy, time) - noise).square().mean()
            regression = coefficient_errors(
                model.regression(condition)[:, None], target,
                coefficient_std_tensor,
            ).mean()
            negative = torch.roll(target, shifts=1, dims=0)
            candidates = torch.cat([target, negative, noisy.detach()], dim=0)
            conditions = torch.cat([condition, condition, condition], dim=0)
            labels = torch.cat([
                torch.ones(len(target), device=device),
                torch.zeros(len(target), device=device),
                torch.zeros(len(target), device=device),
            ])
            compatibility = score_loss(model.score(conditions, candidates), labels)
            loss = diffusion + float(config.get("regression_weight", 0.25)) * regression
            loss = loss + float(config.get("compatibility_weight", 0.1)) * compatibility
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            sums += np.asarray([
                float(loss), float(diffusion), float(regression), float(compatibility),
            ]) * len(target)
        if epoch != 1 and epoch % evaluation_interval:
            continue
        metrics = evaluate(
            model, base_model, valid_tensors, alpha_bars, coefficient_std_tensor,
            config, device,
        )
        row = [epoch, *(sums / len(train_set)), *metrics.values()]
        history.append(row)
        criterion = metrics["selected_mse"]
        if criterion < best - 1e-5:
            best, stale = criterion, 0
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        else:
            stale += evaluation_interval
        if epoch == 1 or epoch % 20 == 0:
            print(
                f"epoch={epoch:03d} train={row[1]:.4f} "
                f"selected={metrics['selected_mse']:.4f} "
                f"oracle={metrics['oracle_mse']:.4f} "
                f"base={float(base['best_valid_loss']):.4f}", flush=True,
            )
        if stale >= int(config.get("early_stopping_patience", 45)):
            break
    model.load_state_dict(best_state)
    final = evaluate(
        model, base_model, valid_tensors, alpha_bars, coefficient_std_tensor,
        config, device,
    )
    payload = {
        "schema": "geometry_pca_latent_diffusion_policy_v1",
        "config": config,
        "dimensions": base["dimensions"],
        "model_state": best_state,
        "base_model_config": base["config"],
        "base_model_state": base["model_state"],
        "pca_mean": pca_mean, "pca_components": components,
        "coefficient_mean": coefficient_mean,
        "coefficient_std": coefficient_std,
        "sequence_shape": base["sequence_shape"],
        "alpha_bars": alpha_bars.detach().cpu(),
        "offline_metrics": dict(final, base_valid_mse=float(base["best_valid_loss"])),
    }
    torch.save(payload, output_dir / "best.pt")
    headers = [
        "epoch", "train_total", "train_diffusion", "train_regression",
        "train_compatibility", *final.keys(),
    ]
    with (output_dir / "metrics.csv").open("w", newline="") as stream:
        writer = csv.writer(stream); writer.writerow(headers); writer.writerows(history)
    (output_dir / "training_summary.json").write_text(
        json.dumps(payload["offline_metrics"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("DIFFUSION_POLICY", output_dir / "best.pt")
    print(json.dumps(payload["offline_metrics"], ensure_ascii=False))


if __name__ == "__main__":
    main()
