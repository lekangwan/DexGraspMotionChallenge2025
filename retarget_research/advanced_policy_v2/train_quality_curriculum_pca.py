#!/usr/bin/env python3
"""从现有PCA策略出发，先吸收近成功数据，再用严格成功数据收敛。"""

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from models import build_pca_model  # noqa: E402
from train_geometry_pca import coefficient_loss, load_trajectories  # noqa: E402


def weighted_loss(prediction, target, coefficient_std, weight):
    error = ((prediction - target) * coefficient_std).square().mean(dim=1)
    error = error / coefficient_std.square().mean()
    return (error * weight).sum() / weight.sum()


@torch.no_grad()
def evaluate(model, valid, coefficient_std, device):
    model.eval()
    task, command, points, interaction, target = valid
    prediction = model(
        task.to(device), command.to(device), points.to(device), None
    )
    return float(coefficient_loss(
        prediction, target.to(device), coefficient_std
    ))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--reference-epochs", type=int)
    parser.add_argument("--strict-epochs", type=int)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    torch.manual_seed(int(config.get("seed", 20260902)))
    device = torch.device(args.device)
    base = torch.load(config["base_checkpoint"], map_location="cpu")
    data_dir = Path(config["data_dir"])
    output_dir = args.output_dir or Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(data_dir / "geometry_normalization.npz", allow_pickle=False) as archive:
        normalization = {name: archive[name].astype(np.float32) for name in archive.files}
    valid_raw = load_trajectories(data_dir, "valid", normalization)
    pca_mean = np.asarray(base["pca_mean"], dtype=np.float32)
    components = np.asarray(base["pca_components"], dtype=np.float32)
    coefficient_mean = np.asarray(base["coefficient_mean"], dtype=np.float32)
    coefficient_std = np.asarray(base["coefficient_std"], dtype=np.float32)

    def targets(sequence):
        coefficients = (sequence - pca_mean) @ components.T
        return ((coefficients - coefficient_mean) / coefficient_std).astype(np.float32)

    valid_target = targets(valid_raw[4])
    valid = tuple(torch.from_numpy(value) for value in (
        valid_raw[0], valid_raw[1], valid_raw[2], valid_raw[3], valid_target,
    ))
    with np.load(config["curriculum_data"], allow_pickle=False) as archive:
        curriculum = {name: archive[name].copy() for name in archive.files}
    target = targets(curriculum["sequence"])
    model = build_pca_model(
        base["config"], base["dimensions"]["task_observation_dim"],
        base["dimensions"]["action_dim"],
    ).to(device)
    model.load_state_dict(base["model_state"])
    coefficient_std_tensor = torch.from_numpy(coefficient_std).to(device)
    baseline = evaluate(model, valid, coefficient_std_tensor, device)
    best, best_state, history = baseline, {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }, [[0, "baseline", baseline]]

    for stage, mask, epochs, learning_rate in (
        ("reference", np.ones(len(target), dtype=bool),
         args.reference_epochs or int(config.get("reference_epochs", 60)),
         float(config.get("reference_learning_rate", 1e-4))),
        ("strict", curriculum["tier"] == 3,
         args.strict_epochs or int(config.get("strict_epochs", 80)),
         float(config.get("strict_learning_rate", 5e-5))),
    ):
        tensors = [
            torch.from_numpy(curriculum[name][mask])
            for name in ("task", "command", "points")
        ] + [torch.from_numpy(target[mask]), torch.from_numpy(curriculum["weight"][mask])]
        loader = DataLoader(
            TensorDataset(*tensors),
            batch_size=min(int(config.get("batch_size", 64)), int(mask.sum())),
            shuffle=True,
        )
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate,
            weight_decay=float(config.get("weight_decay", 1e-3)),
        )
        stale = 0
        for epoch in range(1, epochs + 1):
            model.train()
            for task, command, points, batch_target, weight in loader:
                points = points.to(device)
                jitter = float(config.get("point_jitter_std", 0.02))
                if jitter:
                    points = points + jitter * torch.randn_like(points)
                prediction = model(
                    task.to(device), command.to(device), points, None
                )
                loss = weighted_loss(
                    prediction, batch_target.to(device),
                    coefficient_std_tensor, weight.to(device),
                )
                optimizer.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            if epoch != 1 and epoch % 5:
                continue
            valid_loss = evaluate(model, valid, coefficient_std_tensor, device)
            history.append([epoch, stage, valid_loss])
            if valid_loss < best - 1e-5:
                best, stale = valid_loss, 0
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
            else:
                stale += 5
            if epoch == 1 or epoch % 20 == 0:
                print(
                    f"stage={stage} epoch={epoch:03d} "
                    f"valid={valid_loss:.6f} best={best:.6f}", flush=True,
                )
            if stale >= int(config.get("early_stopping_patience", 30)):
                break
        model.load_state_dict(best_state)
    payload = dict(base)
    payload["config"] = dict(
        base["config"], model_type="geometry_pca_quality_curriculum",
        quality_curriculum=config,
    )
    payload["model_state"] = best_state
    payload["best_valid_loss"] = best
    payload["quality_curriculum_baseline_valid_loss"] = baseline
    torch.save(payload, output_dir / "best.pt")
    with (output_dir / "metrics.csv").open("w", newline="") as stream:
        writer = csv.writer(stream); writer.writerow(["epoch", "stage", "valid_loss"]); writer.writerows(history)
    (output_dir / "training_summary.json").write_text(json.dumps({
        "baseline_valid_loss": baseline, "best_valid_loss": best,
        "relative_improvement": (baseline - best) / baseline,
        "reference_trajectory_count": int(len(target)),
        "strict_trajectory_count": int((curriculum["tier"] == 3).sum()),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(json.loads((output_dir / "training_summary.json").read_text()), ensure_ascii=False))


if __name__ == "__main__":
    main()
