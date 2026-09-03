#!/usr/bin/env python3
"""用PCA压缩整段轨迹，并以初始手腕RBF岭回归预测运动系数。"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def trajectories(data, normalization):
    observations, sequences = [], []
    for trajectory_id in np.unique(data["trajectory_id"]):
        indices = np.flatnonzero(data["trajectory_id"] == trajectory_id)
        actions = data["actions"][indices].astype(np.float32)
        initial = actions[0].copy()
        initial[6:] = 0.0
        observations.append(
            ((data["observations"][indices[0]] - normalization["observation_mean"])
             / normalization["observation_std"])[:6]
        )
        sequences.append(
            (actions - initial - normalization["initial_delta_mean"])
            / normalization["initial_delta_std"]
        )
    return np.stack(observations), np.stack(sequences)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.data_dir / "normalization.npz", allow_pickle=False) as archive:
        normalization = {name: archive[name].astype(np.float32) for name in archive.files}
    splits = {}
    for split in ("train", "valid"):
        with np.load(args.data_dir / f"{split}.npz", allow_pickle=False) as archive:
            splits[split] = {name: archive[name] for name in archive.files}
    train_x, train_y = trajectories(splits["train"], normalization)
    valid_x, valid_y = trajectories(splits["valid"], normalization)
    shape = train_y.shape[1:]
    flat = train_y.reshape(len(train_y), -1)
    mean = flat.mean(axis=0)
    _, _, vt = np.linalg.svd(flat - mean, full_matrices=False)
    pair_distance = np.sum((train_x[:, None] - train_x[None]) ** 2, axis=2)
    median_distance = float(np.median(pair_distance[pair_distance > 0]))
    candidates = []
    best = None
    for rank in (8, 16, 32):
        rank = min(rank, len(train_x) - 1)
        components = vt[:rank]
        coefficients = (flat - mean) @ components.T
        for sigma_scale in (0.5, 1.0, 2.0):
            sigma = max(np.sqrt(median_distance) * sigma_scale, 1e-3)
            kernel = np.exp(-pair_distance / (2.0 * sigma ** 2))
            for ridge in (1e-3, 1e-2, 1e-1):
                dual = np.linalg.solve(
                    kernel + ridge * np.eye(len(kernel)), coefficients
                )
                valid_distance = np.sum(
                    (valid_x[:, None] - train_x[None]) ** 2, axis=2
                )
                valid_kernel = np.exp(-valid_distance / (2.0 * sigma ** 2))
                prediction = mean + (valid_kernel @ dual) @ components
                mse = float(np.mean((prediction - valid_y.reshape(len(valid_y), -1)) ** 2))
                item = (mse, rank, sigma, ridge, components.copy(), dual.copy())
                candidates.append({"mse": mse, "rank": rank, "sigma": sigma, "ridge": ridge})
                if best is None or mse < best[0]:
                    best = item
    mse, rank, sigma, ridge, components, dual = best
    mappings = json.loads((args.data_dir / "mappings.json").read_text(encoding="utf-8"))
    payload = {
        "config": {"model_type": "trajectory_pca_rbf", "rbf_sigma": sigma,
                   "pca_rank": rank, "ridge": ridge},
        "dimensions": {"observation_dim": len(normalization["observation_mean"]),
                       "action_dim": shape[1],
                       "category_count": len(mappings["category_to_id"])},
        "rbf_train_features": train_x.astype(np.float32),
        "rbf_dual": dual.astype(np.float32),
        "pca_mean": mean.astype(np.float32),
        "pca_components": components.astype(np.float32),
        "sequence_shape": shape,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    args.output.with_suffix(".json").write_text(
        json.dumps({"selected_valid_mse": mse, "rank": rank, "sigma": sigma,
                    "ridge": ridge, "candidates": candidates}, indent=2) + "\n"
    )
    print(f"TRAJECTORY_PCA_RBF={args.output} mse={mse:.6f} rank={rank}")


if __name__ == "__main__":
    main()
