#!/usr/bin/env python3
"""在初始手腕近邻中拟合局部线性关系并生成整段轨迹。"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def read_trajectories(path, normalization):
    with np.load(path, allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    features, sequences = [], []
    for trajectory_id in np.unique(data["trajectory_id"]):
        indices = np.flatnonzero(data["trajectory_id"] == trajectory_id)
        actions = data["actions"][indices].astype(np.float32)
        initial = actions[0].copy()
        initial[6:] = 0.0
        features.append(
            ((data["observations"][indices[0]] - normalization["observation_mean"])
             / normalization["observation_std"])[:6]
        )
        sequences.append(
            (actions - initial - normalization["initial_delta_mean"])
            / normalization["initial_delta_std"]
        )
    return np.stack(features).astype(np.float32), np.stack(sequences).astype(np.float32)


def predict(train_x, train_y, query, k, ridge):
    distance = np.sum((train_x - query[None]) ** 2, axis=1)
    nearest = np.argsort(distance)[:k]
    design = np.concatenate([
        np.ones((len(nearest), 1)), train_x[nearest] - query[None]
    ], axis=1)
    weights = 1.0 / (distance[nearest] + 1e-3)
    normal = design.T @ (weights[:, None] * design)
    normal += ridge * np.diag([0.0] + [1.0] * 6)
    target = train_y[nearest].reshape(len(nearest), -1)
    return (np.linalg.pinv(normal) @ (design.T @ (weights[:, None] * target)))[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--local-k", type=int)
    parser.add_argument("--ridge", type=float)
    args = parser.parse_args()
    with np.load(args.data_dir / "normalization.npz", allow_pickle=False) as archive:
        normalization = {name: archive[name].astype(np.float32) for name in archive.files}
    train_x, train_y = read_trajectories(args.data_dir / "train.npz", normalization)
    valid_x, valid_y = read_trajectories(args.data_dir / "valid.npz", normalization)
    candidates, best = [], None
    for k in (7, 10, 15, 20, 30):
        k = min(k, len(train_x))
        for ridge in (0.01, 0.1, 1.0, 10.0):
            prediction = np.stack([
                predict(train_x, train_y, query, k, ridge) for query in valid_x
            ])
            mse = float(np.mean(
                (prediction - valid_y.reshape(len(valid_y), -1)) ** 2
            ))
            candidates.append({"k": k, "ridge": ridge, "mse": mse})
            if best is None or mse < best[0]:
                best = (mse, k, ridge)
    if args.local_k is not None or args.ridge is not None:
        if args.local_k is None or args.ridge is None:
            raise ValueError("固定候选时必须同时提供--local-k和--ridge")
        k = min(args.local_k, len(train_x))
        prediction = np.stack([
            predict(train_x, train_y, query, k, args.ridge) for query in valid_x
        ])
        best = (float(np.mean(
            (prediction - valid_y.reshape(len(valid_y), -1)) ** 2
        )), k, args.ridge)
    mse, k, ridge = best
    mappings = json.loads((args.data_dir / "mappings.json").read_text(encoding="utf-8"))
    payload = {
        "config": {"model_type": "trajectory_local_ridge", "local_k": k, "ridge": ridge},
        "dimensions": {"observation_dim": len(normalization["observation_mean"]),
                       "action_dim": train_y.shape[2],
                       "category_count": len(mappings["category_to_id"])},
        "train_features": train_x,
        "train_sequences": train_y,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    args.output.with_suffix(".json").write_text(json.dumps({
        "selected_valid_mse": mse, "k": k, "ridge": ridge,
        "candidates": candidates,
    }, indent=2) + "\n")
    print(f"TRAJECTORY_LOCAL_RIDGE={args.output} mse={mse:.6f} k={k} ridge={ridge}")


if __name__ == "__main__":
    main()
