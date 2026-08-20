#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import transforms3d

from observations import build_object_shape_descriptor


R_ALIGN = np.array(
    [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32
)


class MappingMLP(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden=64):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def load_pairs(target_dir, manifest, target_dim, hand):
    pairs = []
    for entry in manifest["entries"]:
        source = np.load(Path(entry["source_path"]), allow_pickle=True).item()
        target = np.load(Path(target_dir) / f"{entry['object_name']}.npy",
                         allow_pickle=True).item()
        for i, source_index in enumerate(entry["trajectory_indices"]):
            source_frames = np.asarray(source["grasp_seqs"][source_index], dtype=np.float32)
            target_frames = np.asarray(target["grasp_seqs"][i], dtype=np.float32)
            if target_frames.shape[1] != target_dim:
                raise ValueError(f"目标维度错误: {target_frames.shape}")
            pairs.append((entry["object_name"], int(source_index),
                          source_frames, target_frames))
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hand", choices=("xhand", "linker", "wuji"), required=True)
    parser.add_argument("--source-z-offset", type=float, default=0.4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--train-fraction", type=float, default=0.8)
    args = parser.parse_args()
    args.device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    target_dim = {"xhand": 18, "linker": 12, "wuji": 26}[args.hand]
    joint_dim = {"xhand": 12, "linker": 6, "wuji": 20}[args.hand]
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    pairs = load_pairs(args.target_dir, manifest, target_dim, args.hand)
    names = sorted({name for name, _, _, _ in pairs})
    split_index = int(len(names) * args.train_fraction)
    train_names = set(names[:split_index])
    inputs, outputs = [], []
    shape_cache = {}
    for entry in manifest["entries"]:
        source = np.load(Path(entry["source_path"]), allow_pickle=True).item()
        mesh_path = Path(entry["object_asset_path"]) / "coacd" / "decomposed.obj"
        for source_index in entry["trajectory_indices"]:
            scale = float(np.asarray(source["obj_scale"])[source_index])
            shape_cache[(entry["object_name"], int(source_index))] = (
                build_object_shape_descriptor(mesh_path, scale))
    for name, source_index, source_frames, target_frames in pairs:
        shape = shape_cache[(name, source_index)]
        for frame_index, frame in enumerate(source_frames):
            frame = frame.copy()
            frame[2] += args.source_z_offset
            rotation = transforms3d.euler.euler2mat(*frame[3:6], axes="sxyz")
            euler = np.asarray(
                transforms3d.euler.mat2euler(R_ALIGN @ rotation, axes="sxyz"),
                dtype=np.float32)
            wrist = np.concatenate([frame[:3], euler]).astype(np.float32)
            joints = target_frames[frame_index, 6:]
            inputs.append(np.concatenate([frame[6:], wrist, shape]))
            outputs.append(joints)
    inputs = np.stack(inputs).astype(np.float32)
    outputs = np.stack(outputs).astype(np.float32)
    input_mean = inputs.mean(0)
    input_std = inputs.std(0) + 1e-6
    output_mean = outputs.mean(0)
    output_std = outputs.std(0) + 1e-6
    x_norm = (inputs - input_mean) / input_std
    y_norm = (outputs - output_mean) / output_std
    train_mask = np.array([name in train_names for name, _, _, source_frames in pairs
                           for _ in range(len(source_frames))], dtype=bool)
    model = MappingMLP(inputs.shape[1], joint_dim).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.MSELoss()
    x_tensor = torch.as_tensor(x_norm, dtype=torch.float32, device=args.device)
    y_tensor = torch.as_tensor(y_norm, dtype=torch.float32, device=args.device)
    train_idx = torch.as_tensor(np.flatnonzero(train_mask), device=args.device)
    best_loss = float("inf")
    best_state = None
    for epoch in range(args.epochs):
        model.train()
        permutation = train_idx[torch.randperm(len(train_idx), device=args.device)]
        total = 0.0
        count = 0
        for start in range(0, len(permutation), 512):
            batch = permutation[start:start + 512]
            optimizer.zero_grad()
            loss = loss_fn(model(x_tensor[batch]), y_tensor[batch])
            loss.backward()
            optimizer.step()
            total += loss.item() * len(batch)
            count += len(batch)
        if (epoch + 1) % 50 == 0:
            print(f"epoch {epoch + 1}: train_mse={total / count:.6f}", flush=True)
        if total / count < best_loss:
            best_loss = total / count
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": best_state,
        "input_mean": input_mean,
        "input_std": input_std,
        "output_mean": output_mean,
        "output_std": output_std,
        "train_names": sorted(train_names),
        "hand": args.hand,
        "joint_dim": joint_dim,
        "source_z_offset": args.source_z_offset,
    }, args.output)
    print(f"best_train_mse={best_loss:.6f}")
    print(f"output={args.output.resolve()}")


if __name__ == "__main__":
    main()
