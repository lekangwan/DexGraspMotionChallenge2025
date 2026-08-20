#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import transforms3d

from observations import build_object_shape_descriptor
from train_learned_mapping import R_ALIGN, MappingMLP


KEYPOINT_MAPS = {
    "xhand": "xhand_keypoint_map.json",
    "linker": "linker_o6_keypoint_map.json",
    "wuji": "wuji_keypoint_map.json",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    hand = checkpoint["hand"]
    joint_dim = checkpoint["joint_dim"]
    target_dim = joint_dim + 6
    model = MappingMLP(6 + 22 + 14, joint_dim).to(args.device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    input_mean = torch.as_tensor(checkpoint["input_mean"], device=args.device)
    input_std = torch.as_tensor(checkpoint["input_std"], device=args.device)
    output_mean = torch.as_tensor(checkpoint["output_mean"], device=args.device)
    output_std = torch.as_tensor(checkpoint["output_std"], device=args.device)
    train_names = set(checkpoint["train_names"])
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    retarget_root = Path(__file__).resolve().parents[1] / "retargeting"
    semantics = [
        pair["semantic"]
        for pair in json.loads(
            (retarget_root / "configs" / KEYPOINT_MAPS[hand]).read_text(encoding="utf-8")
        )["pairs"]
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    test_count = 0
    for entry in manifest["entries"]:
        name = entry["object_name"]
        if name in train_names:
            continue
        source = np.load(Path(entry["source_path"]), allow_pickle=True).item()
        indices = np.asarray(entry["trajectory_indices"], dtype=np.int64)
        outputs = []
        mesh_path = Path(entry["object_asset_path"]) / "coacd" / "decomposed.obj"
        for source_index in indices:
            source_frames = np.asarray(
                source["grasp_seqs"][source_index], dtype=np.float32).copy()
            source_frames[:, 2] += float(checkpoint["source_z_offset"])
            shape = build_object_shape_descriptor(
                mesh_path, float(np.asarray(source["obj_scale"])[source_index]))
            frames = np.empty((len(source_frames), target_dim), dtype=np.float32)
            for frame_index, frame in enumerate(source_frames):
                rotation = transforms3d.euler.euler2mat(*frame[3:6], axes="sxyz")
                euler = np.asarray(
                    transforms3d.euler.mat2euler(R_ALIGN @ rotation, axes="sxyz"),
                    dtype=np.float32)
                wrist = np.concatenate([frame[:3], euler]).astype(np.float32)
                x = np.concatenate([frame[6:], wrist, shape]).astype(np.float32)
                x_norm = (torch.as_tensor(x, device=args.device) - input_mean) / input_std
                with torch.no_grad():
                    y_norm = model(x_norm)
                joints = (y_norm * output_std + output_mean).cpu().numpy()
                frames[frame_index, :3] = frame[:3]
                frames[frame_index, 3:6] = euler
                frames[frame_index, 6:] = joints
            outputs.append(frames)
        output_frames = np.stack(outputs).astype(np.float32)
        output = {
            "grasp_seqs": output_frames,
            "optimization_loss_per_frame": np.zeros(
                (len(indices), len(outputs[0])), dtype=np.float32),
            "source_trajectory_indices": indices,
            "obj_rotmat": np.asarray(source["obj_rotmat"])[indices],
            "obj_scale": np.asarray(source["obj_scale"])[indices],
            "retarget_method": "learned_mapping_v1",
            "mapping_semantics": semantics,
            "source_z_offset": float(checkpoint["source_z_offset"]),
        }
        if hand == "wuji":
            output["wuji_joint_names"] = [
                f"finger{finger}_joint{joint}"
                for finger in range(1, 6) for joint in range(1, 5)
            ]
        np.save(args.output_dir / f"{name}.npy", output, allow_pickle=True)
        test_count += 1
    print(f"test_objects={test_count}")
    print(f"output={args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
