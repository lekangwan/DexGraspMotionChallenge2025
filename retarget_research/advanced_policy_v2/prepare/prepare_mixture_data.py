#!/usr/bin/env python3
"""把全部成功/失败重定向轨迹整理为多候选生成与质量判别数据。"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

MODULE = Path(__file__).resolve().parents[1]
PROJECT = MODULE.parents[1]
sys.path.insert(0, str(MODULE))
sys.path.insert(0, str(PROJECT))
from geometry import object_points_in_initial_wrist  # noqa: E402
from retarget_research.advanced_policy.observations import (  # noqa: E402
    build_object_shape_descriptor, build_observation_batch,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hand", choices=["linker", "xhand", "wuji"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    final = PROJECT / "retarget_research/outputs/reboot_synergy_rank5_formal1000_v1/postconfirmed_rank5_v1"
    data_dir = MODULE / f"data/final/{args.hand}"
    audit = json.loads((final / f"audit/{args.hand}_stable_audit.json").read_text(encoding="utf-8"))
    learnability = json.loads((MODULE / "data/final/EXPERT_LEARNABILITY_AUDIT.json").read_text(encoding="utf-8"))
    quality = {
        (row["object_name"], int(row["source_trajectory_index"])): row["learnability_score"]
        for row in learnability["hands"][args.hand]["results"]
    }
    manifest = json.loads((final / f"manifests/{args.hand}.json").read_text(encoding="utf-8"))
    entries = {row["object_name"]: row for row in manifest["entries"]}
    with np.load(data_dir / "geometry_normalization.npz", allow_pickle=False) as archive:
        norm = {name: archive[name].astype(np.float32) for name in archive.files}
    pca = torch.load(
        MODULE / f"runs/candidates_v1/{args.hand}/geometry_pca32/best.pt",
        map_location="cpu",
    )
    pca_mean = np.asarray(pca["pca_mean"], dtype=np.float32)
    components = np.asarray(pca["pca_components"], dtype=np.float32)
    coefficient_mean = np.asarray(pca["coefficient_mean"], dtype=np.float32)
    coefficient_std = np.asarray(pca["coefficient_std"], dtype=np.float32)
    source_cache = {}
    output = {"train": [], "valid": []}
    for item in audit["results"]:
        split = item.get("policy_split")
        if split not in output:
            continue
        name = item["object_name"]
        source_index = int(item["source_trajectory_index"])
        entry = entries[name]
        if name not in source_cache:
            source_cache[name] = np.load(entry["source_path"], allow_pickle=True).item()
        source = source_cache[name]
        scale = float(np.asarray(source["obj_scale"])[source_index])
        rotation = np.asarray(source["obj_rotmat"])[source_index]
        with np.load(item["policy_trace"], allow_pickle=False) as trace:
            arrays = {key: trace[key].copy() for key in trace.files if key != "metadata_json"}
        initial = arrays["policy_action"][0].astype(np.float32).copy()
        initial[6:] = 0.0
        points = object_points_in_initial_wrist(
            entry["object_asset_path"], scale, rotation, initial,
            int(pca["dimensions"]["point_count"]),
        )
        shape = build_object_shape_descriptor(
            Path(entry["object_asset_path"]) / "coacd/decomposed.obj", scale
        )
        observation = build_observation_batch(
            arrays["hand_dof_position"], arrays["hand_dof_velocity"],
            arrays["object_position"], arrays["object_quaternion_xyzw"],
            arrays["object_linear_velocity"], arrays["object_angular_velocity"],
            arrays["object_position"][0], arrays["hand_object_contact_count"],
            shape, 0.15,
        )[0]
        sequence = (
            arrays["policy_action"] - initial - norm["initial_delta_mean"]
        ) / norm["initial_delta_std"]
        coefficient = (sequence.reshape(-1) - pca_mean) @ components.T
        coefficient = (coefficient - coefficient_mean) / coefficient_std
        output[split].append({
            "task_observation": ((observation - norm["observation_mean"]) / norm["observation_std"])[-32:],
            "initial_command": (initial - norm["initial_command_mean"]) / norm["initial_command_std"],
            "object_points": (points - norm["point_mean"]) / norm["point_std"],
            "pca_coefficient": coefficient,
            "success": bool(item["training_eligible"]),
            "quality_weight": float(quality.get((name, source_index), 0.25)),
            "object_name": name,
            "source_index": source_index,
        })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for split, rows in output.items():
        np.savez_compressed(
            args.output_dir / f"mixture_{split}.npz",
            task_observation=np.asarray([row["task_observation"] for row in rows], np.float32),
            initial_command=np.asarray([row["initial_command"] for row in rows], np.float32),
            object_points=np.asarray([row["object_points"] for row in rows], np.float32),
            pca_coefficient=np.asarray([row["pca_coefficient"] for row in rows], np.float32),
            success=np.asarray([row["success"] for row in rows], bool),
            quality_weight=np.asarray([row["quality_weight"] for row in rows], np.float32),
            object_name=np.asarray([row["object_name"] for row in rows]),
            source_index=np.asarray([row["source_index"] for row in rows], np.int16),
        )
        summary[split] = {"count": len(rows), "success_count": sum(row["success"] for row in rows)}
    (args.output_dir / "mixture_summary.json").write_text(
        json.dumps({"hand": args.hand, "splits": summary}, indent=2) + "\n", encoding="utf-8"
    )
    print(args.hand, summary)


if __name__ == "__main__":
    main()
