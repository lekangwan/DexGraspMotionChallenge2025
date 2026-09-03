#!/usr/bin/env python3
"""把参考成功、稳定抬升和严格运输成功整理成类别平衡课程数据。"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from geometry import object_points_in_initial_wrist  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hand", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--unfiltered-data-dir", type=Path, required=True)
    parser.add_argument("--filtered-data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    entries = {item["object_name"]: item for item in manifest["entries"]}
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    quality = {
        (item["object_name"], int(item["source_trajectory_index"])): item
        for item in audit["results"] if item["policy_split"] == "train"
    }
    mappings = json.loads(
        (args.unfiltered_data_dir / "mappings.json").read_text(encoding="utf-8")
    )
    object_by_id = {int(value): key for key, value in mappings["object_to_id"].items()}
    with np.load(args.unfiltered_data_dir / "train.npz", allow_pickle=False) as archive:
        data = {name: archive[name].copy() for name in archive.files}
    with np.load(args.filtered_data_dir / "geometry_normalization.npz", allow_pickle=False) as archive:
        norm = {name: archive[name].astype(np.float32) for name in archive.files}
    source_cache = {}
    records = []
    for trajectory_id in np.unique(data["trajectory_id"]):
        indices = np.flatnonzero(data["trajectory_id"] == trajectory_id)
        object_name = object_by_id[int(data["object_id"][indices[0]])]
        source_index = int(data["source_trajectory_index"][indices[0]])
        item = quality[(object_name, source_index)]
        if not bool(item["reference_isaac_success"]):
            continue
        if bool(item["training_eligible"]):
            tier, weight = 3, 1.0
        elif bool(item["stable_physics_success"]):
            tier, weight = 2, 0.5
        else:
            tier, weight = 1, 0.25
        initial = data["actions"][indices[0]].astype(np.float32).copy()
        initial[6:] = 0.0
        entry = entries[object_name]
        if object_name not in source_cache:
            source_cache[object_name] = np.load(
                entry["source_path"], allow_pickle=True
            ).item()
        source = source_cache[object_name]
        scale = float(np.asarray(source["obj_scale"])[source_index])
        rotation = np.asarray(source["obj_rotmat"])[source_index]
        points = object_points_in_initial_wrist(
            entry["object_asset_path"], scale, rotation, initial, 128, 0.005
        )
        sequence = (
            data["actions"][indices] - initial - norm["initial_delta_mean"]
        ) / norm["initial_delta_std"]
        records.append({
            "task": ((data["observations"][indices[0]] - norm["observation_mean"])
                     / norm["observation_std"])[-32:],
            "command": (initial - norm["initial_command_mean"])
                       / norm["initial_command_std"],
            "points": (points - norm["point_mean"]) / norm["point_std"],
            "sequence": sequence.reshape(-1),
            "category": int(data["category_id"][indices[0]]),
            "tier": tier, "weight": weight,
        })
    category_totals = {}
    for record in records:
        category_totals[record["category"]] = (
            category_totals.get(record["category"], 0.0) + record["weight"]
        )
    weights = np.asarray([
        record["weight"] / category_totals[record["category"]]
        for record in records
    ], dtype=np.float32)
    weights /= weights.mean()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        task=np.asarray([r["task"] for r in records], dtype=np.float32),
        command=np.asarray([r["command"] for r in records], dtype=np.float32),
        points=np.asarray([r["points"] for r in records], dtype=np.float32),
        sequence=np.asarray([r["sequence"] for r in records], dtype=np.float32),
        category=np.asarray([r["category"] for r in records], dtype=np.int64),
        tier=np.asarray([r["tier"] for r in records], dtype=np.int8),
        weight=weights,
    )
    print(json.dumps({
        "hand": args.hand, "trajectory_count": len(records),
        "category_count": len(category_totals),
        "tier_counts": {
            str(tier): sum(r["tier"] == tier for r in records)
            for tier in (1, 2, 3)
        },
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
