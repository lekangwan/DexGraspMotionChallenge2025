#!/usr/bin/env python3
"""为已有策略NPZ增加初始手腕坐标系物体点云和正确的v3专家筛选。"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np

MODULE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = MODULE_ROOT.parents[1]
sys.path.insert(0, str(MODULE_ROOT))

from geometry import object_points_in_initial_wrist  # noqa: E402


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def audit_training_keys(audit_path):
    """读取v3审计，返回允许进入监督训练的物体—轨迹键。"""
    audit = load_json(audit_path)
    if int(audit.get("schema_version", 0)) < 3:
        raise ValueError("训练专家必须来自v3稳定运输审计")
    return {
        (item["object_name"], int(item["source_trajectory_index"]))
        for item in audit["results"]
        if bool(item.get("training_eligible", False))
    }


def filter_split(data, keep_trajectory_ids):
    """按完整轨迹过滤逐步NPZ，避免只删掉轨迹中的部分帧。"""
    mask = np.isin(data["trajectory_id"], np.asarray(sorted(keep_trajectory_ids), dtype=np.int64))
    return {name: value[mask] for name, value in data.items()}


def write_npz(path, arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def prepare(args):
    """复制基础NPZ、按v3质量门重筛train/valid并生成几何sidecar。"""
    manifest = load_json(args.manifest)
    mappings = load_json(args.base_data_dir / "mappings.json")
    entries = {item["object_name"]: item for item in manifest["entries"]}
    id_to_object = {int(value): name for name, value in mappings["object_to_id"].items()}
    eligible = audit_training_keys(args.audit)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "mappings.json").write_text(
        json.dumps(mappings, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    source_cache = {}
    train_initial_commands = []
    train_points = []
    train_relative_actions = []
    summaries = {}

    for split in ("train", "valid", "test"):
        with np.load(args.base_data_dir / f"{split}.npz", allow_pickle=False) as archive:
            data = {name: archive[name].copy() for name in archive.files}
        groups = {
            int(value): np.flatnonzero(data["trajectory_id"] == value)
            for value in np.unique(data["trajectory_id"])
        }
        if split in {"train", "valid"}:
            keep = set()
            for trajectory_id, indices in groups.items():
                object_name = id_to_object[int(data["object_id"][indices[0]])]
                source_index = int(data["source_trajectory_index"][indices[0]])
                if (object_name, source_index) in eligible:
                    keep.add(trajectory_id)
            data = filter_split(data, keep)
            groups = {
                int(value): np.flatnonzero(data["trajectory_id"] == value)
                for value in np.unique(data["trajectory_id"])
            }
        if not groups:
            raise ValueError(f"{split}经过v3质量门后没有轨迹")
        v3_labels = np.zeros(len(data["actions"]), dtype=bool)
        for trajectory_id, indices in groups.items():
            object_name = id_to_object[int(data["object_id"][indices[0]])]
            source_index = int(data["source_trajectory_index"][indices[0]])
            v3_labels[indices] = (object_name, source_index) in eligible
        data["expert_replay_success"] = v3_labels
        write_npz(args.output_dir / f"{split}.npz", data)

        ids, commands, clouds = [], [], []
        for trajectory_id, indices in groups.items():
            object_name = id_to_object[int(data["object_id"][indices[0]])]
            source_index = int(data["source_trajectory_index"][indices[0]])
            entry = entries[object_name]
            if object_name not in source_cache:
                source_cache[object_name] = np.load(entry["source_path"], allow_pickle=True).item()
            source = source_cache[object_name]
            scale = float(np.asarray(source["obj_scale"])[source_index])
            rotation = np.asarray(source["obj_rotmat"])[source_index]
            initial_command = data["actions"][indices[0]].astype(np.float32).copy()
            initial_command[6:] = 0.0
            cloud = object_points_in_initial_wrist(
                entry["object_asset_path"], scale, rotation, initial_command,
                args.point_count, args.clearance,
            )
            ids.append(trajectory_id)
            commands.append(initial_command)
            clouds.append(cloud)
            if split == "train":
                train_initial_commands.append(initial_command)
                train_points.append(cloud)
                train_relative_actions.append(data["actions"][indices] - initial_command)
        write_npz(
            args.output_dir / f"geometry_{split}.npz",
            {
                "trajectory_id": np.asarray(ids, dtype=np.int64),
                "initial_command": np.asarray(commands, dtype=np.float32),
                "object_points": np.asarray(clouds, dtype=np.float32),
            },
        )
        summaries[split] = {
            "trajectory_count": len(ids),
            "step_count": len(data["actions"]),
        }

    with np.load(args.base_data_dir / "normalization.npz", allow_pickle=False) as archive:
        normalization = {name: archive[name].copy() for name in archive.files}
    commands = np.asarray(train_initial_commands, dtype=np.float32)
    points = np.concatenate(train_points, axis=0).astype(np.float32)
    relative = np.concatenate(train_relative_actions, axis=0).astype(np.float32)
    normalization.update({
        "initial_command_mean": commands.mean(axis=0),
        "initial_command_std": np.maximum(commands.std(axis=0), 1e-5),
        "point_mean": points.mean(axis=0),
        "point_std": np.maximum(points.std(axis=0), 1e-5),
        "initial_delta_mean": relative.mean(axis=0),
        "initial_delta_std": np.maximum(relative.std(axis=0), 1e-5),
    })
    write_npz(args.output_dir / "geometry_normalization.npz", normalization)
    summary = {
        "schema_version": 1,
        "hand": args.hand,
        "quality_rule": "train_valid_use_v3_training_eligible; test_unfiltered",
        "category_id_used": False,
        "point_count": int(args.point_count),
        "splits": summaries,
        "audit": str(args.audit.resolve()),
    }
    (args.output_dir / "geometry_dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hand", choices=["linker", "xhand", "wuji"], required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--base-data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--point-count", type=int, default=128)
    parser.add_argument("--clearance", type=float, default=0.005)
    args = parser.parse_args()
    for name in ("manifest", "audit", "base_data_dir", "output_dir"):
        path = getattr(args, name)
        setattr(args, name, path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve())
    summary = prepare(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
