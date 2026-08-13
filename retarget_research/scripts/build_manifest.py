#!/usr/bin/env python3
"""从真实类别inventory冻结可直接批量运行的正式重定向manifest。

输入：每行含物体ID、官方类别、Shadow轨迹npy、碰撞资产目录和轨迹数的CSV。
输出：`entries`协议的50类、100物体、1000轨迹JSON，含哈希和calibration/heldout划分。
内部逻辑：先核对文件、资产和npy字段/形状，再按固定seed逐级抽类别、物体和轨迹。
作用：让三只手run/evaluate入口读取同一份不可因实验结果更换的正式样本名单。
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import random

import numpy as np


REQUIRED_COLUMNS = {
    "object_id",
    "category",
    "trajectory_file",
    "asset_dir",
    "trajectory_count",
}


def sha256(path):
    """分块计算轨迹文件SHA-256。

    输入：本地文件路径。
    输出：64字符十六进制摘要。
    内部逻辑：每次读取1 MiB，避免把大npy整体载入哈希器。
    作用：批处理运行前可检测正式数据是否在抽样后被替换。
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_trajectory_file(path, declared_count):
    """验证一个GraspM3物体文件的必需字段和逐轨迹形状。

    输入：npy路径和inventory声明的轨迹数。
    输出：实际轨迹数、帧数和动作维度。
    内部逻辑：要求字典含`grasp_seqs/obj_scale/obj_rotmat`，动作严格为`(N,70,28)`。
    作用：在抽样前阻止缺字段、数量错误或非标准轨迹进入昂贵批处理。
    """
    data = np.load(path, allow_pickle=True).item()
    required = {"grasp_seqs", "obj_scale", "obj_rotmat"}
    missing = sorted(required - set(data))
    if missing:
        raise ValueError(f"{path}缺少字段: {missing}")
    frames = np.asarray(data["grasp_seqs"])
    if frames.ndim != 3 or frames.shape[1:] != (70, 28):
        raise ValueError(f"{path}的grasp_seqs应为(N,70,28)，实际为{frames.shape}")
    actual_count = int(frames.shape[0])
    if actual_count != int(declared_count):
        raise ValueError(
            f"{path}轨迹数声明{declared_count}，实际{actual_count}"
        )
    for field in ("obj_scale", "obj_rotmat"):
        if len(np.asarray(data[field])) != actual_count:
            raise ValueError(f"{path}的{field}长度不等于轨迹数{actual_count}")
    return actual_count, int(frames.shape[1]), int(frames.shape[2])


def load_inventory(path):
    """读取、规范化并完整验证inventory。

    输入：CSV路径。
    输出：每行含绝对路径、实际形状和文件哈希的物体字典列表。
    内部逻辑：检查表头、非空类别/ID、ID唯一、文件/资产存在并调用npy检查。
    作用：确保类别只能来自显式官方标签，不从易歧义的文件名自动猜测。
    """
    rows = []
    seen_ids = set()
    with Path(path).expanduser().open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        missing_columns = sorted(REQUIRED_COLUMNS - columns)
        if missing_columns:
            raise ValueError(f"inventory缺少列: {missing_columns}")
        for line_number, row in enumerate(reader, start=2):
            object_id = row["object_id"].strip()
            category = row["category"].strip()
            if not object_id or not category:
                raise ValueError(f"inventory第{line_number}行物体ID或类别为空")
            if object_id in seen_ids:
                raise ValueError(f"inventory物体ID重复: {object_id}")
            seen_ids.add(object_id)
            trajectory_file = Path(row["trajectory_file"]).expanduser().resolve()
            asset_dir = Path(row["asset_dir"]).expanduser().resolve()
            if not trajectory_file.is_file():
                raise FileNotFoundError(trajectory_file)
            if not asset_dir.is_dir():
                raise FileNotFoundError(asset_dir)
            required_assets = (
                asset_dir / "coacd" / "coacd_1.urdf",
                asset_dir / "coacd" / "decomposed.obj",
            )
            missing_assets = [str(item) for item in required_assets if not item.is_file()]
            if missing_assets:
                raise FileNotFoundError(
                    f"{object_id}缺少物理/表面资产: {missing_assets}"
                )
            declared_count = int(row["trajectory_count"])
            actual_count, frame_count, action_dimension = inspect_trajectory_file(
                trajectory_file, declared_count
            )
            rows.append(
                {
                    "object_id": object_id,
                    "category": category,
                    "source_path": str(trajectory_file),
                    "source_sha256": sha256(trajectory_file),
                    "object_asset_path": str(asset_dir),
                    "available_trajectory_count": actual_count,
                    "frame_count": frame_count,
                    "action_dimension": action_dimension,
                }
            )
    return rows


def sample_manifest(
    rows,
    seed,
    category_count,
    objects_per_category,
    trajectories_per_object,
    calibration_per_object,
):
    """按类别—物体—轨迹三级固定随机抽样构造manifest。

    输入：已验证inventory、seed及四个数量参数。
    输出：与三只手批处理兼容的正式manifest字典。
    内部逻辑：只保留物体/轨迹数充足类别，抽样后单独随机划calibration和heldout。
    作用：保证恰好50×2×10且调参索引在物理结果出现前已经冻结。
    """
    if not 0 <= calibration_per_object < trajectories_per_object:
        raise ValueError("calibration数量必须在0到每物体轨迹数之间")
    grouped = defaultdict(list)
    for row in rows:
        if row["available_trajectory_count"] >= trajectories_per_object:
            grouped[row["category"]].append(row)
    eligible = sorted(
        category
        for category, values in grouped.items()
        if len(values) >= objects_per_category
    )
    if len(eligible) < category_count:
        raise ValueError(f"合格类别只有{len(eligible)}个，少于要求的{category_count}个")

    rng = random.Random(seed)
    selected_categories = sorted(rng.sample(eligible, category_count))
    entries = []
    for category in selected_categories:
        candidates = sorted(grouped[category], key=lambda item: item["object_id"])
        for row in rng.sample(candidates, objects_per_category):
            sampled = rng.sample(
                range(row["available_trajectory_count"]), trajectories_per_object
            )
            calibration = sorted(
                rng.sample(sampled, calibration_per_object)
            )
            heldout = sorted(set(sampled) - set(calibration))
            entries.append(
                {
                    "object_name": row["object_id"],
                    "category": category,
                    "source_path": row["source_path"],
                    "source_sha256": row["source_sha256"],
                    "object_asset_path": row["object_asset_path"],
                    "available_trajectory_count": row[
                        "available_trajectory_count"
                    ],
                    "frame_count": row["frame_count"],
                    "action_dimension": row["action_dimension"],
                    "trajectory_indices": sorted(sampled),
                    "calibration_indices": calibration,
                    "heldout_indices": heldout,
                }
            )
    entries.sort(key=lambda item: (item["category"], item["object_name"]))
    expected_objects = category_count * objects_per_category
    expected_trajectories = expected_objects * trajectories_per_object
    actual_trajectories = sum(len(item["trajectory_indices"]) for item in entries)
    if len(entries) != expected_objects or actual_trajectories != expected_trajectories:
        raise AssertionError("正式manifest数量不等于配置要求")
    return {
        "schema_version": 2,
        "purpose": "formal_50_category_100_object_1000_trajectory_evaluation",
        "selection_seed": int(seed),
        "category_count": int(category_count),
        "object_count": len(entries),
        "trajectory_count": actual_trajectories,
        "objects_per_category": int(objects_per_category),
        "trajectories_per_object": int(trajectories_per_object),
        "calibration_per_object": int(calibration_per_object),
        "categories": selected_categories,
        "entries": entries,
    }


def main():
    """解析抽样参数、验证inventory并写出正式manifest。

    输入：CSV、输出JSON、seed和类别/物体/轨迹/calibration数量。
    输出：冻结manifest及终端数量摘要。
    内部逻辑：依次调用`load_inventory`和`sample_manifest`，不隐藏任何默认抽样规则。
    作用：作为完整数据到位后启动三手正式批处理的唯一标准入口。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--categories", type=int, default=50)
    parser.add_argument("--objects-per-category", type=int, default=2)
    parser.add_argument("--trajectories-per-object", type=int, default=10)
    parser.add_argument("--calibration-per-object", type=int, default=2)
    args = parser.parse_args()
    rows = load_inventory(args.inventory)
    manifest = sample_manifest(
        rows,
        args.seed,
        args.categories,
        args.objects_per_category,
        args.trajectories_per_object,
        args.calibration_per_object,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"RETARGET_MANIFEST={args.output.resolve()}")
    print(
        f"categories={manifest['category_count']} "
        f"objects={manifest['object_count']} "
        f"trajectories={manifest['trajectory_count']}"
    )


if __name__ == "__main__":
    main()
