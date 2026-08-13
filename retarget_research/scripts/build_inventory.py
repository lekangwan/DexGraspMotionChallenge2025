#!/usr/bin/env python3
"""由官方类别表和两个数据根目录自动生成正式inventory.csv。

输入：仅含`object_id,category`的类别CSV、轨迹npy根目录和同名物体资产根目录。
输出：`build_manifest.py`可直接读取的五列inventory及可选审计JSON。
内部逻辑：按物体ID匹配`ID.npy`与`asset_root/ID`，读取实际轨迹数并检查COACD文件。
作用：避免人工填写100行绝对路径和数量时产生拼写、错配或轨迹数错误。
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def load_category_map(path):
    """读取并验证`object_id,category`官方类别映射。

    输入：类别CSV路径。
    输出：按输入行顺序保存的物体—类别字典列表。
    内部逻辑：拒绝缺列、空值和重复物体ID，不从文件名猜类别。
    作用：把考核要求的“50种不同类别”建立在明确标签上。
    """
    records = []
    seen = set()
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = {"object_id", "category"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"类别表缺列: {sorted(missing)}")
        for line_number, row in enumerate(reader, start=2):
            object_id = row["object_id"].strip()
            category = row["category"].strip()
            if not object_id or not category:
                raise ValueError(f"类别表第{line_number}行存在空值")
            if object_id in seen:
                raise ValueError(f"类别表物体重复: {object_id}")
            seen.add(object_id)
            records.append({"object_id": object_id, "category": category})
    if not records:
        raise ValueError("类别表没有数据行")
    return records


def inspect_object(record, trajectory_root, asset_root):
    """匹配并检查一个物体的轨迹文件和COACD资产。

    输入：类别记录、轨迹根和资产根。
    输出：五列inventory行及轨迹形状审计字段。
    内部逻辑：要求npy字典含标准三字段且`grasp_seqs`为`(N,70,28)`。
    作用：在正式抽样前就发现标签有物体但文件/资产缺失的情况。
    """
    object_id = record["object_id"]
    trajectory = (trajectory_root / f"{object_id}.npy").resolve()
    asset = (asset_root / object_id).resolve()
    if not trajectory.is_file():
        raise FileNotFoundError(trajectory)
    required_assets = [asset / "coacd" / "coacd_1.urdf", asset / "coacd" / "decomposed.obj"]
    missing = [str(path) for path in required_assets if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{object_id}缺少资产: {missing}")
    data = np.load(trajectory, allow_pickle=True).item()
    required_fields = {"grasp_seqs", "obj_scale", "obj_rotmat"}
    missing_fields = required_fields - set(data)
    if missing_fields:
        raise ValueError(f"{trajectory}缺字段: {sorted(missing_fields)}")
    frames = np.asarray(data["grasp_seqs"])
    if frames.ndim != 3 or frames.shape[1:] != (70, 28):
        raise ValueError(f"{trajectory}形状错误: {frames.shape}")
    count = int(frames.shape[0])
    if len(np.asarray(data["obj_scale"])) != count or len(np.asarray(data["obj_rotmat"])) != count:
        raise ValueError(f"{trajectory}物体属性数量与轨迹数不一致")
    return {
        "object_id": object_id,
        "category": record["category"],
        "trajectory_file": str(trajectory),
        "asset_dir": str(asset),
        "trajectory_count": count,
    }


def write_inventory(path, rows):
    """以固定列顺序原子式写inventory CSV。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fields = ["object_id", "category", "trajectory_file", "asset_dir", "trajectory_count"]
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main():
    """解析路径、构建完整inventory并保存数量审计。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--category-map", type=Path, required=True)
    parser.add_argument("--trajectory-root", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path)
    args = parser.parse_args()
    records = load_category_map(args.category_map)
    rows = [
        inspect_object(record, args.trajectory_root.resolve(), args.asset_root.resolve())
        for record in records
    ]
    write_inventory(args.output, rows)
    categories = sorted({row["category"] for row in rows})
    audit = {
        "status": "READY",
        "object_count": len(rows),
        "category_count": len(categories),
        "trajectory_total": sum(int(row["trajectory_count"]) for row in rows),
        "categories": categories,
        "inventory": str(args.output.resolve()),
    }
    if args.audit_output is not None:
        args.audit_output.parent.mkdir(parents=True, exist_ok=True)
        args.audit_output.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    print(f"INVENTORY_READY={args.output.resolve()}")


if __name__ == "__main__":
    main()
