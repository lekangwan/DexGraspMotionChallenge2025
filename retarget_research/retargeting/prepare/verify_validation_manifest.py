#!/usr/bin/env python3
"""审计独立验证manifest的数据完整性、零泄漏和方法冻结状态。

输入：独立验证manifest与默认物体资产根目录。
输出：通过时打印物体/轨迹/排除数量；失败时抛出含具体对象的异常。
内部逻辑：复核源哈希、轨迹形状/索引、资产目录、历史manifest交集和方法配置哈希。
作用：在昂贵优化前阻止数据被替换、开发物体泄漏或冻结参数被静默修改。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from create_pilot_manifest import file_sha256


DEFAULT_OBJECT_ROOT = Path(
    "retarget_research/reference/HandRetargetTask2026/scripts/data/sorting/object_41"
)


def historical_object_names(manifest_paths):
    """收集历史manifest中已经暴露或显式排除的物体名。

    输入：历史manifest绝对或相对路径列表。
    输出：所有`entries/excluded_objects/excluded_tuning_objects`名称的集合。
    内部逻辑：逐文件解析并验证`entries`是列表，再合并三个来源。
    作用：独立验证清单必须与该集合完全无交集。
    """
    names = set()
    for path in manifest_paths:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"历史manifest不存在: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        entries = data.get("entries")
        if not isinstance(entries, list):
            raise ValueError(f"历史manifest缺少entries列表: {path}")
        names.update(str(entry["object_name"]) for entry in entries)
        names.update(str(value) for value in data.get("excluded_objects", []))
        names.update(
            str(value) for value in data.get("excluded_tuning_objects", [])
        )
    return names


def verify_frozen_method(record):
    """核对manifest记录的方法配置仍是冻结时的原文件。

    输入：包含path、sha256和method的`frozen_method`字典。
    输出：解析后的方法名。
    内部逻辑：重新计算配置文件SHA-256并核对配置内部`method`字段。
    作用：防止看完验证结果后修改阈值，却仍沿用旧验证清单名称。
    """
    if not isinstance(record, dict):
        raise ValueError("独立验证manifest缺少frozen_method")
    path = Path(record.get("path", ""))
    if not path.is_file():
        raise FileNotFoundError(f"冻结方法配置不存在: {path}")
    actual_hash = file_sha256(path)
    if actual_hash != record.get("sha256"):
        raise ValueError(f"冻结方法配置哈希已改变: {path}")
    config = json.loads(path.read_text(encoding="utf-8"))
    if str(config.get("method")) != str(record.get("method")):
        raise ValueError(f"冻结方法名称不一致: {path}")
    return str(record["method"])


def verify_validation_manifest(manifest_path, object_root=DEFAULT_OBJECT_ROOT):
    """执行独立验证manifest的完整只读审计。

    输入：待审计manifest路径和物体资产根目录。
    输出：物体数、轨迹数、排除数和冻结方法名摘要。
    内部逻辑：检查条目唯一性、新旧零交集、源文件哈希、`(N,70,28)`、
    轨迹索引范围/去重、每个物体资产，以及冻结方法配置哈希。
    作用：把一次性人工检查固化为每轮长命令之前都能复现的验收门。
    """
    manifest_path = Path(manifest_path)
    object_root = Path(object_root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("验证manifest的entries必须是非空列表")
    names = [str(entry["object_name"]) for entry in entries]
    if len(set(names)) != len(names):
        raise ValueError("验证manifest含重复物体")
    historical = historical_object_names(manifest.get("excluded_manifests", []))
    overlap = sorted(set(names) & historical)
    if overlap:
        raise ValueError(f"验证物体与历史开发物体重合: {overlap}")

    trajectory_count = 0
    for entry in entries:
        source_path = Path(entry["source_path"])
        if not source_path.is_file():
            raise FileNotFoundError(f"源轨迹不存在: {source_path}")
        if file_sha256(source_path) != entry.get("source_sha256"):
            raise ValueError(f"源轨迹哈希已改变: {source_path}")
        data = np.load(source_path, allow_pickle=True).item()
        frames = np.asarray(data["grasp_seqs"])
        if frames.ndim != 3 or frames.shape[1:] != (70, 28):
            raise ValueError(f"{source_path.name}形状错误: {frames.shape}")
        indices = [int(value) for value in entry["trajectory_indices"]]
        if len(indices) != len(set(indices)) or any(
            index < 0 or index >= len(frames) for index in indices
        ):
            raise ValueError(f"{source_path.name}轨迹索引重复或越界: {indices}")
        trajectory_count += len(indices)
        asset_dir = Path(entry.get("object_asset_path", object_root / entry["object_name"]))
        if not asset_dir.is_dir():
            raise FileNotFoundError(f"物体资产目录不存在: {asset_dir}")

    if int(manifest.get("object_count", -1)) != len(entries):
        raise ValueError("manifest的object_count与entries数量不一致")
    if int(manifest.get("trajectory_count", -1)) != trajectory_count:
        raise ValueError("manifest的trajectory_count与索引总数不一致")
    method = verify_frozen_method(manifest.get("frozen_method"))
    return {
        "object_count": len(entries),
        "trajectory_count": trajectory_count,
        "excluded_object_count": len(manifest.get("excluded_objects", [])),
        "frozen_method": method,
    }


def main():
    """解析命令行、运行审计并打印可供工作日志复制的摘要。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--object-root", type=Path, default=DEFAULT_OBJECT_ROOT)
    args = parser.parse_args()
    summary = verify_validation_manifest(args.manifest, args.object_root)
    print("validation_manifest_ok=true")
    for key, value in summary.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
