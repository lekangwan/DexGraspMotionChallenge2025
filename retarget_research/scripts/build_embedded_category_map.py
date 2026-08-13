#!/usr/bin/env python3
"""从GraspM3对象ID中的显式标签生成可审计类别表。

输入：原始对象级轨迹目录、同名COACD资产目录，以及最小轨迹数/每类物体数。
输出：`object_id,category` CSV和记录采纳、合并、排除原因的审计JSON。
内部逻辑：只接受`core-类别-ID`和`sem-类别-ID`，类别按大小写无关规则合并；
同时检查标准轨迹字段、`(N,70,28)`形状和两项必要COACD资产。
作用：利用数据集对象ID已经携带的明确类别标签满足50类抽样，不猜测商品名类别。
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import re

import numpy as np


EMBEDDED_ID = re.compile(r"^(core|sem)-([^-]+)-.+$")


def parse_embedded_category(object_id: str) -> tuple[str, str] | None:
    """解析一个带显式类别的GraspM3对象ID。

    输入：不含`.npy`后缀的对象ID。
    输出：`(来源家族, 规范类别)`；不属于core/sem时返回`None`。
    内部逻辑：读取第二个连字符字段，并用`casefold`合并Bottle/bottle等大小写差异。
    作用：避免把同一语义类别因来源或大小写不同错误计算为两个类别。
    """
    match = EMBEDDED_ID.fullmatch(object_id)
    if match is None:
        return None
    family, raw_category = match.groups()
    return family, raw_category.casefold()


def inspect_record(
    trajectory_path: Path,
    asset_root: Path,
    minimum_trajectories: int,
) -> tuple[dict | None, str]:
    """检查一个轨迹文件能否进入显式类别候选池。

    输入：对象轨迹路径、资产根和每物体最少轨迹数。
    输出：合法时返回类别记录和`accepted`，否则返回`None`及稳定排除原因。
    内部逻辑：先解析ID，再核对COACD资产、三个数组字段、形状与首维一致性。
    作用：让后续inventory不会因为一个坏对象中途失败，并完整记录过滤边界。
    """
    object_id = trajectory_path.stem
    parsed = parse_embedded_category(object_id)
    if parsed is None:
        return None, "unsupported_object_family"
    family, category = parsed
    coacd = asset_root / object_id / "coacd"
    required_assets = (coacd / "coacd_1.urdf", coacd / "decomposed.obj")
    if any(not path.is_file() for path in required_assets):
        return None, "missing_required_assets"
    try:
        data = np.load(trajectory_path, allow_pickle=True).item()
        required = {"grasp_seqs", "obj_scale", "obj_rotmat"}
        if required - set(data):
            return None, "missing_trajectory_fields"
        sequences = np.asarray(data["grasp_seqs"])
        if sequences.ndim != 3 or sequences.shape[1:] != (70, 28):
            return None, "invalid_trajectory_shape"
        trajectory_count = int(sequences.shape[0])
        if (
            len(np.asarray(data["obj_scale"])) != trajectory_count
            or len(np.asarray(data["obj_rotmat"])) != trajectory_count
        ):
            return None, "inconsistent_object_metadata"
        if trajectory_count < minimum_trajectories:
            return None, "too_few_trajectories"
    except (OSError, TypeError, ValueError):
        return None, "unreadable_trajectory"
    return {
        "object_id": object_id,
        "category": category,
        "family": family,
        "trajectory_count": trajectory_count,
    }, "accepted"


def collect_embedded_records(
    trajectory_root: Path,
    asset_root: Path,
    minimum_trajectories: int = 10,
    minimum_objects_per_category: int = 2,
) -> tuple[list[dict], dict]:
    """扫描全部文件并保留物体数足够的显式类别。

    输入：轨迹/资产根和两个最低数量门槛。
    输出：按类别和物体排序的记录，以及完整数量审计字典。
    内部逻辑：逐对象检查后按规范类别分组，整类不足时统一排除，并统计来源。
    作用：在随机抽50类之前先形成确定、可复现且无商品名猜测的候选总体。
    """
    trajectory_root = Path(trajectory_root).resolve()
    asset_root = Path(asset_root).resolve()
    files = sorted(trajectory_root.glob("*.npy"))
    grouped: dict[str, list[dict]] = defaultdict(list)
    reasons: Counter[str] = Counter()
    for path in files:
        record, reason = inspect_record(path, asset_root, minimum_trajectories)
        reasons[reason] += 1
        if record is not None:
            grouped[record["category"]].append(record)

    eligible_categories = {
        category
        for category, values in grouped.items()
        if len(values) >= minimum_objects_per_category
    }
    excluded_small = sum(
        len(values)
        for category, values in grouped.items()
        if category not in eligible_categories
    )
    if excluded_small:
        reasons["category_has_too_few_objects"] += excluded_small
    records = sorted(
        (
            record
            for category in eligible_categories
            for record in grouped[category]
        ),
        key=lambda item: (item["category"], item["object_id"]),
    )
    family_counts = Counter(record["family"] for record in records)
    category_sizes = Counter(record["category"] for record in records)
    audit = {
        "status": "READY" if records else "EMPTY",
        "label_policy": "core/sem embedded category; casefold merge; no mujoco/ddg guessing",
        "trajectory_root": str(trajectory_root),
        "asset_root": str(asset_root),
        "minimum_trajectories": int(minimum_trajectories),
        "minimum_objects_per_category": int(minimum_objects_per_category),
        "scanned_file_count": len(files),
        "eligible_object_count": len(records),
        "eligible_category_count": len(eligible_categories),
        "eligible_trajectory_total": sum(record["trajectory_count"] for record in records),
        "family_counts": dict(sorted(family_counts.items())),
        "excluded_counts": dict(sorted(reasons.items())),
        "category_sizes": dict(sorted(category_sizes.items())),
    }
    return records, audit


def write_category_map(path: Path, records: list[dict]) -> None:
    """把候选记录原子写成下游要求的两列CSV。

    输入：输出路径和带额外审计字段的候选记录。
    输出：只含`object_id,category`的UTF-8 CSV文件。
    内部逻辑：先写同目录临时文件，再整体替换目标文件。
    作用：避免中断时留下被`build_inventory.py`误读的半成品。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["object_id", "category"])
        writer.writeheader()
        writer.writerows(
            {"object_id": item["object_id"], "category": item["category"]}
            for item in records
        )
    temporary.replace(path)


def main() -> None:
    """解析参数、生成类别CSV并保存来源/过滤审计。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory-root", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--minimum-trajectories", type=int, default=10)
    parser.add_argument("--minimum-objects-per-category", type=int, default=2)
    args = parser.parse_args()
    if args.minimum_trajectories <= 0 or args.minimum_objects_per_category <= 0:
        raise ValueError("两个最低数量参数必须为正数")
    records, audit = collect_embedded_records(
        args.trajectory_root,
        args.asset_root,
        args.minimum_trajectories,
        args.minimum_objects_per_category,
    )
    if audit["eligible_category_count"] < 50:
        raise ValueError(
            f"显式标签合格类别只有{audit['eligible_category_count']}个，少于正式要求50"
        )
    write_category_map(args.output, records)
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "eligible_categories": audit["eligible_category_count"],
        "eligible_objects": audit["eligible_object_count"],
        "eligible_trajectories": audit["eligible_trajectory_total"],
        "category_map": str(args.output.resolve()),
        "audit": str(args.audit_output.resolve()),
    }, ensure_ascii=False, indent=2))
    print(f"EMBEDDED_CATEGORY_MAP_READY={args.output.resolve()}")


if __name__ == "__main__":
    main()
