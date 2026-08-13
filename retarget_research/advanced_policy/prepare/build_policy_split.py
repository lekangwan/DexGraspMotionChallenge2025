#!/usr/bin/env python3
"""从正式100物体manifest冻结进阶策略的对象级无泄漏划分。

输入：每类2物体的正式manifest、随机seed和输出JSON。
输出：每类1个训练物体、1个未见测试物体；训练物体内部再划train/valid轨迹。
内部逻辑：用`seed:category`稳定哈希决定两个物体角色；原heldout索引作为策略训练，
原calibration索引作为策略验证，测试物体的10条全部只用于最终闭环测试。
作用：避免同一物体的不同帧或轨迹泄漏到策略测试集，同时保持同类别未见实例评测。
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path


def stable_order(values, seed, namespace):
    """按稳定哈希排列字符串。

    输入：字符串序列、整数seed和命名空间。
    输出：与输入集合相同但顺序确定的列表。
    内部逻辑：排序键为SHA-256(`seed:namespace:value`)，不依赖Python哈希随机化。
    作用：保证不同机器和Python版本生成完全相同的对象角色。
    """
    return sorted(
        values,
        key=lambda value: hashlib.sha256(
            f"{seed}:{namespace}:{value}".encode("utf-8")
        ).hexdigest(),
    )


def build_policy_split(manifest, seed):
    """生成严格的类别内对象级策略划分。

    输入：正式manifest字典和seed。
    输出：含类别角色、逐轨迹split记录及泄漏审计的字典。
    内部逻辑：要求每类恰好2物体；训练物体用heldout做train、calibration做valid，
    测试物体全部轨迹标为test，并验证三个split的`(物体,轨迹)`集合互斥。
    作用：把基本重定向的全量评测名单转为进阶策略可用的冻结协议。
    """
    grouped = defaultdict(list)
    for entry in manifest.get("entries", []):
        grouped[entry.get("category")].append(entry)
    if not grouped or None in grouped:
        raise ValueError("正式manifest每个条目都必须有category")

    categories = []
    records = []
    for category in sorted(grouped):
        entries = grouped[category]
        if len(entries) != 2:
            raise ValueError(f"类别{category}应恰有2个物体，实际{len(entries)}")
        ordered_names = stable_order(
            [entry["object_name"] for entry in entries], seed, category
        )
        by_name = {entry["object_name"]: entry for entry in entries}
        train_entry = by_name[ordered_names[0]]
        test_entry = by_name[ordered_names[1]]
        train_indices = sorted(int(i) for i in train_entry.get("heldout_indices", []))
        valid_indices = sorted(
            int(i) for i in train_entry.get("calibration_indices", [])
        )
        test_indices = sorted(int(i) for i in test_entry["trajectory_indices"])
        if not train_indices or not valid_indices or not test_indices:
            raise ValueError(f"类别{category}缺少train/valid/test轨迹")
        categories.append(
            {
                "category": category,
                "train_object": train_entry["object_name"],
                "test_object": test_entry["object_name"],
                "train_trajectory_count": len(train_indices),
                "valid_trajectory_count": len(valid_indices),
                "test_trajectory_count": len(test_indices),
            }
        )
        for split, entry, indices in (
            ("train", train_entry, train_indices),
            ("valid", train_entry, valid_indices),
            ("test", test_entry, test_indices),
        ):
            records.extend(
                {
                    "split": split,
                    "category": category,
                    "object_name": entry["object_name"],
                    "source_trajectory_index": index,
                }
                for index in indices
            )

    split_keys = {}
    for split in ("train", "valid", "test"):
        split_keys[split] = {
            (item["object_name"], item["source_trajectory_index"])
            for item in records
            if item["split"] == split
        }
    if (
        split_keys["train"] & split_keys["valid"]
        or split_keys["train"] & split_keys["test"]
        or split_keys["valid"] & split_keys["test"]
    ):
        raise AssertionError("策略split存在轨迹泄漏")
    train_objects = {item["object_name"] for item in records if item["split"] != "test"}
    test_objects = {item["object_name"] for item in records if item["split"] == "test"}
    if train_objects & test_objects:
        raise AssertionError("策略训练物体泄漏到测试集")

    return {
        "schema_version": 1,
        "purpose": "target_hand_policy_object_level_split",
        "source_manifest": manifest.get("purpose"),
        "seed": int(seed),
        "category_count": len(categories),
        "train_object_count": len(train_objects),
        "test_object_count": len(test_objects),
        "trajectory_counts": {
            split: len(split_keys[split]) for split in ("train", "valid", "test")
        },
        "categories": categories,
        "records": records,
        "leakage_check": "PASS",
        "protocol_note": (
            "基本任务heldout只表示重定向评测口径；在进阶任务中，训练对象的8条"
            "heldout成为策略train，2条calibration成为valid。测试对象10条始终不可训练。"
        ),
    }


def main():
    """解析路径、构建策略划分并写入JSON。

    输入：`--manifest`、`--output`和可选seed。
    输出：冻结策略split及终端数量摘要。
    内部逻辑：读取manifest后调用纯函数，写盘时保留中文和稳定缩进。
    作用：作为任何目标手训练数据准备前必须执行的第一道防泄漏门。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260813)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    split = build_policy_split(manifest, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(split, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"categories={split['category_count']}")
    print(f"trajectory_counts={split['trajectory_counts']}")
    print("POLICY_SPLIT=READY")


if __name__ == "__main__":
    main()
