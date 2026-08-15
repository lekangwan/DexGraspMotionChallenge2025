#!/usr/bin/env python3
"""从正式数据的train划分冻结Wuji手型约束初筛清单。

输入：正式1000条manifest、策略split、类别数和固定种子。
输出：每类一条、完全不读取物理成功标签的重定向manifest。
内部逻辑：先按train过滤，再用SHA-256稳定排序类别及类内轨迹，默认取20类。
作用：以较低成本比较手型约束，同时不使用valid/test或成功结果调参。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "retarget_research/manifests/formal_50c_100o_1000t_seed20260808.json"
)
DEFAULT_SPLIT = (
    PROJECT_ROOT
    / "retarget_research/advanced_policy/data/formal_v1/policy_split_seed20260813.json"
)


def stable_rank(seed, value):
    """输入种子和字符串，输出跨Python版本稳定的SHA-256排序键。"""
    return hashlib.sha256(f"{int(seed)}|{value}".encode("utf-8")).hexdigest()


def build_screen(source_manifest, policy_split, category_count, seed):
    """构造只含train轨迹且每类一条的初筛manifest。

    输入：两个已解析JSON、目标类别数和种子。
    输出：可直接交给三手批处理入口的manifest字典。
    内部逻辑：用正式manifest提供路径/哈希，用split决定train资格；类别和类内
    轨迹都只按稳定哈希选择，完全不读取任何重放成功率。
    作用：为硬边界与软耦合提供低成本、公平的第一轮比较集。
    """
    entries_by_name = {
        item["object_name"]: item for item in source_manifest["entries"]
    }
    candidates = {}
    for record in policy_split["records"]:
        if record["split"] != "train":
            continue
        name = record["object_name"]
        if name not in entries_by_name:
            raise ValueError(f"split中的物体不在正式manifest: {name}")
        candidate = (
            name,
            int(record["source_trajectory_index"]),
        )
        candidates.setdefault(record["category"], []).append(candidate)
    categories = sorted(
        candidates, key=lambda value: (stable_rank(seed, value), value)
    )
    if not 1 <= int(category_count) <= len(categories):
        raise ValueError(f"类别数必须在1到{len(categories)}之间")
    selected_entries = []
    for category in categories[: int(category_count)]:
        name, source_index = min(
            candidates[category],
            key=lambda item: (
                stable_rank(seed, f"{category}|{item[0]}|{item[1]}"),
                item,
            ),
        )
        entry = dict(entries_by_name[name])
        entry["trajectory_indices"] = [source_index]
        entry["calibration_indices"] = []
        entry["heldout_indices"] = [source_index]
        selected_entries.append(entry)
    return {
        "schema_version": 1,
        "purpose": "wuji_anatomy_constraint_train_only_screen",
        "selection_seed": int(seed),
        "selection_rule": "SHA-256 category and within-category ranking; no physics outcomes",
        "source_manifest": source_manifest.get("purpose"),
        "source_split": policy_split.get("purpose"),
        "allowed_split": "train",
        "category_count": len(selected_entries),
        "object_count": len(selected_entries),
        "trajectory_count": len(selected_entries),
        "categories": [item["category"] for item in selected_entries],
        "entries": selected_entries,
    }


def main():
    """解析路径，生成并打印Wuji手型初筛manifest摘要。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--policy-split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--category-count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    split = json.loads(args.policy_split.read_text(encoding="utf-8"))
    result = build_screen(source, split, args.category_count, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"categories={result['category_count']}")
    print(f"trajectories={result['trajectory_count']}")
    print(f"output={args.output.resolve()}")


if __name__ == "__main__":
    main()
