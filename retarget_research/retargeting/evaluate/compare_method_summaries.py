#!/usr/bin/env python3
"""对齐两份同轨迹物理评测汇总，生成方法A到方法B的配对比较。

输入：基线与改进方法的`manifest_evaluation_summary.json`。
输出：成功率差、逐轨迹新增/丢失成功及分split统计的JSON文件。
内部逻辑：以物体名和源轨迹索引作为稳定主键，先检查样本完全一致，再比较成功布尔值。
作用：避免只比较两个总成功率而掩盖“新增成功同时也丢失旧成功”的情况。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_summary(path: Path) -> dict:
    """读取并检查一份统一物理评测汇总。

    输入：`manifest_evaluation_summary.json`路径。
    输出：包含非空`results`列表的字典。
    内部逻辑：解析JSON并检查配对比较必需的结果字段。
    作用：在比较前尽早暴露路径错误或未完成的评测输出。
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data.get("results"), list) or not data["results"]:
        raise ValueError(f"评测汇总缺少非空results: {path}")
    return data


def trajectory_key(result: dict) -> tuple[str, int]:
    """构造一条源轨迹在不同方法间共享的唯一键。

    输入：统一评测汇总中的单条结果字典。
    输出：`(object_name, source_trajectory_index)`元组。
    内部逻辑：不使用目标目录内的局部索引，避免不同候选保存顺序造成错配。
    作用：保证官方方法和改进方法比较的是同一物体的同一条Shadow轨迹。
    """
    return str(result["object_name"]), int(result["source_trajectory_index"])


def compact_result(result: dict) -> dict:
    """提取报告配对变化所需的少量轨迹字段。

    输入：一条完整评测结果。
    输出：物体、类别、split、源索引、成功与抬升信息字典。
    内部逻辑：删除stdout和文件路径等体积较大的诊断字段。
    作用：让最终比较JSON便于人工阅读并可直接用于报告表格。
    """
    keys = (
        "object_name",
        "category",
        "evaluation_split",
        "source_trajectory_index",
        "success",
        "max_lift_m",
        "final_lift_m",
        "hand_object_contact_steps",
    )
    return {key: result.get(key) for key in keys}


def compare_summaries(baseline: dict, improved: dict) -> dict:
    """计算两种方法在完全相同样本上的配对成功变化。

    输入：官方基线和改进方法的评测汇总字典。
    输出：总体成功差、保持/新增/丢失/共同失败列表及分split统计。
    内部逻辑：验证手类型、manifest和轨迹键集合一致，再逐键比较两个成功标志。
    作用：区分净提升与实际替换掉的成功样本，形成比总成功率更严格的证据。
    """
    if baseline.get("hand") != improved.get("hand"):
        raise ValueError("基线与改进汇总的hand不一致")
    if baseline.get("manifest") != improved.get("manifest"):
        raise ValueError("基线与改进汇总不是同一manifest")

    baseline_by_key = {trajectory_key(item): item for item in baseline["results"]}
    improved_by_key = {trajectory_key(item): item for item in improved["results"]}
    if len(baseline_by_key) != len(baseline["results"]):
        raise ValueError("基线results存在重复轨迹键")
    if len(improved_by_key) != len(improved["results"]):
        raise ValueError("改进results存在重复轨迹键")
    if set(baseline_by_key) != set(improved_by_key):
        missing_in_improved = sorted(set(baseline_by_key) - set(improved_by_key))
        missing_in_baseline = sorted(set(improved_by_key) - set(baseline_by_key))
        raise ValueError(
            f"轨迹集不一致: improved缺{missing_in_improved}, "
            f"baseline缺{missing_in_baseline}"
        )

    groups = {
        "retained_successes": [],
        "gained_successes": [],
        "lost_successes": [],
        "shared_failures": [],
    }
    split_counts: dict[str, dict[str, int]] = {}
    for key in sorted(baseline_by_key):
        before = baseline_by_key[key]
        after = improved_by_key[key]
        before_success = bool(before["success"])
        after_success = bool(after["success"])
        if before_success and after_success:
            group = "retained_successes"
        elif not before_success and after_success:
            group = "gained_successes"
        elif before_success and not after_success:
            group = "lost_successes"
        else:
            group = "shared_failures"
        groups[group].append(
            {"baseline": compact_result(before), "improved": compact_result(after)}
        )
        split = str(after.get("evaluation_split") or "all")
        counters = split_counts.setdefault(
            split,
            {"trajectory_count": 0, "baseline_success_count": 0, "improved_success_count": 0},
        )
        counters["trajectory_count"] += 1
        counters["baseline_success_count"] += int(before_success)
        counters["improved_success_count"] += int(after_success)

    count = len(baseline_by_key)
    baseline_success = sum(bool(item["success"]) for item in baseline_by_key.values())
    improved_success = sum(bool(item["success"]) for item in improved_by_key.values())
    for counters in split_counts.values():
        split_count = counters["trajectory_count"]
        counters["baseline_success_rate"] = counters["baseline_success_count"] / split_count
        counters["improved_success_rate"] = counters["improved_success_count"] / split_count
        counters["success_rate_delta"] = (
            counters["improved_success_rate"] - counters["baseline_success_rate"]
        )
    return {
        "hand": baseline["hand"],
        "manifest": baseline["manifest"],
        "baseline_target_directory": baseline.get("target_directory"),
        "improved_target_directory": improved.get("target_directory"),
        "trajectory_count": count,
        "baseline_success_count": baseline_success,
        "improved_success_count": improved_success,
        "baseline_success_rate": baseline_success / count,
        "improved_success_rate": improved_success / count,
        "success_count_delta": improved_success - baseline_success,
        "success_rate_delta": (improved_success - baseline_success) / count,
        "retained_success_count": len(groups["retained_successes"]),
        "gained_success_count": len(groups["gained_successes"]),
        "lost_success_count": len(groups["lost_successes"]),
        "shared_failure_count": len(groups["shared_failures"]),
        "per_split": split_counts,
        **groups,
    }


def main() -> None:
    """解析两份汇总路径并保存配对比较结果。

    输入：`--baseline-summary`、`--improved-summary`和`--output`命令行参数。
    输出：比较JSON以及终端中的成功数、净变化、新增和丢失数量。
    内部逻辑：依次调用读取、配对比较和JSON序列化函数。
    作用：为验证集和正式1000轨迹复用同一套可审计比较流程。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-summary", type=Path, required=True)
    parser.add_argument("--improved-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    comparison = compare_summaries(
        load_summary(args.baseline_summary), load_summary(args.improved_summary)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"trajectories={comparison['trajectory_count']}")
    print(
        f"success={comparison['baseline_success_count']}"
        f"->{comparison['improved_success_count']}"
    )
    print(f"gained={comparison['gained_success_count']}")
    print(f"lost={comparison['lost_success_count']}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
