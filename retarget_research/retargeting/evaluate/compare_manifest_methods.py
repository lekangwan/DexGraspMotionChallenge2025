#!/usr/bin/env python3
"""在指定小manifest上从一个或多个评测摘要做严格配对比较。

输入：A/B manifest、重复的`方法名 摘要JSON`二元组和输出路径。
输出：各方法小集统计、相对首个基线的新增/丢失成功及逐类别配对记录。
内部逻辑：用`(object_name, source_trajectory_index)`过滤完整或小集摘要并核对无缺失。
作用：复用既有1000条物理结果，同时公平比较本轮新跑的轻量候选结果。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluate_hand_manifest import summarize_results


def manifest_keys(manifest: dict) -> list[tuple[str, int]]:
    """按manifest顺序展开全部轨迹唯一键。

    输入：含`entries/trajectory_indices`的manifest。
    输出：`(物体名, 源轨迹索引)`列表。
    内部逻辑：逐条展开并拒绝重复键，最后核对声明轨迹数。
    作用：建立所有方法必须共同覆盖的配对样本全集。
    """
    keys = [
        (str(entry["object_name"]), int(source_index))
        for entry in manifest.get("entries", [])
        for source_index in entry["trajectory_indices"]
    ]
    if len(keys) != len(set(keys)):
        raise ValueError("manifest包含重复的物体—轨迹键")
    declared = int(manifest.get("trajectory_count", len(keys)))
    if len(keys) != declared:
        raise ValueError(f"manifest展开{len(keys)}条，但声明{declared}条")
    return keys


def select_summary_results(summary: dict, keys: list[tuple[str, int]]) -> list[dict]:
    """从完整或小集评测摘要中抽出manifest指定结果。

    输入：评测摘要和有序目标键列表。
    输出：严格按manifest顺序排列的逐轨迹结果。
    内部逻辑：建立唯一键索引，缺失任一目标键即报错，额外结果则安全忽略。
    作用：允许直接复用正式1000条官方/主方法结果，无需再次物理重放。
    """
    by_key = {}
    for result in summary.get("results", []):
        key = (str(result["object_name"]), int(result["source_trajectory_index"]))
        if key in by_key:
            raise ValueError(f"评测摘要包含重复结果: {key}")
        by_key[key] = result
    missing = [key for key in keys if key not in by_key]
    if missing:
        raise ValueError(f"评测摘要缺少{len(missing)}条manifest结果: {missing[:5]}")
    return [by_key[key] for key in keys]


def compact_metrics(results: list[dict]) -> dict:
    """提取小样本方法比较所需的核心统计。

    输入：与manifest严格对齐的逐轨迹结果。
    输出：成功率、三项均值及逐类别成功布尔字典。
    内部逻辑：复用统一评估器的汇总函数，避免另建统计口径。
    作用：报告简洁指标，同时保留每类一条时可审计的成败分布。
    """
    summary = summarize_results(results)
    return {
        "trajectory_count": summary["trajectory_count"],
        "success_count": summary["success_count"],
        "success_rate": summary["success_rate"],
        "category_macro_success_rate": summary["category_macro_success_rate"],
        "mean_keypoint_distance_m": summary["mean_keypoint_distance_m"],
        "mean_max_lift_m": summary["mean_max_lift_m"],
        "mean_final_lift_m": summary["mean_final_lift_m"],
        "success_by_category": {
            item["category"]: bool(item["success"]) for item in results
        },
    }


def compare_methods(
    manifest: dict, named_summaries: list[tuple[str, dict]]
) -> dict:
    """计算全部方法及其相对首个基线的成对变化。

    输入：小manifest和按命令行顺序给出的`(名称,摘要)`列表。
    输出：各方法统计、配对新增/丢失/共同成功/共同失败记录。
    内部逻辑：所有摘要先过滤到同一有序键；第一个方法固定充当比较基线。
    作用：禁止用不同样本分母比较成功率，并直接暴露净提升来自哪些类别。
    """
    if len(named_summaries) < 2:
        raise ValueError("配对比较至少需要两个方法")
    names = [name for name, _ in named_summaries]
    if len(names) != len(set(names)):
        raise ValueError("方法名不能重复")
    keys = manifest_keys(manifest)
    aligned = {
        name: select_summary_results(summary, keys)
        for name, summary in named_summaries
    }
    baseline_name = names[0]
    baseline = aligned[baseline_name]
    comparisons = {}
    for name in names[1:]:
        candidate = aligned[name]
        paired = []
        for key, base_result, candidate_result in zip(keys, baseline, candidate):
            base_success = bool(base_result["success"])
            candidate_success = bool(candidate_result["success"])
            paired.append(
                {
                    "object_name": key[0],
                    "category": candidate_result.get("category"),
                    "source_trajectory_index": key[1],
                    "baseline_success": base_success,
                    "candidate_success": candidate_success,
                    "outcome": (
                        "added_success"
                        if candidate_success and not base_success
                        else "lost_success"
                        if base_success and not candidate_success
                        else "both_success"
                        if base_success
                        else "both_failure"
                    ),
                }
            )
        counts = {
            outcome: sum(item["outcome"] == outcome for item in paired)
            for outcome in (
                "added_success",
                "lost_success",
                "both_success",
                "both_failure",
            )
        }
        comparisons[name] = {
            "baseline": baseline_name,
            **counts,
            "net_success_change": counts["added_success"] - counts["lost_success"],
            "paired_results": paired,
        }
    return {
        "manifest_purpose": manifest.get("purpose"),
        "trajectory_count": len(keys),
        "baseline_method": baseline_name,
        "methods": {
            name: compact_metrics(results) for name, results in aligned.items()
        },
        "comparisons_to_baseline": comparisons,
    }


def main() -> None:
    """解析manifest和多个摘要，保存JSON并打印最关键配对结果。

    输入：`--manifest`、至少两个`--summary NAME PATH`及`--output`。
    输出：完整配对JSON与每个方法的成功数、相对基线净变化。
    内部逻辑：按参数顺序把第一个summary设为基线，再调用纯比较函数。
    作用：在A组筛选和B组确认完成后提供统一的短命令分析入口。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--summary", nargs=2, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    named_summaries = [
        (name, json.loads(Path(path).read_text(encoding="utf-8")))
        for name, path in args.summary
    ]
    result = compare_methods(manifest, named_summaries)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for name, metrics in result["methods"].items():
        print(
            f"{name}: {metrics['success_count']}/{metrics['trajectory_count']} "
            f"({metrics['success_rate']:.1%})"
        )
    for name, comparison in result["comparisons_to_baseline"].items():
        print(
            f"{name} vs {comparison['baseline']}: "
            f"+{comparison['added_success']} -{comparison['lost_success']} "
            f"net={comparison['net_success_change']:+d}"
        )
    print(f"output={args.output.resolve()}")


if __name__ == "__main__":
    main()
