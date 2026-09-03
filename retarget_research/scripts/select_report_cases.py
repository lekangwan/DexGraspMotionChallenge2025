#!/usr/bin/env python3
"""从专家或策略评测摘要自动选择有代表性的成功与失败视频案例。

输入：`manifest_evaluation_summary.json`或`policy_evaluation_summary.json`及每类数量。
输出：成功、接近成功、抬起后滑落、低抬升失败的去重案例JSON。
内部逻辑：优先保持类别多样；按持续时间、距离阈值和最大—最终高度差进行确定排序。
作用：避免报告只人工挑最好看的轨迹，同时减少录像到少量有解释价值的案例。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def diverse_take(items, count, score_key, reverse=True, used=None):
    """按分数排序后优先选择尚未出现的类别和未使用轨迹。

    输入：结果、数量、分数函数、排序方向和已用键集合。
    输出：选中结果列表；同时原地更新used。
    内部逻辑：第一遍保证类别多样，第二遍再按分数补足。
    作用：有限视频位优先覆盖不同物体类别，而非全来自同一容易类别。
    """
    used = used if used is not None else set()
    ordered = sorted(items, key=score_key, reverse=reverse)
    selected = []
    selected_categories = set()
    for require_new_category in (True, False):
        for item in ordered:
            key = (item["object_name"], int(item["source_trajectory_index"]))
            category = item.get("category")
            if key in used or item in selected:
                continue
            if require_new_category and category in selected_categories:
                continue
            selected.append(item)
            used.add(key)
            selected_categories.add(category)
            if len(selected) >= count:
                return selected
    return selected


def normalize_result(item):
    """把专家/策略摘要的字段差异统一为案例选择所需结构。"""
    return {
        **item,
        "max_lift_m": float(item.get("max_lift_m", 0.0)),
        "final_lift_m": float(item.get("final_lift_m", 0.0)),
        "longest_sustained_lift_time_s": float(item.get("longest_sustained_lift_time_s", 0.0)),
        "hand_object_contact_steps": int(item.get("hand_object_contact_steps", 0)),
        "success": bool(item["success"]),
    }


def choose_cases(results, count_per_group):
    """按照四类物理现象选择互不重复的报告案例。

    输入：统一逐轨迹结果和每组目标数量。
    输出：带选择原因的分组字典。
    内部逻辑：成功取持续/最终高度高者；近失取最接近10 cm和0.5 s者；
    滑落要求曾越过10 cm且最大—最终高度明显；剩余失败取接触多但抬升低者。
    作用：为报告准备“成果、差一点、滑移、未形成抓取”四种可解释视频。
    """
    used = set()
    success = [item for item in results if item["success"]]
    failed = [item for item in results if not item["success"]]
    groups = {}
    groups["success"] = diverse_take(
        success,
        count_per_group,
        lambda item: (item["longest_sustained_lift_time_s"], item["final_lift_m"]),
        True,
        used,
    )
    near = [
        item for item in failed
        if item["max_lift_m"] >= 0.07 or item["longest_sustained_lift_time_s"] >= 0.20
    ]
    groups["near_miss"] = diverse_take(
        near,
        count_per_group,
        lambda item: abs(0.10 - min(item["max_lift_m"], 0.10))
        + max(0.0, 0.50 - item["longest_sustained_lift_time_s"]),
        False,
        used,
    )
    slip = [
        item for item in failed
        if item["max_lift_m"] >= 0.10
        and item["max_lift_m"] - item["final_lift_m"] >= 0.03
    ]
    groups["lift_then_slip"] = diverse_take(
        slip,
        count_per_group,
        lambda item: item["max_lift_m"] - item["final_lift_m"],
        True,
        used,
    )
    groups["low_lift_failure"] = diverse_take(
        failed,
        count_per_group,
        lambda item: (item["hand_object_contact_steps"], -item["max_lift_m"]),
        True,
        used,
    )
    reasons = {
        "success": "strict_success_with_long_sustained_lift",
        "near_miss": "failed_but_close_to_lift_or_sustain_threshold",
        "lift_then_slip": "crossed_lift_threshold_then_lost_at_least_3cm",
        "low_lift_failure": "contact_or_attempt_without_sufficient_lift",
    }
    return {
        name: [{**item, "selection_reason": reasons[name]} for item in items]
        for name, items in groups.items()
    }


def choose_final_retargeting_cases(results, count_per_group):
    """按最终v3协议选择稳定成功、到达但不稳、未到达三类案例。

    输入：带reference/stable/transport字段的最终审计逐轨迹结果。
    输出：三组互不重复且类别尽量多样的案例。
    内部逻辑：成功优先低滑移；不稳定案例优先明显回落或掌物相对运动；
    未到达案例优先选择已有接触但抬升不足者，便于视频解释失败原因。
    作用：确保最终视频与冻结报告口径一致，不再按历史30 cm success字段选片。
    """
    used = set()
    stable = [item for item in results if item.get("training_eligible", False)]
    reached_unstable = [
        item for item in results
        if item.get("reference_isaac_success", False)
        and not item.get("training_eligible", False)
    ]
    not_reached = [
        item for item in results if not item.get("reference_isaac_success", False)
    ]
    groups = {
        "stable_transport_success": diverse_take(
            stable,
            count_per_group,
            lambda item: (
                -float(item.get("max_palm_relative_translation_change_m") or 1.0),
                -float(item.get("max_palm_relative_rotation_change_deg") or 360.0),
                float(item.get("terminal_min_lift_m") or 0.0),
            ),
            True,
            used,
        ),
        "reached_but_unstable": diverse_take(
            reached_unstable,
            count_per_group,
            lambda item: (
                float(item.get("peak_to_final_drop_m") or 0.0),
                float(item.get("max_palm_relative_translation_change_m") or 0.0),
                float(item.get("max_palm_relative_rotation_change_deg") or 0.0),
            ),
            True,
            used,
        ),
        "failed_to_reach": diverse_take(
            not_reached,
            count_per_group,
            lambda item: (
                int(item.get("hand_object_contact_steps", 0)),
                float(item.get("max_lift_m", 0.0)),
            ),
            True,
            used,
        ),
    }
    reasons = {
        "stable_transport_success": "reference_success_and_stable_transport_training_gate",
        "reached_but_unstable": "reached_reference_goal_but_failed_terminal_or_transport_gate",
        "failed_to_reach": "did_not_reach_reference_goal_despite_contact_attempt",
    }
    return {
        name: [{**item, "selection_reason": reasons[name]} for item in items]
        for name, items in groups.items()
    }


def main():
    """读取摘要、选择案例并写出可交给渲染器的JSON。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count-per-group", type=int, default=2)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    results = [normalize_result(item) for item in summary.get("results", [])]
    if not results:
        raise ValueError("评测摘要没有逐轨迹results")
    final_protocol = any("reference_isaac_success" in item for item in results)
    groups = (
        choose_final_retargeting_cases(results, args.count_per_group)
        if final_protocol else choose_cases(results, args.count_per_group)
    )
    output = {
        "schema_version": 1,
        "source_summary": str(args.summary.resolve()),
        "summary_kind": "policy" if "policy_split" in summary else "expert_replay",
        "selection_rule": (
            "final_v3_reference_stability_transport_with_category_diversity"
            if final_protocol else "deterministic_physics_metrics_with_category_diversity"
        ),
        "groups": groups,
        "selected_count": sum(len(items) for items in groups.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print({name: len(items) for name, items in groups.items()})
    print(f"REPORT_CASES={args.output.resolve()}")


if __name__ == "__main__":
    main()
