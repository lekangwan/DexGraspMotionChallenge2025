#!/usr/bin/env python3
"""复核重定向“成功轨迹”是否在抬升阶段仍有真实手物接触。

输入：策略split，以及若干`手名=物理评测摘要`。
输出：按手/划分/类别统计的JSON，另列出官方成功但接触支撑不足的异常轨迹。
内部逻辑：逐步重新计算高度、水平漂移和接触同时成立的最长连续区间，并与官方
要求的持续步数比较；这里只审计现有物理结果，不重新运行Isaac Gym。
作用：防止“物体碰巧飞起”被当作教师数据，并为报告明确成功标签的实际质量。
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np


def longest_true_run(values):
    """输入一维布尔序列，输出其中连续True的最大长度。"""
    longest = current = 0
    for value in np.asarray(values, dtype=bool):
        current = current + 1 if value else 0
        longest = max(longest, current)
    return int(longest)


def parse_hand_reports(values):
    """把重复的`HAND=PATH`参数解析成不重复的手到绝对路径映射。"""
    reports = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--hand-report必须写成HAND=PATH")
        hand, raw_path = value.split("=", 1)
        if hand in reports:
            raise ValueError(f"hand重复: {hand}")
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        reports[hand] = path
    return reports


def audit_one_physics_report(path):
    """计算一条轨迹的接触支撑质量。

    输入：逐轨迹physics JSON路径。
    输出：官方成功、接触支撑成功、最长支撑步数及关键标识。
    内部逻辑：`高度达标 AND 水平漂移合格 AND 接触数>0`必须连续满足官方步数。
    作用：使用比官方最终布尔值更明确的物理证据审计教师标签。
    """
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    positions = np.asarray(report["object_positions_m"], dtype=np.float64)
    contacts = np.asarray(
        report["hand_object_contact_count_per_step"], dtype=np.int64
    )
    initial = np.asarray(report["initial_object_position_m"], dtype=np.float64)
    if len(positions) != len(contacts):
        raise ValueError(f"位置和接触序列长度不一致: {path}")
    lift = positions[:, 2] - initial[2]
    drift = np.linalg.norm(positions[:, :2] - initial[:2], axis=1)
    valid_lift = (
        (lift >= float(report["lift_threshold_m"]))
        & (drift <= float(report["max_allowed_xy_drift_m"]))
    )
    supported = valid_lift & (contacts > 0)
    longest = longest_true_run(supported)
    required = int(report["required_sustain_steps"])
    return {
        "object_name": report["object_name"],
        "source_trajectory_index": int(report["source_trajectory_index"]),
        "official_success": bool(report["success"]),
        "contact_supported_success": longest >= required,
        "longest_contact_supported_steps": longest,
        "required_sustain_steps": required,
        "contact_steps": int(np.count_nonzero(contacts > 0)),
    }


def audit(policy_split_path, hand_reports):
    """汇总所有手的接触支撑审计。

    输入：策略split JSON和手到评测摘要路径映射。
    输出：完整可序列化审计字典。
    内部逻辑：以`物体+源轨迹索引`连接split；分别统计train/valid/test和类别覆盖。
    作用：确认训练教师使用的成功轨迹具有接触证据，并暴露缺专家的类别。
    """
    policy_split_path = Path(policy_split_path).expanduser().resolve()
    split = json.loads(policy_split_path.read_text(encoding="utf-8"))
    record_by_key = {
        (item["object_name"], int(item["source_trajectory_index"])): item
        for item in split["records"]
    }
    output = {
        "schema_version": 1,
        "definition": (
            "contact_supported_success = lift_and_xy_rule AND contact_count>0 "
            "for required_sustain_steps consecutive simulation steps"
        ),
        "policy_split": str(policy_split_path),
        "hands": {},
    }
    for hand, summary_path in hand_reports.items():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        counts = defaultdict(lambda: defaultdict(int))
        category_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        exceptions = []
        seen = set()
        for compact in summary["results"]:
            item = audit_one_physics_report(compact["physics_report"])
            key = (item["object_name"], item["source_trajectory_index"])
            if key not in record_by_key:
                raise ValueError(f"{hand}评测含策略split外轨迹: {key}")
            if key in seen:
                raise ValueError(f"{hand}评测轨迹重复: {key}")
            seen.add(key)
            record = record_by_key[key]
            split_name, category = record["split"], record["category"]
            counts[split_name]["trajectory_count"] += 1
            category_counts[split_name][category]["trajectory_count"] += 1
            for field in ("official_success", "contact_supported_success"):
                counts[split_name][field + "_count"] += int(item[field])
                category_counts[split_name][category][field + "_count"] += int(item[field])
            if item["official_success"] and not item["contact_supported_success"]:
                exceptions.append({**item, "split": split_name, "category": category})
        if seen != set(record_by_key):
            raise ValueError(f"{hand}评测没有覆盖策略split全部轨迹")
        split_results = {}
        for split_name in ("train", "valid", "test"):
            values = dict(counts[split_name])
            total = values["trajectory_count"]
            values["official_success_rate"] = values["official_success_count"] / total
            values["contact_supported_success_rate"] = (
                values["contact_supported_success_count"] / total
            )
            values["missing_official_success_categories"] = sorted(
                category
                for category, category_value in category_counts[split_name].items()
                if category_value["official_success_count"] == 0
            )
            values["missing_contact_supported_categories"] = sorted(
                category
                for category, category_value in category_counts[split_name].items()
                if category_value["contact_supported_success_count"] == 0
            )
            split_results[split_name] = values
        output["hands"][hand] = {
            "evaluation_summary": str(summary_path),
            "splits": split_results,
            "official_success_without_sustained_contact": exceptions,
        }
    return output


def main():
    """解析路径、执行审计、写JSON并输出各手训练集的核心计数。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-split", type=Path, required=True)
    parser.add_argument("--hand-report", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = parse_hand_reports(args.hand_report)
    result = audit(args.policy_split, reports)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    for hand, hand_result in result["hands"].items():
        train = hand_result["splits"]["train"]
        print(
            f"{hand}: official={train['official_success_count']} "
            f"contact_supported={train['contact_supported_success_count']}"
        )
    print(f"EXPERT_QUALITY_AUDIT={output}")


if __name__ == "__main__":
    main()
