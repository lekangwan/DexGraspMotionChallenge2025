#!/usr/bin/env python3
"""为Wuji 50类解剖约束候选选择少量人工视频审查案例。

输入：50类物理评测摘要和逐步策略trace。
输出：4条成功最坏手型、4条干净成功和4条诊断失败的选择JSON及中文索引。
内部逻辑：按真实DIP低于-5.5度的次数/最小值排序成功案例；失败案例分别
覆盖极端反弯与接近30 cm却未稳定的边界情况，不读取test数据。
作用：在扩展正式1000条之前，用最少视频检查数值指标是否符合肉眼判断。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SUMMARY = (
    PROJECT_ROOT
    / "retarget_research/outputs/wuji_anatomy_train50_coupled_v1_evaluation/manifest_evaluation_summary.json"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "retarget_research/advanced_policy/video_audit/wuji_anatomy_train50_v1/wuji_selection.json"
)
SEVERE_BOUND_RAD = np.deg2rad(-5.5)


def enrich_row(row):
    """输入一条评测记录，输出附带真实DIP反弯和物理高度指标的案例字典。"""
    trace_path = Path(row["policy_trace"])
    with np.load(trace_path, allow_pickle=False) as trace:
        metadata = json.loads(str(trace["metadata_json"]))
        names = list(metadata["physics_dof_names"])
        indices = [names.index(f"finger{finger}_joint4") for finger in range(2, 6)]
        actual = np.asarray(trace["hand_dof_position"][:, indices], dtype=np.float64)
    physics = json.loads(Path(row["physics_report"]).read_text(encoding="utf-8"))
    severe = actual < SEVERE_BOUND_RAD
    return {
        "hand": "wuji",
        "object_name": row["object_name"],
        "category": row["category"],
        "source_trajectory_index": int(row["source_trajectory_index"]),
        "target_trajectory_index": int(row["target_trajectory_index"]),
        "physics_report": row["physics_report"],
        "policy_trace": row["policy_trace"],
        "success": bool(row["success"]),
        "max_lift_m": float(row["max_lift_m"]),
        "final_lift_m": float(row["final_lift_m"]),
        "terminal_min_lift_m": float(physics["terminal_min_lift_m"]),
        "distal_severe_count": int(severe.sum()),
        "distal_severe_count_after_step120": int(severe[120:].sum()),
        "minimum_distal_angle_rad": float(actual.min()),
    }


def unique_take(rows, count, used):
    """按已有顺序取指定数量且不重复轨迹，并把所取键写入used集合。"""
    selected = []
    for row in rows:
        key = row["object_name"], row["source_trajectory_index"]
        if key in used:
            continue
        selected.append(row)
        used.add(key)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError(f"只能选出{len(selected)}/{count}条不重复案例")
    return selected


def select_cases(rows):
    """输入全部50类案例，输出三组互不重复且各4条的审查案例。"""
    successful = [row for row in rows if row["success"]]
    failed = [row for row in rows if not row["success"]]
    used = set()
    worst_success = unique_take(
        sorted(
            successful,
            key=lambda row: (
                row["minimum_distal_angle_rad"],
                -row["distal_severe_count"],
                row["object_name"],
            ),
        ),
        4,
        used,
    )
    clean_success = unique_take(
        sorted(
            [row for row in successful if row["distal_severe_count"] == 0],
            key=lambda row: (-row["final_lift_m"], row["object_name"]),
        ),
        4,
        used,
    )
    severe_failures = sorted(
        failed,
        key=lambda row: (
            row["minimum_distal_angle_rad"],
            -row["distal_severe_count"],
            row["object_name"],
        ),
    )
    near_lift_failures = sorted(
        failed,
        key=lambda row: (-row["max_lift_m"], row["object_name"]),
    )
    diagnostic_failures = unique_take(
        severe_failures[:2] + near_lift_failures,
        4,
        used,
    )
    for row in worst_success:
        row["selection_reason"] = "stable_success_with_worst_approach_anatomy"
    for row in clean_success:
        row["selection_reason"] = "stable_success_with_zero_severe_distal_violation"
    for row in diagnostic_failures:
        row["selection_reason"] = "failure_extreme_anatomy_or_near_lift_boundary"
    return {
        "successful_worst_anatomy": worst_success,
        "successful_clean": clean_success,
        "failure_diagnostics": diagnostic_failures,
    }


def write_index(path, groups):
    """把选择分组写成包含高度和关节角的中文观看索引。"""
    lines = [
        "# Wuji 50类候选视频审查索引",
        "",
        "重点观察前2秒普通指远端是否出现肉眼明显反弯，以及成功案例抬升后是否一直保持到结尾。",
        "",
    ]
    for group, rows in groups.items():
        lines.extend([f"## {group}", ""])
        for row in rows:
            lines.append(
                f"- {row['object_name']}[{row['source_trajectory_index']}]："
                f"success={row['success']}，max/final={row['max_lift_m']:.3f}/"
                f"{row['final_lift_m']:.3f} m，DIP最小={np.degrees(row['minimum_distal_angle_rad']):.1f}°，"
                f"明显越界{row['distal_severe_count']}次。"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    """解析参数、确定12条案例并写出渲染器可直接读取的选择文件。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    rows = [enrich_row(row) for row in summary["results"]]
    groups = select_cases(rows)
    result = {
        "schema_version": 1,
        "summary_kind": "expert_replay",
        "selection_rule": "train50 only; 4 worst-anatomy successes + 4 clean successes + 4 diagnostic failures; no test data",
        "groups": groups,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_index(args.output.parent / "WATCHING_INDEX.md", groups)
    print(f"selected={sum(len(rows) for rows in groups.values())}")
    print(f"output={args.output.resolve()}")


if __name__ == "__main__":
    main()
