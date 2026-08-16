#!/usr/bin/env python3
"""为Wuji拇指零空间50类确认选择12条针对性视频案例。

输入：新旧方法比较、0.05物理评测摘要和拇指手型审计。
输出：5条新增运输成功、3条运输退化、3条最坏拇指接近段和1条干净成功，
以及中文观看索引。
内部逻辑：全部使用冻结train50结果；配对集合由严格运输键决定，拇指异常按
近90度比例/中位角排序并排除重复，干净案例从剩余运输成功中确定性选择。
作用：在正式1000条前用最少视频同时检查真实提升、滑移代价和局部手型风险。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_COMPARISON = (
    PROJECT_ROOT
    / "retarget_research/outputs/wuji_method_reassessment_v1/train50_final_comparison/wuji_anatomy_ablation_summary.json"
)
DEFAULT_PHYSICS = (
    PROJECT_ROOT
    / "retarget_research/outputs/wuji_method_reassessment_v1/thumb_nullspace_n005_train50_evaluation/manifest_evaluation_summary.json"
)
DEFAULT_THUMB = (
    PROJECT_ROOT
    / "retarget_research/outputs/wuji_method_reassessment_v1/thumb_nullspace_n005_train50_analysis/wuji_thumb_nullspace_summary.json"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "retarget_research/advanced_policy/video_audit/wuji_thumb_nullspace_train50_v1/wuji_selection.json"
)


def key(row):
    """输入案例字典，输出`(物体名,源索引)`唯一键。"""
    return row["object_name"], int(row["source_trajectory_index"])


def selection_item(row, thumb_row, reason, transport_success):
    """合并物理/手型指标，输出渲染器可直接读取的案例。"""
    return {
        "hand": "wuji",
        "object_name": row["object_name"],
        "category": row.get("category"),
        "source_trajectory_index": int(row["source_trajectory_index"]),
        "target_trajectory_index": int(row["target_trajectory_index"]),
        "physics_report": row["physics_report"],
        "policy_trace": row["policy_trace"],
        "success": bool(row["success"]),
        "transport_success": bool(transport_success),
        "max_lift_m": float(row["max_lift_m"]),
        "final_lift_m": float(row["final_lift_m"]),
        "thumb_joint4_median_deg": float(thumb_row["angles"]["median_deg"]),
        "thumb_joint4_near_90_ratio": float(
            thumb_row["angles"]["near_85_to_95_ratio"]
        ),
        "thumb_tip_max_displacement_mm": float(
            thumb_row["thumb_tip_displacement"]["maximum_mm"]
        ),
        "selection_reason": reason,
    }


def take_unique(rows, count, used):
    """从已排序案例中取指定数量不重复项并更新used。"""
    selected = []
    for row in rows:
        if key(row) in used:
            continue
        selected.append(row)
        used.add(key(row))
        if len(selected) == count:
            return selected
    raise ValueError(f"只能选出{len(selected)}/{count}条不重复案例")


def select_cases(comparison, physics, thumb):
    """根据严格运输配对和拇指指标，构造四组共12条案例。"""
    old = comparison["point_coupled_50"]
    new = comparison["thumb_nullspace_n005_50"]
    old_transport = {tuple(value) for value in old["transport_keys"]}
    new_transport = {tuple(value) for value in new["transport_keys"]}
    physics_by_key = {key(row): row for row in physics["results"]}
    thumb_by_key = {key(row): row for row in thumb["results"]}
    if set(physics_by_key) != set(thumb_by_key):
        raise ValueError("物理与拇指审计的轨迹集不一致")

    gains = sorted(new_transport - old_transport)
    regressions = sorted(old_transport - new_transport)
    if len(gains) != 5 or len(regressions) != 3:
        raise ValueError(
            f"预期配对新增5/退化3，实际{len(gains)}/{len(regressions)}"
        )
    used = set(gains) | set(regressions)
    thumb_ranked = sorted(
        thumb["results"],
        key=lambda row: (
            -row["angles"]["near_85_to_95_ratio"],
            -row["angles"]["median_deg"],
            row["object_name"],
        ),
    )
    worst_thumb = take_unique(thumb_ranked, 3, used)
    clean_ranked = sorted(
        [
            thumb_by_key[item]
            for item in new_transport
            if thumb_by_key[item]["angles"]["near_85_to_95_ratio"] == 0.0
        ],
        key=lambda row: (
            row["thumb_tip_displacement"]["maximum_mm"],
            row["object_name"],
        ),
    )
    clean = take_unique(clean_ranked, 1, used)

    def build(keys_or_rows, reason):
        chosen_keys = [
            value if isinstance(value, tuple) else key(value)
            for value in keys_or_rows
        ]
        return [
            selection_item(
                physics_by_key[item], thumb_by_key[item], reason,
                item in new_transport,
            )
            for item in chosen_keys
        ]

    return {
        "new_transport_gains": build(gains, "new_strict_transport_success_vs_point"),
        "stable_but_transport_regressions": build(
            regressions, "still_lifted_30cm_but_palm_relative_transport_regressed"
        ),
        "worst_thumb_approach": build(
            worst_thumb, "highest_remaining_approach_near_90_ratio"
        ),
        "representative_clean_success": build(
            clean, "strict_transport_success_with_zero_near_90_frames"
        ),
    }


def write_index(path, groups):
    """把四组案例及重点观察项写成中文观看索引。"""
    lines = [
        "# Wuji拇指零空间 50类视频审查",
        "",
        "观察三点：拇指最后两节是否仍长时间形成生硬直角；物体是否稳定抬升到结尾；退化组是否有明显掌物相对滑移。",
        "",
    ]
    for group, rows in groups.items():
        lines.extend([f"## {group}", ""])
        for row in rows:
            lines.append(
                f"- {row['object_name']}[{row['source_trajectory_index']}]："
                f"stable={row['success']}，transport={row['transport_success']}，"
                f"final={row['final_lift_m']:.3f} m，joint4中位="
                f"{row['thumb_joint4_median_deg']:.1f}°，近90度="
                f"{row['thumb_joint4_near_90_ratio']:.1%}。"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    """解析三份报告，写出12条选择JSON和中文观看索引。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", type=Path, default=DEFAULT_COMPARISON)
    parser.add_argument("--physics-summary", type=Path, default=DEFAULT_PHYSICS)
    parser.add_argument("--thumb-summary", type=Path, default=DEFAULT_THUMB)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    comparison = json.loads(args.comparison.read_text(encoding="utf-8"))
    physics = json.loads(args.physics_summary.read_text(encoding="utf-8"))
    thumb = json.loads(args.thumb_summary.read_text(encoding="utf-8"))
    groups = select_cases(comparison, physics, thumb)
    result = {
        "schema_version": 1,
        "summary_kind": "expert_replay",
        "selection_rule": (
            "train50 only; all 5 strict gains + all 3 strict regressions + "
            "3 worst remaining thumb cases + 1 clean success; no test data"
        ),
        "groups": groups,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_index(args.output.parent / "WATCHING_INDEX.md", groups)
    print(f"selected={sum(len(rows) for rows in groups.values())}")
    print(f"output={args.output.resolve()}")


if __name__ == "__main__":
    main()
