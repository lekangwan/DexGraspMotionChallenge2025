#!/usr/bin/env python3
"""汇总Wuji手型2×2消融的物理成功、运输稳定性和真实关节姿态。

输入：各候选的manifest评测摘要、策略trace及重定向npy。
输出：逐候选JSON、中文Markdown以及相对旧基线的逐轨迹得失。
内部逻辑：复用冻结的掌物相对运输判据，并统一统计四根普通指PIP/DIP
低于-5度的比例；另用0.5度容差区分PhysX数值穿透与明显反弯。
作用：在同一train20屏幕上选择唯一Wuji重定向方法，避免凭单个视频挑方法。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from audit_stable_success import transport_metrics


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_STUDY = (
    PROJECT_ROOT
    / "retarget_research/retargeting/configs/wuji_anatomy_ablation_v1.json"
)
DEFAULT_EVALUATION_ROOT = (
    PROJECT_ROOT / "retarget_research/outputs/wuji_anatomy_ablation_v1_evaluation"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "retarget_research/outputs/wuji_anatomy_ablation_v1_analysis"
)
DISTAL_JOINTS = [f"finger{finger}_joint4" for finger in range(2, 6)]
PIP_JOINTS = [f"finger{finger}_joint3" for finger in range(2, 6)]
BOUND_RAD = np.deg2rad(-5.0)
SEVERE_BOUND_RAD = np.deg2rad(-5.5)


def trajectory_key(row):
    """输入评测记录，输出可跨候选对齐的对象名与源轨迹索引。"""
    return row["object_name"], int(row["source_trajectory_index"])


def joint_violation_counts(trace_path, joint_names):
    """统计一条trace中指定真实关节的全过程与末段反弯。

    输入：trace路径和关节名列表。
    输出：样本总数、严格低于-5度和明显低于-5.5度的全程/末段次数。
    内部逻辑：从trace元数据恢复物理DOF顺序，只读取实际关节位置而非命令。
    作用：检查优化限制在接触仿真中是否仍能保持自然手型。
    """
    with np.load(trace_path, allow_pickle=False) as trace:
        actual = np.asarray(trace["hand_dof_position"], dtype=np.float64)
        metadata = json.loads(str(trace["metadata_json"]))
    names = list(metadata["physics_dof_names"])
    indices = [names.index(name) for name in joint_names]
    values = actual[:, indices]
    terminal = values[-30:]
    return {
        "all_value_count": int(values.size),
        "all_strict_count": int((values < BOUND_RAD).sum()),
        "all_severe_count": int((values < SEVERE_BOUND_RAD).sum()),
        "terminal_value_count": int(terminal.size),
        "terminal_strict_count": int((terminal < BOUND_RAD).sum()),
        "terminal_severe_count": int((terminal < SEVERE_BOUND_RAD).sum()),
        "minimum_rad": float(values.min()),
    }


def mean_optimization_loss(target_directory):
    """输入候选npy目录，输出全部轨迹和帧上的平均关键点优化损失。"""
    losses = []
    for path in sorted(target_directory.glob("*.npy")):
        data = np.load(path, allow_pickle=True).item()
        losses.append(np.asarray(data["optimization_loss_per_frame"], dtype=np.float64))
    if not losses:
        raise ValueError(f"没有找到重定向npy: {target_directory}")
    return float(np.concatenate([value.reshape(-1) for value in losses]).mean())


def summarize_candidate(name, summary_path, protocol):
    """读取一个候选并汇总稳定成功、运输成功、优化误差与PIP/DIP反弯。"""
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    stable_keys, transport_keys = set(), set()
    distal_totals = pip_totals = None
    for row in summary["results"]:
        key = trajectory_key(row)
        physics = json.loads(Path(row["physics_report"]).read_text(encoding="utf-8"))
        stable = bool(physics["success"])
        transport = transport_metrics(
            Path(row["policy_trace"]),
            protocol["transport_stability"],
            int(protocol["height_and_hold"]["terminal_hold_steps"]),
        )["transport_stability_success"]
        if stable:
            stable_keys.add(key)
        if stable and transport:
            transport_keys.add(key)
        distal = joint_violation_counts(Path(row["policy_trace"]), DISTAL_JOINTS)
        pip = joint_violation_counts(Path(row["policy_trace"]), PIP_JOINTS)
        if distal_totals is None:
            distal_totals = {key: value for key, value in distal.items() if key != "minimum_rad"}
            distal_totals["minimum_rad"] = distal["minimum_rad"]
            pip_totals = {key: value for key, value in pip.items() if key != "minimum_rad"}
            pip_totals["minimum_rad"] = pip["minimum_rad"]
        else:
            for field in distal_totals:
                if field == "minimum_rad":
                    distal_totals[field] = min(distal_totals[field], distal[field])
                    pip_totals[field] = min(pip_totals[field], pip[field])
                else:
                    distal_totals[field] += distal[field]
                    pip_totals[field] += pip[field]
    target_directory = Path(summary["target_directory"])
    return {
        "name": name,
        "trajectory_count": len(summary["results"]),
        "stable_success_count": len(stable_keys),
        "transport_success_count": len(transport_keys),
        "mean_optimization_loss": mean_optimization_loss(target_directory),
        "mean_final_lift_m": float(summary["mean_final_lift_m"]),
        "stable_keys": sorted([list(key) for key in stable_keys]),
        "transport_keys": sorted([list(key) for key in transport_keys]),
        "distal_actual": distal_totals,
        "pip_actual": pip_totals,
    }


def add_baseline_differences(summaries):
    """输入全部候选摘要，原地加入相对旧基线的稳定/运输新增与丢失轨迹。"""
    baseline = summaries["legacy_unconstrained"]
    for item in summaries.values():
        for metric in ("stable", "transport"):
            current = {tuple(key) for key in item[f"{metric}_keys"]}
            reference = {tuple(key) for key in baseline[f"{metric}_keys"]}
            item[f"{metric}_gained_vs_legacy"] = sorted([list(key) for key in current - reference])
            item[f"{metric}_lost_vs_legacy"] = sorted([list(key) for key in reference - current])


def ratio(item, prefix, severe=False):
    """把某个关节统计字典转换成全程或末段越界比例。"""
    level = "severe" if severe else "strict"
    return item[f"{prefix}_{level}_count"] / item[f"{prefix}_value_count"]


def write_markdown(path, summaries):
    """把机器可读摘要转成便于人工决策的中文对比表。"""
    lines = [
        "# Wuji解剖约束2×2消融结果",
        "",
        "严格越界指真实关节低于-5度；明显越界指低于-5.5度，用于排除PhysX的微小数值穿透。运输成功还要求稳定30 cm且掌物相对位姿不过度滑移。",
        "",
        "| 候选 | 稳定30 cm | 运输成功 | 平均优化loss | DIP全程明显越界 | DIP末段明显越界 | PIP末段明显越界 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summaries.values():
        distal = item["distal_actual"]
        pip = item["pip_actual"]
        lines.append(
            f"| {item['name']} | {item['stable_success_count']}/20 | "
            f"{item['transport_success_count']}/20 | {item['mean_optimization_loss']:.5f} | "
            f"{ratio(distal, 'all', True):.2%} | {ratio(distal, 'terminal', True):.2%} | "
            f"{ratio(pip, 'terminal', True):.2%} |"
        )
    lines.extend(["", "## 相对旧基线的逐轨迹变化", ""])
    for name, item in summaries.items():
        if name == "legacy_unconstrained":
            continue
        gains = [f"{key[0]}[{key[1]}]" for key in item["transport_gained_vs_legacy"]]
        losses = [f"{key[0]}[{key[1]}]" for key in item["transport_lost_vs_legacy"]]
        lines.append(f"- `{name}`：运输新增 {', '.join(gains) or '无'}；丢失 {', '.join(losses) or '无'}。")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    """解析路径，汇总所有已完成候选并写出JSON和Markdown。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", type=Path, default=DEFAULT_STUDY)
    parser.add_argument("--evaluation-root", type=Path, default=DEFAULT_EVALUATION_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    study = json.loads(args.study.read_text(encoding="utf-8"))
    protocol_path = PROJECT_ROOT / "retarget_research/retargeting/configs/stable_success_protocol_v2.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    summaries = {}
    for candidate in study["candidates"]:
        name = candidate["name"]
        summary_path = args.evaluation_root / name / "manifest_evaluation_summary.json"
        if summary_path.exists():
            summaries[name] = summarize_candidate(name, summary_path, protocol)
    add_baseline_differences(summaries)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_json = args.output_dir / "wuji_anatomy_ablation_summary.json"
    output_json.write_text(json.dumps(summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(args.output_dir / "WUJI_ANATOMY_ABLATION_RESULTS.md", summaries)
    for item in summaries.values():
        print(
            f"{item['name']}: stable={item['stable_success_count']}/20 "
            f"transport={item['transport_success_count']}/20 "
            f"loss={item['mean_optimization_loss']:.5f} "
            f"terminal_distal_severe={ratio(item['distal_actual'], 'terminal', True):.2%}"
        )
    print(f"OUTPUT={args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
