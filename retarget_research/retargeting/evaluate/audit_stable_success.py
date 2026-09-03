#!/usr/bin/env python3
"""从已保存的三手PhysX曲线重算稳定成功和训练可用性。

输入：三手正式1000条摘要、逐轨迹物体/接触JSON、策略trace、对象split和冻结阈值。
输出：每手含1000条新标签的JSON、可用轨迹键JSON、三手总表和中文Markdown。
内部逻辑：先检查末段30步高度/回落/波动/接触，再把物体位姿变到手掌坐标系
检查运输期滑移；最后叠加手型解剖学门，Wuji在修复反向远端弯曲前隔离。
作用：不重跑PhysX就能剔除“抓起后掉落”和“抓形不可用”的伪专家。
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "retarget_research/retargeting/configs/stable_success_protocol_v2.json"
)
DEFAULT_SPLIT = (
    PROJECT_ROOT
    / "retarget_research/advanced_policy/data/formal_v1/policy_split_seed20260813.json"
)
DEFAULT_REPORTS = {
    "linker": PROJECT_ROOT / "retarget_research/outputs/formal_1000/linker_object_centric_3mm_v1_evaluation/manifest_evaluation_summary.json",
    "xhand": PROJECT_ROOT / "retarget_research/outputs/formal_1000/xhand_official_evaluation/manifest_evaluation_summary.json",
    "wuji": PROJECT_ROOT / "retarget_research/outputs/formal_1000/wuji_v1_evaluation/manifest_evaluation_summary.json",
}


def trajectory_key(item):
    """把逐轨迹记录转成稳定的对象名与源索引键。"""
    return item["object_name"], int(item["source_trajectory_index"])


def longest_true_run(values):
    """输入布尔序列，输出其中最长连续True段的步数。"""
    longest = current = 0
    for value in np.asarray(values, dtype=bool):
        current = current + 1 if value else 0
        longest = max(longest, current)
    return int(longest)


def reference_isaac_metrics(report, protocol):
    """按参考仓库Isaac主评测条件重算累计成功和末段成功。"""
    positions = np.asarray(report["object_positions_m"], dtype=np.float64)
    goal = np.asarray(protocol["goal_world_position_m"], dtype=np.float64)
    radius = float(protocol["goal_radius_m"])
    hold_steps = int(protocol["terminal_hold_steps"])
    reached = (
        (np.linalg.norm(positions - goal[None, :], axis=1) <= radius)
        | (positions[:, 2] >= goal[2])
    )
    hits = np.flatnonzero(reached)
    return {
        "reference_isaac_success": bool(len(hits)),
        "reference_isaac_terminal_success": bool(reached[-hold_steps:].all()),
        "reference_isaac_first_success_step": int(hits[0]) if len(hits) else None,
    }


def terminal_metrics(report, protocol):
    """重算单条轨迹的末段稳定抓取指标。

    输入：原物理JSON字典和高度/保持阈值。
    输出：旧成功、末段最低高度、峰值回落、波动、接触及新成功。
    内部逻辑：直接重用已保存的240步位置/接触曲线，不再运行仿真。
    作用：把“曾经越过阈值”和“最后仍稳定抓住”分开。
    """
    positions = np.asarray(report["object_positions_m"], dtype=np.float64)
    initial = np.asarray(report["initial_object_position_m"], dtype=np.float64)
    contacts = np.asarray(report["hand_object_contact_count_per_step"]) > 0
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"物体位置应为(T,3)，实际{positions.shape}")
    if contacts.shape != (len(positions),):
        raise ValueError("接触曲线与位置曲线不等长")
    hold_steps = int(protocol["terminal_hold_steps"])
    if not 1 <= hold_steps <= len(positions):
        raise ValueError("末段保持步数越界")
    lift = positions[:, 2] - initial[2]
    drift = np.linalg.norm(positions[:, :2] - initial[None, :2], axis=1)
    valid = (
        (lift >= float(protocol["lift_threshold_m"]))
        & (drift <= float(report["max_allowed_xy_drift_m"]))
    )
    terminal_lift = lift[-hold_steps:]
    terminal_contact_ratio = float(contacts[-hold_steps:].mean())
    peak_drop = float(lift.max() - lift[-1])
    terminal_range = float(np.ptp(terminal_lift))
    legacy = bool(
        longest_true_run(valid) >= int(report["required_sustain_steps"])
    )
    stable = bool(
        legacy
        and valid[-hold_steps:].all()
        and peak_drop <= float(protocol["max_peak_to_final_drop_m"])
        and terminal_range <= float(protocol["max_terminal_lift_range_m"])
        and terminal_contact_ratio >= float(protocol["min_terminal_contact_ratio"])
    )
    return {
        "legacy_success_recomputed": legacy,
        "terminal_min_lift_m": float(terminal_lift.min()),
        "terminal_lift_range_m": terminal_range,
        "peak_to_final_drop_m": peak_drop,
        "terminal_contact_ratio": terminal_contact_ratio,
        "stable_physics_success": stable,
    }


def transport_metrics(trace_path, protocol, terminal_hold_steps):
    """在手掌坐标系中测量物体离桌后的相对滑移。

    输入：对齐策略trace、运输阈值和末段长度。
    输出：起点、最大掌物相对平移/旋转及通过标志。
    内部逻辑：从物体抬升5 cm且有接触时开始，用实际腕XYZ+欧拉角
    把物体变到掌坐标系，相对末段中位姿测量最大偏移。
    作用：区分“手和物体一起运动”与“物体在手里滑动/翻滚”。
    """
    with np.load(trace_path, allow_pickle=False) as trace:
        hand_pose = np.asarray(trace["hand_dof_position"][:, :6], dtype=np.float64)
        object_position = np.asarray(trace["object_position"], dtype=np.float64)
        object_rotation = Rotation.from_quat(
            np.asarray(trace["object_quaternion_xyzw"], dtype=np.float64)
        )
        contacts = np.asarray(trace["hand_object_contact_count"]) > 0
    lifted = object_position[:, 2] - object_position[0, 2]
    start_candidates = np.flatnonzero(
        (lifted >= float(protocol["start_after_lift_m"])) & contacts
    )
    if not len(start_candidates):
        return {
            "transport_start_step": None,
            "max_palm_relative_translation_change_m": None,
            "max_palm_relative_rotation_change_deg": None,
            "transport_stability_success": False,
        }
    start = int(start_candidates[0])
    hand_rotation = Rotation.from_euler("xyz", hand_pose[:, 3:6])
    local_position = hand_rotation.inv().apply(
        object_position - hand_pose[:, :3]
    )
    terminal_position = np.median(local_position[-terminal_hold_steps:], axis=0)
    translation_change = float(
        np.linalg.norm(local_position[start:] - terminal_position, axis=1).max()
    )
    local_rotation = hand_rotation.inv() * object_rotation
    terminal_rotation = local_rotation[-1]
    rotation_change = float(
        np.degrees((terminal_rotation.inv() * local_rotation[start:]).magnitude()).max()
    )
    success = bool(
        translation_change
        <= float(protocol["max_palm_relative_translation_change_m"])
        and rotation_change
        <= float(protocol["max_palm_relative_rotation_change_deg"])
    )
    return {
        "transport_start_step": start,
        "max_palm_relative_translation_change_m": translation_change,
        "max_palm_relative_rotation_change_deg": rotation_change,
        "transport_stability_success": success,
    }


def reason_labels(terminal, transport, anatomy_passes, height_protocol):
    """根据各道门和冻结阈值输出可能包含多项的失败原因。"""
    reasons = []
    if not terminal["legacy_success_recomputed"]:
        reasons.append("never_sustained_legacy_lift")
    if terminal["terminal_min_lift_m"] < float(height_protocol["lift_threshold_m"]):
        reasons.append("not_lifted_through_terminal_hold")
    if terminal["peak_to_final_drop_m"] > float(
        height_protocol["max_peak_to_final_drop_m"]
    ):
        reasons.append("large_peak_to_final_drop")
    if terminal["terminal_lift_range_m"] > float(
        height_protocol["max_terminal_lift_range_m"]
    ):
        reasons.append("still_sliding_in_terminal_hold")
    if terminal["terminal_contact_ratio"] < float(
        height_protocol["min_terminal_contact_ratio"]
    ):
        reasons.append("terminal_contact_interrupted")
    if not transport["transport_stability_success"]:
        reasons.append("palm_relative_transport_slip")
    if not anatomy_passes:
        reasons.append("hand_anatomy_quarantined")
    return reasons


def audit_wuji_distal_flexion(rows):
    """统计Wuji四根普通手指远端关节的反向弯曲。

    输入：含`policy_trace`路径的1000条原评测记录。
    输出：finger2到5的joint4首步中位数、负值比例和贴近负下界的轨迹数。
    内部逻辑：Wuji手指动作为20维，每指4关节；因此普通指远端索引是
    7/11/15/19，小于-0.45 rad视为贴近URDF的-0.4932 rad负下界。
    作用：把视频中的“指尖反人类弯曲”转成可复核的数据证据。
    """
    names = (
        "finger2_joint4",
        "finger3_joint4",
        "finger4_joint4",
        "finger5_joint4",
    )
    indices = (7, 11, 15, 19)
    values = []
    for row in rows:
        with np.load(row["policy_trace"], allow_pickle=False) as trace:
            action = np.asarray(trace["policy_action"][:, 6:], dtype=np.float64)
        if action.ndim != 2 or action.shape[1] != 20:
            raise ValueError(f"Wuji手指动作应为(T,20)，实际{action.shape}")
        values.append(action[:, indices])
    stacked = np.stack(values)
    result = {}
    for column, name in enumerate(names):
        joint = stacked[:, :, column]
        result[name] = {
            "first_step_median_rad": float(np.median(joint[:, 0])),
            "first_step_below_minus_0p1_count": int((joint[:, 0] < -0.1).sum()),
            "negative_timestep_ratio": float((joint < 0).mean()),
            "trajectory_reaching_below_minus_0p45_count": int(
                (joint < -0.45).any(axis=1).sum()
            ),
            "minimum_rad": float(joint.min()),
        }
    return {
        "joint_semantics": "four_non_thumb_distal_flexion_joints",
        "trajectory_count": len(rows),
        "negative_direction_interpretation": (
            "visual reverse distal bending; optimization exploited URDF negative range"
        ),
        "per_joint": result,
    }


def audit_hand(hand, source_summary, split_by_key, config):
    """重算一只手的1000条轨迹并汇总新成功率。

    输入：手名、旧摘要、轨迹split表和完整协议。
    输出：含新指标的每条结果和分组汇总。
    内部逻辑：终态门、掌物滑移门、手型门串联，任一失败都不进训练集。
    作用：同时报告物理稳定能力和当前真正可用的专家数量。
    """
    summary = json.loads(source_summary.read_text(encoding="utf-8"))
    height_protocol = config["height_and_hold"]
    reference_protocol = config.get("reference_isaac")
    transport_protocol = config["transport_stability"]
    anatomy_value = config["training_anatomy_gate"][hand]
    anatomy_passes = not anatomy_value.startswith("quarantine")
    rows = []
    reason_counts = Counter()
    for original in summary["results"]:
        physics = json.loads(
            Path(original["physics_report"]).read_text(encoding="utf-8")
        )
        terminal = terminal_metrics(physics, height_protocol)
        reference = (
            reference_isaac_metrics(physics, reference_protocol)
            if reference_protocol is not None
            else {}
        )
        transport = transport_metrics(
            Path(original["policy_trace"]),
            transport_protocol,
            int(height_protocol["terminal_hold_steps"]),
        )
        transport_quality = bool(
            terminal["stable_physics_success"]
            and transport["transport_stability_success"]
        )
        training_eligible = bool(transport_quality and anatomy_passes)
        reasons = reason_labels(
            terminal, transport, anatomy_passes, height_protocol
        )
        reason_counts.update(reasons)
        rows.append(
            {
                **original,
                "legacy_success_from_source": bool(original["success"]),
                **reference,
                **terminal,
                **transport,
                "transport_quality_success": transport_quality,
                "anatomy_gate": anatomy_value,
                "training_eligible": training_eligible,
                "policy_split": split_by_key.get(trajectory_key(original), "unknown"),
                "quality_failure_reasons": reasons,
            }
        )
    split_counts = {}
    for split in sorted({row["policy_split"] for row in rows}):
        selected = [row for row in rows if row["policy_split"] == split]
        split_counts[split] = {
            "trajectory_count": len(selected),
            "stable_physics_count": sum(row["stable_physics_success"] for row in selected),
            "transport_quality_count": sum(row["transport_quality_success"] for row in selected),
            "training_eligible_count": sum(row["training_eligible"] for row in selected),
        }
    anatomy_diagnostics = (
        audit_wuji_distal_flexion(summary["results"])
        if hand == "wuji"
        else None
    )
    result = {
        "schema_version": int(config.get("schema_version", 2)),
        "hand": hand,
        "source_summary": str(source_summary.resolve()),
        "success_protocol": config.get(
            "protocol_id", "stable_30cm_terminal_and_palm_relative_transport_v2"
        ),
        "anatomy_gate": anatomy_value,
        "trajectory_count": len(rows),
        "source_10cm_success_count": sum(
            row["legacy_success_from_source"] for row in rows
        ),
        "legacy_success_count": sum(row["legacy_success_recomputed"] for row in rows),
        "stable_physics_success_count": sum(row["stable_physics_success"] for row in rows),
        "transport_quality_success_count": sum(row["transport_quality_success"] for row in rows),
        "training_eligible_count": sum(row["training_eligible"] for row in rows),
        "stable_physics_success_rate": float(np.mean([row["stable_physics_success"] for row in rows])),
        "transport_quality_success_rate": float(np.mean([row["transport_quality_success"] for row in rows])),
        "training_eligible_rate": float(np.mean([row["training_eligible"] for row in rows])),
        "transport_quality_category_count": len({row["category"] for row in rows if row["transport_quality_success"]}),
        "transport_quality_object_count": len({row["object_name"] for row in rows if row["transport_quality_success"]}),
        "failure_reason_counts": dict(sorted(reason_counts.items())),
        "anatomy_diagnostics": anatomy_diagnostics,
        "per_policy_split": split_counts,
        "results": rows,
    }
    if reference_protocol is not None:
        result.update({
            "reference_isaac_success_count": sum(
                row["reference_isaac_success"] for row in rows
            ),
            "reference_isaac_terminal_success_count": sum(
                row["reference_isaac_terminal_success"] for row in rows
            ),
        })
    return result


def write_markdown(path, summaries, config_path):
    """把三手旧/终态/运输/可训练数量写成简明中文表格。"""
    config = json.loads(config_path.read_text(encoding="utf-8"))
    lift_cm = float(config["height_and_hold"]["lift_threshold_m"]) * 100.0
    lift_label = f"{lift_cm:g} cm"
    lines = [
        f"# 稳定成功轨迹重审 v{config.get('schema_version', 2)}",
        "",
        f"协议：`{config_path.resolve()}`",
        "",
        f"| 手 | 原10 cm旧报告 | 曾连续达到{lift_label} | 末段稳定{lift_label} | 运输不滑移 | 当前可训练 | 运输质量覆盖类别/物体 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for hand in ("linker", "xhand", "wuji"):
        item = summaries[hand]
        total = item["trajectory_count"]
        lines.append(
            f"| {hand} | {item['source_10cm_success_count']}/{total} | "
            f"{item['legacy_success_count']}/{total} | "
            f"{item['stable_physics_success_count']}/{total} | "
            f"{item['transport_quality_success_count']}/{total} | "
            f"{item['training_eligible_count']}/{total} | "
            f"{item['transport_quality_category_count']}/{item['transport_quality_object_count']} |"
        )
    lines.append("")
    quarantined = [
        hand for hand in ("linker", "xhand", "wuji")
        if summaries[hand]["training_eligible_count"]
        < summaries[hand]["transport_quality_success_count"]
    ]
    if quarantined:
        lines.append(
            "手型门仍排除部分轨迹的手："
            + "、".join(f"`{hand}`" for hand in quarantined)
            + "。"
        )
    else:
        lines.append(
            "三只手当前均已通过各自的手型检查，因此可训练数等于运输质量通过数。"
        )
    lines.extend(
        [
            "",
            "进阶训练只使用“运输不滑移且通过手型检查”的轨迹；不使用旧的10 cm中途越线标签。",
        ]
    )
    wuji_joints = summaries["wuji"]["anatomy_diagnostics"]["per_joint"]
    lines.extend(
        [
            "",
            "## Wuji反向远端弯曲证据",
            "",
            "| 关节 | 第一执行步中位数 | 整体负值比例 | 曾超过-0.45 rad的轨迹 |",
            "|---|---:|---:|---:|",
        ]
    )
    for name, item in wuji_joints.items():
        lines.append(
            f"| {name} | {item['first_step_median_rad']:.3f} rad | "
            f"{item['negative_timestep_ratio']:.1%} | "
            f"{item['trajectory_reaching_below_minus_0p45_count']}/1000 |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    """解析路径，秒级重算三手1000条并写出审计包。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--policy-split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--linker-report", type=Path, default=DEFAULT_REPORTS["linker"])
    parser.add_argument("--xhand-report", type=Path, default=DEFAULT_REPORTS["xhand"])
    parser.add_argument("--wuji-report", type=Path, default=DEFAULT_REPORTS["wuji"])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    split = json.loads(args.policy_split.read_text(encoding="utf-8"))
    split_by_key = {trajectory_key(item): item["split"] for item in split["records"]}
    report_paths = {
        "linker": args.linker_report,
        "xhand": args.xhand_report,
        "wuji": args.wuji_report,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for hand in ("linker", "xhand", "wuji"):
        summary = audit_hand(hand, report_paths[hand], split_by_key, config)
        summaries[hand] = summary
        (args.output_dir / f"{hand}_stable_audit.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        accepted = [
            [row["object_name"], int(row["source_trajectory_index"])]
            for row in summary["results"]
            if row["training_eligible"]
        ]
        (args.output_dir / f"{hand}_training_eligible_keys.json").write_text(
            json.dumps(accepted, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        total = summary["trajectory_count"]
        print(
            f"{hand}: source10cm={summary['source_10cm_success_count']}/{total} "
            f"reached={summary['legacy_success_count']}/{total} "
            f"stable={summary['stable_physics_success_count']}/{total} "
            f"transport={summary['transport_quality_success_count']}/{total} "
            f"training={summary['training_eligible_count']}/{total}",
            flush=True,
        )
    compact = {
        hand: {key: value for key, value in summary.items() if key != "results"}
        for hand, summary in summaries.items()
    }
    (args.output_dir / "three_hand_stable_audit_summary.json").write_text(
        json.dumps(compact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_markdown(args.output_dir / "STABLE_SUCCESS_AUDIT.md", summaries, args.config)
    print(f"STABLE_SUCCESS_AUDIT={args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
