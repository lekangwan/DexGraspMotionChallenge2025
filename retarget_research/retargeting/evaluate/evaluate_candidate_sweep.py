#!/usr/bin/env python3
"""顺序评测一组冻结候选，并汇总成可比较的超参数搜索表。

输入：搜索JSON，其中记录hand、manifest、候选目录、评测目录和参数。
输出：每个候选的统一PhysX报告，以及按主指标排列的搜索汇总JSON。
内部逻辑：逐候选调用共享manifest评测器，验证摘要的手、数据和目标路径，再提取成功与连续指标。
作用：让1000条正式评测前的小样本调参可续跑、可审计，不需人工拼接成功轨迹。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


EVALUATE_DIR = Path(__file__).resolve().parent
RETARGET_ROOT = EVALUATE_DIR.parent
PROJECT_ROOT = RETARGET_ROOT.parent.parent
EVALUATE_SCRIPT = EVALUATE_DIR / "evaluate_hand_manifest.py"
LINKER_ACTIVE_DOFS = (
    "rh_index_mcp_pitch",
    "rh_middle_mcp_pitch",
    "rh_pinky_mcp_pitch",
    "rh_ring_mcp_pitch",
    "rh_thumb_cmc_yaw",
    "rh_thumb_cmc_pitch",
)
LINKER_MIMIC_DOFS = (
    "rh_index_dip",
    "rh_middle_dip",
    "rh_pinky_dip",
    "rh_ring_dip",
    "rh_thumb_ip",
)


def resolve_project_path(value: str) -> Path:
    """把搜索配置中的路径解析为绝对路径。

    输入：绝对路径，或从项目根目录开始的相对路径字符串。
    输出：已解析的绝对`Path`。
    内部逻辑：相对路径固定与项目根拼接，不依赖启动时工作目录。
    作用：保证用户从不同目录执行时仍读写同一组实验文件。
    """
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def expected_trajectory_count(manifest_path: Path) -> int:
    """计算manifest中冻结的轨迹总数。

    输入：manifest JSON路径。
    输出：所有条目`trajectory_indices`长度之和。
    内部逻辑：解析JSON并显式求和，拒绝空样本。
    作用：续跑时判断已有评测是否真正完整。
    """
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    count = sum(len(entry["trajectory_indices"]) for entry in manifest["entries"])
    if count <= 0:
        raise ValueError(f"manifest没有轨迹: {manifest_path}")
    return count


def load_matching_summary(
    summary_path: Path,
    hand: str,
    manifest_path: Path,
    target_dir: Path,
    trajectory_count: int,
    expected_physics_options: dict | None = None,
) -> dict | None:
    """读取与当前候选完全匹配的已有评测摘要。

    输入：摘要路径、手类型、manifest、候选目录、期望轨迹数和可选物理参数子集。
    输出：全部字段匹配时返回字典，否则返回`None`。
    内部逻辑：检查文件存在后对比四个续跑关键字段，并逐项核对指定物理参数。
    作用：只跳过真正同配置的完整结果，防止同一轨迹在PD搜索中复用错误增益。
    """
    if not summary_path.is_file():
        return None
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        physics_options = summary.get("physics_options", {})
        physics_matches = all(
            physics_options.get(key) == value
            for key, value in (expected_physics_options or {}).items()
        )
        if (
            summary.get("hand") == hand
            and Path(summary["manifest"]).resolve() == manifest_path
            and Path(summary["target_directory"]).resolve() == target_dir
            and int(summary["trajectory_count"]) == trajectory_count
            and physics_matches
        ):
            return summary
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return None
    return None


def linker_tracking_metrics(summary: dict) -> dict:
    """汇总Linker命令角与实际关节角之间的误差。

    输入：含逐轨迹物理报告路径的manifest评测摘要。
    输出：主动轴、mimic轴平均绝对误差和所有手指轴最坏瞬时误差。
    内部逻辑：读取每条物理JSON，对各组DOF的MAE取等权平均，对峰值取最大值。
    作用：解释PD候选是否真的改善跟踪，避免仅凭成功率给刚度变化编造原因。
    """
    active_errors = []
    mimic_errors = []
    peak_errors = []
    for result in summary.get("results", []):
        report_path = Path(result["physics_report"])
        report = json.loads(report_path.read_text(encoding="utf-8"))
        mean_by_dof = report.get("mean_absolute_tracking_error_by_dof", {})
        max_by_dof = report.get("max_absolute_tracking_error_by_dof", {})
        active_errors.extend(float(mean_by_dof[name]) for name in LINKER_ACTIVE_DOFS)
        mimic_errors.extend(float(mean_by_dof[name]) for name in LINKER_MIMIC_DOFS)
        peak_errors.extend(
            float(max_by_dof[name]) for name in (*LINKER_ACTIVE_DOFS, *LINKER_MIMIC_DOFS)
        )

    def mean_or_none(values):
        """输入一组数值并返回算术平均；空组返回None。"""
        return sum(values) / len(values) if values else None

    return {
        "mean_active_tracking_error_rad": mean_or_none(active_errors),
        "mean_mimic_tracking_error_rad": mean_or_none(mimic_errors),
        "worst_finger_tracking_error_rad": max(peak_errors) if peak_errors else None,
    }


def geometry_step_metrics(summary: dict) -> dict:
    """汇总候选轨迹相邻帧之间的最大跳变量。

    输入：含逐轨迹几何报告路径的manifest评测摘要。
    输出：关节L2、手腕平移和手腕旋转最大步长在全部轨迹上的平均值。
    内部逻辑：读取每条几何JSON，对三个逐轨迹最大值分别取算术平均。
    作用：在时序消融中量化轨迹是否变平滑，并与物理成功率联合判断。
    """
    fields = {
        "mean_max_joint_step_l2_rad": "max_joint_step_l2_rad",
        "mean_max_wrist_translation_step_m": "max_wrist_translation_step_m",
        "mean_max_wrist_rotation_step_rad": "max_wrist_rotation_step_rad",
    }
    values = {output_name: [] for output_name in fields}
    for result in summary.get("results", []):
        report = json.loads(Path(result["geometry_report"]).read_text(encoding="utf-8"))
        for output_name, report_name in fields.items():
            if report_name in report:
                values[output_name].append(float(report[report_name]))
    return {
        name: (sum(items) / len(items) if items else None)
        for name, items in values.items()
    }


def compact_metrics(name: str, parameters: dict, summary: dict) -> dict:
    """从一份完整评测摘要中提取搜索表字段。

    输入：候选名、超参数字典和manifest评测摘要。
    输出：候选参数、成功数/率、宏平均、几何和抬升指标的紧凑字典。
    内部逻辑：仅复制报告选择所需字段，保留原始摘要路径便于追溯。
    作用：让多个候选在一页JSON中直接比较。
    """
    metrics = {
        "name": name,
        "parameters": parameters,
        "summary_path": summary["_summary_path"],
        "trajectory_count": int(summary["trajectory_count"]),
        "success_count": int(summary["success_count"]),
        "success_rate": float(summary["success_rate"]),
        "object_macro_success_rate": summary.get("object_macro_success_rate"),
        "category_macro_success_rate": summary.get("category_macro_success_rate"),
        "mean_keypoint_distance_m": summary.get("mean_keypoint_distance_m"),
        "mean_max_lift_m": summary.get("mean_max_lift_m"),
        "mean_final_lift_m": summary.get("mean_final_lift_m"),
    }
    if summary.get("hand") in {"linker", "linker11"}:
        metrics.update(linker_tracking_metrics(summary))
    metrics.update(geometry_step_metrics(summary))
    return metrics


def ranking_key(metrics: dict) -> tuple:
    """生成用于展示而非自动决策的候选排序键。

    输入：`compact_metrics`生成的候选指标。
    输出：按成功数、物体宏平均、最终和最大抬升降序的tuple。
    内部逻辑：将缺失的连续指标当作负无穷，不让空值变成优势。
    作用：给出可读初排；最终选择仍需检查配对回退和邻近参数稳定性。
    """
    def safe(value):
        """将可空指标转为可排序浮点，缺失值排在最后。"""
        return float("-inf") if value is None else float(value)

    return (
        int(metrics["success_count"]),
        safe(metrics["object_macro_success_rate"]),
        safe(metrics["mean_final_lift_m"]),
        safe(metrics["mean_max_lift_m"]),
    )


def evaluate_candidate(candidate: dict, config: dict, args, count: int) -> dict:
    """评测或续跑跳过一个冻结候选。

    输入：候选字典、全局搜索配置、命令行参数和期望轨迹数。
    输出：带`_summary_path`的完整manifest评测摘要。
    内部逻辑：优先验证可续跑摘要；不匹配时调用共享评测器并再次验收。
    作用：使数分钟的批量物理搜索可以安全中断后重启。
    """
    hand = str(config["hand"])
    manifest_path = resolve_project_path(config["manifest"])
    target_dir = resolve_project_path(candidate["target_dir"])
    evaluation_dir = resolve_project_path(candidate["evaluation_dir"])
    summary_path = evaluation_dir / "manifest_evaluation_summary.json"
    expected_physics_options = candidate.get("expected_physics_options", {})
    summary = (
        load_matching_summary(
            summary_path,
            hand,
            manifest_path,
            target_dir,
            count,
            expected_physics_options,
        )
        if args.resume
        else None
    )
    if summary is None:
        command = [
            sys.executable,
            str(EVALUATE_SCRIPT),
            "--hand",
            hand,
            "--manifest",
            str(manifest_path),
            "--target-dir",
            str(target_dir),
            "--output-dir",
            str(evaluation_dir),
            "--workers",
            str(args.workers),
            *[str(value) for value in candidate.get("evaluation_args", [])],
        ]
        print(f"\n=== {candidate['name']} ===", flush=True)
        subprocess.run(command, check=True)
        summary = load_matching_summary(
            summary_path,
            hand,
            manifest_path,
            target_dir,
            count,
            expected_physics_options,
        )
        if summary is None:
            raise RuntimeError(f"候选评测未产生完整匹配摘要: {candidate['name']}")
    else:
        print(f"=== {candidate['name']}: resume skip ===", flush=True)
    summary["_summary_path"] = str(summary_path)
    return summary


def main() -> None:
    """解析搜索配置，运行候选评测并写出排序汇总。

    输入：`--config`、`--output`、单候选worker数和可选`--resume`。
    输出：搜索汇总JSON，终端同时打印按主指标排序的简表。
    内部逻辑：验证候选名唯一，顺序执行以避免多个PhysX任务抢占CPU，最后提取和排序指标。
    作用：作为Linker、XHand和Wuji共用的小样本超参数搜索评测入口。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    candidates = config.get("candidates", [])
    names = [str(candidate["name"]) for candidate in candidates]
    if not candidates or len(names) != len(set(names)):
        raise ValueError("搜索候选不能为空且name必须唯一")
    count = expected_trajectory_count(resolve_project_path(config["manifest"]))
    metrics = []
    for candidate in candidates:
        summary = evaluate_candidate(candidate, config, args, count)
        metrics.append(
            compact_metrics(
                str(candidate["name"]), candidate.get("parameters", {}), summary
            )
        )
    ranked = sorted(metrics, key=ranking_key, reverse=True)
    output = {
        "study_name": config.get("study_name"),
        "hand": config["hand"],
        "manifest": str(resolve_project_path(config["manifest"])),
        "trajectory_count": count,
        "selection_rule": config.get("selection_rule"),
        "ranked_candidates": ranked,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("\n=== ranking for inspection ===")
    for item in ranked:
        print(
            f"{item['name']}: success={item['success_count']}/"
            f"{item['trajectory_count']} final={item['mean_final_lift_m']}"
        )
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
