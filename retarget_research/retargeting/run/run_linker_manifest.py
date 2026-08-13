#!/usr/bin/env python3
"""按冻结manifest批量运行Linker O6或11自由度增强重定向实验。

输入：manifest、输出目录、并行数及一组固定Linker优化超参数。
输出：每物体候选`.npy`和记录命令、耗时、stdout、状态的汇总JSON。
内部逻辑：先验证源文件SHA-256，再并行调用单文件重定向入口处理固定索引。
作用：把人工逐条命令变成可复现批处理，并防止运行时悄悄更换源数据。
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


RUN_DIR = Path(__file__).resolve().parent
RETARGET_SCRIPT = RUN_DIR / "retarget_linker_keypoints.py"
DEFAULT_OBJECT_ROOT = (
    RUN_DIR.parents[1]
    / "reference"
    / "HandRetargetTask2026"
    / "scripts"
    / "data"
    / "sorting"
    / "object_41"
)


def file_sha256(path):
    """计算待运行源文件的SHA-256。

    输入：本地文件路径。
    输出：64字符十六进制摘要。
    逻辑：按1 MiB分块读取，避免大文件一次性载入内存。
    作用：确认运行时数据与manifest冻结时完全相同。
    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def verify_entry(entry):
    """验证一个manifest条目的路径、哈希和索引范围。

    输入：包含源路径、哈希、可用数量和轨迹索引的条目字典。
    输出：已解析的源Path；错误时抛出明确异常。
    逻辑：依次检查文件存在、哈希一致、索引非空且处于冻结范围。
    作用：在昂贵优化开始前阻止数据漂移或错误索引。
    """
    source = Path(entry["source_path"])
    if not source.is_file():
        raise FileNotFoundError(f"源文件不存在: {source}")
    actual_hash = file_sha256(source)
    if actual_hash != entry["source_sha256"]:
        raise ValueError(f"源文件哈希变化: {source}")
    indices = entry["trajectory_indices"]
    available = int(entry["available_trajectory_count"])
    if not indices or min(indices) < 0 or max(indices) >= available:
        raise ValueError(f"轨迹索引越界: {entry['object_name']} {indices}")
    return source


def build_command(entry, source, output, args):
    """构造一个物体的单文件Linker重定向命令。

    输入：manifest条目、源/输出路径和批处理超参数。
    输出：可直接交给`subprocess.run`的字符串列表。
    逻辑：固定解释器和单文件入口，追加manifest索引及所有显式方法参数。
    作用：使汇总报告能够完整复现每个子任务，而不依赖隐藏默认值。
    """
    object_asset = Path(
        entry.get("object_asset_path", args.object_root / entry["object_name"])
    )
    command = [
        sys.executable,
        str(RETARGET_SCRIPT),
        "--source",
        str(source),
        "--output",
        str(output),
        "--trajectory-indices",
        *[str(index) for index in entry["trajectory_indices"]],
        "--maxeval",
        str(args.maxeval),
        "--joint-mode",
        args.joint_mode,
        "--source-z-offset",
        str(args.source_z_offset),
        "--joint-temporal-weight",
        str(args.joint_temporal_weight),
        "--translation-temporal-weight",
        str(args.translation_temporal_weight),
        "--rotation-temporal-weight",
        str(args.rotation_temporal_weight),
        "--contact-start-frame",
        str(args.contact_start_frame),
        "--late-tip-weight",
        str(args.late_tip_weight),
        "--late-thumb-weight",
        str(args.late_thumb_weight),
        "--late-structure-weight",
        str(args.late_structure_weight),
        "--expert-contact-threshold",
        str(args.expert_contact_threshold),
        "--expert-contact-weight",
        str(args.expert_contact_weight),
        "--object-root",
        str(object_asset.parent),
        "--object-name",
        object_asset.name,
        "--object-clearance",
        str(args.object_clearance),
        "--target-surface-weight",
        str(args.target_surface_weight),
        "--surface-activation-min-expert-tips",
        str(args.surface_activation_min_expert_tips),
        "--phase-contact-weight",
        str(args.phase_contact_weight),
        "--phase-normal-weight",
        str(args.phase_normal_weight),
        "--phase-penetration-weight",
        str(args.phase_penetration_weight),
        "--phase-joint-hold-weight",
        str(args.phase_joint_hold_weight),
        "--phase-joint-prior-weight",
        str(args.phase_joint_prior_weight),
        "--phase-joint-delta-bound",
        str(args.phase_joint_delta_bound),
        "--phase-contact-threshold",
        str(args.phase_contact_threshold),
        "--phase-min-contact-tips",
        str(args.phase_min_contact_tips),
        "--phase-lift-delta",
        str(args.phase_lift_delta),
        "--phase-region-neighbors",
        str(args.phase_region_neighbors),
        "--phase-contact-offset",
        str(args.phase_contact_offset),
        "--phase-min-signed-distance",
        str(args.phase_min_signed_distance),
        "--opposition-candidate-neighbors",
        str(args.opposition_candidate_neighbors),
        "--opposition-distance-scale",
        str(args.opposition_distance_scale),
        "--opposition-weight",
        str(args.opposition_weight),
        "--opposition-refine-frames",
        str(args.opposition_refine_frames),
        "--reachable-pad-alignment-weight",
        str(args.reachable_pad_alignment_weight),
        "--reachable-min-opposing-fingers",
        str(args.reachable_min_opposing_fingers),
        "--grip-tighten-thumb-pitch",
        str(args.grip_tighten_thumb_pitch),
        "--grip-tighten-fingers",
        str(args.grip_tighten_fingers),
    ]
    if args.include_thumb_middle:
        command.append("--include-thumb-middle")
    if args.include_finger_middle:
        command.append("--include-finger-middle")
    if args.contact_pad_config is not None:
        command.extend(["--contact-pad-config", str(args.contact_pad_config)])
    if args.initial_target_dir is not None:
        initial_target = args.initial_target_dir / f"{entry['object_name']}.npy"
        command.extend(["--initial-target", str(initial_target)])
    if args.phase_only_refinement:
        command.append("--phase-only-refinement")
    if args.phase_joint_only:
        command.append("--phase-joint-only")
    if args.freeze_lift_grasp:
        command.append("--freeze-lift-grasp")
    if args.carry_lift_wrist_residual:
        command.append("--carry-lift-wrist-residual")
    if args.carry_lift_joint_residual:
        command.append("--carry-lift-joint-residual")
    if args.reachable_opposition:
        command.append("--reachable-opposition")
    return command


def existing_output_matches(output, entry, args):
    """检查已有候选文件是否与当前manifest和方法配置一致。

    输入：输出路径、manifest条目和本次批处理参数。
    输出：全部关键字段匹配时为True，否则为False。
    逻辑：读取npy元数据，核对源索引、轨迹数、迭代数、点集与时序权重。
    作用：支持安全续跑，只跳过真正由同一实验配置生成的完整结果。
    """
    if not output.is_file():
        return False
    try:
        data = np.load(output, allow_pickle=True).item()
        indices_match = np.array_equal(
            np.asarray(data["source_trajectory_indices"]),
            np.asarray(entry["trajectory_indices"]),
        )
        expected_dimension = 12 if args.joint_mode == "coupled6" else 17
        shape_match = np.asarray(data["grasp_seqs"]).shape == (
            len(entry["trajectory_indices"]),
            70,
            expected_dimension,
        )
        expected_pad_config = (
            None
            if args.contact_pad_config is None
            else str(args.contact_pad_config.resolve())
        )
        expected_initial = (
            None
            if args.initial_target_dir is None
            else str(
                (args.initial_target_dir / f"{entry['object_name']}.npy").resolve()
            )
        )
        return bool(
            indices_match
            and shape_match
            and str(data.get("joint_mode", "coupled6")) == args.joint_mode
            and int(data["maxeval"]) == args.maxeval
            and bool(data["include_thumb_middle"])
            == (
                args.include_thumb_middle
                or args.include_finger_middle
                or args.joint_mode == "independent11"
            )
            and bool(data.get("include_finger_middle", False))
            == args.include_finger_middle
            and float(data["joint_temporal_weight"])
            == args.joint_temporal_weight
            and float(data["translation_temporal_weight"])
            == args.translation_temporal_weight
            and float(data["rotation_temporal_weight"])
            == args.rotation_temporal_weight
            and int(data["contact_start_frame"]) == args.contact_start_frame
            and float(data["late_tip_weight"]) == args.late_tip_weight
            and float(data["late_thumb_weight"]) == args.late_thumb_weight
            and float(data["late_structure_weight"])
            == args.late_structure_weight
            and float(data["expert_contact_threshold"])
            == args.expert_contact_threshold
            and float(data["expert_contact_weight"])
            == args.expert_contact_weight
            and float(data["object_clearance"]) == args.object_clearance
            and float(data["target_surface_weight"]) == args.target_surface_weight
            and int(data["surface_activation_min_expert_tips"])
            == args.surface_activation_min_expert_tips
            and data["contact_pad_config"] == expected_pad_config
            and data["initial_target"] == expected_initial
            and bool(data["phase_only_refinement"])
            == args.phase_only_refinement
            and bool(data.get("phase_joint_only", False))
            == args.phase_joint_only
            and bool(data["freeze_lift_grasp"]) == args.freeze_lift_grasp
            and bool(data["carry_lift_wrist_residual"])
            == args.carry_lift_wrist_residual
            and bool(data.get("carry_lift_joint_residual", False))
            == args.carry_lift_joint_residual
            and float(data["phase_contact_weight"])
            == args.phase_contact_weight
            and float(data["phase_normal_weight"]) == args.phase_normal_weight
            and float(data["phase_penetration_weight"])
            == args.phase_penetration_weight
            and float(data["phase_joint_hold_weight"])
            == args.phase_joint_hold_weight
            and float(data["phase_joint_prior_weight"])
            == args.phase_joint_prior_weight
            and float(data.get("phase_joint_delta_bound", 0.0))
            == args.phase_joint_delta_bound
            and float(data["phase_contact_threshold"])
            == args.phase_contact_threshold
            and int(data["phase_min_contact_tips"])
            == args.phase_min_contact_tips
            and float(data["phase_lift_delta"]) == args.phase_lift_delta
            and int(data["phase_region_neighbors"])
            == args.phase_region_neighbors
            and float(data["phase_contact_offset"]) == args.phase_contact_offset
            and float(data["phase_min_signed_distance"])
            == args.phase_min_signed_distance
            and int(data["opposition_candidate_neighbors"])
            == args.opposition_candidate_neighbors
            and float(data["opposition_distance_scale"])
            == args.opposition_distance_scale
            and float(data["opposition_weight"]) == args.opposition_weight
            and int(data["opposition_refine_frames"])
            == args.opposition_refine_frames
            and bool(data.get("reachable_opposition", False))
            == args.reachable_opposition
            and float(data.get("reachable_pad_alignment_weight", 1.0))
            == args.reachable_pad_alignment_weight
            and int(data.get("reachable_min_opposing_fingers", 2))
            == args.reachable_min_opposing_fingers
            and float(data["grip_tighten_thumb_pitch"])
            == args.grip_tighten_thumb_pitch
            and float(data["grip_tighten_fingers"])
            == args.grip_tighten_fingers
        )
    except (KeyError, OSError, TypeError, ValueError):
        return False


def run_entry(entry, args):
    """执行并记录一个物体的全部冻结轨迹。

    输入：单个manifest条目和批处理参数。
    输出：包含退出码、耗时、命令、stdout/stderr和输出路径的字典。
    逻辑：验证条目、创建输出路径、同步运行子进程并捕获文本日志。
    作用：为并行调度提供互不共享状态的最小任务单元。
    """
    source = verify_entry(entry)
    output = args.output_dir / f"{entry['object_name']}.npy"
    output.parent.mkdir(parents=True, exist_ok=True)
    command = build_command(entry, source, output, args)
    if args.resume and existing_output_matches(output, entry, args):
        return {
            "object_name": entry["object_name"],
            "trajectory_indices": entry["trajectory_indices"],
            "trajectory_count": len(entry["trajectory_indices"]),
            "output": str(output.resolve()),
            "command": command,
            "elapsed_seconds": 0.0,
            "return_code": 0,
            "stdout": "skipped: matching output already exists",
            "stderr": "",
            "success": True,
            "skipped_existing": True,
        }
    start = time.perf_counter()
    process = subprocess.run(command, text=True, capture_output=True, check=False)
    elapsed = time.perf_counter() - start
    return {
        "object_name": entry["object_name"],
        "trajectory_indices": entry["trajectory_indices"],
        "trajectory_count": len(entry["trajectory_indices"]),
        "output": str(output.resolve()),
        "command": command,
        "elapsed_seconds": elapsed,
        "return_code": process.returncode,
        "stdout": process.stdout,
        "stderr": process.stderr,
        "success": process.returncode == 0 and output.is_file(),
        "skipped_existing": False,
    }


def main():
    """解析参数、并行运行manifest并写出批处理摘要。

    输入：manifest/输出路径、worker数量和完整优化器参数。
    输出：`manifest_run_summary.json`；任一子任务失败时最终退出非零。
    逻辑：先顺序验证全部条目，再用线程池管理独立子进程并按物体名汇总。
    作用：作为run分区中从冻结样本到批量Linker候选轨迹的标准入口。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--maxeval", type=int, default=100)
    parser.add_argument(
        "--joint-mode",
        choices=["coupled6", "independent11"],
        default="coupled6",
        help="真实O6六主动轴，或相同外形的11轴完全解耦诊断模型",
    )
    parser.add_argument("--source-z-offset", type=float, default=0.4)
    parser.add_argument("--include-thumb-middle", action="store_true")
    parser.add_argument(
        "--include-finger-middle",
        action="store_true",
        help="加入四指中段并自动加入拇指中段，形成15点密集消融",
    )
    parser.add_argument("--joint-temporal-weight", type=float, default=0.0)
    parser.add_argument("--translation-temporal-weight", type=float, default=0.0)
    parser.add_argument("--rotation-temporal-weight", type=float, default=0.0)
    parser.add_argument("--contact-start-frame", type=int, default=-1)
    parser.add_argument("--late-tip-weight", type=float, default=1.0)
    parser.add_argument("--late-thumb-weight", type=float, default=1.0)
    parser.add_argument("--late-structure-weight", type=float, default=1.0)
    parser.add_argument("--expert-contact-threshold", type=float, default=-1.0)
    parser.add_argument("--expert-contact-weight", type=float, default=1.0)
    parser.add_argument("--object-root", type=Path, default=DEFAULT_OBJECT_ROOT)
    parser.add_argument("--object-clearance", type=float, default=0.005)
    parser.add_argument("--target-surface-weight", type=float, default=0.0)
    parser.add_argument(
        "--surface-activation-min-expert-tips", type=int, default=2
    )
    parser.add_argument("--contact-pad-config", type=Path)
    parser.add_argument(
        "--initial-target-dir",
        type=Path,
        help="按物体名读取第一阶段Linker候选的目录",
    )
    parser.add_argument("--phase-only-refinement", action="store_true")
    parser.add_argument("--phase-joint-only", action="store_true")
    parser.add_argument("--freeze-lift-grasp", action="store_true")
    parser.add_argument("--carry-lift-wrist-residual", action="store_true")
    parser.add_argument("--carry-lift-joint-residual", action="store_true")
    parser.add_argument("--phase-contact-weight", type=float, default=3.0)
    parser.add_argument("--phase-normal-weight", type=float, default=0.02)
    parser.add_argument("--phase-penetration-weight", type=float, default=1.0)
    parser.add_argument("--phase-joint-hold-weight", type=float, default=0.1)
    parser.add_argument("--phase-joint-prior-weight", type=float, default=1.0)
    parser.add_argument("--phase-joint-delta-bound", type=float, default=0.0)
    parser.add_argument("--phase-contact-threshold", type=float, default=0.02)
    parser.add_argument("--phase-min-contact-tips", type=int, default=2)
    parser.add_argument("--phase-lift-delta", type=float, default=0.03)
    parser.add_argument("--phase-region-neighbors", type=int, default=32)
    parser.add_argument("--phase-contact-offset", type=float, default=-0.001)
    parser.add_argument("--phase-min-signed-distance", type=float, default=-0.003)
    parser.add_argument("--opposition-candidate-neighbors", type=int, default=0)
    parser.add_argument("--opposition-distance-scale", type=float, default=0.03)
    parser.add_argument("--opposition-weight", type=float, default=1.0)
    parser.add_argument("--opposition-refine-frames", type=int, default=4)
    parser.add_argument("--reachable-opposition", action="store_true")
    parser.add_argument("--reachable-pad-alignment-weight", type=float, default=1.0)
    parser.add_argument("--reachable-min-opposing-fingers", type=int, default=2)
    parser.add_argument("--grip-tighten-thumb-pitch", type=float, default=0.0)
    parser.add_argument("--grip-tighten-fingers", type=float, default=0.0)
    args = parser.parse_args()

    if args.phase_only_refinement and args.initial_target_dir is None:
        parser.error("--phase-only-refinement必须同时提供--initial-target-dir")
    if args.phase_joint_only and args.initial_target_dir is None:
        parser.error("--phase-joint-only必须同时提供--initial-target-dir")
    if args.freeze_lift_grasp and args.initial_target_dir is None:
        parser.error("--freeze-lift-grasp必须同时提供--initial-target-dir")
    if args.carry_lift_wrist_residual and not args.freeze_lift_grasp:
        parser.error("--carry-lift-wrist-residual必须同时启用--freeze-lift-grasp")
    if args.carry_lift_joint_residual and args.initial_target_dir is None:
        parser.error("--carry-lift-joint-residual必须同时提供--initial-target-dir")
    if args.carry_lift_joint_residual and args.freeze_lift_grasp:
        parser.error("动态关节残差传播与--freeze-lift-grasp不能同时启用")
    if args.reachable_opposition:
        if args.initial_target_dir is None or args.contact_pad_config is None:
            parser.error("--reachable-opposition需要初始候选目录和指腹配置")
        if args.opposition_candidate_neighbors < 1:
            parser.error("--reachable-opposition需要正的候选点数")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    entries = manifest["entries"]
    for entry in entries:
        verify_entry(entry)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started_at = time.perf_counter()
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_entry, entry, args): entry for entry in entries}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                f"{result['object_name']}: success={result['success']} "
                f"time={result['elapsed_seconds']:.2f}s",
                flush=True,
            )
    results.sort(key=lambda item: item["object_name"])
    summary = {
        "manifest": str(args.manifest.resolve()),
        "manifest_purpose": manifest.get("purpose"),
        "object_count": len(results),
        "trajectory_count": sum(item["trajectory_count"] for item in results),
        "workers": args.workers,
        "wall_time_seconds": time.perf_counter() - started_at,
        "all_successful": all(item["success"] for item in results),
        "method": {
            "maxeval": args.maxeval,
            "joint_mode": args.joint_mode,
            "source_z_offset": args.source_z_offset,
            "include_thumb_middle": (
                args.include_thumb_middle
                or args.include_finger_middle
                or args.joint_mode == "independent11"
            ),
            "include_finger_middle": args.include_finger_middle,
            "joint_temporal_weight": args.joint_temporal_weight,
            "translation_temporal_weight": args.translation_temporal_weight,
            "rotation_temporal_weight": args.rotation_temporal_weight,
            "contact_start_frame": args.contact_start_frame,
            "late_tip_weight": args.late_tip_weight,
            "late_thumb_weight": args.late_thumb_weight,
            "late_structure_weight": args.late_structure_weight,
            "expert_contact_threshold": args.expert_contact_threshold,
            "expert_contact_weight": args.expert_contact_weight,
            "object_root": str(args.object_root.resolve()),
            "object_clearance": args.object_clearance,
            "target_surface_weight": args.target_surface_weight,
            "surface_activation_min_expert_tips": (
                args.surface_activation_min_expert_tips
            ),
            "contact_pad_config": (
                None
                if args.contact_pad_config is None
                else str(args.contact_pad_config.resolve())
            ),
            "initial_target_dir": (
                None
                if args.initial_target_dir is None
                else str(args.initial_target_dir.resolve())
            ),
            "phase_only_refinement": args.phase_only_refinement,
            "phase_joint_only": args.phase_joint_only,
            "freeze_lift_grasp": args.freeze_lift_grasp,
            "carry_lift_wrist_residual": args.carry_lift_wrist_residual,
            "carry_lift_joint_residual": args.carry_lift_joint_residual,
            "phase_contact_weight": args.phase_contact_weight,
            "phase_normal_weight": args.phase_normal_weight,
            "phase_penetration_weight": args.phase_penetration_weight,
            "phase_joint_hold_weight": args.phase_joint_hold_weight,
            "phase_joint_prior_weight": args.phase_joint_prior_weight,
            "phase_joint_delta_bound": args.phase_joint_delta_bound,
            "phase_contact_threshold": args.phase_contact_threshold,
            "phase_min_contact_tips": args.phase_min_contact_tips,
            "phase_lift_delta": args.phase_lift_delta,
            "phase_region_neighbors": args.phase_region_neighbors,
            "phase_contact_offset": args.phase_contact_offset,
            "phase_min_signed_distance": args.phase_min_signed_distance,
            "opposition_candidate_neighbors": (
                args.opposition_candidate_neighbors
            ),
            "opposition_distance_scale": args.opposition_distance_scale,
            "opposition_weight": args.opposition_weight,
            "opposition_refine_frames": args.opposition_refine_frames,
            "reachable_opposition": args.reachable_opposition,
            "reachable_pad_alignment_weight": (
                args.reachable_pad_alignment_weight
            ),
            "reachable_min_opposing_fingers": (
                args.reachable_min_opposing_fingers
            ),
            "grip_tighten_thumb_pitch": args.grip_tighten_thumb_pitch,
            "grip_tighten_fingers": args.grip_tighten_fingers,
        },
        "results": results,
    }
    summary_path = args.output_dir / "manifest_run_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"all_successful={summary['all_successful']}")
    print(f"output={summary_path}")
    if not summary["all_successful"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
