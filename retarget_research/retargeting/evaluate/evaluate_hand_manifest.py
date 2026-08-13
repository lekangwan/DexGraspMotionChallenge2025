#!/usr/bin/env python3
"""按同一冻结manifest统一评估XHand、Linker或Wuji候选。

输入：手类型、manifest、对应候选目录、报告目录和并行数。
输出：逐轨迹几何/物理JSON及轨迹微平均、物体宏平均、类别宏平均汇总。
内部逻辑：按手选择几何和PhysX入口，核对候选索引与维度后并行调用。
作用：保证三只手共享样本、指标和汇总方式，避免维护三套统计口径。
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


EVALUATE_DIR = Path(__file__).resolve().parent
HAND_SPECS = {
    "linker": {
        "dimension": 12,
        "geometry": EVALUATE_DIR / "evaluate_linker_geometry.py",
        "physics": EVALUATE_DIR / "replay_linker_isaac.py",
    },
    "linker11": {
        "dimension": 17,
        "geometry": EVALUATE_DIR / "evaluate_linker_geometry.py",
        "physics": EVALUATE_DIR / "replay_linker_isaac.py",
    },
    "xhand": {
        "dimension": 18,
        "geometry": EVALUATE_DIR / "evaluate_xhand_geometry.py",
        "physics": EVALUATE_DIR / "replay_xhand_isaac.py",
    },
    "wuji": {
        "dimension": 26,
        "geometry": EVALUATE_DIR / "evaluate_wuji_geometry.py",
        "physics": EVALUATE_DIR / "replay_wuji_isaac.py",
    },
}


def verify_target(entry, target_path, hand):
    """核对一个目标手候选与manifest索引、帧数和动作维度。

    输入：manifest条目、候选路径和手类型。
    输出：候选轨迹数量；不匹配时抛出异常。
    内部逻辑：检查`source_trajectory_indices`和`(N,70,D)`，D由手类型确定。
    作用：防止错文件、错顺序或不完整候选进入物理汇总。
    """
    if not target_path.is_file():
        raise FileNotFoundError(f"候选文件不存在: {target_path}")
    data = np.load(target_path, allow_pickle=True).item()
    expected = np.asarray(entry["trajectory_indices"], dtype=np.int64)
    actual = np.asarray(data["source_trajectory_indices"], dtype=np.int64)
    if not np.array_equal(actual, expected):
        raise ValueError(
            f"候选源索引不一致: {entry['object_name']} {actual} vs {expected}"
        )
    expected_shape = (len(expected), 70, HAND_SPECS[hand]["dimension"])
    frames = np.asarray(data["grasp_seqs"])
    if frames.shape != expected_shape:
        raise ValueError(f"{hand}候选形状错误: {frames.shape} vs {expected_shape}")
    return len(expected)


def linker_adaptive_gain_decision(
    target_path,
    target_index,
    scale_threshold=0.06,
    joint_std_threshold=0.25,
):
    """按物体尺度和闭合姿态不均匀程度决定是否提高O6控制增益。

    输入：候选npy、轨迹索引、尺度阈值和抬升首帧关节标准差阈值。
    输出：是否使用高增益，以及可审计的scale、关节标准差和抬升帧。
    内部逻辑：读取生成轨迹时保存的抬升帧，在该帧计算6个主动关节角标准差；
    只有小尺度物体且各指闭合差异明显时才触发高增益。
    作用：用运行前可见的几何/轨迹状态统一选择PD，而不是按物理成败逐条切换。
    """
    data = np.load(target_path, allow_pickle=True).item()
    frames = np.asarray(data["grasp_seqs"][target_index], dtype=np.float32)
    scales = np.asarray(data["obj_scale"], dtype=np.float64)
    metadata = data.get("squeeze_phase_metadata", data.get("phase_metadata"))
    if metadata is None or len(metadata) <= target_index:
        raise ValueError("自适应Linker增益需要候选中的阶段元数据")
    phase = metadata[target_index]
    if phase is None or "lift_start_frame" not in phase:
        raise ValueError("阶段元数据缺少lift_start_frame")
    lift_start = int(phase["lift_start_frame"])
    if not 0 <= lift_start < len(frames):
        raise ValueError(f"lift_start_frame越界: {lift_start}")
    scale = float(scales[target_index])
    joint_std = float(np.std(frames[lift_start, 6:]))
    use_high_gain = bool(
        scale <= float(scale_threshold) + 1e-8
        and joint_std >= float(joint_std_threshold)
    )
    return {
        "use_high_gain": use_high_gain,
        "object_scale": scale,
        "lift_start_frame": lift_start,
        "lift_joint_std_rad": joint_std,
        "scale_threshold": float(scale_threshold),
        "joint_std_threshold_rad": float(joint_std_threshold),
    }


def run_command(command):
    """执行单个评估命令并在失败时保留完整诊断。

    输入：子进程参数列表。
    输出：标准输出文本。
    内部逻辑：捕获stdout/stderr，非零退出时抛出含完整上下文的异常。
    作用：让并行评估不会静默漏掉失败任务。
    """
    process = subprocess.run(command, text=True, capture_output=True, check=False)
    if process.returncode != 0:
        raise RuntimeError(
            f"命令失败({process.returncode}): {' '.join(command)}\n"
            f"stdout:\n{process.stdout}\nstderr:\n{process.stderr}"
        )
    return process.stdout


def load_completed_evaluation(
    hand,
    entry,
    target_path,
    target_index,
    output_dir,
    physics_extra_args=(),
    control_decision=None,
    policy_trace_dir=None,
):
    """严格核对并恢复一条已完成的几何/PhysX评测。

    输入：手类型、manifest条目、候选路径/索引、报告与trace目录及物理参数。
    输出：可直接参与汇总的result；缺文件或任一元数据不匹配时返回None。
    内部逻辑：同时核对源/目标绝对路径、双索引、物体、Linker PD、
    必需数值字段和240步对齐trace，然后从两份JSON重建结果。
    作用：中断后只重跑缺失/不匹配的轨迹，不把旧方法或残缺文件误当正式结果。
    """
    source_index = int(entry["trajectory_indices"][target_index])
    item_dir = output_dir / entry["object_name"]
    geometry_path = item_dir / f"source_{source_index}_geometry.json"
    physics_path = item_dir / f"source_{source_index}_physics.json"
    trace_path = (
        None
        if policy_trace_dir is None
        else policy_trace_dir
        / entry["object_name"]
        / f"source_{source_index}_trace.npz"
    )
    required = [geometry_path, physics_path]
    if trace_path is not None:
        required.append(trace_path)
    if not all(path.is_file() for path in required):
        return None
    try:
        geometry = json.loads(geometry_path.read_text(encoding="utf-8"))
        physics = json.loads(physics_path.read_text(encoding="utf-8"))
        expected_source = Path(entry["source_path"]).resolve()
        expected_target = target_path.resolve()
        for report in (geometry, physics):
            if Path(report["source"]).resolve() != expected_source:
                return None
            if Path(report["target"]).resolve() != expected_target:
                return None
            if int(report["source_trajectory_index"]) != source_index:
                return None
            if int(report["target_trajectory_index"]) != target_index:
                return None
        if physics["object_name"] != entry["object_name"]:
            return None
        if int(physics["target_dimensions"]) != HAND_SPECS[hand]["dimension"]:
            return None
        expected_physics = {
            name[2:].replace("-", "_"): float(value)
            for name, value in zip(physics_extra_args[0::2], physics_extra_args[1::2])
        }
        if any(
            not np.isclose(float(physics.get(name, np.nan)), value)
            for name, value in expected_physics.items()
        ):
            return None
        metric_names = (
            "keypoint_mean_distance_m",
            "keypoint_max_distance_m",
            "max_joint_step_l2_rad",
        )
        physics_names = (
            "max_lift_m",
            "final_lift_m",
            "hand_object_contact_steps",
            "longest_sustained_lift_time_s",
        )
        values = [geometry[name] for name in metric_names] + [
            physics[name] for name in physics_names
        ]
        if not np.isfinite(np.asarray(values, dtype=np.float64)).all():
            return None
        if trace_path is not None:
            with np.load(trace_path, allow_pickle=False) as trace:
                metadata = json.loads(str(trace["metadata_json"].item()))
                if metadata.get("trace_alignment") != "pre_action_state_to_command_v1":
                    return None
                if metadata.get("hand") != hand:
                    return None
                if metadata.get("object_name") != entry["object_name"]:
                    return None
                if Path(metadata["source"]).resolve() != expected_source:
                    return None
                if Path(metadata["target"]).resolve() != expected_target:
                    return None
                if int(metadata["source_trajectory_index"]) != source_index:
                    return None
                if int(metadata["target_trajectory_index"]) != target_index:
                    return None
                lengths = {len(trace[name]) for name in trace.files if name != "metadata_json"}
                if lengths != {240}:
                    return None
                if not np.isfinite(trace["policy_action"]).all():
                    return None
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return None
    result = {
        "hand": hand,
        "object_name": entry["object_name"],
        "category": entry.get("category"),
        "evaluation_split": (
            "calibration"
            if source_index in entry.get("calibration_indices", [])
            else "heldout"
            if source_index in entry.get("heldout_indices", [])
            else "all"
        ),
        "source_trajectory_index": source_index,
        "target_trajectory_index": target_index,
        "geometry_report": str(geometry_path.resolve()),
        "physics_report": str(physics_path.resolve()),
        "keypoint_mean_distance_m": geometry["keypoint_mean_distance_m"],
        "keypoint_max_distance_m": geometry["keypoint_max_distance_m"],
        "max_joint_step_l2_rad": geometry["max_joint_step_l2_rad"],
        "max_lift_m": physics["max_lift_m"],
        "final_lift_m": physics["final_lift_m"],
        "hand_object_contact_steps": physics["hand_object_contact_steps"],
        "longest_sustained_lift_time_s": physics[
            "longest_sustained_lift_time_s"
        ],
        "success": bool(physics["success"]),
        "finger_stiffness": physics.get("finger_stiffness"),
        "finger_damping": physics.get("finger_damping"),
        "mimic_stiffness": physics.get("mimic_stiffness"),
        "mimic_damping": physics.get("mimic_damping"),
        "adaptive_control_decision": control_decision,
        "elapsed_seconds": 0.0,
        "geometry_stdout": "resumed: validated existing report",
        "physics_stdout": "resumed: validated existing report",
        "resumed_existing": True,
    }
    if trace_path is not None:
        result["policy_trace"] = str(trace_path.resolve())
    return result


def evaluate_trajectory(
    hand,
    entry,
    target_path,
    target_index,
    output_dir,
    physics_extra_args=(),
    control_decision=None,
    policy_trace_dir=None,
    resume=False,
):
    """评估一条固定源—目标轨迹对。

    输入：手类型、manifest条目、候选、内部索引、报告目录、物理参数和可选轨迹目录。
    输出：几何误差、平滑性、物理抬升、接触与成功摘要。
    内部逻辑：先调用对应几何入口，再以相同索引调用对应PhysX入口。
    作用：构成可并行且跨手字段一致的最小评估任务。
    """
    if resume:
        completed = load_completed_evaluation(
            hand,
            entry,
            target_path,
            target_index,
            output_dir,
            physics_extra_args,
            control_decision,
            policy_trace_dir,
        )
        if completed is not None:
            return completed
    spec = HAND_SPECS[hand]
    source_index = int(entry["trajectory_indices"][target_index])
    item_dir = output_dir / entry["object_name"]
    item_dir.mkdir(parents=True, exist_ok=True)
    geometry_path = item_dir / f"source_{source_index}_geometry.json"
    physics_path = item_dir / f"source_{source_index}_physics.json"
    common = [
        "--source",
        entry["source_path"],
        "--target",
        str(target_path),
        "--source-index",
        str(source_index),
        "--target-index",
        str(target_index),
    ]
    geometry_command = [
        sys.executable,
        str(spec["geometry"]),
        *common,
        "--output",
        str(geometry_path),
    ]
    physics_command = [
        sys.executable,
        str(spec["physics"]),
        *common,
        "--output",
        str(physics_path),
        "--object-name",
        entry["object_name"],
    ]
    trace_path = None
    if policy_trace_dir is not None:
        trace_path = (
            policy_trace_dir
            / entry["object_name"]
            / f"source_{source_index}_trace.npz"
        )
        physics_command.extend(["--trace-output", str(trace_path)])
    if entry.get("object_asset_path"):
        physics_command.extend(["--object-dir", entry["object_asset_path"]])
    physics_command.extend(list(physics_extra_args))
    started = time.perf_counter()
    geometry_stdout = run_command(geometry_command)
    physics_stdout = run_command(physics_command)
    geometry = json.loads(geometry_path.read_text(encoding="utf-8"))
    physics = json.loads(physics_path.read_text(encoding="utf-8"))
    result = {
        "hand": hand,
        "object_name": entry["object_name"],
        "category": entry.get("category"),
        "evaluation_split": (
            "calibration"
            if source_index in entry.get("calibration_indices", [])
            else "heldout"
            if source_index in entry.get("heldout_indices", [])
            else "all"
        ),
        "source_trajectory_index": source_index,
        "target_trajectory_index": target_index,
        "geometry_report": str(geometry_path.resolve()),
        "physics_report": str(physics_path.resolve()),
        "keypoint_mean_distance_m": geometry["keypoint_mean_distance_m"],
        "keypoint_max_distance_m": geometry["keypoint_max_distance_m"],
        "max_joint_step_l2_rad": geometry["max_joint_step_l2_rad"],
        "max_lift_m": physics["max_lift_m"],
        "final_lift_m": physics["final_lift_m"],
        "hand_object_contact_steps": physics["hand_object_contact_steps"],
        "longest_sustained_lift_time_s": physics[
            "longest_sustained_lift_time_s"
        ],
        "success": physics["success"],
        "finger_stiffness": physics.get("finger_stiffness"),
        "finger_damping": physics.get("finger_damping"),
        "mimic_stiffness": physics.get("mimic_stiffness"),
        "mimic_damping": physics.get("mimic_damping"),
        "adaptive_control_decision": control_decision,
        "elapsed_seconds": time.perf_counter() - started,
        "geometry_stdout": geometry_stdout,
        "physics_stdout": physics_stdout,
        "resumed_existing": False,
    }
    if trace_path is not None:
        if not trace_path.is_file():
            raise RuntimeError(f"物理评测没有生成策略轨迹: {trace_path}")
        result["policy_trace"] = str(trace_path.resolve())
    return result


def summarize_results(results):
    """把逐轨迹结果汇总成微平均、物体宏平均和类别宏平均。

    输入：同一目标手的全部轨迹结果。
    输出：成功数/率、平均误差/抬升、失败清单、逐物体/类别/split统计。
    内部逻辑：先按轨迹计数，再分别平均物体成功率和官方类别成功率。
    作用：避免轨迹较多或物体较多的类别在正式1000条结果中支配结论。
    """
    if not results:
        raise ValueError("没有可汇总的轨迹")
    per_object = {}
    for name in sorted({item["object_name"] for item in results}):
        selected = [item for item in results if item["object_name"] == name]
        count = sum(bool(item["success"]) for item in selected)
        per_object[name] = {
            "trajectory_count": len(selected),
            "success_count": count,
            "success_rate": count / len(selected),
            "mean_keypoint_distance_m": float(
                np.mean([item["keypoint_mean_distance_m"] for item in selected])
            ),
            "mean_max_lift_m": float(
                np.mean([item["max_lift_m"] for item in selected])
            ),
        }
    success_count = sum(bool(item["success"]) for item in results)
    per_category = {}
    known_categories = sorted(
        {item.get("category") for item in results if item.get("category")}
    )
    for category in known_categories:
        selected = [item for item in results if item.get("category") == category]
        count = sum(bool(item["success"]) for item in selected)
        object_names = sorted({item["object_name"] for item in selected})
        per_category[category] = {
            "object_count": len(object_names),
            "trajectory_count": len(selected),
            "success_count": count,
            "success_rate": count / len(selected),
            "object_macro_success_rate": float(
                np.mean([per_object[name]["success_rate"] for name in object_names])
            ),
        }
    per_split = {}
    for split in sorted({item.get("evaluation_split", "all") for item in results}):
        selected = [
            item for item in results if item.get("evaluation_split", "all") == split
        ]
        count = sum(bool(item["success"]) for item in selected)
        per_split[split] = {
            "trajectory_count": len(selected),
            "success_count": count,
            "success_rate": count / len(selected),
        }
    return {
        "trajectory_count": len(results),
        "success_count": success_count,
        "success_rate": success_count / len(results),
        "trajectory_micro_success_rate": success_count / len(results),
        "object_macro_success_rate": float(
            np.mean([values["success_rate"] for values in per_object.values()])
        ),
        "category_macro_success_rate": (
            None
            if not per_category
            else float(
                np.mean([values["success_rate"] for values in per_category.values()])
            )
        ),
        "mean_keypoint_distance_m": float(
            np.mean([item["keypoint_mean_distance_m"] for item in results])
        ),
        "mean_max_lift_m": float(np.mean([item["max_lift_m"] for item in results])),
        "mean_final_lift_m": float(
            np.mean([item["final_lift_m"] for item in results])
        ),
        "failed_trajectories": [
            {
                "object_name": item["object_name"],
                "category": item.get("category"),
                "evaluation_split": item.get("evaluation_split", "all"),
                "source_trajectory_index": item["source_trajectory_index"],
            }
            for item in results
            if not item["success"]
        ],
        "per_object": per_object,
        "per_category": per_category,
        "per_split": per_split,
    }


def main():
    """解析参数、评估整个manifest并保存统一汇总。

    输入：`--hand`、manifest、候选目录、报告目录和worker数。
    输出：逐轨迹报告及`manifest_evaluation_summary.json`。
    内部逻辑：先验证全部候选，再并行提交每条固定轨迹，最后统一汇总。
    作用：作为三只目标手冻结开发集的共同evaluate入口。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--hand", choices=sorted(HAND_SPECS), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--policy-trace-dir",
        type=Path,
        help="可选目录；为每条物理重放同时保存进阶策略专家NPZ",
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="严格验证已有几何/PhysX/trace后跳过已完成轨迹",
    )
    parser.add_argument("--linker-finger-stiffness", type=float, default=120.0)
    parser.add_argument("--linker-finger-damping", type=float, default=5.0)
    parser.add_argument("--linker-mimic-stiffness", type=float, default=120.0)
    parser.add_argument("--linker-mimic-damping", type=float, default=5.0)
    parser.add_argument("--linker-adaptive-gains", action="store_true")
    parser.add_argument("--linker-adaptive-scale-threshold", type=float, default=0.06)
    parser.add_argument("--linker-adaptive-joint-std-threshold", type=float, default=0.25)
    parser.add_argument("--linker-high-stiffness", type=float, default=400.0)
    parser.add_argument("--linker-high-damping", type=float, default=20.0)
    args = parser.parse_args()

    linker_physics_args = [
        "--finger-stiffness",
        str(args.linker_finger_stiffness),
        "--finger-damping",
        str(args.linker_finger_damping),
        "--mimic-stiffness",
        str(args.linker_mimic_stiffness),
        "--mimic-damping",
        str(args.linker_mimic_damping),
    ]
    physics_extra_args = linker_physics_args if args.hand in {"linker", "linker11"} else []

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    tasks = []
    for entry in manifest["entries"]:
        target_path = args.target_dir / f"{entry['object_name']}.npy"
        count = verify_target(entry, target_path, args.hand)
        for index in range(count):
            selected_physics_args = list(physics_extra_args)
            decision = None
            if args.linker_adaptive_gains and args.hand == "linker":
                decision = linker_adaptive_gain_decision(
                    target_path,
                    index,
                    args.linker_adaptive_scale_threshold,
                    args.linker_adaptive_joint_std_threshold,
                )
                if decision["use_high_gain"]:
                    selected_physics_args = [
                        "--finger-stiffness", str(args.linker_high_stiffness),
                        "--finger-damping", str(args.linker_high_damping),
                        "--mimic-stiffness", str(args.linker_high_stiffness),
                        "--mimic-damping", str(args.linker_high_damping),
                    ]
            tasks.append(
                (entry, target_path, index, selected_physics_args, decision)
            )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                evaluate_trajectory,
                args.hand,
                entry,
                target_path,
                target_index,
                args.output_dir,
                selected_physics_args,
                decision,
                args.policy_trace_dir,
                args.resume,
            ): (entry, target_index)
            for entry, target_path, target_index, selected_physics_args, decision in tasks
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                f"{result['object_name']}[{result['source_trajectory_index']}]: "
                f"success={result['success']} lift={result['max_lift_m']:.4f}m",
                flush=True,
            )
    results.sort(key=lambda item: (item["object_name"], item["source_trajectory_index"]))
    summary = {
        "hand": args.hand,
        "manifest": str(args.manifest.resolve()),
        "target_directory": str(args.target_dir.resolve()),
        "workers": args.workers,
        "policy_trace_directory": (
            None
            if args.policy_trace_dir is None
            else str(args.policy_trace_dir.resolve())
        ),
        "physics_options": {
            "linker_finger_stiffness": args.linker_finger_stiffness,
            "linker_finger_damping": args.linker_finger_damping,
            "linker_mimic_stiffness": args.linker_mimic_stiffness,
            "linker_mimic_damping": args.linker_mimic_damping,
            "adaptive_gains": args.linker_adaptive_gains,
            "adaptive_scale_threshold": args.linker_adaptive_scale_threshold,
            "adaptive_joint_std_threshold": (
                args.linker_adaptive_joint_std_threshold
            ),
            "high_stiffness": args.linker_high_stiffness,
            "high_damping": args.linker_high_damping,
        }
        if args.hand in {"linker", "linker11"}
        else {},
        "wall_time_seconds": time.perf_counter() - started,
        "resumed_trajectory_count": sum(
            int(item["resumed_existing"]) for item in results
        ),
        **summarize_results(results),
        "results": results,
    }
    summary_path = args.output_dir / "manifest_evaluation_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"success={summary['success_count']}/{summary['trajectory_count']}")
    print(f"success_rate={summary['success_rate']:.4f}")
    print(f"output={summary_path}")


if __name__ == "__main__":
    main()
