#!/usr/bin/env python3
"""按冻结manifest批量评估Linker几何误差与CPU PhysX抓取成功。

输入：manifest、对应候选轨迹目录、报告目录和并行数。
输出：每条轨迹的几何/物理JSON，以及总体成功率和均值汇总JSON。
内部逻辑：核对候选源索引后，为每条固定轨迹依次调用两个独立评估入口。
作用：把manifest从“抽样名单”落实为可重复的完整批量评价，而非手工挑案例。
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
GEOMETRY_SCRIPT = EVALUATE_DIR / "evaluate_linker_geometry.py"
PHYSICS_SCRIPT = EVALUATE_DIR / "replay_linker_isaac.py"


def verify_target(entry, target_path):
    """核对候选文件与manifest冻结索引是否一致。

    输入：manifest物体条目和对应Linker候选npy路径。
    输出：候选轨迹数量；不匹配时抛出异常。
    逻辑：检查文件存在、`source_trajectory_indices`及`(N,70,12)`形状。
    作用：防止评估错文件、错顺序或不完整的批量输出。
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
    frames = np.asarray(data["grasp_seqs"])
    if frames.shape != (len(expected), 70, 12):
        raise ValueError(f"Linker候选形状错误: {target_path} {frames.shape}")
    return len(expected)


def run_command(command):
    """执行一个评估子命令并在失败时提供完整日志。

    输入：subprocess字符串列表。
    输出：子进程stdout字符串。
    逻辑：捕获stdout/stderr，退出码非零时抛出包含两者的RuntimeError。
    作用：让并行批处理不会静默吞掉某条轨迹的评估错误。
    """
    process = subprocess.run(command, text=True, capture_output=True, check=False)
    if process.returncode != 0:
        raise RuntimeError(
            f"命令失败({process.returncode}): {' '.join(command)}\n"
            f"stdout:\n{process.stdout}\nstderr:\n{process.stderr}"
        )
    return process.stdout


def evaluate_trajectory(entry, target_path, target_index, output_dir):
    """评估manifest中的一条源—目标轨迹对。

    输入：物体条目、候选路径、目标内部索引和报告根目录。
    输出：包含几何与物理核心指标、报告路径和耗时的字典。
    逻辑：用manifest源索引调用几何评估，再以完全相同索引调用物理重放。
    作用：形成批量调度中互不共享状态、可并行的最小评估单元。
    """
    source_index = int(entry["trajectory_indices"][target_index])
    item_dir = output_dir / entry["object_name"]
    item_dir.mkdir(parents=True, exist_ok=True)
    geometry_path = item_dir / f"source_{source_index}_geometry.json"
    physics_path = item_dir / f"source_{source_index}_physics.json"
    geometry_command = [
        sys.executable,
        str(GEOMETRY_SCRIPT),
        "--source",
        entry["source_path"],
        "--target",
        str(target_path),
        "--source-index",
        str(source_index),
        "--target-index",
        str(target_index),
        "--output",
        str(geometry_path),
    ]
    physics_command = [
        sys.executable,
        str(PHYSICS_SCRIPT),
        "--source",
        entry["source_path"],
        "--target",
        str(target_path),
        "--source-index",
        str(source_index),
        "--target-index",
        str(target_index),
        "--output",
        str(physics_path),
    ]
    start = time.perf_counter()
    geometry_stdout = run_command(geometry_command)
    physics_stdout = run_command(physics_command)
    elapsed = time.perf_counter() - start
    geometry = json.loads(geometry_path.read_text(encoding="utf-8"))
    physics = json.loads(physics_path.read_text(encoding="utf-8"))
    return {
        "object_name": entry["object_name"],
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
        "elapsed_seconds": elapsed,
        "geometry_stdout": geometry_stdout,
        "physics_stdout": physics_stdout,
    }


def summarize_results(results):
    """汇总逐轨迹结果为总体和逐物体指标。

    输入：所有轨迹评估结果列表。
    输出：成功数/率、平均误差/抬升、失败清单和逐物体成功率字典。
    逻辑：总体按轨迹计数，逐物体按各自固定轨迹数独立统计。
    作用：同时提供总体趋势和物体间差异，避免只展示单个成功案例。
    """
    successes = [bool(item["success"]) for item in results]
    object_names = sorted({item["object_name"] for item in results})
    per_object = {}
    for name in object_names:
        selected = [item for item in results if item["object_name"] == name]
        success_count = sum(bool(item["success"]) for item in selected)
        per_object[name] = {
            "trajectory_count": len(selected),
            "success_count": success_count,
            "success_rate": success_count / len(selected),
            "mean_keypoint_distance_m": float(
                np.mean([item["keypoint_mean_distance_m"] for item in selected])
            ),
            "mean_max_lift_m": float(
                np.mean([item["max_lift_m"] for item in selected])
            ),
        }
    return {
        "trajectory_count": len(results),
        "success_count": sum(successes),
        "success_rate": sum(successes) / len(results),
        "mean_keypoint_distance_m": float(
            np.mean([item["keypoint_mean_distance_m"] for item in results])
        ),
        "mean_max_lift_m": float(
            np.mean([item["max_lift_m"] for item in results])
        ),
        "mean_final_lift_m": float(
            np.mean([item["final_lift_m"] for item in results])
        ),
        "failed_trajectories": [
            {
                "object_name": item["object_name"],
                "source_trajectory_index": item["source_trajectory_index"],
            }
            for item in results
            if not item["success"]
        ],
        "per_object": per_object,
    }


def main():
    """解析参数、并行评估整个manifest并保存汇总。

    输入：manifest、候选目录、报告目录和worker数。
    输出：逐轨迹报告及`manifest_evaluation_summary.json`。
    逻辑：先验证全部候选，再提交每条固定轨迹任务，完成后统一汇总。
    作用：作为evaluate分区中Linker开发集成功率的标准入口。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    tasks = []
    for entry in manifest["entries"]:
        target_path = args.target_dir / f"{entry['object_name']}.npy"
        count = verify_target(entry, target_path)
        tasks.extend((entry, target_path, index) for index in range(count))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started_at = time.perf_counter()
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                evaluate_trajectory, entry, target_path, target_index, args.output_dir
            ): (entry, target_index)
            for entry, target_path, target_index in tasks
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                f"{result['object_name']}[{result['source_trajectory_index']}]: "
                f"success={result['success']} lift={result['max_lift_m']:.4f}m",
                flush=True,
            )
    results.sort(
        key=lambda item: (item["object_name"], item["source_trajectory_index"])
    )
    aggregate = summarize_results(results)
    summary = {
        "manifest": str(args.manifest.resolve()),
        "target_directory": str(args.target_dir.resolve()),
        "workers": args.workers,
        "wall_time_seconds": time.perf_counter() - started_at,
        **aggregate,
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
