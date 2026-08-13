#!/usr/bin/env python3
"""用小规模PhysX黑盒搜索诊断Linker O6邻近抓形上界。

输入：单条Shadow源、现有12维候选、源/目标索引、物体名和搜索规模。
输出：每个残差候选的轨迹/物理报告、排序摘要及最佳单候选轨迹。
内部逻辑：在6个主动关节上叠加闭合期渐进残差，并行调用标准PhysX重放器评估。
作用：只回答“当前抓形附近是否存在成功解”，不得作为逐轨迹物理选优的正式成绩。
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
REPLAY_SCRIPT = EVALUATE_DIR / "replay_linker_isaac.py"
JOINT_LOWER = np.zeros(6, dtype=np.float32)
JOINT_UPPER = np.asarray([1.36, 0.58, 1.60, 1.60, 1.60, 1.60], dtype=np.float32)


def generate_residuals(count, scale, seed):
    """生成包含零点、坐标方向和随机方向的确定性6维残差。

    输入：候选总数、最大绝对幅度和随机种子。
    输出：`(count,6)`数组，第一行为全零，其余均位于`[-scale,scale]`。
    内部逻辑：先加入每个维度的正负半尺度方向，再用固定种子均匀样本补足。
    作用：同时检查单关节敏感性和多关节协同，避免纯随机搜索漏掉简单方向。
    """
    if count < 1 or scale <= 0:
        raise ValueError("count必须为正且scale必须大于0")
    values = [np.zeros(6, dtype=np.float32)]
    for joint_index in range(6):
        for sign in (-1.0, 1.0):
            residual = np.zeros(6, dtype=np.float32)
            residual[joint_index] = sign * float(scale) * 0.5
            values.append(residual)
            if len(values) == count:
                return np.stack(values)
    generator = np.random.default_rng(int(seed))
    while len(values) < count:
        values.append(
            generator.uniform(-scale, scale, size=6).astype(np.float32)
        )
    return np.stack(values)


def generate_wrist_probes(translation_scale, rotation_scale):
    """生成零点及手腕6维坐标正负探针。

    输入：XYZ平移幅度（米）和欧拉角幅度（弧度）。
    输出：固定13个`[x,y,z,roll,pitch,yaw]`残差。
    内部逻辑：第一项全零，随后依次加入每个维度的负/正坐标方向。
    作用：用极少物理查询诊断毫米级腕位姿误差，而不与手指随机搜索混合。
    """
    if translation_scale <= 0 or rotation_scale <= 0:
        raise ValueError("手腕平移和旋转探针幅度必须为正")
    values = [np.zeros(6, dtype=np.float32)]
    scales = [translation_scale] * 3 + [rotation_scale] * 3
    for index, scale in enumerate(scales):
        for sign in (-1.0, 1.0):
            value = np.zeros(6, dtype=np.float32)
            value[index] = sign * float(scale)
            values.append(value)
    return np.stack(values)


def apply_residual(
    frames, close_start, lift_start, residual, wrist_residual=None
):
    """把独立关节残差从闭合开始渐进叠加到一条O6轨迹。

    输入：`(T,12)`轨迹、闭合/抬升帧、6维关节残差和可选6维手腕残差。
    输出：保持手腕不变、关节满足真实限位的新轨迹。
    内部逻辑：闭合前系数0，闭合到抬升线性增至1，抬升后保持；最后逐关节裁剪。
    作用：搜索邻近抓形时避免接近阶段提前碰撞，也不冻结原动态关节变化。
    """
    result = np.asarray(frames, dtype=np.float32).copy()
    residual = np.asarray(residual, dtype=np.float32)
    wrist_residual = np.zeros(6, dtype=np.float32) if wrist_residual is None else np.asarray(wrist_residual, dtype=np.float32)
    if result.ndim != 2 or result.shape[1] != 12 or residual.shape != (6,) or wrist_residual.shape != (6,):
        raise ValueError("frames必须为(T,12)，关节/手腕残差必须为(6,)")
    progress = np.zeros(len(result), dtype=np.float32)
    progress[close_start : lift_start + 1] = np.linspace(
        0.0, 1.0, lift_start - close_start + 1, dtype=np.float32
    )
    progress[lift_start + 1 :] = 1.0
    result[:, 6:] = np.clip(
        result[:, 6:] + progress[:, None] * residual[None, :],
        JOINT_LOWER,
        JOINT_UPPER,
    )
    result[:, :6] += progress[:, None] * wrist_residual[None, :]
    return result


def report_rank(report):
    """生成严格成功优先的物理报告排序键。

    输入：标准Linker PhysX JSON字典。
    输出：可直接比较的元组。
    内部逻辑：依次比较成功、持续步数、最大抬升、最终抬升和接触步数。
    作用：避免只追逐一次高度尖峰，并保持与项目候选分析的成功定义一致。
    """
    return (
        int(bool(report["success"])),
        int(report["longest_sustained_lift_steps"]),
        float(report["max_lift_m"]),
        float(report["final_lift_m"]),
        int(report["hand_object_contact_steps"]),
    )


def evaluate_candidate(
    index, residual, wrist_residual, base_data, base_frames, phases, args
):
    """保存并物理重放一个残差候选。

    输入：候选编号/残差、基线数据/帧、阶段和命令行参数。
    输出：含残差、报告路径、核心指标、命令输出和耗时的记录。
    内部逻辑：生成只含目标轨迹的npy，再调用未经修改的标准CPU PhysX入口。
    作用：为线程池提供完全独立、可审计的最小物理查询单元。
    """
    started = time.perf_counter()
    candidate_dir = args.output_dir / "candidates"
    report_dir = args.output_dir / "reports"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = candidate_dir / f"candidate_{index:03d}.npy"
    report_path = report_dir / f"candidate_{index:03d}.json"
    output = dict(base_data)
    output["grasp_seqs"] = apply_residual(
        base_frames,
        phases["close_start_frame"],
        phases["lift_start_frame"],
        residual,
        wrist_residual,
    )[None, ...]
    output["source_trajectory_indices"] = np.asarray(
        [args.source_index], dtype=np.int64
    )
    output["physics_search_residual"] = np.asarray(residual, dtype=np.float32)
    output["physics_search_wrist_residual"] = np.asarray(
        wrist_residual, dtype=np.float32
    )
    output["method"] = "linker_o6_physics_residual_upper_bound"
    np.save(candidate_path, output, allow_pickle=True)
    command = [
        sys.executable,
        str(REPLAY_SCRIPT),
        "--source", str(args.source),
        "--target", str(candidate_path),
        "--source-index", str(args.source_index),
        "--target-index", "0",
        "--object-name", args.object_name,
        "--output", str(report_path),
    ]
    process = subprocess.run(command, text=True, capture_output=True, check=False)
    if process.returncode != 0 or not report_path.is_file():
        raise RuntimeError(
            f"候选{index}物理重放失败: {process.stderr or process.stdout}"
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return {
        "candidate_index": int(index),
        "residual_rad": np.asarray(residual).tolist(),
        "wrist_residual": np.asarray(wrist_residual).tolist(),
        "candidate_path": str(candidate_path.resolve()),
        "report_path": str(report_path.resolve()),
        "success": bool(report["success"]),
        "max_lift_m": float(report["max_lift_m"]),
        "final_lift_m": float(report["final_lift_m"]),
        "longest_sustained_lift_steps": int(
            report["longest_sustained_lift_steps"]
        ),
        "hand_object_contact_steps": int(report["hand_object_contact_steps"]),
        "rank": report_rank(report),
        "elapsed_seconds": time.perf_counter() - started,
    }


def main():
    """解析参数、并行搜索并保存上界摘要。

    输入：源/基线/索引、阶段元数据、候选规模、幅度、线程数和输出目录。
    输出：全部候选JSON、按严格物理指标选出的best.npy和`search_summary.json`。
    内部逻辑：验证基线索引后生成固定残差集，用线程池调度独立PhysX进程并排序。
    作用：在不修改正式方法的前提下诊断近失轨迹是否存在邻近成功抓形。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--source-index", type=int, required=True)
    parser.add_argument("--target-index", type=int, required=True)
    parser.add_argument("--object-name", required=True)
    parser.add_argument("--close-start-frame", type=int, required=True)
    parser.add_argument("--lift-start-frame", type=int, required=True)
    parser.add_argument("--count", type=int, default=31)
    parser.add_argument("--scale", type=float, default=0.16)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--probe-wrist", action="store_true")
    parser.add_argument("--wrist-translation-scale", type=float, default=0.003)
    parser.add_argument("--wrist-rotation-scale", type=float, default=0.03)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    data = np.load(args.target, allow_pickle=True).item()
    all_frames = np.asarray(data["grasp_seqs"], dtype=np.float32)
    if not 0 <= args.target_index < len(all_frames):
        parser.error("--target-index越界")
    if not 0 <= args.close_start_frame < args.lift_start_frame < all_frames.shape[1]:
        parser.error("阶段帧必须满足0<=close<lift<T")
    if args.probe_wrist:
        wrist_residuals = generate_wrist_probes(
            args.wrist_translation_scale, args.wrist_rotation_scale
        )
        residuals = np.zeros_like(wrist_residuals)
    else:
        residuals = generate_residuals(args.count, args.scale, args.seed)
        wrist_residuals = np.zeros_like(residuals)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    phases = {
        "close_start_frame": args.close_start_frame,
        "lift_start_frame": args.lift_start_frame,
    }
    results = []
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                evaluate_candidate,
                index,
                residual,
                wrist_residuals[index],
                data,
                all_frames[args.target_index],
                phases,
                args,
            ): index
            for index, residual in enumerate(residuals)
        }
        for future in as_completed(futures):
            item = future.result()
            results.append(item)
            print(
                f"candidate={item['candidate_index']:03d} "
                f"success={item['success']} lift={item['max_lift_m']:.4f}m",
                flush=True,
            )
    results.sort(key=lambda item: item["candidate_index"])
    best = max(results, key=lambda item: tuple(item["rank"]))
    best_data = np.load(best["candidate_path"], allow_pickle=True).item()
    best_path = args.output_dir / "best.npy"
    np.save(best_path, best_data, allow_pickle=True)
    summary = {
        "status": "diagnostic_upper_bound_not_formal_method",
        "source": str(args.source.resolve()),
        "target": str(args.target.resolve()),
        "source_index": args.source_index,
        "target_index": args.target_index,
        "object_name": args.object_name,
        "count": args.count,
        "scale": args.scale,
        "seed": args.seed,
        "workers": args.workers,
        "close_start_frame": args.close_start_frame,
        "lift_start_frame": args.lift_start_frame,
        "best_candidate_index": best["candidate_index"],
        "best_residual_rad": best["residual_rad"],
        "best_wrist_residual": best["wrist_residual"],
        "best_success": best["success"],
        "best_max_lift_m": best["max_lift_m"],
        "best_sustained_steps": best["longest_sustained_lift_steps"],
        "best_path": str(best_path.resolve()),
        "wall_time_seconds": time.perf_counter() - started,
        "results": results,
    }
    summary_path = args.output_dir / "search_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"best={best['candidate_index']} success={best['success']}")
    print(f"output={summary_path}")


if __name__ == "__main__":
    main()
