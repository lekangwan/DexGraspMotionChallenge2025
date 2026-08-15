#!/usr/bin/env python3
"""在50类均衡valid闭环上比较T100与T70统一学生并冻结唯一候选。

输入：手名、冻结流水线配置和并行worker数。
输出：两份各50条PhysX闭环摘要，以及记录排名与选中checkpoint的JSON。
内部逻辑：每类从已排序valid取第一条，两个学生使用完全相同的轨迹和评测参数；
按类别宏成功率、轨迹成功率、平均最终抬升、平均接触步数依次比较。
作用：在Online-R1采集前完成唯一学生选择，同时严格不读取对象级test。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


POLICY_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = POLICY_ROOT.parents[1]
DEFAULT_PIPELINE = POLICY_ROOT / "configs" / "full_pipeline_v1.json"
HAND_SPECS = POLICY_ROOT / "configs" / "hand_data_specs_v4.json"
MANIFEST = PROJECT_ROOT / "retarget_research" / "manifests" / "formal_50c_100o_1000t_seed20260808.json"
POLICY_SPLIT = POLICY_ROOT / "data" / "formal_v1" / "policy_split_seed20260813.json"
EVALUATE = POLICY_ROOT / "evaluate_policy_manifest.py"


def project_path(value):
    """把项目相对路径转换为绝对Path。"""
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def run_evaluation(hand, checkpoint, target_dir, data_dir, output_dir, workers):
    """完成或复用一个学生的50类均衡valid评测并返回摘要。

    输入：手、checkpoint、候选/数据/输出目录与worker数。
    输出：解析后的`policy_evaluation_summary.json`。
    内部逻辑：始终传`--split valid --max-tasks-per-category 1 --resume`；
    单轨迹seed由共享评测器稳定生成。
    作用：确保T100/T70只改变模型权重，不改变轨迹、初态或随机数协议。
    """
    summary_path = output_dir / "policy_evaluation_summary.json"
    if not summary_path.is_file():
        command = [
            sys.executable,
            "-u",
            str(EVALUATE),
            "--hand",
            hand,
            "--manifest",
            str(MANIFEST),
            "--policy-split",
            str(POLICY_SPLIT),
            "--target-dir",
            str(target_dir),
            "--checkpoint",
            str(checkpoint),
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(output_dir),
            "--split",
            "valid",
            "--max-tasks-per-category",
            "1",
            "--workers",
            str(workers),
            "--device",
            "cpu",
            "--resume",
        ]
        print("RUN:", " ".join(command), flush=True)
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    else:
        print(f"SKIP complete evaluation: {summary_path}", flush=True)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if int(summary["trajectory_count"]) != 50:
        raise ValueError(f"均衡valid应为50条，实际为{summary['trajectory_count']}")
    if int(summary.get("max_tasks_per_category", -1)) != 1:
        raise ValueError("已有摘要不是每类1条协议")
    if Path(summary["checkpoint"]).resolve() != checkpoint.resolve():
        raise ValueError("已有摘要属于不同checkpoint")
    return summary


def ranking_key(summary):
    """把冻结的主指标和三个tie-breaker转换为可降序比较元组。"""
    return (
        float(summary["category_macro_success_rate"]),
        float(summary["trajectory_micro_success_rate"]),
        float(summary["mean_final_lift_m"]),
        float(summary["mean_hand_object_contact_steps"]),
    )


def main():
    """运行两个学生、按冻结规则排序并写选择合同。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", type=Path, default=DEFAULT_PIPELINE)
    parser.add_argument("--hand", choices=["linker", "xhand", "wuji"], required=True)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if args.workers <= 0:
        raise ValueError("workers必须为正整数")
    pipeline = json.loads(args.pipeline.read_text(encoding="utf-8"))
    specs = json.loads(HAND_SPECS.read_text(encoding="utf-8"))
    run_root = project_path(pipeline["run_root"])
    data_dir = project_path(pipeline["data_root"]) / args.hand
    target_dir = project_path(specs["hands"][args.hand]["target_dir"])
    candidates = {}
    for suffix in ("t100", "t70"):
        checkpoint = run_root / f"{args.hand}_student_{suffix}_v1" / "best.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        output_dir = run_root / f"{args.hand}_student_{suffix}_v1" / "closed_loop_valid_50"
        summary = run_evaluation(
            args.hand, checkpoint, target_dir, data_dir, output_dir, args.workers
        )
        candidates[suffix] = {
            "checkpoint": str(checkpoint.resolve()),
            "evaluation_summary": str((output_dir / "policy_evaluation_summary.json").resolve()),
            "trajectory_count": int(summary["trajectory_count"]),
            "success_count": int(summary["success_count"]),
            "category_macro_success_rate": float(summary["category_macro_success_rate"]),
            "trajectory_micro_success_rate": float(summary["trajectory_micro_success_rate"]),
            "mean_final_lift_m": float(summary["mean_final_lift_m"]),
            "mean_hand_object_contact_steps": float(summary["mean_hand_object_contact_steps"]),
            "ranking_key": list(ranking_key(summary)),
        }
    selected = max(candidates, key=lambda name: tuple(candidates[name]["ranking_key"]))
    result = {
        "schema_version": 1,
        "hand": args.hand,
        "protocol": "valid_split_first_sorted_trajectory_per_category; no_test_access",
        "ranking_rule": [
            "category_macro_success_rate",
            "trajectory_micro_success_rate",
            "mean_final_lift_m",
            "mean_hand_object_contact_steps",
        ],
        "candidates": candidates,
        "selected": selected,
        "selected_checkpoint": candidates[selected]["checkpoint"],
    }
    output = run_root / f"{args.hand}_student_valid_selection.json"
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"selected={selected} success={candidates[selected]['success_count']}/50",
        flush=True,
    )
    print(f"STUDENT_VALID_SELECTION={output.resolve()}")


if __name__ == "__main__":
    main()
