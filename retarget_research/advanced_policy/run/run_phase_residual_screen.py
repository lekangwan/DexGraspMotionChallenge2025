#!/usr/bin/env python3
"""训练阶段条件动作增量策略，并在10条成功专家valid轨迹上快速筛查。

输入：手名和worker数；使用现有成功train数据及冻结目标手候选。
输出：PhaseResidual checkpoint、loss曲线和10条闭环诊断摘要。
内部逻辑：先补充train动作delta统计，再训练“状态+上一命令+阶段→delta”；只从
成功专家valid中按每类最多1条取前10条，检查它是否至少恢复平滑接触/抬升。
作用：在重建昂贵Soup/教师/Online流水线前，用最小物理成本验证新动作表示。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


POLICY_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = POLICY_ROOT.parents[1]
TRAIN = POLICY_ROOT / "train.py"
EVALUATE = POLICY_ROOT / "evaluate_policy_manifest.py"
ADD_STATS = POLICY_ROOT / "prepare" / "add_residual_action_stats.py"
HAND_SPECS = POLICY_ROOT / "configs" / "hand_data_specs_v4.json"
MANIFEST = PROJECT_ROOT / "retarget_research" / "manifests" / "formal_50c_100o_1000t_seed20260808.json"
POLICY_SPLIT = POLICY_ROOT / "data" / "formal_v1" / "policy_split_seed20260813.json"


def project_path(value):
    """把项目相对路径转换为绝对Path。"""
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def run_checked(command):
    """前台运行子阶段，输出完整命令并在非零退出时停止。"""
    print("RUN:", " ".join(str(value) for value in command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def write_config(path, config):
    """写冻结screen配置，已有不同内容时拒绝混用旧checkpoint。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(config, ensure_ascii=False, indent=2) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != text:
        raise ValueError(f"已有screen配置发生变化: {path}")
    path.write_text(text, encoding="utf-8")


def main():
    """补统计、训练残差策略、运行10条专家成功闭环并输出摘要。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--hand", choices=["linker", "xhand", "wuji"], required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.workers <= 0:
        raise ValueError("workers必须为正整数")
    data_dir = POLICY_ROOT / "data" / "formal_v1" / args.hand
    run_root = POLICY_ROOT / "runs" / "full_pipeline_v1"
    output_dir = run_root / f"{args.hand}_phase_residual_screen_v1"
    config_path = run_root / "_configs" / f"{args.hand}_phase_residual_screen.json"
    specs = json.loads(HAND_SPECS.read_text(encoding="utf-8"))
    target_dir = project_path(specs["hands"][args.hand]["target_dir"])

    run_checked([sys.executable, "-u", str(ADD_STATS), "--data-dir", str(data_dir)])
    config = {
        "experiment_name": output_dir.name,
        "hand": args.hand,
        "model_type": "phase_residual",
        "seed": 20260815,
        "device": "cuda",
        "epochs": 100,
        "batch_size": 512,
        "num_workers": 4,
        "learning_rate": 0.0003,
        "weight_decay": 0.0001,
        "gradient_clip": 1.0,
        "early_stopping_patience": 15,
        "minimum_improvement": 0.000001,
        "category_embedding_dim": 16,
        "hidden_dims": [256, 256, 256],
        "dropout": 0.05,
        "motion_steps": 210,
        "data_dir": str(data_dir.resolve()),
        "output_dir": str(output_dir.resolve())
    }
    write_config(config_path, config)
    best = output_dir / "best.pt"
    summary = output_dir / "training_summary.json"
    if not (best.is_file() and summary.is_file()):
        command = [sys.executable, "-u", str(TRAIN), "--config", str(config_path)]
        last = output_dir / "last.pt"
        if last.is_file():
            command.extend(["--resume", str(last)])
        run_checked(command)
    else:
        print(f"SKIP complete training: {output_dir}", flush=True)

    evaluation_dir = output_dir / "closed_loop_expert_valid_10"
    evaluation_summary = evaluation_dir / "policy_evaluation_summary.json"
    if not evaluation_summary.is_file():
        run_checked(
            [
                sys.executable,
                "-u",
                str(EVALUATE),
                "--hand",
                args.hand,
                "--manifest",
                str(MANIFEST),
                "--policy-split",
                str(POLICY_SPLIT),
                "--target-dir",
                str(target_dir),
                "--checkpoint",
                str(best),
                "--data-dir",
                str(data_dir),
                "--output-dir",
                str(evaluation_dir),
                "--split",
                "valid",
                "--expert-success-only",
                "--max-tasks-per-category",
                "1",
                "--max-tasks",
                "10",
                "--workers",
                str(args.workers),
                "--device",
                "cpu",
                "--resume",
            ]
        )
    result = json.loads(evaluation_summary.read_text(encoding="utf-8"))
    print(
        f"PHASE_RESIDUAL_SCREEN={args.hand} success={result['success_count']}/10 "
        f"final_lift={result['mean_final_lift_m']:.6f} "
        f"contacts={result['mean_hand_object_contact_steps']:.2f}"
    )


if __name__ == "__main__":
    main()
