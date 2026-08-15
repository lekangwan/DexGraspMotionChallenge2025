#!/usr/bin/env python3
"""从已冻结统一学生继续完成Online-R1、Temporal3和Diffusion对照。

输入：手名、通过valid选定的student checkpoint和冻结流水线配置。
输出：50类均衡在线轨迹、聚合NPZ、Online-R1学生、Temporal3及Diffusion模型。
内部逻辑：学生控制PhysX、固定类别教师逐状态标注；随后按25/75混合微调；
Temporal从Online-R1无损warm start，Diffusion在同一聚合数据上从头训练。
作用：完成旧Shadow流水线后半段，同时保持三只目标手独立训练和可续跑。
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
AGGREGATE = POLICY_ROOT / "prepare" / "aggregate_online_data.py"
TRAIN = POLICY_ROOT / "train.py"


def project_path(value):
    """把项目相对路径转为绝对Path。"""
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def run_checked(command):
    """前台运行一个依赖阶段并在失败时立即停止。"""
    print("RUN:", " ".join(str(value) for value in command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def write_config(path, config):
    """原子依赖语义地写派生配置；已有不同配置时拒绝续跑混用。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(config, ensure_ascii=False, indent=2) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != text:
        raise ValueError(f"冻结派生配置发生变化: {path}")
    path.write_text(text, encoding="utf-8")


def train_stage(config_path, output_dir):
    """完成、续训或跳过一个共享训练阶段并返回best checkpoint。"""
    best, summary, last = (
        output_dir / "best.pt",
        output_dir / "training_summary.json",
        output_dir / "last.pt",
    )
    if best.is_file() and summary.is_file():
        print(f"SKIP complete: {output_dir}", flush=True)
        return best
    command = [sys.executable, "-u", str(TRAIN), "--config", str(config_path)]
    if last.is_file():
        command.extend(["--resume", str(last)])
    run_checked(command)
    if not best.is_file() or not summary.is_file():
        raise RuntimeError(f"训练产物不完整: {output_dir}")
    return best


def training_config(pipeline, hand, model_type, output_dir):
    """返回一个继承公共参数的目标手训练配置。"""
    config = dict(pipeline["common"])
    config.update(
        {
            "experiment_name": output_dir.name,
            "hand": hand,
            "model_type": model_type,
            "seed": int(pipeline["seed_primary"]),
            "data_dir": str(project_path(pipeline["data_root"]) / hand),
            "output_dir": str(output_dir),
        }
    )
    return config


def main():
    """解析选定学生，依次采集Online-R1并训练后三个模型。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", type=Path, default=DEFAULT_PIPELINE)
    parser.add_argument("--hand", choices=["linker", "xhand", "wuji"], required=True)
    parser.add_argument("--student-checkpoint", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    pipeline = json.loads(args.pipeline.read_text(encoding="utf-8"))
    specs = json.loads(HAND_SPECS.read_text(encoding="utf-8"))
    student = args.student_checkpoint.expanduser().resolve()
    if not student.is_file():
        raise FileNotFoundError(student)
    run_root = project_path(pipeline["run_root"])
    data_dir = project_path(pipeline["data_root"]) / args.hand
    teacher = run_root / f"{args.hand}_category_teacher_v1" / "best.pt"
    if not teacher.is_file():
        raise FileNotFoundError(teacher)
    target_dir = project_path(specs["hands"][args.hand]["target_dir"])
    config_root = run_root / "_configs"

    collection_reports = run_root / f"{args.hand}_online_r1_collection"
    online_dir = run_root / f"{args.hand}_online_r1_raw"
    collection_summary = collection_reports / "policy_evaluation_summary.json"
    if not collection_summary.is_file():
        stage = pipeline["stages"]["online_r1"]
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
                str(student),
                "--teacher-checkpoint",
                str(teacher),
                "--online-data-dir",
                str(online_dir),
                "--data-dir",
                str(data_dir),
                "--output-dir",
                str(collection_reports),
                "--split",
                stage["rollout_split"],
                "--max-tasks-per-category",
                str(stage["max_tasks_per_category"]),
                "--workers",
                str(args.workers),
                "--device",
                "cpu",
                "--resume",
            ]
        )
    else:
        print(f"SKIP complete online collection: {collection_summary}", flush=True)

    online_data = run_root / f"{args.hand}_online_r1_data_v1.npz"
    if not online_data.is_file():
        run_checked(
            [
                sys.executable,
                "-u",
                str(AGGREGATE),
                "--online-dir",
                str(online_dir),
                "--data-dir",
                str(data_dir),
                "--output",
                str(online_data),
            ]
        )
    else:
        print(f"SKIP existing aggregate: {online_data}", flush=True)

    online_stage = pipeline["stages"]["online_r1"]
    online_output = run_root / f"{args.hand}_online_r1_student_v1"
    online_config = training_config(pipeline, args.hand, "online_student", online_output)
    online_config.update(
        {key: value for key, value in online_stage.items() if key not in {"rollout_split", "max_tasks_per_category"}}
    )
    online_config.update(
        {
            "online_data_path": str(online_data),
            "init_checkpoint": str(student),
        }
    )
    online_config_path = config_root / f"{args.hand}_online_r1_student.json"
    write_config(online_config_path, online_config)
    online_checkpoint = train_stage(online_config_path, online_output)

    temporal_stage = pipeline["stages"]["temporal3"]
    temporal_output = run_root / f"{args.hand}_temporal3_v1"
    temporal_config = training_config(pipeline, args.hand, "temporal3", temporal_output)
    temporal_config.update(temporal_stage)
    temporal_config.update(
        {
            "online_data_path": str(online_data),
            "init_checkpoint": str(online_checkpoint),
        }
    )
    temporal_config_path = config_root / f"{args.hand}_temporal3.json"
    write_config(temporal_config_path, temporal_config)
    temporal_checkpoint = train_stage(temporal_config_path, temporal_output)

    diffusion_stage = pipeline["stages"]["diffusion"]
    diffusion_output = run_root / f"{args.hand}_diffusion_v1"
    diffusion_config = training_config(pipeline, args.hand, "diffusion", diffusion_output)
    diffusion_config.update(diffusion_stage)
    diffusion_config["online_data_path"] = str(online_data)
    diffusion_config_path = config_root / f"{args.hand}_diffusion.json"
    write_config(diffusion_config_path, diffusion_config)
    diffusion_checkpoint = train_stage(diffusion_config_path, diffusion_output)

    result = {
        "hand": args.hand,
        "selected_offline_student": str(student),
        "category_teacher": str(teacher),
        "online_collection_summary": str(collection_summary),
        "online_data": str(online_data),
        "online_r1_student": str(online_checkpoint),
        "temporal3": str(temporal_checkpoint),
        "diffusion": str(diffusion_checkpoint),
    }
    output = run_root / f"{args.hand}_online_temporal_diffusion_summary.json"
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"ONLINE_TEMPORAL_DIFFUSION={output}")


if __name__ == "__main__":
    main()
