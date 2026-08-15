#!/usr/bin/env python3
"""顺序运行单只手的BC第二seed、Soup、类别教师和两种统一学生。

输入：`full_pipeline_v1.json`和`--hand`；第一seed复用已经完成的正式BC。
输出：第二seed checkpoint、Soup、类别教师、train/valid教师标签、T100/T70学生。
内部逻辑：按依赖顺序生成派生配置并调用共享训练入口；完整阶段自动跳过，部分
训练从last.pt严格续训，Soup/标签若存在则不覆盖。
作用：把离线蒸馏阶段变成一次可续跑命令；三只手之间无依赖，可由用户并行运行。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


POLICY_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = POLICY_ROOT.parents[1]
TRAIN_SCRIPT = POLICY_ROOT / "train.py"
SOUP_SCRIPT = POLICY_ROOT / "prepare" / "make_model_soup.py"
LABEL_SCRIPT = POLICY_ROOT / "prepare" / "generate_teacher_labels.py"
DEFAULT_PIPELINE = POLICY_ROOT / "configs" / "full_pipeline_v1.json"


def project_path(value):
    """把配置内项目相对路径转为绝对路径。"""
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def run_checked(command):
    """以前台、无缓冲方式运行子阶段，非零退出立即停止整条依赖链。"""
    print("RUN:", " ".join(str(value) for value in command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def write_config(path, config):
    """写派生训练配置；已有文件必须逐字段相同，防止续跑时参数漂移。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(config, ensure_ascii=False, indent=2) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != text:
        raise ValueError(f"已有派生配置与当前冻结配置不同: {path}")
    path.write_text(text, encoding="utf-8")


def train_stage(config_path, output_dir):
    """运行、续训或跳过一个训练阶段。

    输入：派生配置路径和输出目录。
    输出：best.pt路径。
    内部逻辑：summary+best同时存在视为完成；只有last时传`--resume`，否则新训练。
    作用：用户中断后可重复同一总命令，而不会从头覆盖已经完成的模型。
    """
    best = output_dir / "best.pt"
    summary = output_dir / "training_summary.json"
    if best.is_file() and summary.is_file():
        print(f"SKIP complete: {output_dir}", flush=True)
        return best
    command = [sys.executable, "-u", str(TRAIN_SCRIPT), "--config", str(config_path)]
    last = output_dir / "last.pt"
    if last.is_file():
        command.extend(["--resume", str(last)])
    run_checked(command)
    if not best.is_file() or not summary.is_file():
        raise RuntimeError(f"训练返回但产物不完整: {output_dir}")
    return best


def base_training_config(pipeline, hand, model_type, seed, output_dir):
    """合并公共参数与阶段参数，返回共享train.py可读取的配置字典。"""
    config = dict(pipeline["common"])
    config.update(
        {
            "experiment_name": output_dir.name,
            "hand": hand,
            "model_type": model_type,
            "seed": int(seed),
            "data_dir": str(project_path(pipeline["data_root"]) / hand),
            "output_dir": str(output_dir),
        }
    )
    return config


def run_hand(pipeline, hand):
    """完成一只手全部离线蒸馏依赖。

    输入：冻结流水线配置和手名。
    输出：各阶段checkpoint路径摘要。
    内部逻辑：先获得第二seed，再与现有第一seed等权Soup；教师从Soup初始化，
    教师标签按原split顺序生成；最后独立训练T100与T70学生。
    作用：为后续valid选择和Online-R1提供两个学生候选及唯一固定教师。
    """
    data_dir = project_path(pipeline["data_root"]) / hand
    run_root = project_path(pipeline["run_root"])
    config_root = run_root / "_configs"
    primary = PROJECT_ROOT / "retarget_research" / "advanced_policy" / "runs" / "formal_v1" / f"{hand}_bc_v1" / "best.pt"
    if not primary.is_file():
        raise FileNotFoundError(primary)

    second_dir = run_root / f"{hand}_bc_seed{pipeline['seed_secondary']}"
    second_config = base_training_config(
        pipeline, hand, "bc", pipeline["seed_secondary"], second_dir
    )
    second_config.update(pipeline["stages"]["bc_second_seed"])
    second_path = config_root / f"{hand}_bc_second.json"
    write_config(second_path, second_config)
    second = train_stage(second_path, second_dir)

    soup_dir = run_root / f"{hand}_bc_soup_v1"
    soup = soup_dir / "model_soup.pt"
    if not soup.is_file():
        weights = pipeline["stages"]["bc_soup"]["weights"]
        command = [
            sys.executable,
            "-u",
            str(SOUP_SCRIPT),
            "--ingredient",
            str(primary),
            "--ingredient",
            str(second),
            "--output",
            str(soup),
        ]
        for weight in weights:
            command.extend(["--weight", str(weight)])
        run_checked(command)
    else:
        print(f"SKIP existing Soup: {soup}", flush=True)

    teacher_dir = run_root / f"{hand}_category_teacher_v1"
    teacher_config = base_training_config(
        pipeline, hand, "category_teacher", pipeline["seed_primary"], teacher_dir
    )
    teacher_config.update(pipeline["stages"]["category_teacher"])
    teacher_config["init_checkpoint"] = str(soup)
    teacher_path = config_root / f"{hand}_category_teacher.json"
    write_config(teacher_path, teacher_config)
    teacher = train_stage(teacher_path, teacher_dir)

    label_dir = run_root / f"{hand}_teacher_labels_v1"
    for split in ("train", "valid"):
        label = label_dir / f"{split}.npz"
        if label.is_file():
            print(f"SKIP existing labels: {label}", flush=True)
            continue
        run_checked(
            [
                sys.executable,
                "-u",
                str(LABEL_SCRIPT),
                "--checkpoint",
                str(teacher),
                "--data-dir",
                str(data_dir),
                "--split",
                split,
                "--output",
                str(label),
                "--device",
                pipeline["common"]["device"],
            ]
        )

    students = {}
    student_stage = pipeline["stages"]["student"]
    for teacher_weight in student_stage["teacher_weights"]:
        suffix = f"t{int(round(teacher_weight * 100))}"
        student_dir = run_root / f"{hand}_student_{suffix}_v1"
        student_config = base_training_config(
            pipeline, hand, "student", pipeline["seed_primary"], student_dir
        )
        student_config.update(
            {
                key: value
                for key, value in student_stage.items()
                if key != "teacher_weights"
            }
        )
        student_config.update(
            {
                "teacher_weight": float(teacher_weight),
                "teacher_label_dir": str(label_dir),
                "init_checkpoint": str(soup),
            }
        )
        config_path = config_root / f"{hand}_student_{suffix}.json"
        write_config(config_path, student_config)
        students[suffix] = str(train_stage(config_path, student_dir))
    return {
        "hand": hand,
        "primary_bc": str(primary),
        "secondary_bc": str(second),
        "soup": str(soup),
        "category_teacher": str(teacher),
        "students": students,
    }


def main():
    """解析单手参数、执行离线阶段并写阶段摘要。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", type=Path, default=DEFAULT_PIPELINE)
    parser.add_argument("--hand", choices=["linker", "xhand", "wuji"], required=True)
    args = parser.parse_args()
    pipeline = json.loads(args.pipeline.read_text(encoding="utf-8"))
    result = run_hand(pipeline, args.hand)
    run_root = project_path(pipeline["run_root"])
    output = run_root / f"{args.hand}_offline_stage_summary.json"
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"OFFLINE_DISTILLATION={output}")


if __name__ == "__main__":
    main()
