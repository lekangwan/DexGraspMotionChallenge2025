"""Generate coherent teacher labels and train two Task-ID student controls."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    REPO_ROOT
    / "custom_tools/configs/unified_student_taskid_scaled20_v1.yaml"
)
INIT_CHECKPOINT = (
    REPO_ROOT
    / "custom_tools/runs/bc/model_soups/"
    / "noise005_s2025_s2026_weighted2to1.ckpt"
)
LABELS = (
    REPO_ROOT
    / "custom_tools/data/distillation/"
    / "routed_teacher_scaled20_standard_train1726.npz"
)
TEACHERS = {
    "bottle": (
        REPO_ROOT
        / "custom_tools/runs/bc/"
        / "category_expert_bottle_scaled20_noise005_soup_seed2025_e40_v1/"
        / "epoch=029-step=8790.ckpt"
    ),
    "mug": (
        REPO_ROOT
        / "custom_tools/runs/bc/"
        / "category_expert_mug_scaled20_noise005_soup_seed2025_e40_v1/"
        / "epoch=009-step=2250.ckpt"
    ),
    "bowl": (
        REPO_ROOT
        / "custom_tools/runs/bc/"
        / "category_expert_bowl_scaled20_noise005_soup_seed2025_e40_v1/"
        / "epoch=039-step=7440.ckpt"
    ),
    "camera": (
        REPO_ROOT
        / "custom_tools/runs/bc/"
        / "category_expert_camera_scaled20_noise005_soup_seed2025_e40_v1/"
        / "epoch=039-step=9520.ckpt"
    ),
}
RUNS = (
    ("unified_student_taskid_scaled20_t100_seed2025_e20_v1", 1.00),
    ("unified_student_taskid_scaled20_t70_demo30_seed2025_e20_v1", 0.70),
)
EXPECTED_FRAME_SAMPLES = 1726 * 70


def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-free-vram-mb", type=int, default=4500)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def require_inputs():
    required = [CONFIG, INIT_CHECKPOINT, *TEACHERS.values()]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing stage inputs: {}".format(missing))
    if any("objectbalanced" in str(path).lower() for path in TEACHERS.values()):
        raise RuntimeError(
            "Post-hoc object-balanced experts are forbidden in this pool")


def label_command(cli):
    command = [
        sys.executable,
        "-u",
        str(REPO_ROOT / "custom_tools/generate_routed_teacher_labels.py"),
        "--config",
        str(CONFIG),
        "--output",
        str(LABELS),
        "--batch-size",
        "512",
        "--min-free-vram-mb",
        str(cli.min_free_vram_mb),
    ]
    for category, checkpoint in TEACHERS.items():
        command.extend([
            "--teacher", "{}={}".format(category, checkpoint)])
    return command


def train_command(cli, run_name, teacher_weight):
    return [
        sys.executable,
        "-u",
        str(REPO_ROOT / "custom_tools/train_bc.py"),
        "--config",
        str(CONFIG),
        "--run-name",
        run_name,
        "--seed",
        "2025",
        "--num-epochs",
        "20",
        "--teacher-weight",
        str(teacher_weight),
        "--init-checkpoint",
        str(INIT_CHECKPOINT),
        "--min-free-vram-mb",
        str(cli.min_free_vram_mb),
    ]


def verify_existing_labels():
    data = np.load(LABELS, allow_pickle=False)
    if data["teacher_actions"].shape != (EXPECTED_FRAME_SAMPLES, 28):
        raise RuntimeError(
            "Existing teacher labels have unexpected shape: {}".format(
                data["teacher_actions"].shape))
    actual = {
        str(category): str(Path(path).resolve())
        for category, path in zip(
            data["teacher_categories"].tolist(),
            data["teacher_checkpoints"].tolist())
    }
    expected = {
        category: str(path.resolve())
        for category, path in TEACHERS.items()
    }
    if actual != expected:
        raise RuntimeError(
            "Existing labels came from a different teacher pool")


def run_checked(command):
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    subprocess.run(
        command, cwd=str(REPO_ROOT), env=environment, check=True)


def main():
    cli = parse_cli()
    require_inputs()
    plan = {
        "purpose": (
            "Compare pure routed-teacher distillation with the predeclared "
            "70/30 teacher/demonstration target using explicit Task ID."),
        "teacher_pool_rule": (
            "All four teachers are standard 20-object category experts; "
            "no post-hoc object-balanced expert is allowed."),
        "task_id_order": ["bottle", "mug", "bowl", "camera"],
        "teacher_checkpoints": {
            category: str(path) for category, path in TEACHERS.items()},
        "teacher_labels": str(LABELS),
        "initialization": str(INIT_CHECKPOINT),
        "runs": [
            {
                "run_name": run_name,
                "teacher_weight": teacher_weight,
                "demo_weight": 1.0 - teacher_weight,
                "epochs": 20,
            }
            for run_name, teacher_weight in RUNS
        ],
    }
    plan_path = (
        REPO_ROOT / "custom_tools/results/taskid_offline_stage_plan_v1.json")
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    with plan_path.open("w", encoding="utf-8") as handle:
        json.dump(plan, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    commands = [label_command(cli)] + [
        train_command(cli, run_name, weight)
        for run_name, weight in RUNS
    ]
    print("Validated coherent teacher pool; plan: {}".format(plan_path))
    if cli.dry_run:
        for command in commands:
            print(" ".join(command))
        return

    if LABELS.is_file():
        verify_existing_labels()
        print("[SKIP] verified teacher labels: {}".format(LABELS))
    else:
        run_checked(commands[0])
        verify_existing_labels()

    for (run_name, _), command in zip(RUNS, commands[1:]):
        run_dir = REPO_ROOT / "custom_tools/runs/bc" / run_name
        last_checkpoint = run_dir / "last.ckpt"
        resource_summary = run_dir / "resource_summary.yaml"
        if last_checkpoint.is_file() and resource_summary.is_file():
            print("[SKIP] completed run: {}".format(run_name))
            continue
        existing = list(run_dir.glob("*.ckpt"))
        if existing:
            raise RuntimeError(
                "Partial run needs inspection before resuming: {}".format(
                    run_dir))
        run_checked(command)
        if not last_checkpoint.is_file() or not resource_summary.is_file():
            raise RuntimeError(
                "Training returned without completion files: {}".format(
                    run_dir))
    print("TASKID_OFFLINE_STAGE=COMPLETE")


if __name__ == "__main__":
    main()
