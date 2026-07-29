"""Run the frozen Temporal3 gated-residual Stage-1 experiment."""

import argparse
from pathlib import Path
import subprocess
import sys

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
BC_CHECKPOINT = (
    REPO_ROOT / "custom_tools/runs/bc/"
    "unified_student_taskid_temporal3_seed2025_e4_v1/"
    "epoch=003-step=5152.ckpt")
BC_CONFIG = (
    REPO_ROOT / "custom_tools/configs/"
    "unified_student_taskid_temporal3_v1.yaml")
RESIDUAL_CONFIG = (
    REPO_ROOT / "custom_tools/configs/"
    "residual_ppo_temporal3_anchored_i50.yaml")
TRAJECTORY_ROOT = (
    REPO_ROOT / "dexgrasp/dataset/"
    "scaled_category_final_v1_preprocessed")
AUDIT = (
    REPO_ROOT / "custom_tools/results/"
    "temporal3_residual_train495_audit_seed2025.yaml")
SELECTION = (
    REPO_ROOT / "custom_tools/configs/"
    "residual_temporal3_anchored64_selection.yaml")
DEVELOPMENT_SELECTION = (
    REPO_ROOT / "custom_tools/configs/scaled_development_all12.yaml")
RUN_NAME = "temporal3_gated_residual_anchored64_seed2025_i50_v1"
RUN_DIR = REPO_ROOT / "custom_tools/runs/residual_ppo" / RUN_NAME
RESULT_DIR = (
    REPO_ROOT / "custom_tools/results/"
    "temporal3_residual_stage1_development_v1")
BASELINE = (
    REPO_ROOT / "custom_tools/results/"
    "temporal3_residual_zero_dev_parity_seed2025.yaml")


def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--min-free-vram-mb", type=int, default=4500)
    parser.add_argument("--max-attempts", type=int, default=5)
    return parser.parse_args()


def execute(command):
    print("RUN {}".format(" ".join(str(value) for value in command)), flush=True)
    subprocess.run(
        [str(value) for value in command], cwd=str(REPO_ROOT), check=True)


def validate_selection():
    with SELECTION.open("r", encoding="utf-8") as handle:
        selection = yaml.safe_load(handle)
    if selection.get("status") != "frozen_stage1_selection":
        raise ValueError("Temporal3 curriculum is not frozen")
    if selection.get("total_environments") != 64:
        raise ValueError("Expected 64 curriculum environments")
    if selection.get("total_anchor_environments") != 32:
        raise ValueError("Expected 32 genuine-success anchors")


def evaluate(label, checkpoint, cli):
    output = RESULT_DIR / "{}.yaml".format(label)
    if output.exists():
        print("REUSE {}".format(output), flush=True)
        return output
    execute([
        PYTHON, "-u", REPO_ROOT / "custom_tools/evaluate_residual_isolated.py",
        "--residual-checkpoint", checkpoint,
        "--residual-config", RESIDUAL_CONFIG,
        "--bc-checkpoint", BC_CHECKPOINT,
        "--bc-config", BC_CONFIG,
        "--trajectory-root", TRAJECTORY_ROOT,
        "--trajectory-selection", DEVELOPMENT_SELECTION,
        "--num-trajectories", "0",
        "--output", output,
        "--seed", cli.seed,
        "--min-free-vram-mb", cli.min_free_vram_mb,
        "--max-attempts", cli.max_attempts,
    ])
    return output


def summarize(outputs):
    rows = []
    sources = [("baseline_zero_residual", BASELINE)] + list(outputs.items())
    for label, path in sources:
        with path.open("r", encoding="utf-8") as handle:
            result = yaml.safe_load(handle)
        rows.append({
            "label": label,
            "path": str(path),
            "total_success_count": int(result["total_success_count"]),
            "total_trajectory_count": int(result["total_trajectory_count"]),
            "macro_success_rate": float(
                result["macro_official_peak_success_rate"]),
            "macro_mean_maximum_lift_m": float(
                result["macro_mean_maximum_lift_m"]),
            "macro_failure_rate": float(result["macro_failure_rate"]),
            "category_macro_success_rates": result[
                "category_macro_success_rates"],
        })
    baseline = rows[0]["macro_success_rate"]
    for row in rows:
        row["macro_gain_over_paired_baseline"] = (
            row["macro_success_rate"] - baseline)
        row["passes_two_point_repeat_threshold"] = (
            row["label"] != "baseline_zero_residual"
            and row["macro_gain_over_paired_baseline"] >= 0.02)
    summary = {
        "status": "complete",
        "stage": "temporal3_behavior_anchored_gated_residual_stage1",
        "final_holdout_accessed": False,
        "selection": str(SELECTION),
        "run_dir": str(RUN_DIR),
        "paired_baseline": str(BASELINE),
        "repeat_threshold_macro_gain": 0.02,
        "rows": rows,
    }
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    output = RESULT_DIR / "summary.yaml"
    with output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(summary, handle, allow_unicode=True, sort_keys=False)
    print("\nlabel,success,macro,lift_m,failure,gain,repeat", flush=True)
    for row in rows:
        print("{},{}/{},{:.6f},{:.6f},{:.6f},{:+.6f},{}".format(
            row["label"], row["total_success_count"],
            row["total_trajectory_count"], row["macro_success_rate"],
            row["macro_mean_maximum_lift_m"], row["macro_failure_rate"],
            row["macro_gain_over_paired_baseline"],
            row["passes_two_point_repeat_threshold"]), flush=True)
    print("Saved summary: {}".format(output), flush=True)


def main():
    cli = parse_cli()
    if cli.seed != 2025:
        raise ValueError("Stage-1 training seed is frozen to 2025")
    if not SELECTION.exists():
        execute([
            PYTHON,
            REPO_ROOT / "custom_tools/select_temporal3_residual_curriculum.py",
            "--audit", AUDIT,
            "--output", SELECTION,
        ])
    validate_selection()

    if not (RUN_DIR / "last.pt").exists():
        if RUN_DIR.exists():
            raise RuntimeError(
                "An incomplete training directory already exists: {}".format(
                    RUN_DIR))
        execute([
            PYTHON, "-u", REPO_ROOT / "custom_tools/train_residual_ppo.py",
            "--config", RESIDUAL_CONFIG,
            "--bc-checkpoint", BC_CHECKPOINT,
            "--bc-config", BC_CONFIG,
            "--trajectory-root", TRAJECTORY_ROOT,
            "--trajectory-selection", SELECTION,
            "--run-name", RUN_NAME,
            "--min-free-vram-mb", cli.min_free_vram_mb,
        ])
    else:
        print("REUSE completed training: {}".format(RUN_DIR), flush=True)

    outputs = {}
    for iteration in (10, 25, 50):
        label = "iteration_{:04d}".format(iteration)
        checkpoint = RUN_DIR / "checkpoints/{}.pt".format(label)
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        outputs[label] = evaluate(label, checkpoint, cli)
    summarize(outputs)


if __name__ == "__main__":
    main()
