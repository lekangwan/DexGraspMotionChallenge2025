"""Complete the no-Task-ID ablation through Online-R1 and Temporal3.

This script starts only after run_missing_teacher_comparisons.py has selected
the no-Task-ID offline checkpoint on the frozen development split. It never
uses the final holdout for model selection or reporting.
"""

import argparse
from pathlib import Path
import statistics
import subprocess
import sys

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
MISSING_SUMMARY = (
    ROOT / "custom_tools/results/missing_teacher_comparisons_v1/summary.yaml")
COLLECTOR = (
    ROOT / "custom_tools/collect_taskid_online_scaled20_isolated.py")
TRAIN = ROOT / "custom_tools/train_bc.py"
EVALUATE = ROOT / "custom_tools/evaluate_bc_checkpoints_isolated.py"
RESIDUAL_CONFIG = ROOT / "custom_tools/configs/residual_ppo_stage1.yaml"
TRAJECTORY_ROOT = (
    ROOT / "dexgrasp/dataset/scaled_category_final_v1_preprocessed")
DEV_SELECTION = ROOT / "custom_tools/configs/scaled_development_all12.yaml"

OFFLINE_CONFIG = (
    ROOT / "custom_tools/configs/unified_student_notask_scaled20_v1.yaml")
R1_CONFIG = (
    ROOT / "custom_tools/configs/"
    "unified_student_notask_online_r1_scaled20_v1.yaml")
TEMPORAL_CONFIG = (
    ROOT / "custom_tools/configs/unified_student_notask_temporal3_v1.yaml")

R1_DATA = (
    ROOT / "custom_tools/data/distillation/"
    "online_notask_scaled20_r1_train4.npz")
R1_PARTS = (
    ROOT / "custom_tools/data/distillation/"
    "online_notask_scaled20_r1_train4_parts")
R2_DATA = (
    ROOT / "custom_tools/data/distillation/"
    "online_notask_scaled20_r2_train4_offset4.npz")
R2_PARTS = (
    ROOT / "custom_tools/data/distillation/"
    "online_notask_scaled20_r2_train4_offset4_parts")
AGGREGATED = (
    ROOT / "custom_tools/data/distillation/"
    "online_notask_scaled20_r1_r2_aggregated.npz")

R1_RUN_NAME = "unified_student_notask_online_r1_frac025_seed2025_e10_v1"
R1_RUN = ROOT / "custom_tools/runs/bc" / R1_RUN_NAME
R1_EPOCHS = (2, 4, 6, 8, 10)
TEMPORAL_RUN_NAME = "unified_student_notask_temporal3_seed2025_e4_v1"
TEMPORAL_RUN = ROOT / "custom_tools/runs/bc" / TEMPORAL_RUN_NAME
TEMPORAL_EPOCHS = (1, 2, 3, 4)
OUTPUT = ROOT / "custom_tools/results/notask_full_pipeline_ablation_v1"
SEEDS = (2025, 2026, 2027)

TASKID_REPEAT_SUMMARY = (
    ROOT / "custom_tools/results/taskid_temporal3_repeats_v1/summary.yaml")


def load_yaml(path):
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def run(command, dry_run=False):
    command = [str(item) for item in command]
    print("RUN: {}".format(" ".join(command)), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=str(ROOT), check=True)


def checkpoint(run_dir, epoch):
    paths = list(run_dir.glob(
        "epoch={:03d}-step=*.ckpt".format(epoch - 1)))
    if len(paths) != 1:
        raise RuntimeError(
            "Expected one epoch {} checkpoint in {}".format(epoch, run_dir))
    return paths[0].resolve()


def train_if_needed(cli, config, run_name, run_dir, init):
    last = run_dir / "last.ckpt"
    resource = run_dir / "resource_summary.yaml"
    if last.is_file() and resource.is_file():
        print("[REUSE TRAINING] {}".format(run_name), flush=True)
        return
    command = [
        PYTHON, "-u", TRAIN,
        "--config", config,
        "--run-name", run_name,
        "--min-free-vram-mb", cli.min_free_vram_mb,
    ]
    if last.is_file():
        command.extend(["--resume-checkpoint", last])
    else:
        command.extend(["--init-checkpoint", init])
    run(command, cli.dry_run)


def collect(cli, student, config, output, parts, offset):
    if output.is_file() and output.with_suffix(".yaml").is_file():
        print("[REUSE COLLECTION] {}".format(output), flush=True)
        return
    run([
        PYTHON, "-u", COLLECTOR,
        "--student-checkpoint", student,
        "--bc-config", config,
        "--output", output,
        "--parts-dir", parts,
        "--trajectories-per-object", 4,
        "--trajectory-start-offset", offset,
        "--horizon", 69,
        "--seed", 2025,
        "--min-free-vram-mb", cli.min_free_vram_mb,
        "--max-attempts", cli.max_attempts,
    ], cli.dry_run)


def read_result(path, expected_checkpoint):
    row = load_yaml(path)["checkpoint_results"][0]
    if Path(row["checkpoint"]).resolve() != expected_checkpoint.resolve():
        raise RuntimeError("Checkpoint mismatch in {}".format(path))
    return row


def evaluate(cli, label, model, config, seed, group):
    output = OUTPUT / group / "{}_seed{}.yaml".format(label, seed)
    if output.is_file():
        return read_result(output, model)
    run([
        PYTHON, "-u", EVALUATE,
        "--checkpoint", model,
        "--bc-config", config,
        "--residual-config", RESIDUAL_CONFIG,
        "--trajectory-root", TRAJECTORY_ROOT,
        "--object-selection", DEV_SELECTION,
        "--output", output,
        "--seed", seed,
        "--min-free-vram-mb", cli.min_free_vram_mb,
        "--max-attempts", cli.max_attempts,
    ], cli.dry_run)
    if cli.dry_run:
        return None
    return read_result(output, model)


def rank_key(row):
    return (
        float(row["macro_official_peak_success_rate"]),
        float(row["macro_mean_maximum_lift_m"]),
        -float(row["macro_failure_rate"]),
    )


def select_epoch(cli, label, run_dir, config, epochs):
    rows = []
    for epoch in epochs:
        model = checkpoint(run_dir, epoch)
        row = evaluate(
            cli, "{}_epoch{:02d}".format(label, epoch),
            model, config, 2025, "screening")
        rows.append((epoch, model, row))
    return max(rows, key=lambda item: rank_key(item[2])), rows


def repeat_summary(label, model, rows):
    macros = [
        float(row["macro_official_peak_success_rate"]) for row in rows]
    overalls = [
        float(row["overall_official_peak_success_rate"]) for row in rows]
    lifts = [float(row["macro_mean_maximum_lift_m"]) for row in rows]
    failures = [float(row["macro_failure_rate"]) for row in rows]
    categories = {}
    for category in ("bottle", "mug", "bowl", "camera"):
        values = [
            float(row["category_macro_success_rates"][category])
            for row in rows]
        categories[category] = {
            "mean": statistics.mean(values),
            "std": statistics.pstdev(values),
            "values": values,
        }
    return {
        "label": label,
        "checkpoint": str(model),
        "success_counts": [
            int(row["total_success_count"]) for row in rows],
        "trajectory_count_per_seed": int(rows[0]["total_trajectory_count"]),
        "overall_success_mean": statistics.mean(overalls),
        "overall_success_std": statistics.pstdev(overalls),
        "macro_success_mean": statistics.mean(macros),
        "macro_success_std": statistics.pstdev(macros),
        "lift_mean_m": statistics.mean(lifts),
        "failure_mean": statistics.mean(failures),
        "category_success": categories,
    }


def merge_online_rounds():
    if AGGREGATED.is_file() and AGGREGATED.with_suffix(".yaml").is_file():
        print("[REUSE AGGREGATION] {}".format(AGGREGATED), flush=True)
        return
    arrays = {}
    sources = []
    for path in (R1_DATA, R2_DATA):
        data = np.load(path, allow_pickle=False)
        sources.append(data)
    try:
        if not np.array_equal(
                sources[0]["object_ids"], sources[1]["object_ids"]):
            raise RuntimeError("R1/R2 object ordering differs")
        keys = (
            "observations", "teacher_actions", "student_actions",
            "category_indices", "object_indices", "trajectory_indices",
            "frame_indices")
        pair_sets = [
            set(zip(
                data["object_indices"].tolist(),
                data["trajectory_indices"].tolist()))
            for data in sources]
        if pair_sets[0] & pair_sets[1]:
            raise RuntimeError("No-Task-ID R1/R2 trajectories overlap")
        for key in keys:
            arrays[key] = np.concatenate(
                [data[key] for data in sources], axis=0)
        arrays["object_ids"] = sources[0]["object_ids"].copy()
        np.savez_compressed(AGGREGATED, **arrays)
        summary = {
            "method": "no-Task-ID DAgger R1/R2 aggregation",
            "training_split_only": True,
            "formal_final_holdout_used": False,
            "source_rounds": [str(R1_DATA), str(R2_DATA)],
            "sample_count": int(len(arrays["observations"])),
            "trajectory_pair_overlap": 0,
        }
        with AGGREGATED.with_suffix(".yaml").open(
                "w", encoding="utf-8") as handle:
            yaml.safe_dump(summary, handle, sort_keys=False)
    finally:
        for data in sources:
            data.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-free-vram-mb", type=int, default=4500)
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    cli = parser.parse_args()

    required = (
        MISSING_SUMMARY, COLLECTOR, TRAIN, EVALUATE,
        RESIDUAL_CONFIG, TRAJECTORY_ROOT, DEV_SELECTION,
        OFFLINE_CONFIG, R1_CONFIG, TEMPORAL_CONFIG,
        TASKID_REPEAT_SUMMARY)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Run the first missing-comparison stage before this script: {}"
            .format(missing))
    first_summary = load_yaml(MISSING_SUMMARY)
    offline = Path(
        first_summary["taskid_ablation"]
        ["without_taskid_three_seed"]["checkpoint"]).resolve()
    if not offline.is_file():
        raise FileNotFoundError(offline)
    OUTPUT.mkdir(parents=True, exist_ok=True)

    collect(cli, offline, OFFLINE_CONFIG, R1_DATA, R1_PARTS, 0)
    if cli.dry_run:
        print("DRY_RUN stops after the first dependent command.", flush=True)
        return

    train_if_needed(
        cli, R1_CONFIG, R1_RUN_NAME, R1_RUN, offline)
    r1_best, r1_screen = select_epoch(
        cli, "notask_online_r1", R1_RUN, R1_CONFIG, R1_EPOCHS)

    collect(cli, r1_best[1], R1_CONFIG, R2_DATA, R2_PARTS, 4)
    merge_online_rounds()

    train_if_needed(
        cli, TEMPORAL_CONFIG, TEMPORAL_RUN_NAME,
        TEMPORAL_RUN, r1_best[1])
    temporal_best, temporal_screen = select_epoch(
        cli, "notask_temporal3", TEMPORAL_RUN,
        TEMPORAL_CONFIG, TEMPORAL_EPOCHS)

    repeated = {}
    for label, selected, config in (
        ("no_taskid_online_r1", r1_best, R1_CONFIG),
        ("no_taskid_temporal3", temporal_best, TEMPORAL_CONFIG),
    ):
        rows = [selected[2]]
        for seed in (2026, 2027):
            rows.append(evaluate(
                cli, label, selected[1], config, seed, "repeats"))
        repeated[label] = repeat_summary(label, selected[1], rows)

    taskid_summary = load_yaml(TASKID_REPEAT_SUMMARY)
    taskid_nodes = {
        row["label"]: row for row in taskid_summary["nodes"]}
    summary = {
        "status": "complete",
        "formal_final_holdout_result": False,
        "final_holdout_accessed": False,
        "purpose": (
            "Controlled full-pipeline ablation of the four-way category "
            "one-hot; no post-hoc final-holdout selection."),
        "no_taskid_offline_checkpoint": str(offline),
        "r1_screening": [
            {
                "epoch": epoch,
                "macro_success_rate": float(
                    row["macro_official_peak_success_rate"]),
                "checkpoint": str(model),
            }
            for epoch, model, row in r1_screen
        ],
        "temporal3_screening": [
            {
                "epoch": epoch,
                "macro_success_rate": float(
                    row["macro_official_peak_success_rate"]),
                "checkpoint": str(model),
            }
            for epoch, model, row in temporal_screen
        ],
        "without_taskid": repeated,
        "with_taskid_existing_controls": {
            "online_r1": taskid_nodes["online_r1"],
            "temporal3": taskid_nodes["temporal3"],
        },
    }
    with (OUTPUT / "summary.yaml").open(
            "w", encoding="utf-8") as handle:
        yaml.safe_dump(summary, handle, allow_unicode=True, sort_keys=False)
    print("NOTASK_FULL_PIPELINE_ABLATION=COMPLETE", flush=True)
    for label, row in repeated.items():
        print("{} macro={:.2f}+/-{:.2f}% success={}".format(
            label, 100 * row["macro_success_mean"],
            100 * row["macro_success_std"],
            row["success_counts"]), flush=True)


if __name__ == "__main__":
    main()
