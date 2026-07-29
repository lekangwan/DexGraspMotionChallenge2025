"""Run the missing provided-BC comparison and Task-ID ablation.

Model selection uses only the frozen 12-object development set. The already
accessed 8-object final holdout is evaluated only after the data-matched
provided BC epoch has been selected, and is never used to choose a model.
"""

import argparse
from pathlib import Path
import statistics
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
TRAIN = ROOT / "custom_tools/train_bc.py"
EVALUATE = ROOT / "custom_tools/evaluate_bc_checkpoints_isolated.py"
RESIDUAL_CONFIG = ROOT / "custom_tools/configs/residual_ppo_stage1.yaml"
TRAJECTORY_ROOT = (
    ROOT / "dexgrasp/dataset/scaled_category_final_v1_preprocessed")
DEV_SELECTION = ROOT / "custom_tools/configs/scaled_development_all12.yaml"
FINAL_SELECTION = ROOT / "custom_tools/configs/scaled_final_holdout_all8.yaml"
OFFICIAL_CHECKPOINT = (
    ROOT / "ActionDiffusion/bc/saved_models/"
    "1obj_seq2000_DexRep_pro100_start_uniform_vis_action_dsam_mod/last.ckpt")
SOUP_CHECKPOINT = (
    ROOT / "custom_tools/runs/bc/model_soups/"
    "noise005_s2025_s2026_weighted2to1.ckpt")

OFFICIAL_CONFIG = (
    ROOT / "custom_tools/configs/official_bc_scaled20_matched_v1.yaml")
OFFICIAL_RUN = (
    ROOT / "custom_tools/runs/bc/"
    "official_bc_scaled20_matched_seed2025_e100_v1")
OFFICIAL_EPOCHS = (20, 40, 60, 80, 100)

NOTASK_CONFIG = (
    ROOT / "custom_tools/configs/unified_student_notask_scaled20_v1.yaml")
NOTASK_RUN = (
    ROOT / "custom_tools/runs/bc/"
    "unified_student_notask_scaled20_t100_seed2025_e20_v1")
NOTASK_EPOCHS = (5, 10, 15, 20)

TASKID_CONFIG = (
    ROOT / "custom_tools/configs/unified_student_taskid_scaled20_v1.yaml")
TASKID_CHECKPOINT = (
    ROOT / "custom_tools/runs/bc/"
    "unified_student_taskid_scaled20_t100_seed2025_e20_v1/"
    "epoch=014-step=14145.ckpt")
TASKID_EXISTING_RESULTS = (
    ROOT / "custom_tools/results/taskid_offline_development_v1")

OUTPUT = ROOT / "custom_tools/results/missing_teacher_comparisons_v1"
SEEDS = (2025, 2026, 2027)


def load_yaml(path):
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def checkpoint(run_dir, epoch):
    matches = list(run_dir.glob(
        "epoch={:03d}-step=*.ckpt".format(epoch - 1)))
    if len(matches) != 1:
        raise RuntimeError(
            "Expected one epoch {} checkpoint in {}".format(epoch, run_dir))
    return matches[0].resolve()


def run(command, dry_run=False):
    print(" ".join(str(item) for item in command), flush=True)
    if not dry_run:
        subprocess.run(
            [str(item) for item in command], cwd=str(ROOT), check=True)


def train_if_needed(cli, config, run_name, run_dir, init_checkpoint):
    last_checkpoint = run_dir / "last.ckpt"
    resource_summary = run_dir / "resource_summary.yaml"
    if last_checkpoint.is_file() and resource_summary.is_file():
        print("[REUSE TRAINING] {}".format(run_name), flush=True)
        return
    command = [
        PYTHON, "-u", TRAIN,
        "--config", config,
        "--run-name", run_name,
        "--min-free-vram-mb", cli.min_free_vram_mb,
    ]
    if last_checkpoint.is_file():
        command.extend(["--resume-checkpoint", last_checkpoint])
        print("[RESUME TRAINING] {}".format(run_name), flush=True)
    else:
        command.extend(["--init-checkpoint", init_checkpoint])
    run(command, cli.dry_run)


def evaluate(cli, label, model_checkpoint, config, selection, seed, split):
    result = OUTPUT / split / (
        "{}_seed{}.yaml".format(label, seed))
    if result.is_file():
        loaded = load_yaml(result)["checkpoint_results"][0]
        if Path(loaded["checkpoint"]).resolve() != model_checkpoint.resolve():
            raise RuntimeError("Checkpoint mismatch in {}".format(result))
        print("[REUSE EVAL] {} {} seed={}".format(
            label, split, seed), flush=True)
        return loaded
    run([
        PYTHON, "-u", EVALUATE,
        "--checkpoint", model_checkpoint,
        "--bc-config", config,
        "--residual-config", RESIDUAL_CONFIG,
        "--trajectory-root", TRAJECTORY_ROOT,
        "--object-selection", selection,
        "--output", result,
        "--seed", seed,
        "--min-free-vram-mb", cli.min_free_vram_mb,
        "--max-attempts", cli.max_attempts,
    ], cli.dry_run)
    if cli.dry_run:
        return None
    return load_yaml(result)["checkpoint_results"][0]


def rank_key(row):
    return (
        float(row["macro_official_peak_success_rate"]),
        float(row["macro_mean_maximum_lift_m"]),
        -float(row["macro_failure_rate"]),
    )


def summarize_repeats(label, checkpoint_path, rows):
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
        "checkpoint": str(checkpoint_path),
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-free-vram-mb", type=int, default=4500)
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    cli = parser.parse_args()

    required = (
        TRAIN, EVALUATE, RESIDUAL_CONFIG, TRAJECTORY_ROOT,
        DEV_SELECTION, FINAL_SELECTION, OFFICIAL_CHECKPOINT,
        SOUP_CHECKPOINT, OFFICIAL_CONFIG, NOTASK_CONFIG,
        TASKID_CONFIG, TASKID_CHECKPOINT)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing inputs: {}".format(missing))
    OUTPUT.mkdir(parents=True, exist_ok=True)

    train_if_needed(
        cli, OFFICIAL_CONFIG,
        "official_bc_scaled20_matched_seed2025_e100_v1",
        OFFICIAL_RUN, OFFICIAL_CHECKPOINT)
    train_if_needed(
        cli, NOTASK_CONFIG,
        "unified_student_notask_scaled20_t100_seed2025_e20_v1",
        NOTASK_RUN, SOUP_CHECKPOINT)
    if cli.dry_run:
        print("DRY_RUN_TRAINING_COMMANDS_COMPLETE", flush=True)
        return

    official_screen = []
    for epoch in OFFICIAL_EPOCHS:
        path = checkpoint(OFFICIAL_RUN, epoch)
        row = evaluate(
            cli, "matched_official_epoch{:03d}".format(epoch),
            path, OFFICIAL_CONFIG, DEV_SELECTION, 2025, "development")
        official_screen.append((epoch, path, row))
    official_best = max(official_screen, key=lambda item: rank_key(item[2]))

    notask_screen = []
    for epoch in NOTASK_EPOCHS:
        path = checkpoint(NOTASK_RUN, epoch)
        row = evaluate(
            cli, "notask_epoch{:03d}".format(epoch),
            path, NOTASK_CONFIG, DEV_SELECTION, 2025, "development")
        notask_screen.append((epoch, path, row))
    notask_best = max(notask_screen, key=lambda item: rank_key(item[2]))

    development_repeats = {}
    for label, path, config, seed2025_row in (
        ("matched_official_bc", official_best[1], OFFICIAL_CONFIG,
         official_best[2]),
        ("offline_student_no_taskid", notask_best[1], NOTASK_CONFIG,
         notask_best[2]),
    ):
        rows = [seed2025_row]
        for seed in (2026, 2027):
            rows.append(evaluate(
                cli, label, path, config, DEV_SELECTION, seed,
                "development_repeats"))
        development_repeats[label] = summarize_repeats(label, path, rows)

    taskid_rows = []
    for seed in SEEDS:
        existing = (
            TASKID_EXISTING_RESULTS / "seed{}".format(seed)
            / "t100_epoch15.yaml")
        aggregate = load_yaml(existing)
        row = aggregate["checkpoint_results"][0]
        if Path(row["checkpoint"]).resolve() != TASKID_CHECKPOINT.resolve():
            raise RuntimeError(
                "Existing Task-ID checkpoint mismatch in {}".format(existing))
        print("[REUSE EXISTING TASK-ID] seed={}".format(seed), flush=True)
        taskid_rows.append(row)
    development_repeats["offline_student_with_taskid"] = summarize_repeats(
        "offline_student_with_taskid", TASKID_CHECKPOINT, taskid_rows)

    final_rows = []
    for seed in SEEDS:
        final_rows.append(evaluate(
            cli, "matched_official_bc", official_best[1],
            OFFICIAL_CONFIG, FINAL_SELECTION, seed, "final_reporting"))
    final_matched_official = summarize_repeats(
        "matched_official_bc", official_best[1], final_rows)

    summary = {
        "status": "complete",
        "selection_rule": (
            "Epochs selected on frozen 12-object development set only; "
            "final holdout is reporting-only."),
        "matched_official_bc": {
            "controlled_features": (
                "provided single-frame network and provided action loss; "
                "same 80-object training data; no custom enhancements"),
            "screening": [
                {
                    "epoch": epoch,
                    "checkpoint": str(path),
                    "macro_success_rate": float(
                        row["macro_official_peak_success_rate"]),
                    "mean_maximum_lift_m": float(
                        row["macro_mean_maximum_lift_m"]),
                }
                for epoch, path, row in official_screen
            ],
            "selected_epoch": official_best[0],
            "development_three_seed": development_repeats[
                "matched_official_bc"],
            "final_three_seed": final_matched_official,
        },
        "taskid_ablation": {
            "controlled_features": (
                "same Soup initialization, routed teacher labels, 80-object "
                "data, noise, sampler, loss, epochs, and evaluation; only "
                "the four-way category one-hot differs"),
            "no_taskid_screening": [
                {
                    "epoch": epoch,
                    "checkpoint": str(path),
                    "macro_success_rate": float(
                        row["macro_official_peak_success_rate"]),
                    "mean_maximum_lift_m": float(
                        row["macro_mean_maximum_lift_m"]),
                }
                for epoch, path, row in notask_screen
            ],
            "selected_no_taskid_epoch": notask_best[0],
            "without_taskid_three_seed": development_repeats[
                "offline_student_no_taskid"],
            "with_taskid_three_seed": development_repeats[
                "offline_student_with_taskid"],
        },
    }
    with (OUTPUT / "summary.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(summary, handle, allow_unicode=True, sort_keys=False)

    print("MISSING_TEACHER_COMPARISONS=COMPLETE", flush=True)
    print("matched official selected epoch={}".format(
        official_best[0]), flush=True)
    print("no-Task-ID selected epoch={}".format(
        notask_best[0]), flush=True)
    for key, row in development_repeats.items():
        print("{} dev macro={:.2f}+/-{:.2f}% success={}".format(
            key, 100 * row["macro_success_mean"],
            100 * row["macro_success_std"],
            row["success_counts"]), flush=True)
    print("matched official final macro={:.2f}+/-{:.2f}% success={}".format(
        100 * final_matched_official["macro_success_mean"],
        100 * final_matched_official["macro_success_std"],
        final_matched_official["success_counts"]), flush=True)


if __name__ == "__main__":
    main()
