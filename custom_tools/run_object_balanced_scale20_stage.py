"""Train and evaluate the scale-20 object-balanced category expert ablation."""

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import statistics
import subprocess
import sys

import yaml

from custom_tools import run_scaled_category_development_matrix as matrix


REPO_ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = matrix.CATEGORIES
EPOCHS = (5, 10, 15, 20, 25, 30, 35, 40)
TRAIN_CONFIG = (
    REPO_ROOT
    / "custom_tools/configs/category_expert_bc_scaled20_objectbalanced_v1.yaml"
)
MANIFEST = REPO_ROOT / "custom_tools/configs/scaled_category_split_final_v1.json"
DATASET_SUMMARY = (
    REPO_ROOT / "custom_tools/results/scaled_bc20_dataset_summary_v1.json"
)
DEFAULT_OUTPUT = (
    REPO_ROOT / "custom_tools/results/object_balanced_scale20_development_v1"
)
OLD_SUMMARY = (
    REPO_ROOT / "custom_tools/results/scaled_category_development_v1/summary.yaml"
)


def load_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def trajectory_counts():
    summary = load_json(DATASET_SUMMARY)
    counts = {}
    for category in CATEGORIES:
        rows = [row for row in summary["objects"]
                if row["category"] == category]
        if len(rows) != 20:
            raise RuntimeError("Expected 20 {} objects".format(category))
        counts[category] = {
            "train": sum(int(row["train_count"]) for row in rows),
            "valid": sum(int(row["valid_count"]) for row in rows),
        }
    return counts


def run_name(category):
    return (
        "category_expert_{}_scaled20_objectbalanced_noise005_soup_"
        "seed2025_e40_v1".format(category)
    )


def train_command(category, counts, min_free_vram_mb):
    return [
        sys.executable,
        "-u",
        str(REPO_ROOT / "custom_tools/train_bc.py"),
        "--config", str(TRAIN_CONFIG),
        "--run-name", run_name(category),
        "--seed", "2025",
        "--num-epochs", "40",
        "--seq-num", str(counts[category]["train"]),
        "--val-seq-num", str(counts[category]["valid"]),
        "--train-category", category,
        "--category-train-size", "20",
        "--category-manifest", str(MANIFEST),
        "--init-checkpoint", str(matrix.SOUP),
        "--object-balanced-sampling",
        "--min-free-vram-mb", str(min_free_vram_mb),
    ]


def replace_initialization_with_resume(command, checkpoint):
    command = list(command)
    index = command.index("--init-checkpoint")
    del command[index:index + 2]
    command.extend(["--resume-checkpoint", str(checkpoint)])
    return command


def validate_completed_run(category, directory):
    for name in ("last.ckpt", "resolved_config.yaml", "run_metadata.yaml"):
        path = directory / name
        if not path.is_file():
            raise FileNotFoundError(path)
    with (directory / "resolved_config.yaml").open(
            "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if config.get("expert_category") != category:
        raise RuntimeError("Wrong category in {}".format(directory))
    if len(config.get("train_obj_code_list", [])) != 20:
        raise RuntimeError("Wrong object count in {}".format(directory))
    if not config.get("object_balanced_sampling", False):
        raise RuntimeError("Object balancing is disabled in {}".format(directory))
    if int(config.get("checkpoint_every_n_epochs")) != 5:
        raise RuntimeError("Checkpoint interval changed in {}".format(directory))
    for epoch in EPOCHS:
        matrix.checkpoint_for_epoch(directory, epoch)


def train_all(counts, min_free_vram_mb, dry_run):
    directories = {}
    for index, category in enumerate(CATEGORIES, 1):
        directory = REPO_ROOT / "custom_tools/runs/bc" / run_name(category)
        directories[category] = directory
        command = train_command(category, counts, min_free_vram_mb)
        print("TRAIN {}/4 {} train={} valid={}".format(
            index, category, counts[category]["train"],
            counts[category]["valid"]), flush=True)
        if dry_run:
            print("  " + " ".join(command), flush=True)
            continue
        if (directory / "last.ckpt").is_file():
            validate_completed_run(category, directory)
            print("  [REUSE] {}".format(directory), flush=True)
            continue
        partial = sorted(directory.glob("epoch=*-step=*.ckpt"))
        if partial:
            command = replace_initialization_with_resume(command, partial[-1])
            print("  [RESUME] {}".format(partial[-1]), flush=True)
        subprocess.run(command, cwd=str(REPO_ROOT), check=True)
        validate_completed_run(category, directory)
    return directories


def summarize(output_root, seeds, candidates_by_category, expected,
              finalists_by_category, evaluation_mode):
    raw = []
    for seed in seeds:
        for category in CATEGORIES:
            for candidate in candidates_by_category[category]:
                output = (
                    output_root / "seed{}".format(seed) / category
                    / (candidate["label"] + ".yaml")
                )
                if not output.is_file():
                    continue
                result = matrix.read_result(output, candidate["checkpoint"])
                raw.append({
                    "seed": seed,
                    "category": category,
                    "epoch": candidate["epoch"],
                    "success_count": int(result["total_success_count"]),
                    "trajectory_count": int(result["total_trajectory_count"]),
                    "macro_success_rate": float(
                        result["macro_official_peak_success_rate"]),
                    "mean_maximum_lift_m": float(
                        result["macro_mean_maximum_lift_m"]),
                    "failure_rate": float(result["macro_failure_rate"]),
                    "result": str(output),
                })
    raw_path = output_root / "all_completed_evaluations.csv"
    fields = list(raw[0]) if raw else ["seed", "category", "epoch"]
    with raw_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(raw)

    groups = {}
    for row in raw:
        groups.setdefault((row["category"], row["epoch"]), []).append(row)
    category_rows = []
    for (category, epoch), rows in groups.items():
        rates = [row["macro_success_rate"] for row in rows]
        category_rows.append({
            "category": category,
            "epoch": epoch,
            "completed_seed_count": len(rows),
            "total_success_count": sum(row["success_count"] for row in rows),
            "total_trajectory_count": sum(
                row["trajectory_count"] for row in rows),
            "mean_macro_success_rate": statistics.mean(rates),
            "seed_std_macro_success_rate": (
                statistics.pstdev(rates) if len(rates) > 1 else 0.0),
            "mean_maximum_lift_m": statistics.mean([
                row["mean_maximum_lift_m"] for row in rows]),
            "mean_failure_rate": statistics.mean([
                row["failure_rate"] for row in rows]),
        })
    category_rows.sort(key=lambda row: (
        row["category"], -row["mean_macro_success_rate"],
        -row["mean_maximum_lift_m"], row["mean_failure_rate"], row["epoch"]))
    with (output_root / "category_epoch_summary.csv").open(
            "w", encoding="utf-8", newline="") as handle:
        fields = list(category_rows[0]) if category_rows else ["category", "epoch"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(category_rows)

    global_rows = []
    for epoch in EPOCHS:
        per_seed = []
        for seed in seeds:
            rows = [row for row in raw
                    if row["epoch"] == epoch and row["seed"] == seed]
            if len(rows) != len(CATEGORIES):
                continue
            per_seed.append({
                "macro": statistics.mean([
                    row["macro_success_rate"] for row in rows]),
                "lift": statistics.mean([
                    row["mean_maximum_lift_m"] for row in rows]),
                "failure": statistics.mean([
                    row["failure_rate"] for row in rows]),
                "success": sum(row["success_count"] for row in rows),
                "trajectories": sum(row["trajectory_count"] for row in rows),
            })
        if per_seed:
            rates = [row["macro"] for row in per_seed]
            global_rows.append({
                "epoch": epoch,
                "completed_seed_count": len(per_seed),
                "total_success_count": sum(row["success"] for row in per_seed),
                "total_trajectory_count": sum(
                    row["trajectories"] for row in per_seed),
                "mean_macro_success_rate": statistics.mean(rates),
                "seed_std_macro_success_rate": (
                    statistics.pstdev(rates) if len(rates) > 1 else 0.0),
                "mean_maximum_lift_m": statistics.mean([
                    row["lift"] for row in per_seed]),
                "mean_failure_rate": statistics.mean([
                    row["failure"] for row in per_seed]),
            })
    global_rows.sort(key=lambda row: (
        -row["mean_macro_success_rate"], -row["mean_maximum_lift_m"],
        row["mean_failure_rate"], row["epoch"]))
    with (output_root / "global_epoch_summary.csv").open(
            "w", encoding="utf-8", newline="") as handle:
        fields = list(global_rows[0]) if global_rows else ["epoch"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(global_rows)

    with OLD_SUMMARY.open("r", encoding="utf-8") as handle:
        previous = yaml.safe_load(handle)["best_available_per_category"]
    best = {}
    for category in CATEGORIES:
        finalist_epochs = {
            candidate["epoch"]
            for candidate in finalists_by_category[category]
        }
        options = [
            row for row in category_rows
            if row["category"] == category
            and row["epoch"] in finalist_epochs
            and row["completed_seed_count"] == len(seeds)
        ]
        selected = options[0] if options else None
        old = previous[category]
        best[category] = {
            "object_balanced": selected,
            "previous_best": old,
            "macro_success_rate_change": (
                selected["mean_macro_success_rate"]
                - float(old["mean_macro_success_rate"])) if selected else None,
        }
    summary = {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "status": "complete" if len(raw) == expected else "partial",
        "completed_evaluations": len(raw),
        "expected_evaluations": expected,
        "training_seed": 2025,
        "simulation_seeds": list(seeds),
        "evaluation_mode": evaluation_mode,
        "finalist_epochs_by_category": {
            category: [candidate["epoch"]
                       for candidate in finalists_by_category[category]]
            for category in CATEGORIES
        },
        "object_balanced_sampling": True,
        "only_intended_change": "equal expected sampling probability per object",
        "formal_final_holdout_result": False,
        "final_holdout_accessed": False,
        "global_ranking": global_rows,
        "best_comparison_per_category": best,
    }
    with (output_root / "summary.yaml").open(
            "w", encoding="utf-8") as handle:
        yaml.safe_dump(summary, handle, allow_unicode=True, sort_keys=False)
    return summary


def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=(2025, 2026, 2027))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--min-free-vram-mb", type=int, default=4500)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--max-consecutive-failures", type=int, default=3)
    parser.add_argument(
        "--full-matrix", action="store_true",
        help="Evaluate every epoch with every simulation seed instead of the fast funnel.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    cli = parse_cli()
    seeds = tuple(dict.fromkeys(cli.seeds))
    for path in (TRAIN_CONFIG, MANIFEST, DATASET_SUMMARY, OLD_SUMMARY):
        if not path.is_file():
            raise FileNotFoundError(path)
    counts = trajectory_counts()
    directories = train_all(counts, cli.min_free_vram_mb, cli.dry_run)
    screening_count = len(CATEGORIES) * len(EPOCHS)
    finalist_count = 2
    expected_evaluations = (
        len(seeds) * screening_count if cli.full_matrix
        else screening_count
        + max(0, len(seeds) - 1) * len(CATEGORIES) * finalist_count
    )
    if cli.dry_run:
        print("DRY_RUN_COMPLETE mode={} training_runs=4 "
              "aggregate_evaluations={} fresh_object_processes={}".format(
                  "full" if cli.full_matrix else "fast_funnel",
                  expected_evaluations, expected_evaluations * 3), flush=True)
        return

    output_root = Path(cli.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    protocol = load_json(matrix.PROTOCOL)
    selections = matrix.write_selections(protocol, output_root)
    candidates_by_category = {
        category: [{
            "label": "epoch{:02d}".format(epoch),
            "epoch": epoch,
            "checkpoint": matrix.checkpoint_for_epoch(
                directories[category], epoch),
        } for epoch in EPOCHS]
        for category in CATEGORIES
    }
    plan = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "training_runs": 4,
        "training_seed": 2025,
        "simulation_seeds": list(seeds),
        "epochs": list(EPOCHS),
        "aggregate_evaluations": expected_evaluations,
        "fresh_object_processes": expected_evaluations * 3,
        "evaluation_mode": "full_matrix" if cli.full_matrix else "fast_funnel",
        "screening_seed": seeds[0],
        "finalists_per_category": (
            len(EPOCHS) if cli.full_matrix else finalist_count),
        "final_holdout_accessed": False,
    }
    with (output_root / "plan.json").open("w", encoding="utf-8") as handle:
        json.dump(plan, handle, indent=2)
        handle.write("\n")

    matrix.require_free_vram_with_nvidia_smi(cli.min_free_vram_mb)
    failures = []
    consecutive_failures = 0
    completed = 0
    abort = False

    def evaluate_jobs(seed, category_candidates):
        nonlocal completed, consecutive_failures, abort
        for category in CATEGORIES:
            for candidate in category_candidates[category]:
                completed += 1
                print("EVAL_PROGRESS {}/{}".format(
                    completed, expected_evaluations), flush=True)
                output = (
                    output_root / "seed{}".format(seed) / category
                    / (candidate["label"] + ".yaml")
                )
                output.parent.mkdir(parents=True, exist_ok=True)
                ok = matrix.evaluate_candidate(
                    candidate, category, seed, selections[category], output,
                    cli.min_free_vram_mb, cli.max_attempts, False)
                if ok:
                    consecutive_failures = 0
                    continue
                failures.append({
                    "seed": seed, "category": category,
                    "epoch": candidate["epoch"],
                })
                consecutive_failures += 1
                if consecutive_failures >= cli.max_consecutive_failures:
                    abort = True
                    print("[ABORT] consecutive evaluation failures", flush=True)
                    break
            if abort:
                break

    screening_seed = seeds[0]
    evaluate_jobs(screening_seed, candidates_by_category)
    finalists_by_category = {category: [] for category in CATEGORIES}
    if not abort:
        for category in CATEGORIES:
            ranked = []
            for candidate in candidates_by_category[category]:
                output = (
                    output_root / "seed{}".format(screening_seed) / category
                    / (candidate["label"] + ".yaml")
                )
                result = matrix.read_result(output, candidate["checkpoint"])
                ranked.append((
                    -float(result["macro_official_peak_success_rate"]),
                    -float(result["macro_mean_maximum_lift_m"]),
                    float(result["macro_failure_rate"]),
                    candidate["epoch"], candidate,
                ))
            ranked.sort(key=lambda item: item[:4])
            finalists_by_category[category] = (
                list(candidates_by_category[category]) if cli.full_matrix
                else [item[-1] for item in ranked[:finalist_count]]
            )
            print("FINALISTS {}: {}".format(
                category,
                [candidate["epoch"]
                 for candidate in finalists_by_category[category]]), flush=True)
        with (output_root / "screening_finalists.json").open(
                "w", encoding="utf-8") as handle:
            json.dump({
                category: [candidate["epoch"]
                           for candidate in finalists_by_category[category]]
                for category in CATEGORIES
            }, handle, indent=2)
            handle.write("\n")
        for seed in seeds[1:]:
            evaluate_jobs(seed, finalists_by_category)
            if abort:
                break
    summary = summarize(
        output_root, seeds, candidates_by_category, expected_evaluations,
        finalists_by_category,
        "full_matrix" if cli.full_matrix else "fast_funnel")
    with (output_root / "failures.json").open("w", encoding="utf-8") as handle:
        json.dump(failures, handle, indent=2)
        handle.write("\n")
    print("OBJECT_BALANCED_STAGE_STATUS={} failures={}".format(
        summary["status"], len(failures)), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
