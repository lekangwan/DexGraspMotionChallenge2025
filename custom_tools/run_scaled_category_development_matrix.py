"""Run the frozen development-set matrix for scaled category experts.

Every checkpoint/object pair is evaluated in a fresh Isaac Gym process through
``evaluate_bc_checkpoints_isolated.py``.  The final holdout objects are checked
for disjointness but are never passed to an evaluator.
"""

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import statistics
import subprocess
import sys

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = ("bottle", "mug", "bowl", "camera")
DEFAULT_EPOCHS = (10, 20, 30, 40)
PROTOCOL = REPO_ROOT / "custom_tools/configs/scaled_evaluation_protocol_v1.json"
TRAJECTORY_ROOT = REPO_ROOT / "dexgrasp/dataset/scaled_category_final_v1_preprocessed"
BC_CONFIG = REPO_ROOT / "custom_tools/configs/category_expert_bc_scaled20_v1.yaml"
RESIDUAL_CONFIG = REPO_ROOT / "custom_tools/configs/residual_ppo_stage1.yaml"
SOUP = (
    REPO_ROOT
    / "custom_tools/runs/bc/model_soups/"
    / "noise005_s2025_s2026_weighted2to1.ckpt"
)


def require_free_vram_with_nvidia_smi(min_free_vram_mb):
    """Fail once up front instead of launching hundreds of doomed workers."""
    command = [
        "nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"
    ]
    completed = subprocess.run(
        command, check=False, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)
    if completed.returncode != 0:
        raise RuntimeError("nvidia-smi VRAM check failed: {}".format(
            completed.stderr.strip()))
    values = [int(line.strip()) for line in completed.stdout.splitlines()
              if line.strip()]
    if not values:
        raise RuntimeError("nvidia-smi returned no GPU memory values")
    free_mb = values[0]
    print("GPU memory before matrix: {} MiB free".format(free_mb), flush=True)
    if free_mb < min_free_vram_mb:
        raise RuntimeError(
            "Only {} MiB VRAM is free; need at least {} MiB."
            .format(free_mb, min_free_vram_mb))


def load_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def checkpoint_for_epoch(run_dir, epoch):
    matches = sorted(run_dir.glob("epoch={:03d}-step=*.ckpt".format(epoch - 1)))
    if len(matches) != 1:
        raise RuntimeError(
            "Expected one epoch {} checkpoint in {}, got {}".format(
                epoch, run_dir, len(matches)
            )
        )
    return matches[0].resolve()


def run_directory(category, scale):
    if scale == 4:
        name = "category_expert_{}_noise005_soup_seed2025_e40_v1".format(category)
    else:
        name = (
            "category_expert_{}_scaled{}_noise005_soup_seed2025_e40_v1"
            .format(category, scale)
        )
    return REPO_ROOT / "custom_tools/runs/bc" / name


def validate_run(category, scale, run_dir):
    config_path = run_dir / "resolved_config.yaml"
    metadata_path = run_dir / "run_metadata.yaml"
    last_path = run_dir / "last.ckpt"
    for path in (config_path, metadata_path, last_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = yaml.safe_load(handle)
    if config.get("expert_category") != category:
        raise RuntimeError("Wrong expert category in {}".format(config_path))
    if len(config.get("train_obj_code_list", [])) != scale:
        raise RuntimeError("Wrong object count in {}".format(config_path))
    if int(config.get("seed")) != 2025:
        raise RuntimeError("Unexpected training seed in {}".format(config_path))
    if Path(metadata.get("init_checkpoint", "")).resolve() != SOUP.resolve():
        raise RuntimeError("Unexpected initialization in {}".format(metadata_path))


def build_candidates(category, epochs):
    candidates = [{
        "label": "soup_init",
        "scale": 0,
        "epoch": 0,
        "checkpoint": SOUP.resolve(),
    }]
    for scale in (4, 10, 20):
        run_dir = run_directory(category, scale)
        validate_run(category, scale, run_dir)
        for epoch in epochs:
            candidates.append({
                "label": "scale{}_epoch{:02d}".format(scale, epoch),
                "scale": scale,
                "epoch": epoch,
                "checkpoint": checkpoint_for_epoch(run_dir, epoch),
            })
    return candidates


def write_selections(protocol, output_root):
    selection_dir = output_root / "selections"
    selection_dir.mkdir(parents=True, exist_ok=True)
    all_development = set()
    all_holdout = set()
    paths = {}
    for category in CATEGORIES:
        category_protocol = protocol["categories"][category]
        development = list(category_protocol["development"])
        holdout = list(category_protocol["final_holdout"])
        if len(development) != 3 or len(holdout) != 2:
            raise RuntimeError("Frozen protocol count changed for {}".format(category))
        if set(development) & set(holdout):
            raise RuntimeError("Development/holdout overlap for {}".format(category))
        all_development.update(development)
        all_holdout.update(holdout)
        for object_id in development:
            trajectory = TRAJECTORY_ROOT / (object_id + ".npy")
            if not trajectory.is_file():
                raise FileNotFoundError(trajectory)
        selection = {
            "status": "frozen_scaled_development_only",
            "category": category,
            "object_ids": development,
            "final_holdout_accessed": False,
            "source_protocol": str(PROTOCOL),
        }
        path = selection_dir / (category + "_development.yaml")
        with path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(selection, handle, sort_keys=False)
        paths[category] = path
    if all_development & all_holdout:
        raise RuntimeError("Global development/final-holdout overlap")
    if len(all_development) != 12 or len(all_holdout) != 8:
        raise RuntimeError("Frozen protocol global counts changed")
    return paths


def read_result(path, expected_checkpoint):
    with path.open("r", encoding="utf-8") as handle:
        aggregate = yaml.safe_load(handle)
    rows = aggregate.get("checkpoint_results", [])
    if len(rows) != 1:
        raise RuntimeError("Expected one checkpoint result in {}".format(path))
    result = rows[0]
    if Path(result["checkpoint"]).resolve() != Path(expected_checkpoint).resolve():
        raise RuntimeError("Checkpoint mismatch in {}".format(path))
    return result


def evaluate_candidate(candidate, category, seed, selection, output,
                       min_free_vram_mb, max_attempts, dry_run):
    if output.exists():
        read_result(output, candidate["checkpoint"])
        print("[REUSE] {} seed={} {}".format(category, seed, candidate["label"]),
              flush=True)
        return True
    command = [
        sys.executable,
        "-u",
        str(REPO_ROOT / "custom_tools/evaluate_bc_checkpoints_isolated.py"),
        "--checkpoint", str(candidate["checkpoint"]),
        "--bc-config", str(BC_CONFIG),
        "--residual-config", str(RESIDUAL_CONFIG),
        "--trajectory-root", str(TRAJECTORY_ROOT),
        "--object-selection", str(selection),
        "--output", str(output),
        "--seed", str(seed),
        "--min-free-vram-mb", str(min_free_vram_mb),
        "--max-attempts", str(max_attempts),
    ]
    print("[RUN] {} seed={} {}".format(category, seed, candidate["label"]),
          flush=True)
    if dry_run:
        print("  " + " ".join(command), flush=True)
        return True
    completed = subprocess.run(command, cwd=str(REPO_ROOT), check=False)
    if completed.returncode != 0:
        print("[FAILED] {} seed={} {} exit={}".format(
            category, seed, candidate["label"], completed.returncode), flush=True)
        return False
    read_result(output, candidate["checkpoint"])
    return True


def mean(values):
    return sum(values) / len(values)


def summarize(output_root, seeds, candidates_by_category):
    raw_rows = []
    for seed in seeds:
        for category in CATEGORIES:
            for candidate in candidates_by_category[category]:
                output = (
                    output_root / "seed{}".format(seed) / category
                    / (candidate["label"] + ".yaml")
                )
                if not output.is_file():
                    continue
                result = read_result(output, candidate["checkpoint"])
                raw_rows.append({
                    "seed": seed,
                    "category": category,
                    "label": candidate["label"],
                    "scale": candidate["scale"],
                    "epoch": candidate["epoch"],
                    "success_count": int(result["total_success_count"]),
                    "trajectory_count": int(result["total_trajectory_count"]),
                    "macro_success_rate": float(
                        result["macro_official_peak_success_rate"]),
                    "macro_mean_maximum_lift_m": float(
                        result["macro_mean_maximum_lift_m"]),
                    "macro_failure_rate": float(result["macro_failure_rate"]),
                    "result": str(output),
                })

    raw_path = output_root / "all_completed_evaluations.csv"
    fields = list(raw_rows[0]) if raw_rows else [
        "seed", "category", "label", "scale", "epoch", "success_count",
        "trajectory_count", "macro_success_rate", "macro_mean_maximum_lift_m",
        "macro_failure_rate", "result"]
    with raw_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(raw_rows)

    grouped = {}
    for row in raw_rows:
        grouped.setdefault((row["category"], row["label"]), []).append(row)
    category_rows = []
    for (category, label), rows in grouped.items():
        rates = [row["macro_success_rate"] for row in rows]
        category_rows.append({
            "category": category,
            "label": label,
            "scale": rows[0]["scale"],
            "epoch": rows[0]["epoch"],
            "completed_seed_count": len(rows),
            "total_success_count": sum(row["success_count"] for row in rows),
            "total_trajectory_count": sum(row["trajectory_count"] for row in rows),
            "mean_macro_success_rate": mean(rates),
            "seed_std_macro_success_rate": (
                statistics.pstdev(rates) if len(rates) > 1 else 0.0),
            "mean_maximum_lift_m": mean([
                row["macro_mean_maximum_lift_m"] for row in rows]),
            "mean_failure_rate": mean([
                row["macro_failure_rate"] for row in rows]),
        })
    category_rows.sort(key=lambda row: (
        row["category"], -row["mean_macro_success_rate"],
        -row["mean_maximum_lift_m"], row["mean_failure_rate"], row["epoch"]))
    category_path = output_root / "category_candidate_summary.csv"
    with category_path.open("w", encoding="utf-8", newline="") as handle:
        fields = list(category_rows[0]) if category_rows else ["category", "label"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(category_rows)

    global_groups = {}
    for row in raw_rows:
        global_groups.setdefault((row["label"], row["seed"]), []).append(row)
    global_by_label = {}
    for (label, seed), rows in global_groups.items():
        if len(rows) != len(CATEGORIES):
            continue
        global_by_label.setdefault(label, []).append({
            "seed": seed,
            "scale": rows[0]["scale"],
            "epoch": rows[0]["epoch"],
            "success_count": sum(row["success_count"] for row in rows),
            "trajectory_count": sum(row["trajectory_count"] for row in rows),
            "macro_success_rate": mean([
                row["macro_success_rate"] for row in rows]),
            "mean_lift": mean([
                row["macro_mean_maximum_lift_m"] for row in rows]),
            "failure_rate": mean([row["macro_failure_rate"] for row in rows]),
        })
    global_rows = []
    for label, rows in global_by_label.items():
        rates = [row["macro_success_rate"] for row in rows]
        global_rows.append({
            "label": label,
            "scale": rows[0]["scale"],
            "epoch": rows[0]["epoch"],
            "completed_seed_count": len(rows),
            "total_success_count": sum(row["success_count"] for row in rows),
            "total_trajectory_count": sum(row["trajectory_count"] for row in rows),
            "mean_macro_success_rate": mean(rates),
            "seed_std_macro_success_rate": (
                statistics.pstdev(rates) if len(rates) > 1 else 0.0),
            "mean_maximum_lift_m": mean([row["mean_lift"] for row in rows]),
            "mean_failure_rate": mean([row["failure_rate"] for row in rows]),
        })
    global_rows.sort(key=lambda row: (
        -row["mean_macro_success_rate"], -row["mean_maximum_lift_m"],
        row["mean_failure_rate"], row["epoch"]))
    global_path = output_root / "global_candidate_summary.csv"
    with global_path.open("w", encoding="utf-8", newline="") as handle:
        fields = list(global_rows[0]) if global_rows else ["label"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(global_rows)

    summary = {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "status": (
            "complete" if len(raw_rows) == len(seeds) * len(CATEGORIES) * 13
            else "partial"),
        "formal_final_holdout_result": False,
        "final_holdout_accessed": False,
        "completed_evaluations": len(raw_rows),
        "expected_evaluations": len(seeds) * len(CATEGORIES) * 13,
        "seeds": list(seeds),
        "selection_metric": (
            "category/object macro official_peak success; lift and failure are "
            "tie-break diagnostics only"),
        "global_ranking": global_rows,
        "best_available_per_category": {
            category: next(
                (row for row in category_rows if row["category"] == category), None)
            for category in CATEGORIES
        },
    }
    with (output_root / "summary.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(summary, handle, allow_unicode=True, sort_keys=False)
    return summary


def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=(2025, 2026, 2027))
    parser.add_argument("--epochs", type=int, nargs="+", default=DEFAULT_EPOCHS)
    parser.add_argument(
        "--output-root",
        default=str(REPO_ROOT / "custom_tools/results/scaled_category_development_v1"),
    )
    parser.add_argument("--min-free-vram-mb", type=int, default=4500)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--max-consecutive-failures", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    cli = parse_cli()
    seeds = tuple(dict.fromkeys(cli.seeds))
    epochs = tuple(dict.fromkeys(cli.epochs))
    if epochs != DEFAULT_EPOCHS:
        raise ValueError("Frozen matrix epochs must be {}".format(DEFAULT_EPOCHS))
    for path in (PROTOCOL, TRAJECTORY_ROOT, BC_CONFIG, RESIDUAL_CONFIG, SOUP):
        if not path.exists():
            raise FileNotFoundError(path)
    protocol = load_json(PROTOCOL)
    if protocol.get("status") != "frozen_before_scaled_training":
        raise RuntimeError("Evaluation protocol is not the pre-training frozen version")
    output_root = Path(cli.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    selections = write_selections(protocol, output_root)
    candidates_by_category = {
        category: build_candidates(category, epochs) for category in CATEGORIES
    }
    if not cli.dry_run:
        require_free_vram_with_nvidia_smi(cli.min_free_vram_mb)

    plan = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "seeds": list(seeds),
        "epochs": list(epochs),
        "categories": list(CATEGORIES),
        "candidate_count_per_category": 13,
        "aggregate_evaluation_count": len(seeds) * len(CATEGORIES) * 13,
        "fresh_object_process_count": len(seeds) * len(CATEGORIES) * 13 * 3,
        "development_object_count": 12,
        "final_holdout_object_count": 8,
        "final_holdout_accessed": False,
    }
    with (output_root / "plan.json").open("w", encoding="utf-8") as handle:
        json.dump(plan, handle, indent=2)
        handle.write("\n")
    print("MATRIX_PLAN {}".format(plan), flush=True)

    failures = []
    consecutive_failures = 0
    abort_remaining = False
    completed = 0
    total = plan["aggregate_evaluation_count"]
    for seed in seeds:
        for category in CATEGORIES:
            for candidate in candidates_by_category[category]:
                completed += 1
                print("MATRIX_PROGRESS {}/{}".format(completed, total), flush=True)
                output = (
                    output_root / "seed{}".format(seed) / category
                    / (candidate["label"] + ".yaml")
                )
                output.parent.mkdir(parents=True, exist_ok=True)
                ok = evaluate_candidate(
                    candidate, category, seed, selections[category], output,
                    cli.min_free_vram_mb, cli.max_attempts, cli.dry_run,
                )
                if not ok:
                    failures.append({
                        "seed": seed,
                        "category": category,
                        "label": candidate["label"],
                    })
                    consecutive_failures += 1
                    if consecutive_failures >= cli.max_consecutive_failures:
                        print(
                            "[ABORT] {} consecutive evaluations failed; keep all "
                            "completed object files and rerun after checking GPU/PhysX."
                            .format(consecutive_failures), flush=True)
                        abort_remaining = True
                        break
                else:
                    consecutive_failures = 0
            if abort_remaining:
                break
        if abort_remaining:
            break

    if cli.dry_run:
        print("DRY_RUN_COMPLETE", flush=True)
        return
    summary = summarize(output_root, seeds, candidates_by_category)
    with (output_root / "failures.json").open("w", encoding="utf-8") as handle:
        json.dump(failures, handle, indent=2)
        handle.write("\n")
    print("MATRIX_STATUS={} failures={}".format(
        summary["status"], len(failures)), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
