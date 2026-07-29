"""Repeat the three competitive Task-ID students on the frozen dev split."""

import argparse
from pathlib import Path
import statistics
import subprocess
import sys

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
BC_CONFIG = (
    REPO_ROOT
    / "custom_tools/configs/unified_student_taskid_scaled20_v1.yaml"
)
RESIDUAL_CONFIG = (
    REPO_ROOT / "custom_tools/configs/residual_ppo_stage1.yaml"
)
TRAJECTORY_ROOT = (
    REPO_ROOT / "dexgrasp/dataset/scaled_category_final_v1_preprocessed"
)
SELECTION = (
    REPO_ROOT / "custom_tools/configs/scaled_development_all12.yaml"
)
OUTPUT_ROOT = (
    REPO_ROOT / "custom_tools/results/taskid_offline_development_v1"
)
CANDIDATES = {
    "t100_epoch15": (
        REPO_ROOT / "custom_tools/runs/bc/"
        / "unified_student_taskid_scaled20_t100_seed2025_e20_v1/"
        / "epoch=014-step=14145.ckpt"
    ),
    "t70_demo30_epoch15": (
        REPO_ROOT / "custom_tools/runs/bc/"
        / "unified_student_taskid_scaled20_t70_demo30_seed2025_e20_v1/"
        / "epoch=014-step=14145.ckpt"
    ),
    "t70_demo30_epoch20": (
        REPO_ROOT / "custom_tools/runs/bc/"
        / "unified_student_taskid_scaled20_t70_demo30_seed2025_e20_v1/"
        / "epoch=019-step=18860.ckpt"
    ),
}
SEEDS = (2025, 2026, 2027)


def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-free-vram-mb", type=int, default=4500)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_yaml(path):
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def read_result(path, checkpoint):
    aggregate = load_yaml(path)
    if aggregate.get("final_holdout_accessed", False):
        raise RuntimeError("A development result claims holdout access")
    rows = aggregate.get("checkpoint_results", [])
    if len(rows) != 1:
        raise RuntimeError("Expected one result in {}".format(path))
    if Path(rows[0]["checkpoint"]).resolve() != checkpoint.resolve():
        raise RuntimeError("Checkpoint mismatch in {}".format(path))
    return rows[0]


def run_one(cli, label, checkpoint, seed):
    output = OUTPUT_ROOT / "seed{}".format(seed) / (label + ".yaml")
    if output.is_file():
        read_result(output, checkpoint)
        print("[REUSE] {} seed={}".format(label, seed), flush=True)
        return
    command = [
        sys.executable,
        "-u",
        str(REPO_ROOT / "custom_tools/evaluate_bc_checkpoints_isolated.py"),
        "--checkpoint",
        str(checkpoint),
        "--bc-config",
        str(BC_CONFIG),
        "--residual-config",
        str(RESIDUAL_CONFIG),
        "--trajectory-root",
        str(TRAJECTORY_ROOT),
        "--object-selection",
        str(SELECTION),
        "--output",
        str(output),
        "--seed",
        str(seed),
        "--min-free-vram-mb",
        str(cli.min_free_vram_mb),
        "--max-attempts",
        str(cli.max_attempts),
    ]
    print("[RUN] {} seed={}".format(label, seed), flush=True)
    if cli.dry_run:
        print(" ".join(command), flush=True)
        return
    subprocess.run(command, cwd=str(REPO_ROOT), check=True)
    read_result(output, checkpoint)


def summarize():
    rows = []
    for label, checkpoint in CANDIDATES.items():
        seed_results = [
            read_result(
                OUTPUT_ROOT / "seed{}".format(seed) / (label + ".yaml"),
                checkpoint)
            for seed in SEEDS
        ]
        macro = [
            float(item["macro_official_peak_success_rate"])
            for item in seed_results]
        lift = [
            float(item["macro_mean_maximum_lift_m"])
            for item in seed_results]
        failure = [
            float(item["macro_failure_rate"])
            for item in seed_results]
        categories = {}
        for category in ("bottle", "mug", "bowl", "camera"):
            values = [
                float(item["category_macro_success_rates"][category])
                for item in seed_results]
            categories[category] = {
                "mean": statistics.mean(values),
                "std": statistics.pstdev(values),
                "values": values,
            }
        rows.append({
            "label": label,
            "checkpoint": str(checkpoint),
            "seeds": list(SEEDS),
            "success_counts": [
                int(item["total_success_count"]) for item in seed_results],
            "trajectory_count_per_seed": int(
                seed_results[0]["total_trajectory_count"]),
            "macro_success_mean": statistics.mean(macro),
            "macro_success_std": statistics.pstdev(macro),
            "macro_success_values": macro,
            "lift_mean_m": statistics.mean(lift),
            "failure_mean": statistics.mean(failure),
            "category_success": categories,
        })
    rows.sort(
        key=lambda row: (
            row["macro_success_mean"],
            row["lift_mean_m"],
            -row["failure_mean"]),
        reverse=True)
    summary = {
        "status": "complete",
        "formal_final_holdout_result": False,
        "final_holdout_accessed": False,
        "selection": str(SELECTION),
        "seeds": list(SEEDS),
        "ranking": rows,
    }
    output = OUTPUT_ROOT / "repeat_summary.yaml"
    with output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(summary, handle, allow_unicode=True, sort_keys=False)
    print("TASKID_OFFLINE_REPEATS=COMPLETE")
    for rank, row in enumerate(rows, 1):
        print(
            "#{:02d} {} success={} macro={:.2f}+/-{:.2f}% "
            "lift={:.3f}m failure={:.2f}%".format(
                rank, row["label"], row["success_counts"],
                100 * row["macro_success_mean"],
                100 * row["macro_success_std"],
                row["lift_mean_m"], 100 * row["failure_mean"]))


def main():
    cli = parse_cli()
    missing = [
        str(path) for path in (
            BC_CONFIG, RESIDUAL_CONFIG, TRAJECTORY_ROOT, SELECTION,
            *CANDIDATES.values())
        if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing repeat inputs: {}".format(missing))
    for seed in SEEDS:
        for label, checkpoint in CANDIDATES.items():
            run_one(cli, label, checkpoint, seed)
    if not cli.dry_run:
        summarize()


if __name__ == "__main__":
    main()
