"""Repeat the selected temporal3 policy and compare it with online R1."""

import argparse
from pathlib import Path
import statistics
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT / "custom_tools/configs/"
    / "unified_student_taskid_temporal3_v1.yaml"
)
CHECKPOINT = (
    ROOT / "custom_tools/runs/bc/"
    / "unified_student_taskid_temporal3_seed2025_e4_v1/"
    / "epoch=003-step=5152.ckpt"
)
SELECTION = ROOT / "custom_tools/configs/scaled_development_all12.yaml"
TRAJECTORY_ROOT = (
    ROOT / "dexgrasp/dataset/scaled_category_final_v1_preprocessed"
)
RESIDUAL_CONFIG = ROOT / "custom_tools/configs/residual_ppo_stage1.yaml"
SOURCE_RESULT = (
    ROOT / "custom_tools/results/taskid_temporal3_development_v1/"
    / "temporal3_epoch04.yaml"
)
OUTPUT_ROOT = ROOT / "custom_tools/results/taskid_temporal3_repeats_v1"
R1_ROOT = ROOT / "custom_tools/results/taskid_online_r1_repeats_v1"
R1_SEED2025 = (
    ROOT / "custom_tools/results/taskid_online_r1_development_v1/"
    / "online_25pct_epoch02.yaml"
)
SEEDS = (2025, 2026, 2027)
CATEGORIES = ("bottle", "mug", "bowl", "camera")


def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-free-vram-mb", type=int, default=4500)
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_yaml(path):
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def read_result(path, checkpoint):
    aggregate = load_yaml(path)
    rows = aggregate.get("checkpoint_results", [])
    if len(rows) != 1:
        raise RuntimeError("Expected one checkpoint result in {}".format(path))
    if Path(rows[0]["checkpoint"]).resolve() != checkpoint.resolve():
        raise RuntimeError("Checkpoint mismatch in {}".format(path))
    return rows[0]


def temporal_output(seed):
    if seed == 2025:
        return SOURCE_RESULT
    return OUTPUT_ROOT / "seed{}".format(seed) / "temporal3_epoch04.yaml"


def evaluate(cli, seed):
    output = temporal_output(seed)
    if output.is_file():
        read_result(output, CHECKPOINT)
        print("[REUSE] temporal3 seed={}".format(seed), flush=True)
        return
    command = [
        sys.executable,
        "-u",
        str(ROOT / "custom_tools/evaluate_bc_checkpoints_isolated.py"),
        "--checkpoint",
        str(CHECKPOINT),
        "--bc-config",
        str(CONFIG),
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
    print("[RUN] temporal3 seed={}".format(seed), flush=True)
    if cli.dry_run:
        print(" ".join(command), flush=True)
        return
    subprocess.run(command, cwd=str(ROOT), check=True)
    read_result(output, CHECKPOINT)


def r1_output(seed):
    if seed == 2025:
        return R1_SEED2025
    return (
        R1_ROOT / "seed{}".format(seed) / "online_25pct_epoch02.yaml"
    )


def aggregate(label, checkpoint, paths):
    results = [read_result(path, checkpoint) for path in paths]
    macros = [
        float(item["macro_official_peak_success_rate"]) for item in results
    ]
    categories = {}
    for category in CATEGORIES:
        values = [
            float(item["category_macro_success_rates"][category])
            for item in results
        ]
        categories[category] = {
            "mean": statistics.mean(values),
            "std": statistics.pstdev(values),
            "values": values,
        }
    return {
        "label": label,
        "checkpoint": str(checkpoint),
        "success_counts": [
            int(item["total_success_count"]) for item in results
        ],
        "trajectory_count_per_seed": int(results[0]["total_trajectory_count"]),
        "macro_success_mean": statistics.mean(macros),
        "macro_success_std": statistics.pstdev(macros),
        "macro_success_values": macros,
        "lift_mean_m": statistics.mean(
            float(item["macro_mean_maximum_lift_m"]) for item in results
        ),
        "failure_mean": statistics.mean(
            float(item["macro_failure_rate"]) for item in results
        ),
        "category_success": categories,
        "outputs": [str(path) for path in paths],
    }


def summarize():
    r1_checkpoint = (
        ROOT / "custom_tools/runs/bc/"
        / "unified_student_taskid_online_r1_frac025_seed2025_e10_v1/"
        / "epoch=001-step=2232.ckpt"
    )
    r1 = aggregate(
        "online_r1",
        r1_checkpoint,
        [r1_output(seed) for seed in SEEDS],
    )
    temporal = aggregate(
        "temporal3",
        CHECKPOINT,
        [temporal_output(seed) for seed in SEEDS],
    )
    category_deltas = {
        category: (
            temporal["category_success"][category]["mean"]
            - r1["category_success"][category]["mean"]
        )
        for category in CATEGORIES
    }
    summary = {
        "status": "complete",
        "stage": "three-seed temporal3 confirmation",
        "formal_final_holdout_result": False,
        "final_holdout_accessed": False,
        "seeds": list(SEEDS),
        "nodes": [r1, temporal],
        "temporal_minus_r1_macro": (
            temporal["macro_success_mean"] - r1["macro_success_mean"]
        ),
        "temporal_minus_r1_category": category_deltas,
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_ROOT / "summary.yaml").open(
        "w", encoding="utf-8"
    ) as handle:
        yaml.safe_dump(summary, handle, allow_unicode=True, sort_keys=False)
    print("TASKID_TEMPORAL3_REPEATS=COMPLETE", flush=True)
    for node in (r1, temporal):
        print(
            "{} success={} macro={:.2f}+/-{:.2f}% lift={:.3f}m "
            "failure={:.2f}%".format(
                node["label"],
                node["success_counts"],
                100 * node["macro_success_mean"],
                100 * node["macro_success_std"],
                node["lift_mean_m"],
                100 * node["failure_mean"],
            ),
            flush=True,
        )


def main():
    cli = parse_cli()
    if cli.max_attempts < 1:
        raise ValueError("--max-attempts must be positive")
    required = [
        CONFIG, CHECKPOINT, SELECTION, TRAJECTORY_ROOT, RESIDUAL_CONFIG,
        SOURCE_RESULT, R1_SEED2025,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing temporal repeat inputs: {}".format(
            missing))
    for seed in SEEDS:
        evaluate(cli, seed)
    if not cli.dry_run:
        summarize()


if __name__ == "__main__":
    main()
