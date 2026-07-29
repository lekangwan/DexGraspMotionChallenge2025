"""Repeat the selected Task-ID online student and compare serial mainline nodes."""

import argparse
from pathlib import Path
import statistics
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
ONLINE_CONFIG = (
    ROOT / "custom_tools/configs/"
    / "unified_student_taskid_online_r1_scaled20_v1.yaml"
)
RESIDUAL_CONFIG = ROOT / "custom_tools/configs/residual_ppo_stage1.yaml"
TRAJECTORY_ROOT = (
    ROOT / "dexgrasp/dataset/scaled_category_final_v1_preprocessed"
)
SELECTION = ROOT / "custom_tools/configs/scaled_development_all12.yaml"
ONLINE_CHECKPOINT = (
    ROOT / "custom_tools/runs/bc/"
    / "unified_student_taskid_online_r1_frac025_seed2025_e10_v1/"
    / "epoch=001-step=2232.ckpt"
)
ONLINE_STAGE_ROOT = (
    ROOT / "custom_tools/results/taskid_online_r1_development_v1"
)
OFFLINE_ROOT = (
    ROOT / "custom_tools/results/taskid_offline_development_v1"
)
TEACHER_ROOT = (
    ROOT / "custom_tools/results/scaled_category_development_v1"
)
OUTPUT_ROOT = (
    ROOT / "custom_tools/results/taskid_online_r1_repeats_v1"
)
SEEDS = (2025, 2026, 2027)
CATEGORIES = ("bottle", "mug", "bowl", "camera")
TEACHER_LABELS = {
    "bottle": "scale20_epoch30",
    "mug": "scale20_epoch10",
    "bowl": "scale20_epoch40",
    "camera": "scale20_epoch40",
}


def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-free-vram-mb", type=int, default=4500)
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=5,
        help="Fresh-process retries for occasional PhysX initialization errors.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_yaml(path):
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def checkpoint_result(path, expected_checkpoint=None):
    aggregate = load_yaml(path)
    rows = aggregate.get("checkpoint_results", [])
    if len(rows) != 1:
        raise RuntimeError("Expected one checkpoint result in {}".format(path))
    if expected_checkpoint is not None:
        actual = Path(rows[0]["checkpoint"]).resolve()
        if actual != expected_checkpoint.resolve():
            raise RuntimeError("Checkpoint mismatch in {}".format(path))
    return rows[0]


def online_output(seed):
    if seed == 2025:
        return ONLINE_STAGE_ROOT / "online_25pct_epoch02.yaml"
    return OUTPUT_ROOT / "seed{}".format(seed) / "online_25pct_epoch02.yaml"


def evaluate_online(cli, seed):
    output = online_output(seed)
    if output.is_file():
        checkpoint_result(output, ONLINE_CHECKPOINT)
        print("[REUSE] selected online student seed={}".format(seed), flush=True)
        return
    command = [
        sys.executable,
        "-u",
        str(ROOT / "custom_tools/evaluate_bc_checkpoints_isolated.py"),
        "--checkpoint",
        str(ONLINE_CHECKPOINT),
        "--bc-config",
        str(ONLINE_CONFIG),
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
    print("[RUN] selected online student seed={}".format(seed), flush=True)
    if cli.dry_run:
        print(" ".join(command), flush=True)
        return
    subprocess.run(command, cwd=str(ROOT), check=True)
    checkpoint_result(output, ONLINE_CHECKPOINT)


def aggregate_student(label, checkpoint, paths):
    results = [checkpoint_result(path, checkpoint) for path in paths]
    macros = [
        float(item["macro_official_peak_success_rate"]) for item in results
    ]
    category_summary = {}
    for category in CATEGORIES:
        values = [
            float(item["category_macro_success_rates"][category])
            for item in results
        ]
        category_summary[category] = {
            "mean": statistics.mean(values),
            "std": statistics.pstdev(values),
            "values": values,
        }
    return {
        "label": label,
        "policy_count": 1,
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
        "category_success": category_summary,
        "outputs": [str(path) for path in paths],
    }


def teacher_result_path(seed, category):
    return (
        TEACHER_ROOT
        / "seed{}".format(seed)
        / category
        / (TEACHER_LABELS[category] + ".yaml")
    )


def aggregate_teacher_pool():
    seed_rows = []
    for seed in SEEDS:
        category_results = {
            category: checkpoint_result(teacher_result_path(seed, category))
            for category in CATEGORIES
        }
        seed_rows.append(
            {
                "success_count": sum(
                    int(item["total_success_count"])
                    for item in category_results.values()
                ),
                "trajectory_count": sum(
                    int(item["total_trajectory_count"])
                    for item in category_results.values()
                ),
                "macro_success": statistics.mean(
                    float(item["macro_official_peak_success_rate"])
                    for item in category_results.values()
                ),
                "lift": statistics.mean(
                    float(item["macro_mean_maximum_lift_m"])
                    for item in category_results.values()
                ),
                "failure": statistics.mean(
                    float(item["macro_failure_rate"])
                    for item in category_results.values()
                ),
                "categories": {
                    category: float(
                        category_results[category][
                            "macro_official_peak_success_rate"
                        ]
                    )
                    for category in CATEGORIES
                },
            }
        )
    macros = [item["macro_success"] for item in seed_rows]
    return {
        "label": "fixed_20object_category_teacher_pool",
        "policy_count": 4,
        "routing": "category Task ID selects one fixed expert",
        "teacher_labels": TEACHER_LABELS,
        "success_counts": [item["success_count"] for item in seed_rows],
        "trajectory_count_per_seed": seed_rows[0]["trajectory_count"],
        "macro_success_mean": statistics.mean(macros),
        "macro_success_std": statistics.pstdev(macros),
        "macro_success_values": macros,
        "lift_mean_m": statistics.mean(item["lift"] for item in seed_rows),
        "failure_mean": statistics.mean(
            item["failure"] for item in seed_rows
        ),
        "category_success": {
            category: {
                "mean": statistics.mean(
                    item["categories"][category] for item in seed_rows
                ),
                "std": statistics.pstdev(
                    item["categories"][category] for item in seed_rows
                ),
                "values": [
                    item["categories"][category] for item in seed_rows
                ],
            }
            for category in CATEGORIES
        },
        "outputs": [
            str(teacher_result_path(seed, category))
            for seed in SEEDS
            for category in CATEGORIES
        ],
    }


def summarize():
    offline_checkpoint = (
        ROOT / "custom_tools/runs/bc/"
        / "unified_student_taskid_scaled20_t100_seed2025_e20_v1/"
        / "epoch=014-step=14145.ckpt"
    )
    teacher = aggregate_teacher_pool()
    offline = aggregate_student(
        "offline_taskid_student",
        offline_checkpoint,
        [
            OFFLINE_ROOT
            / "seed{}".format(seed)
            / "t100_epoch15.yaml"
            for seed in SEEDS
        ],
    )
    online = aggregate_student(
        "online_r1_25pct_epoch02",
        ONLINE_CHECKPOINT,
        [online_output(seed) for seed in SEEDS],
    )
    nodes = [teacher, offline, online]
    summary = {
        "status": "complete",
        "stage": "serial Task-ID mainline comparison after online round 1",
        "formal_final_holdout_result": False,
        "final_holdout_accessed": False,
        "selection": str(SELECTION),
        "seeds": list(SEEDS),
        "metric": "object-macro official peak success",
        "serial_nodes": nodes,
        "online_minus_offline_macro": (
            online["macro_success_mean"] - offline["macro_success_mean"]
        ),
        "online_minus_teacher_macro": (
            online["macro_success_mean"] - teacher["macro_success_mean"]
        ),
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary_path = OUTPUT_ROOT / "summary.yaml"
    with summary_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(summary, handle, allow_unicode=True, sort_keys=False)
    print("TASKID_ONLINE_R1_REPEATS=COMPLETE", flush=True)
    for node in nodes:
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
        ONLINE_CONFIG,
        RESIDUAL_CONFIG,
        TRAJECTORY_ROOT,
        SELECTION,
        ONLINE_CHECKPOINT,
        online_output(2025),
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing repeat inputs: {}".format(missing))
    for seed in SEEDS:
        evaluate_online(cli, seed)
    if not cli.dry_run:
        summarize()


if __name__ == "__main__":
    main()
