"""Evaluate the pre-locked R1 and Temporal3 policies on the final holdout.

The model identities, primary model, objects, seeds, and metric are frozen in
configuration files before this script is run. Results are for reporting only:
they must not be used to select another checkpoint or tune the policy.
"""

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "custom_tools/configs/taskid_final_model_lock_v1.yaml"
PROTOCOL = ROOT / "custom_tools/configs/scaled_evaluation_protocol_v1.json"
SELECTION = ROOT / "custom_tools/configs/scaled_final_holdout_all8.yaml"
TRAJECTORY_ROOT = (
    ROOT / "dexgrasp/dataset/scaled_category_final_v1_preprocessed"
)
RESIDUAL_CONFIG = ROOT / "custom_tools/configs/residual_ppo_stage1.yaml"
OUTPUT_ROOT = ROOT / "custom_tools/results/taskid_locked_final_holdout_v1"
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


def absolute_from_root(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_frozen_inputs(lock):
    if lock["status"] != "locked_before_final_holdout_evaluation":
        raise RuntimeError("Final model lock status is invalid")
    if lock["final_holdout_accessed_at_lock_time"] is not False:
        raise RuntimeError("Lock must predate final holdout access")
    if lock["post_holdout_training_or_selection_allowed"] is not False:
        raise RuntimeError("Post-holdout model selection must be prohibited")
    if lock["models"]["primary"]["label"] != "temporal3":
        raise RuntimeError("Temporal3 must remain the pre-locked primary model")

    with PROTOCOL.open(encoding="utf-8") as handle:
        protocol = json.load(handle)
    selection = load_yaml(SELECTION)
    expected = {
        object_id
        for category in protocol["categories"].values()
        for object_id in category["final_holdout"]
    }
    actual = set(selection["object_ids"])
    development = {
        object_id
        for category in protocol["categories"].values()
        for object_id in category["development"]
    }
    if actual != expected or actual & development or len(actual) != 8:
        raise RuntimeError("Final holdout selection does not match protocol")
    if selection["must_not_be_used_for_model_selection"] is not True:
        raise RuntimeError("Final holdout usage rule is missing")

    for model in lock["models"].values():
        checkpoint = absolute_from_root(model["checkpoint"])
        config = absolute_from_root(model["config"])
        if not checkpoint.is_file() or not config.is_file():
            raise FileNotFoundError(
                "Missing locked model input: {} or {}".format(
                    checkpoint, config))
        if sha256(checkpoint) != model["checkpoint_sha256"]:
            raise RuntimeError("Checkpoint hash changed: {}".format(checkpoint))
    for object_id in selection["object_ids"]:
        trajectory = TRAJECTORY_ROOT / (object_id + ".npy")
        if not trajectory.is_file():
            raise FileNotFoundError(trajectory)


def output_path(label, seed):
    return OUTPUT_ROOT / "seed{}".format(seed) / (label + ".yaml")


def read_result(path, checkpoint):
    aggregate = load_yaml(path)
    rows = aggregate.get("checkpoint_results", [])
    if len(rows) != 1:
        raise RuntimeError("Expected one checkpoint result in {}".format(path))
    if Path(rows[0]["checkpoint"]).resolve() != checkpoint.resolve():
        raise RuntimeError("Checkpoint mismatch in {}".format(path))
    return rows[0]


def evaluate(cli, label, model, seed):
    checkpoint = absolute_from_root(model["checkpoint"])
    config = absolute_from_root(model["config"])
    output = output_path(label, seed)
    if output.is_file():
        read_result(output, checkpoint)
        print("[REUSE] {} seed={}".format(label, seed), flush=True)
        return
    command = [
        sys.executable,
        "-u",
        str(ROOT / "custom_tools/evaluate_bc_checkpoints_isolated.py"),
        "--checkpoint", str(checkpoint),
        "--bc-config", str(config),
        "--residual-config", str(RESIDUAL_CONFIG),
        "--trajectory-root", str(TRAJECTORY_ROOT),
        "--object-selection", str(SELECTION),
        "--output", str(output),
        "--seed", str(seed),
        "--min-free-vram-mb", str(cli.min_free_vram_mb),
        "--max-attempts", str(cli.max_attempts),
    ]
    print("[RUN] {} seed={}".format(label, seed), flush=True)
    if cli.dry_run:
        print(" ".join(command), flush=True)
        return
    subprocess.run(command, cwd=str(ROOT), check=True)
    read_result(output, checkpoint)


def aggregate(label, model):
    checkpoint = absolute_from_root(model["checkpoint"])
    paths = [output_path(label, seed) for seed in SEEDS]
    results = [read_result(path, checkpoint) for path in paths]
    macro_values = [
        float(item["macro_official_peak_success_rate"]) for item in results
    ]
    overall_values = [
        float(item["overall_official_peak_success_rate"]) for item in results
    ]
    category_success = {}
    for category in CATEGORIES:
        values = [
            float(item["category_macro_success_rates"][category])
            for item in results
        ]
        category_success[category] = {
            "mean": statistics.mean(values),
            "std": statistics.pstdev(values),
            "values": values,
        }
    return {
        "label": label,
        "role": "primary" if label == "temporal3" else "baseline",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": model["checkpoint_sha256"],
        "success_counts": [
            int(item["total_success_count"]) for item in results
        ],
        "trajectory_count_per_seed": int(results[0]["total_trajectory_count"]),
        "overall_success_mean": statistics.mean(overall_values),
        "overall_success_std": statistics.pstdev(overall_values),
        "overall_success_values": overall_values,
        "macro_success_mean": statistics.mean(macro_values),
        "macro_success_std": statistics.pstdev(macro_values),
        "macro_success_values": macro_values,
        "lift_mean_m": statistics.mean(
            float(item["macro_mean_maximum_lift_m"]) for item in results),
        "failure_mean": statistics.mean(
            float(item["macro_failure_rate"]) for item in results),
        "category_success": category_success,
        "outputs": [str(path) for path in paths],
    }


def summarize(lock):
    baseline = aggregate("online_r1", lock["models"]["baseline"])
    primary = aggregate("temporal3", lock["models"]["primary"])
    summary = {
        "status": "complete",
        "stage": "one-time locked final holdout evaluation",
        "formal_final_holdout_result": True,
        "final_holdout_accessed": True,
        "result_may_be_used_for_further_model_selection": False,
        "primary_model_was_locked_before_evaluation": "temporal3",
        "primary_metric": lock["primary_metric"],
        "seeds": list(SEEDS),
        "object_count": 8,
        "nodes": [baseline, primary],
        "temporal_minus_r1_macro": (
            primary["macro_success_mean"] - baseline["macro_success_mean"]),
        "temporal_minus_r1_overall": (
            primary["overall_success_mean"] - baseline["overall_success_mean"]),
        "temporal_minus_r1_lift_m": (
            primary["lift_mean_m"] - baseline["lift_mean_m"]),
        "temporal_minus_r1_failure": (
            primary["failure_mean"] - baseline["failure_mean"]),
        "temporal_minus_r1_macro_by_seed": [
            primary_value - baseline_value
            for primary_value, baseline_value in zip(
                primary["macro_success_values"],
                baseline["macro_success_values"])
        ],
        "temporal_minus_r1_category": {
            category: (
                primary["category_success"][category]["mean"]
                - baseline["category_success"][category]["mean"])
            for category in CATEGORIES
        },
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_ROOT / "summary.yaml").open(
        "w", encoding="utf-8"
    ) as handle:
        yaml.safe_dump(summary, handle, allow_unicode=True, sort_keys=False)
    print("TASKID_LOCKED_FINAL_HOLDOUT=COMPLETE", flush=True)
    for node in (baseline, primary):
        print(
            "{} success={} overall={:.2f}+/-{:.2f}% "
            "macro={:.2f}+/-{:.2f}% lift={:.3f}m failure={:.2f}%".format(
                node["label"],
                node["success_counts"],
                100 * node["overall_success_mean"],
                100 * node["overall_success_std"],
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
        LOCK, PROTOCOL, SELECTION, TRAJECTORY_ROOT, RESIDUAL_CONFIG,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing final evaluation inputs: {}".format(
            missing))
    lock = load_yaml(LOCK)
    validate_frozen_inputs(lock)
    for seed in SEEDS:
        for role in ("baseline", "primary"):
            model = lock["models"][role]
            evaluate(cli, model["label"], model, seed)
    if not cli.dry_run:
        summarize(lock)


if __name__ == "__main__":
    main()
