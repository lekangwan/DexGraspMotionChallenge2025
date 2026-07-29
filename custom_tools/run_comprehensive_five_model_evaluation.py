"""Evaluate and plot five locked serial-policy nodes on three data splits.

The script never trains or selects a checkpoint.  It reuses previously verified
results when the checkpoint path matches, runs only missing isolated evaluations,
then exports one auditable YAML/CSV summary and report-ready figures.
"""

import argparse
import collections
import csv
import hashlib
from pathlib import Path
import statistics
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = (
    ROOT / "custom_tools/configs/"
    "comprehensive_five_model_evaluation_v1.yaml")
EVALUATOR = ROOT / "custom_tools/evaluate_bc_checkpoints_isolated.py"
RESIDUAL_CONFIG = ROOT / "custom_tools/configs/residual_ppo_stage1.yaml"
FULL_TRAJECTORY_ROOT = (
    ROOT / "dexgrasp/dataset/scaled_category_final_v1_preprocessed")
DEV_SELECTION = ROOT / "custom_tools/configs/scaled_development_all12.yaml"
FINAL_SELECTION = ROOT / "custom_tools/configs/scaled_final_holdout_all8.yaml"
SEEN80_PREPARER = ROOT / "custom_tools/run_taskid_locked_seen80_validation.py"
SEEN80_ROOT = (
    ROOT / "custom_tools/results/taskid_locked_seen80_validation_v1/"
    "seen80_sim_validation_trajectories")
SEEN80_SELECTION = (
    ROOT / "custom_tools/results/taskid_locked_seen80_validation_v1/"
    "seen80_selection.yaml")
OUTPUT = ROOT / "custom_tools/results/comprehensive_five_model_evaluation_v1"
CATEGORIES = ("bottle", "mug", "bowl", "camera")
SPLIT_ORDER = ("seen80", "development12", "final8")


def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-free-vram-mb", type=int, default=4500)
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_yaml(path):
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def absolute(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def validate_protocol(protocol):
    if protocol["status"] != "locked_reporting_protocol":
        raise RuntimeError("Comprehensive evaluation protocol is not locked")
    if protocol["primary_metric"] != (
            "object_macro_official_peak_success_rate"):
        raise RuntimeError("Unexpected primary metric")
    labels = [model["label"] for model in protocol["models"]]
    expected = [
        "official_bc", "bc_soup", "offline_student", "online_r1",
        "temporal3"]
    if labels != expected:
        raise RuntimeError("Model order changed: {}".format(labels))
    for model in protocol["models"]:
        checkpoint = absolute(model["checkpoint"])
        config = absolute(model["config"])
        if not checkpoint.is_file() or not config.is_file():
            raise FileNotFoundError(
                "Missing model input: {} or {}".format(checkpoint, config))
        if sha256(checkpoint) != model["checkpoint_sha256"]:
            raise RuntimeError("Checkpoint hash changed: {}".format(checkpoint))


def prepare_seen80(cli):
    command = [
        sys.executable, "-u", str(SEEN80_PREPARER),
        "--seed", "2025",
        "--min-free-vram-mb", str(cli.min_free_vram_mb),
        "--max-attempts", str(cli.max_attempts),
        "--dry-run",
    ]
    subprocess.run(command, cwd=str(ROOT), check=True)
    if not SEEN80_ROOT.is_dir() or not SEEN80_SELECTION.is_file():
        raise RuntimeError("Seen80 staging was not created")
    selection = load_yaml(SEEN80_SELECTION)
    if len(selection["object_ids"]) != 80:
        raise RuntimeError("Seen80 selection must contain 80 objects")


def split_inputs(split):
    if split == "seen80":
        return SEEN80_ROOT, SEEN80_SELECTION
    if split == "development12":
        return FULL_TRAJECTORY_ROOT, DEV_SELECTION
    if split == "final8":
        return FULL_TRAJECTORY_ROOT, FINAL_SELECTION
    raise KeyError(split)


def previous_result(split, label, seed):
    """Return a verified historical result path when one already exists."""
    if split == "seen80" and label == "temporal3" and seed == 2025:
        return (
            ROOT / "custom_tools/results/"
            "taskid_locked_seen80_validation_v1/seed2025/temporal3.yaml")

    if split == "development12":
        if label == "official_bc":
            if seed == 2025:
                return (
                    ROOT / "custom_tools/results/"
                    "missing_teacher_comparisons_v1/development/"
                    "matched_official_epoch080_seed2025.yaml")
            return (
                ROOT / "custom_tools/results/"
                "missing_teacher_comparisons_v1/development_repeats/"
                "matched_official_bc_seed{}.yaml".format(seed))
        if label == "offline_student":
            return (
                ROOT / "custom_tools/results/taskid_offline_development_v1/"
                "seed{}/t100_epoch15.yaml".format(seed))
        if label == "online_r1":
            if seed == 2025:
                return (
                    ROOT / "custom_tools/results/"
                    "taskid_online_r1_development_v1/"
                    "online_25pct_epoch02.yaml")
            return (
                ROOT / "custom_tools/results/taskid_online_r1_repeats_v1/"
                "seed{}/online_25pct_epoch02.yaml".format(seed))
        if label == "temporal3":
            if seed == 2025:
                return (
                    ROOT / "custom_tools/results/"
                    "taskid_temporal3_development_v1/"
                    "temporal3_epoch04.yaml")
            return (
                ROOT / "custom_tools/results/taskid_temporal3_repeats_v1/"
                "seed{}/temporal3_epoch04.yaml".format(seed))

    if split == "final8":
        if label == "official_bc":
            return (
                ROOT / "custom_tools/results/"
                "missing_teacher_comparisons_v1/final_reporting/"
                "matched_official_bc_seed{}.yaml".format(seed))
        if label in ("online_r1", "temporal3"):
            return (
                ROOT / "custom_tools/results/taskid_locked_final_holdout_v1/"
                "seed{}/{}.yaml".format(seed, label))
    return None


def result_row(path, expected_checkpoint):
    if not path or not path.is_file():
        return None
    aggregate = load_yaml(path)
    rows = aggregate.get("checkpoint_results", [])
    if len(rows) != 1:
        raise RuntimeError("Expected one checkpoint result in {}".format(path))
    actual = Path(rows[0]["checkpoint"]).resolve()
    if actual != expected_checkpoint.resolve():
        raise RuntimeError(
            "Checkpoint mismatch in {}: {} != {}".format(
                path, actual, expected_checkpoint))
    return rows[0]


def evaluate(cli, split, model, seed):
    checkpoint = absolute(model["checkpoint"])
    reused = previous_result(split, model["label"], seed)
    row = result_row(reused, checkpoint)
    if row is not None:
        print("[REUSE] {} {} seed={} <- {}".format(
            split, model["label"], seed, reused), flush=True)
        return row, reused

    trajectory_root, selection = split_inputs(split)
    output = (
        OUTPUT / "raw" / split / "seed{}".format(seed)
        / "{}.yaml".format(model["label"]))
    row = result_row(output, checkpoint)
    if row is not None:
        print("[REUSE] {} {} seed={}".format(
            split, model["label"], seed), flush=True)
        return row, output

    command = [
        sys.executable, "-u", str(EVALUATOR),
        "--checkpoint", str(checkpoint),
        "--bc-config", str(absolute(model["config"])),
        "--residual-config", str(RESIDUAL_CONFIG),
        "--trajectory-root", str(trajectory_root),
        "--object-selection", str(selection),
        "--output", str(output),
        "--seed", str(seed),
        "--min-free-vram-mb", str(cli.min_free_vram_mb),
        "--max-attempts", str(cli.max_attempts),
    ]
    print("[RUN] {} {} seed={}".format(
        split, model["label"], seed), flush=True)
    print(" ".join(command), flush=True)
    if cli.dry_run:
        return None, output
    subprocess.run(command, cwd=str(ROOT), check=True)
    return result_row(output, checkpoint), output


def summarize_model(model, seed_rows, sources):
    macros = [
        float(row["macro_official_peak_success_rate"]) for row in seed_rows]
    overalls = [
        float(row["overall_official_peak_success_rate"]) for row in seed_rows]
    lifts = [float(row["macro_mean_maximum_lift_m"]) for row in seed_rows]
    failures = [float(row["macro_failure_rate"]) for row in seed_rows]
    category_success = {}
    for category in CATEGORIES:
        values = [
            float(row["category_macro_success_rates"][category])
            for row in seed_rows]
        category_success[category] = {
            "mean": statistics.mean(values),
            "std": statistics.pstdev(values),
            "values": values,
        }
    return {
        "label": model["label"],
        "display_name": model["display_name"],
        "checkpoint": str(absolute(model["checkpoint"])),
        "checkpoint_sha256": model["checkpoint_sha256"],
        "success_counts": [
            int(row["total_success_count"]) for row in seed_rows],
        "trajectory_count_per_seed": int(
            seed_rows[0]["total_trajectory_count"]),
        "overall_success_mean": statistics.mean(overalls),
        "overall_success_std": statistics.pstdev(overalls),
        "macro_success_mean": statistics.mean(macros),
        "macro_success_std": statistics.pstdev(macros),
        "macro_success_values": macros,
        "lift_mean_m": statistics.mean(lifts),
        "lift_std_m": statistics.pstdev(lifts),
        "failure_mean": statistics.mean(failures),
        "category_success": category_success,
        "sources": [str(path) for path in sources],
    }


def export_csv(protocol, summary):
    output = OUTPUT / "metrics.csv"
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "split", "split_role", "model", "display_name",
            "object_count", "trajectory_count_per_seed",
            "macro_success_mean", "macro_success_std",
            "overall_success_mean", "overall_success_std",
            "lift_mean_m", "lift_std_m", "failure_mean",
            "bottle_success", "mug_success", "bowl_success",
            "camera_success"])
        writer.writeheader()
        for split in SPLIT_ORDER:
            for row in summary["splits"][split]["models"]:
                writer.writerow({
                    "split": split,
                    "split_role": protocol["splits"][split]["role"],
                    "model": row["label"],
                    "display_name": row["display_name"],
                    "object_count": protocol["splits"][split]["object_count"],
                    "trajectory_count_per_seed": (
                        row["trajectory_count_per_seed"]),
                    "macro_success_mean": row["macro_success_mean"],
                    "macro_success_std": row["macro_success_std"],
                    "overall_success_mean": row["overall_success_mean"],
                    "overall_success_std": row["overall_success_std"],
                    "lift_mean_m": row["lift_mean_m"],
                    "lift_std_m": row["lift_std_m"],
                    "failure_mean": row["failure_mean"],
                    "bottle_success": (
                        row["category_success"]["bottle"]["mean"]),
                    "mug_success": row["category_success"]["mug"]["mean"],
                    "bowl_success": row["category_success"]["bowl"]["mean"],
                    "camera_success": (
                        row["category_success"]["camera"]["mean"]),
                })


def export_plots(summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    models = summary["splits"]["seen80"]["models"]
    labels = [row["display_name"] for row in models]
    x = np.arange(len(labels))
    colors = {
        "seen80": "#2878B5",
        "development12": "#F39B35",
        "final8": "#C82423",
    }
    names = {
        "seen80": "Seen-80 held-out",
        "development12": "Development-12 (selection)",
        "final8": "Final unseen-8",
    }
    styles = {
        "seen80": "-",
        "development12": "--",
        "final8": "-",
    }

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.8))
    for split in SPLIT_ORDER:
        rows = summary["splits"][split]["models"]
        success = [100.0 * row["macro_success_mean"] for row in rows]
        success_std = [100.0 * row["macro_success_std"] for row in rows]
        lift = [100.0 * row["lift_mean_m"] for row in rows]
        lift_std = [100.0 * row["lift_std_m"] for row in rows]
        # Official BC is a data-matched control branch, not the direct parent
        # of the historical Soup checkpoint.  Keep the control point visually
        # separate and connect only the proposed Soup-to-Temporal3 pipeline.
        axes[0].errorbar(
            x[:1], success[:1], yerr=success_std[:1], marker="s",
            capsize=3, color=colors[split], linestyle="none")
        axes[0].errorbar(
            x[1:], success[1:], yerr=success_std[1:], marker="o", capsize=3,
            color=colors[split], linestyle=styles[split],
            linewidth=2, label=names[split])
        axes[1].errorbar(
            x[:1], lift[:1], yerr=lift_std[:1], marker="s",
            capsize=3, color=colors[split], linestyle="none")
        axes[1].errorbar(
            x[1:], lift[1:], yerr=lift_std[1:], marker="o", capsize=3,
            color=colors[split], linestyle=styles[split],
            linewidth=2, label=names[split])
    axes[0].set_ylabel("Object-macro official success (%)")
    axes[1].set_ylabel("Mean maximum lift (cm)")
    for axis in axes:
        axis.set_xticks(x)
        axis.set_xticklabels(labels, rotation=12, ha="right")
        axis.set_xlabel("Locked serial policy node")
        axis.grid(axis="y", alpha=0.25)
        axis.axvline(0.5, color="#777777", linestyle=":", linewidth=1)
        axis.text(
            0.1, 0.97, "Baseline", transform=axis.get_xaxis_transform(),
            ha="center", va="top", fontsize=8, color="#555555")
        axis.text(
            2.5, 0.97, "Proposed serial pipeline",
            transform=axis.get_xaxis_transform(),
            ha="center", va="top", fontsize=8, color="#555555")
        axis.legend(fontsize=8)
    axes[0].set_title("Closed-loop success")
    axes[1].set_title("Object lifting")
    fig.suptitle(
        "Five-model evaluation on seen, development, and final-unseen splits")
    fig.tight_layout()
    fig.savefig(OUTPUT / "five_model_split_curves.png", dpi=220)
    plt.close(fig)

    final_rows = summary["splits"]["final8"]["models"]
    width = 0.16
    fig, axis = plt.subplots(figsize=(10.8, 4.8))
    for index, category in enumerate(CATEGORIES):
        values = [
            100.0 * row["category_success"][category]["mean"]
            for row in final_rows]
        axis.bar(
            x + (index - 1.5) * width, values, width=width,
            label=category.capitalize())
    axis.set_xticks(x)
    axis.set_xticklabels(labels, rotation=12, ha="right")
    axis.set_ylabel("Object-macro official success (%)")
    axis.set_title("Final unseen-8 category success")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT / "final8_category_comparison.png", dpi=220)
    plt.close(fig)


def main():
    cli = parse_cli()
    protocol = load_yaml(PROTOCOL)
    validate_protocol(protocol)
    prepare_seen80(cli)
    seeds = [int(seed) for seed in protocol["simulation_seeds"]]
    collected = collections.defaultdict(dict)
    sources = collections.defaultdict(dict)

    for split in SPLIT_ORDER:
        for model in protocol["models"]:
            rows = []
            paths = []
            for seed in seeds:
                row, path = evaluate(cli, split, model, seed)
                if row is not None:
                    rows.append(row)
                paths.append(path)
            collected[split][model["label"]] = rows
            sources[split][model["label"]] = paths

    if cli.dry_run:
        print("COMPREHENSIVE_FIVE_MODEL_DRY_RUN=COMPLETE", flush=True)
        return

    summary = {
        "status": "complete",
        "protocol": str(PROTOCOL),
        "primary_metric": protocol["primary_metric"],
        "simulation_seeds": seeds,
        "interpretation": {
            "seen80": (
                "Same object instances as optimizer training, but trajectories "
                "were excluded from optimizer updates."),
            "development12": (
                "Used for checkpoint selection; report as selection history, "
                "not unbiased generalization."),
            "final8": (
                "Unseen reporting split; no post-holdout model selection is "
                "allowed."),
        },
        "splits": {},
    }
    for split in SPLIT_ORDER:
        model_summaries = []
        for model in protocol["models"]:
            rows = collected[split][model["label"]]
            if len(rows) != len(seeds):
                raise RuntimeError(
                    "Incomplete rows for {} {}".format(split, model["label"]))
            model_summaries.append(summarize_model(
                model, rows, sources[split][model["label"]]))
        summary["splits"][split] = {
            "role": protocol["splits"][split]["role"],
            "object_count": protocol["splits"][split]["object_count"],
            "model_selection_allowed": False,
            "models": model_summaries,
        }

    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "summary.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(summary, handle, allow_unicode=True, sort_keys=False)
    export_csv(protocol, summary)
    export_plots(summary)
    print("COMPREHENSIVE_FIVE_MODEL_EVALUATION=COMPLETE", flush=True)
    print("summary={}".format(OUTPUT / "summary.yaml"), flush=True)
    print("figure={}".format(
        OUTPUT / "five_model_split_curves.png"), flush=True)


if __name__ == "__main__":
    main()
