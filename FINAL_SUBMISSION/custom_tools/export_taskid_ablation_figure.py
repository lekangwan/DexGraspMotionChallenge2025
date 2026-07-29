"""Export a report-ready Task-ID ablation figure and its source metrics."""

import csv
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
OFFLINE_SUMMARY = (
    ROOT / "custom_tools/results/missing_teacher_comparisons_v1/summary.yaml")
PIPELINE_SUMMARY = (
    ROOT / "custom_tools/results/notask_full_pipeline_ablation_v1/summary.yaml")
OUTPUT = ROOT / "custom_tools/results/taskid_ablation_report_v1"
STAGES = ("Offline student", "Online-R1", "Temporal3")
CATEGORIES = ("bottle", "mug", "bowl", "camera")


def load_yaml(path):
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def metric(row):
    return {
        "mean": float(row["macro_success_mean"]),
        "std": float(row["macro_success_std"]),
    }


def main():
    offline = load_yaml(OFFLINE_SUMMARY)["taskid_ablation"]
    pipeline = load_yaml(PIPELINE_SUMMARY)
    if pipeline["status"] != "complete":
        raise RuntimeError("Task-ID full-pipeline ablation is incomplete")
    if pipeline["formal_final_holdout_result"] is not False:
        raise RuntimeError("Expected a development-set-only ablation")

    without = pipeline["without_taskid"]
    with_task = pipeline["with_taskid_existing_controls"]
    stage_rows = {
        "without_taskid": [
            metric(offline["without_taskid_three_seed"]),
            metric(without["no_taskid_online_r1"]),
            metric(without["no_taskid_temporal3"]),
        ],
        "with_taskid": [
            metric(offline["with_taskid_three_seed"]),
            metric(with_task["online_r1"]),
            metric(with_task["temporal3"]),
        ],
    }
    category_rows = {
        "without_taskid": {
            category: without["no_taskid_temporal3"][
                "category_success"][category]
            for category in CATEGORIES
        },
        "with_taskid": {
            category: with_task["temporal3"]["category_success"][category]
            for category in CATEGORIES
        },
    }

    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "metrics.csv").open(
            "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["section", "label", "condition", "mean", "std"])
        for condition, rows in stage_rows.items():
            for label, row in zip(STAGES, rows):
                writer.writerow([
                    "pipeline_stage", label, condition,
                    row["mean"], row["std"]])
        for condition, rows in category_rows.items():
            for category, row in rows.items():
                writer.writerow([
                    "temporal3_category", category, condition,
                    row["mean"], row["std"]])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    x = np.arange(len(STAGES))
    colors = {
        "without_taskid": "#7A7A7A",
        "with_taskid": "#2878B5",
    }
    labels = {
        "without_taskid": "Without Task ID",
        "with_taskid": "With Task ID",
    }
    figure, axes = plt.subplots(1, 2, figsize=(12.6, 4.8))

    for condition in ("without_taskid", "with_taskid"):
        rows = stage_rows[condition]
        means = [100.0 * row["mean"] for row in rows]
        stds = [100.0 * row["std"] for row in rows]
        axes[0].errorbar(
            x, means, yerr=stds, marker="o", markersize=7,
            linewidth=2.2, capsize=4, color=colors[condition],
            label=labels[condition])
        for index, value in enumerate(means):
            offset = -1.45 if condition == "without_taskid" else 0.75
            axes[0].text(
                index, value + offset, "{:.2f}%".format(value),
                ha="center", va="bottom", fontsize=8,
                color=colors[condition])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(STAGES)
    axes[0].set_ylabel("Object-macro official success (%)")
    axes[0].set_title("Pipeline-stage success")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(loc="upper left", fontsize=9)
    axes[0].set_ylim(15, 42)

    category_x = np.arange(len(CATEGORIES))
    width = 0.34
    no_task_means = [
        100.0 * float(category_rows["without_taskid"][category]["mean"])
        for category in CATEGORIES]
    task_means = [
        100.0 * float(category_rows["with_taskid"][category]["mean"])
        for category in CATEGORIES]
    no_task_std = [
        100.0 * float(category_rows["without_taskid"][category]["std"])
        for category in CATEGORIES]
    task_std = [
        100.0 * float(category_rows["with_taskid"][category]["std"])
        for category in CATEGORIES]
    axes[1].bar(
        category_x - width / 2, no_task_means, width,
        yerr=no_task_std, capsize=3, color=colors["without_taskid"],
        label=labels["without_taskid"])
    axes[1].bar(
        category_x + width / 2, task_means, width,
        yerr=task_std, capsize=3, color=colors["with_taskid"],
        label=labels["with_taskid"])
    axes[1].set_xticks(category_x)
    axes[1].set_xticklabels([value.capitalize() for value in CATEGORIES])
    axes[1].set_ylabel("Object-macro official success (%)")
    axes[1].set_title("Temporal3 category success")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(loc="upper right", fontsize=9)
    axes[1].set_ylim(0, 52)

    figure.suptitle(
        "Task-ID ablation on Development-12 "
        "(313 trajectories, 3 simulation seeds)",
        fontsize=12)
    figure.text(
        0.5, 0.01,
        "Development/selection set only — not the final unseen-8 test.",
        ha="center", va="bottom", fontsize=9, color="#A33A2B")
    figure.tight_layout(rect=(0, 0.045, 1, 0.94))
    figure.savefig(OUTPUT / "taskid_ablation_curve.png", dpi=240)
    figure.savefig(OUTPUT / "taskid_ablation_curve.pdf")
    plt.close(figure)
    print("TASKID_ABLATION_FIGURE=COMPLETE")
    print("png={}".format(OUTPUT / "taskid_ablation_curve.png"))
    print("pdf={}".format(OUTPUT / "taskid_ablation_curve.pdf"))


if __name__ == "__main__":
    main()
