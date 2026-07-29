"""Export report assets for the locked Task-ID -> Online-R1 -> Temporal3 line.

This script is reporting-only.  It reads the already locked training audit,
the one-time final holdout evaluation, and the Temporal3 TensorBoard log.
It never trains a model or changes the final model lock.
"""

import csv
from pathlib import Path
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


ROOT = Path(__file__).resolve().parents[1]
TRAINING_SUMMARY = (
    ROOT / "custom_tools/results/training_mainline_three_seed_v1/summary.yaml")
FINAL_ROOT = ROOT / "custom_tools/results/taskid_locked_final_holdout_v1"
FINAL_SUMMARY = FINAL_ROOT / "summary.yaml"
EVENT_DIR = (
    ROOT / "custom_tools/runs/bc/"
    "unified_student_taskid_temporal3_seed2025_e4_v1/"
    "tensorboard_logs/lightning_logs/version_0")
RESIDUAL_METRICS = (
    ROOT / "custom_tools/runs/residual_ppo/"
    "temporal3_gated_residual_anchored64_seed2025_i50_v1/metrics.csv")
RESIDUAL_EVALUATION = (
    ROOT / "custom_tools/results/"
    "temporal3_residual_stage1_development_v1/summary.yaml")
OUTPUT = ROOT / "custom_tools/results/taskid_final_report_assets_v1"
CATEGORIES = ("bottle", "mug", "bowl", "camera")
NODE_LABELS = {
    "bc_soup": "BC Soup",
    "category_teacher_pool": "Category\nexperts",
    "offline_taskid_student": "Offline\nTask-ID",
    "online_r1_student": "Online-R1",
    "temporal3_student": "Temporal3",
}
MODEL_LABELS = {"online_r1": "Online-R1", "temporal3": "Temporal3"}


def load_yaml(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_csv(path, rows):
    if not rows:
        raise ValueError("Cannot write an empty CSV")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_inputs(training, final):
    if training.get("status") != "complete":
        raise RuntimeError("Training audit is incomplete")
    if final.get("status") != "complete":
        raise RuntimeError("Final holdout evaluation is incomplete")
    if final.get("formal_final_holdout_result") is not True:
        raise RuntimeError("Input is not the formal final holdout result")
    if final.get("result_may_be_used_for_further_model_selection") is not False:
        raise RuntimeError("Final result is not marked reporting-only")
    if final.get("primary_model_was_locked_before_evaluation") != "temporal3":
        raise RuntimeError("Temporal3 was not locked before final evaluation")


def plot_training_loss():
    accumulator = EventAccumulator(
        str(EVENT_DIR), size_guidance={"scalars": 0})
    accumulator.Reload()
    train = accumulator.Scalars("train_loss_epoch")
    valid = accumulator.Scalars("val_loss")
    if len(train) != len(valid) or not train:
        raise RuntimeError("Unexpected Temporal3 epoch-loss records")
    rows = [
        {
            "epoch": index + 1,
            "optimizer_step": train_event.step,
            "train_loss": train_event.value,
            "validation_loss": valid_event.value,
        }
        for index, (train_event, valid_event) in enumerate(zip(train, valid))
    ]
    write_csv(OUTPUT / "temporal3_training_loss.csv", rows)
    figure, axis = plt.subplots(figsize=(6.4, 4.0))
    epochs = [row["epoch"] for row in rows]
    axis.plot(
        epochs, [row["train_loss"] for row in rows],
        marker="o", linewidth=2, label="Train loss")
    axis.plot(
        epochs, [row["validation_loss"] for row in rows],
        marker="s", linewidth=2, label="Validation loss")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Action imitation loss")
    axis.set_xticks(epochs)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(OUTPUT / "temporal3_training_loss.png", dpi=220)
    plt.close(figure)


def rolling_mean(values, window):
    return [
        statistics.mean(values[max(0, index - window + 1):index + 1])
        for index in range(len(values))]


def plot_residual_reward_success():
    with RESIDUAL_METRICS.open(encoding="utf-8") as handle:
        training = list(csv.DictReader(handle))
    evaluation = load_yaml(RESIDUAL_EVALUATION)
    iterations = [int(row["iteration"]) for row in training]
    rewards = [float(row["reward_reward_mean"]) for row in training]
    evaluation_rows = evaluation["rows"]
    validation_iterations = [
        0 if row["label"] == "baseline_zero_residual"
        else int(row["label"].split("_")[-1])
        for row in evaluation_rows]
    validation_success = [
        100 * float(row["macro_success_rate"]) for row in evaluation_rows]

    figure, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
    axes[0].plot(
        iterations, rewards, color="#7aa6c2", alpha=0.45,
        linewidth=1, label="Per iteration")
    axes[0].plot(
        iterations, rolling_mean(rewards, 5), color="#2f6f9f",
        linewidth=2, label="5-iteration mean")
    axes[0].axhline(0, color="black", linewidth=0.8, alpha=0.5)
    axes[0].set_xlabel("Residual-PPO iteration")
    axes[0].set_ylabel("Custom training reward")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(
        validation_iterations, validation_success,
        marker="o", linewidth=2, color="#b35c44")
    axes[1].axhline(
        validation_success[0], color="#777777", linestyle="--",
        linewidth=1.2, label="Zero-residual baseline")
    axes[1].set_xlabel("Residual-PPO iteration")
    axes[1].set_ylabel("Development macro success (%)")
    axes[1].set_xticks(validation_iterations)
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(
        OUTPUT / "temporal3_residual_reward_success.png", dpi=220)
    plt.close(figure)


def plot_serial_mainline(training):
    rows = []
    for index, node in enumerate(training["nodes"]):
        rows.append({
            "serial_index": index,
            "node": node["label"],
            "macro_success_mean": node["macro_success_mean"],
            "macro_success_std": node["macro_success_std"],
            "mean_maximum_lift_m": node["lift_mean_m"],
            "failure_rate": node["failure_mean"],
        })
    write_csv(OUTPUT / "training_mainline_metrics.csv", rows)
    x = list(range(len(rows)))
    figure, axis = plt.subplots(figsize=(7.8, 4.2))
    axis.errorbar(
        x, [100 * row["macro_success_mean"] for row in rows],
        yerr=[100 * row["macro_success_std"] for row in rows],
        marker="o", markersize=7, linewidth=2.2, capsize=4,
        color="#2f6f9f")
    axis.set_xticks(
        x, [NODE_LABELS[row["node"]] for row in rows])
    axis.set_ylabel("Object-macro official success (%)")
    axis.set_xlabel("Serial training node (diagnostic subset only)")
    axis.set_title(
        "16 seen-object diagnostic subset — not final test performance",
        fontsize=10)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(OUTPUT / "training_mainline_success.png", dpi=220)
    plt.close(figure)


def final_runs(final):
    runs = {}
    for node in final["nodes"]:
        label = node["label"]
        runs[label] = [
            load_yaml(path)["checkpoint_results"][0]
            for path in node["outputs"]]
    return runs


def plot_final_overall(final, runs):
    rows = []
    for node in final["nodes"]:
        label = node["label"]
        model_runs = runs[label]
        rows.append({
            "model": label,
            "macro_success_mean": node["macro_success_mean"],
            "macro_success_std": node["macro_success_std"],
            "overall_success_mean": node["overall_success_mean"],
            "overall_success_std": node["overall_success_std"],
            "mean_maximum_lift_m": node["lift_mean_m"],
            "mean_maximum_lift_std_m": statistics.pstdev(
                run["macro_mean_maximum_lift_m"] for run in model_runs),
            "failure_rate": node["failure_mean"],
        })
    write_csv(OUTPUT / "final_model_metrics.csv", rows)
    x = list(range(len(rows)))
    labels = [MODEL_LABELS[row["model"]] for row in rows]
    colors = ("#8c8c8c", "#2f78b7")
    figure, axes = plt.subplots(1, 2, figsize=(8.6, 3.9))
    axes[0].bar(
        x, [100 * row["macro_success_mean"] for row in rows],
        yerr=[100 * row["macro_success_std"] for row in rows],
        capsize=5, color=colors)
    axes[0].set_ylabel("Object-macro success (%)")
    axes[0].set_xticks(x, labels)
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(
        x, [100 * row["mean_maximum_lift_m"] for row in rows],
        yerr=[100 * row["mean_maximum_lift_std_m"] for row in rows],
        capsize=5, color=colors)
    axes[1].set_ylabel("Mean maximum lift (cm)")
    axes[1].set_xticks(x, labels)
    axes[1].grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(OUTPUT / "final_model_comparison.png", dpi=220)
    plt.close(figure)


def plot_final_categories(final):
    rows = []
    for node in final["nodes"]:
        for category in CATEGORIES:
            record = node["category_success"][category]
            rows.append({
                "model": node["label"],
                "category": category,
                "success_mean": record["mean"],
                "success_std": record["std"],
            })
    write_csv(OUTPUT / "final_category_metrics.csv", rows)
    x = list(range(len(CATEGORIES)))
    width = 0.35
    figure, axis = plt.subplots(figsize=(7.3, 4.1))
    for model_index, model in enumerate(("online_r1", "temporal3")):
        selected = [row for row in rows if row["model"] == model]
        positions = [
            value + (model_index - 0.5) * width for value in x]
        axis.bar(
            positions, [100 * row["success_mean"] for row in selected],
            yerr=[100 * row["success_std"] for row in selected],
            width=width, capsize=4, label=MODEL_LABELS[model],
            color=("#8c8c8c", "#2f78b7")[model_index])
    axis.set_xticks(x, [category.capitalize() for category in CATEGORIES])
    axis.set_ylabel("Object-macro official success (%)")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(OUTPUT / "final_category_success.png", dpi=220)
    plt.close(figure)


def write_object_metrics(runs):
    rows = []
    for model, model_runs in runs.items():
        object_maps = [
            {item["object_id"]: item for item in run["objects"]}
            for run in model_runs]
        for object_id in object_maps[0]:
            records = [mapping[object_id] for mapping in object_maps]
            success = [
                float(record["official_peak_success_rate"])
                for record in records]
            lift = [
                float(record["mean_maximum_lift_m"])
                for record in records]
            failure = [
                float(record["failure_rate"]) for record in records]
            rows.append({
                "model": model,
                "category": records[0]["category"],
                "object_id": object_id,
                "trajectory_count": records[0]["trajectory_count"],
                "success_mean": statistics.mean(success),
                "success_std": statistics.pstdev(success),
                "lift_mean_m": statistics.mean(lift),
                "failure_mean": statistics.mean(failure),
            })
    write_csv(OUTPUT / "final_object_metrics.csv", rows)


def select_stable_render_cases(runs):
    temporal_runs = runs["temporal3"]
    object_maps = [
        {item["object_id"]: item for item in run["objects"]}
        for run in temporal_runs]
    cases = []
    for category in CATEGORIES:
        object_ids = sorted(
            object_id for object_id, item in object_maps[0].items()
            if item["category"] == category)
        selected = None
        for object_id in object_ids:
            records = [mapping[object_id] for mapping in object_maps]
            success_sets = [
                set(record["official_peak_success_local_indices"])
                for record in records]
            trajectory_count = int(records[0]["trajectory_count"])
            stable_success = set.intersection(*success_sets)
            stable_failure = (
                set(range(trajectory_count)).difference(
                    set.union(*success_sets)))
            if stable_success and stable_failure:
                selected = (object_id, min(stable_success), min(stable_failure))
                break
        if selected is None:
            raise RuntimeError(
                "No stable success/failure pair for {}".format(category))
        object_id, success_index, failure_index = selected
        cases.extend((
            {
                "category": category,
                "outcome": "stable_success",
                "object_id": object_id,
                "trajectory_index": int(success_index),
            },
            {
                "category": category,
                "outcome": "stable_failure",
                "object_id": object_id,
                "trajectory_index": int(failure_index),
            },
        ))
    result = {
        "status": "reporting_only_after_locked_final_evaluation",
        "source": str(FINAL_SUMMARY),
        "model": "locked_temporal3",
        "selection_rule": (
            "For each category, sort final object IDs; choose the first object "
            "with both outcomes, then the smallest trajectory index that "
            "succeeded in all three seeds and the smallest index that never "
            "succeeded in any seed. Cases are not used for model selection."),
        "cases": cases,
    }
    with (OUTPUT / "final_render_cases.yaml").open(
            "w", encoding="utf-8") as handle:
        yaml.safe_dump(result, handle, allow_unicode=True, sort_keys=False)


def main():
    training = load_yaml(TRAINING_SUMMARY)
    final = load_yaml(FINAL_SUMMARY)
    validate_inputs(training, final)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    runs = final_runs(final)
    plot_training_loss()
    plot_residual_reward_success()
    plot_serial_mainline(training)
    plot_final_overall(final, runs)
    plot_final_categories(final)
    write_object_metrics(runs)
    select_stable_render_cases(runs)
    print("TASKID_FINAL_REPORT_ASSETS=COMPLETE")
    print("OUTPUT={}".format(OUTPUT))


if __name__ == "__main__":
    main()
