"""Export locked final-test figures, tables, and stable render cases."""

import csv
from pathlib import Path
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml


ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "custom_tools/results/evaluations/final_unseen_v2_locked"
SUMMARY = FINAL / "final_summary.yaml"
OUTPUT = ROOT / "custom_tools/results/final_report_assets"
CATEGORIES = ("bottle", "mug", "bowl", "camera")
METHODS = (
    ("soup_baseline", "BC Soup"),
    ("unified_online_t70", "Unified 70%"),
    ("unified_online_t85", "Unified 85%"),
    ("routed_teacher_pool", "Routed experts"),
)


def load(path):
    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def single_runs(label):
    return [load(FINAL / "single_networks" / f"{label}_r{repeat}.yaml")
            ["checkpoint_results"][0] for repeat in (1, 2, 3)]


def routed_runs():
    runs = []
    for repeat in (1, 2, 3):
        objects = []
        for category in CATEGORIES:
            result = load(FINAL / "routed_teacher_pool" /
                          f"repeat{repeat}_{category}.yaml")
            objects.extend(result["checkpoint_results"][0]["objects"])
        runs.append(objects)
    return runs


def object_macro(objects, key):
    return statistics.mean(float(item[key]) for item in objects)


def summarize():
    rows = []
    run_objects = {}
    for key, label in METHODS[:3]:
        runs = single_runs(key)
        run_objects[key] = [run["objects"] for run in runs]
        rows.append({
            "method": key,
            "label": label,
            "mean_success_count": statistics.mean(
                run["total_success_count"] for run in runs),
            "success_count_std": statistics.pstdev(
                run["total_success_count"] for run in runs),
            "mean_macro_success_rate": statistics.mean(
                run["macro_official_peak_success_rate"] for run in runs),
            "macro_success_rate_std": statistics.pstdev(
                run["macro_official_peak_success_rate"] for run in runs),
            "mean_macro_lift_m": statistics.mean(
                run["macro_mean_maximum_lift_m"] for run in runs),
            "macro_lift_m_std": statistics.pstdev(
                run["macro_mean_maximum_lift_m"] for run in runs),
            "mean_macro_failure_rate": statistics.mean(
                run["macro_failure_rate"] for run in runs),
        })
    routed = routed_runs()
    run_objects["routed_teacher_pool"] = routed
    summary = load(SUMMARY)["routed_teacher_pool_results"]
    rows.append({
        "method": "routed_teacher_pool",
        "label": "Routed experts",
        "mean_success_count": summary["mean_success_count"],
        "success_count_std": summary["success_count_population_std"],
        "mean_macro_success_rate": summary["mean_macro_success_rate"],
        "macro_success_rate_std": summary[
            "macro_success_rate_population_std"],
        "mean_macro_lift_m": summary["mean_macro_lift_m"],
        "macro_lift_m_std": statistics.pstdev(
            object_macro(objects, "mean_maximum_lift_m") for objects in routed),
        "mean_macro_failure_rate": summary["mean_macro_failure_rate"],
    })
    return rows, run_objects


def write_summary(rows):
    path = OUTPUT / "final_method_summary.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return path


def plot_methods(rows):
    x = list(range(len(rows)))
    labels = [row["label"] for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    axes[0].bar(x, [100 * row["mean_macro_success_rate"] for row in rows],
                yerr=[100 * row["macro_success_rate_std"] for row in rows],
                capsize=4, color=("#7f8c8d", "#3498db", "#9b59b6", "#e67e22"))
    axes[0].set_ylabel("Official success rate (%)")
    axes[0].set_xticks(x, labels, rotation=18, ha="right")
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(x, [100 * row["mean_macro_lift_m"] for row in rows],
                yerr=[100 * row["macro_lift_m_std"] for row in rows],
                capsize=4, color=("#7f8c8d", "#3498db", "#9b59b6", "#e67e22"))
    axes[1].set_ylabel("Mean maximum lift (cm)")
    axes[1].set_xticks(x, labels, rotation=18, ha="right")
    axes[1].grid(axis="y", alpha=0.25)
    figure.tight_layout()
    path = OUTPUT / "final_method_comparison.png"
    figure.savefig(path, dpi=200)
    plt.close(figure)
    return path


def category_rates(run_objects):
    result = {}
    for method, runs in run_objects.items():
        result[method] = {}
        for category in CATEGORIES:
            per_run = []
            for objects in runs:
                selected = [item for item in objects
                            if item["category"] == category]
                per_run.append(object_macro(
                    selected, "official_peak_success_rate"))
            result[method][category] = statistics.mean(per_run)
    return result


def plot_categories(run_objects):
    rates = category_rates(run_objects)
    x = list(range(len(CATEGORIES)))
    width = 0.19
    figure, axis = plt.subplots(figsize=(9.2, 4.5))
    colors = ("#7f8c8d", "#3498db", "#9b59b6", "#e67e22")
    for index, (key, label) in enumerate(METHODS):
        offset = (index - 1.5) * width
        axis.bar([value + offset for value in x],
                 [100 * rates[key][category] for category in CATEGORIES],
                 width=width, label=label, color=colors[index])
    axis.set_xticks(x, [item.capitalize() for item in CATEGORIES])
    axis.set_ylabel("Official success rate (%)")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(ncol=2)
    figure.tight_layout()
    path = OUTPUT / "final_category_success.png"
    figure.savefig(path, dpi=200)
    plt.close(figure)
    return path


def stable_cases():
    cases = []
    routed = routed_runs()
    for category in CATEGORIES:
        object_ids = sorted({item["object_id"] for item in routed[0]
                             if item["category"] == category})
        successes = []
        failures = []
        for object_id in object_ids:
            records = []
            for objects in routed:
                item = next(value for value in objects
                            if value["object_id"] == object_id)
                success = set(item["official_peak_success_source_indices"])
                lifts = dict(zip(item["trajectory_indices"],
                                 item["diagnostic_maximum_lift_m_by_trajectory"]))
                records.append((success, lifts))
            common = set.intersection(*(record[0] for record in records))
            all_indices = set.intersection(*(set(record[1]) for record in records))
            never = all_indices - set.union(*(record[0] for record in records))
            for outcome, indices, destination in (
                    ("success", common, successes),
                    ("failure", never, failures)):
                for index in indices:
                    mean_lift = statistics.mean(record[1][index]
                                                for record in records)
                    if -0.03 <= mean_lift <= 0.40:
                        destination.append({
                            "category": category,
                            "outcome": outcome,
                            "object_id": object_id,
                            "trajectory_index": int(index),
                            "mean_maximum_lift_m": float(mean_lift),
                            "stable_across_repeats": True,
                        })
        if not successes or not failures:
            raise RuntimeError(f"No stable success/failure for {category}")
        cases.append(max(successes, key=lambda item: item[
            "mean_maximum_lift_m"]))
        cases.append(max(failures, key=lambda item: item[
            "mean_maximum_lift_m"]))
    result = {
        "source": str(SUMMARY),
        "selection_rule": (
            "Highest mean-lift trajectory with the same outcome in all three "
            "locked routed-teacher repeats."),
        "cases": cases,
    }
    path = OUTPUT / "final_routed_cases.yaml"
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(result, handle, allow_unicode=True, sort_keys=False)
    return path


def main():
    if load(SUMMARY).get("post_evaluation_training_allowed") is not False:
        raise RuntimeError("Final summary is not report-only")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows, run_objects = summarize()
    outputs = (write_summary(rows), plot_methods(rows),
               plot_categories(run_objects), stable_cases())
    for path in outputs:
        print(path.resolve())


if __name__ == "__main__":
    main()
