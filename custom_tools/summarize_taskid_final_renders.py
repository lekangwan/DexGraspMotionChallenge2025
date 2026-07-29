"""Summarize actual render outcomes and build a report-ready case montage."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import yaml


ROOT = Path(__file__).resolve().parents[1]
RENDERS = ROOT / "custom_tools/results/taskid_final_report_renders_v1"
OUTPUT = ROOT / "custom_tools/results/taskid_final_report_assets_v1"


def load(path):
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def collect_rows():
    rows = []
    for path in sorted(RENDERS.glob("*.yaml")):
        result = load(path)
        item = result["objects"][0]
        success = int(item["official_peak_success_count"]) == 1
        case_dir = RENDERS / path.stem
        image = case_dir / (
            "env000_success.png" if success else "env000_final.png")
        if not image.is_file():
            raise FileNotFoundError(image)
        rows.append({
            "case": path.stem,
            "category": item["object_id"].split("-", 2)[1],
            "object_id": item["object_id"],
            "trajectory_index": item["trajectory_indices"][0],
            "actual_outcome": "success" if success else "failure",
            "maximum_lift_m": item[
                "diagnostic_maximum_lift_m_by_trajectory"][0],
            "explicit_failure": bool(item["diagnostic_failure_rate"] > 0),
            "result_yaml": str(path),
            "representative_image": str(image),
        })
    if len(rows) != 8:
        raise RuntimeError("Expected eight rendered cases")
    return rows


def write_rows(rows):
    with (OUTPUT / "final_render_outcomes.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "status": "complete",
        "source": str(RENDERS),
        "labels_use_actual_single_environment_replay": True,
        "success_count": sum(
            row["actual_outcome"] == "success" for row in rows),
        "failure_count": sum(
            row["actual_outcome"] == "failure" for row in rows),
        "cases": rows,
    }
    with (OUTPUT / "final_render_outcomes.yaml").open(
            "w", encoding="utf-8") as handle:
        yaml.safe_dump(summary, handle, allow_unicode=True, sort_keys=False)


def select_montage(rows):
    successes = [
        row for row in rows if row["actual_outcome"] == "success"]
    if len(successes) < 2:
        raise RuntimeError("Need two actual success renders")
    selected = sorted(successes, key=lambda row: row["category"])[:2]
    for category in ("bowl", "camera"):
        failures = [
            row for row in rows
            if row["category"] == category
            and row["actual_outcome"] == "failure"]
        if not failures:
            raise RuntimeError("Missing {} failure render".format(category))
        selected.append(max(
            failures,
            key=lambda row: (
                row["explicit_failure"], row["maximum_lift_m"])))
    return selected


def make_montage(rows):
    selected = select_montage(rows)
    figure, axes = plt.subplots(2, 2, figsize=(9.2, 6.8))
    for axis, row in zip(axes.flat, selected):
        axis.imshow(mpimg.imread(row["representative_image"]))
        axis.axis("off")
        axis.set_title(
            "{} {} — max lift {:.1f} cm".format(
                row["category"].capitalize(), row["actual_outcome"],
                100 * float(row["maximum_lift_m"])),
            fontsize=11)
    figure.tight_layout()
    figure.savefig(OUTPUT / "representative_render_cases.png", dpi=220)
    plt.close(figure)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = collect_rows()
    write_rows(rows)
    make_montage(rows)
    print(
        "TASKID_FINAL_RENDER_SUMMARY=COMPLETE successes={} failures={}".format(
            sum(row["actual_outcome"] == "success" for row in rows),
            sum(row["actual_outcome"] == "failure" for row in rows)))
    print("OUTPUT={}".format(OUTPUT))


if __name__ == "__main__":
    main()
