"""Build the final ShadowHand ablation table and presentation figure."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "custom_tools/results/restart_shadow_final_ablation_v1"
SUMMARY = RESULT / "summary.yaml"
ORDER = (
    ("released_official_bc", "Released BC"),
    ("matched_official_bc", "Matched BC"),
    ("temporal3", "Temporal3"),
    ("temporal3_demo80", "+ Demo80"),
    ("chunk8_no_ensemble", "+ Chunk8"),
    ("chunk8_equal", "+ Ensemble"),
    ("chunk8_equal_lift", "+ Lift"),
)


def main():
    data = yaml.safe_load(SUMMARY.read_text(encoding="utf-8"))
    rows = []
    for key, display in ORDER:
        item = data["methods"][key]
        rows.append({
            "method": key,
            "display_name": display,
            "official_macro_mean_percent": 100.0 * item["official_macro_rate"]["mean"],
            "official_macro_std_percent": 100.0 * item["official_macro_rate"]["std"],
            "official_overall_mean_percent": 100.0 * item["official_overall_rate"]["mean"],
            "stable_official_mean_percent": 100.0 * item["stable_official_rate"]["mean"],
            "seed2025_count": item["official_counts"][0],
            "seed2026_count": item["official_counts"][1],
            "seed2027_count": item["official_counts"][2],
        })

    with (RESULT / "final_ablation_table.csv").open(
            "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    labels = [row["display_name"] for row in rows]
    official = np.asarray([row["official_macro_mean_percent"] for row in rows])
    errors = np.asarray([row["official_macro_std_percent"] for row in rows])
    stable = np.asarray([row["stable_official_mean_percent"] for row in rows])
    colors = ["#A9B4C2", "#7C8DA5", "#5276A7", "#7396C8",
              "#55A6A6", "#24A39A", "#E28E4A"]

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig, axes = plt.subplots(2, 1, figsize=(10.8, 7.2), sharex=True)
    x = np.arange(len(rows))
    axes[0].bar(x, official, yerr=errors, capsize=3, color=colors,
                edgecolor="white", linewidth=0.8)
    axes[0].set_ylabel("Official macro success (%)")
    axes[0].set_ylim(0, 45)
    axes[0].grid(axis="y", alpha=0.22, linewidth=0.8)
    for index, value in enumerate(official):
        axes[0].text(index, value + errors[index] + 0.8,
                     "{:.2f}".format(value), ha="center", va="bottom",
                     fontsize=10)

    axes[1].bar(x, stable, color=colors, edgecolor="white", linewidth=0.8)
    axes[1].set_ylabel("Stable official success (%)")
    axes[1].set_ylim(0, 45)
    axes[1].set_xticks(x, labels, rotation=18, ha="right")
    axes[1].grid(axis="y", alpha=0.22, linewidth=0.8)
    for index, value in enumerate(stable):
        axes[1].text(index, value + 0.8, "{:.2f}".format(value),
                     ha="center", va="bottom", fontsize=10)

    fig.suptitle("ShadowHand Final8: baseline and component ablation",
                 fontsize=15, fontweight="semibold")
    fig.text(0.5, 0.005,
             "Mean over simulation seeds 2025/2026/2027; 216 trajectories per seed",
             ha="center", fontsize=9, color="#4B5563")
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(RESULT / "final_ablation.png", dpi=240, bbox_inches="tight")
    fig.savefig(RESULT / "final_ablation.svg", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
