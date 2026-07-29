"""Repeat paired BC versus framewise-k1 retrieval on held-out motions."""

import argparse
from datetime import datetime
from pathlib import Path
import statistics
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "dexgrasp/dataset/bc_multicategory_train"
VALID = ROOT / "dexgrasp/dataset/bc_multicategory_valid"
OUTPUT = ROOT / "custom_tools/results/retrieval_paired_repeats"

OBJECTS = {
    "bottle_dc005": (
        "core-bottle-dc005c019fbfb32c90071898148dca0e",
        ROOT / "custom_tools/runs/bc/category_expert_bottle_noise005_soup_seed2025_e40_v1/epoch=039-step=2560.ckpt"),
    "mug_b4ae": (
        "core-mug-b4ae56d6638d5338de671f28c83d2dcb",
        ROOT / "custom_tools/runs/bc/model_soups/noise005_s2025_s2026_weighted2to1.ckpt"),
    "camera_82819": (
        "core-camera-82819e1201d2dc583a3e53900c6cbba",
        ROOT / "custom_tools/runs/bc/category_expert_camera_noise005_soup_seed2025_e40_v1/epoch=009-step=500.ckpt"),
}


def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--min-free-vram-mb", type=int, default=4500)
    return parser.parse_args()


def main():
    cli = parse_cli()
    if cli.repeats < 2:
        raise ValueError("Use at least two repeats")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for repeat in range(1, cli.repeats + 1):
        for label, (object_id, checkpoint) in OBJECTS.items():
            output = OUTPUT / "{}_r{}.yaml".format(label, repeat)
            if not output.is_file():
                command = [
                    sys.executable,
                    ROOT / "custom_tools/evaluate_phase_retrieval_bc.py",
                    "--object-id", object_id,
                    "--reference-root", TRAIN,
                    "--evaluation-root", VALID,
                    "--bc-checkpoint", checkpoint,
                    "--candidate-profile", "paired",
                    "--output", output,
                    "--seed", "2025",
                    "--min-free-vram-mb", str(cli.min_free_vram_mb),
                ]
                print("RUN repeat={} object={}".format(repeat, label), flush=True)
                subprocess.run(list(map(str, command)), cwd=str(ROOT), check=True)
            else:
                print("REUSE {}".format(output), flush=True)
            with output.open(encoding="utf-8") as handle:
                result = yaml.safe_load(handle)
            by_method = {item["label"]: item for item in result["ranked_results"]}
            for method in ("bc", "framewise_k1"):
                rows.append({
                    "repeat": repeat,
                    "object": label,
                    "method": method,
                    "success_count": by_method[method][
                        "official_peak_success_count"],
                    "trajectory_count": result["evaluation_trajectory_count"],
                    "mean_lift_m": by_method[method]["mean_maximum_lift_m"],
                })
    summary = []
    for object_label in OBJECTS:
        for method in ("bc", "framewise_k1"):
            selected = [row for row in rows if row["object"] == object_label
                        and row["method"] == method]
            counts = [row["success_count"] for row in selected]
            lifts = [row["mean_lift_m"] for row in selected]
            total = selected[0]["trajectory_count"]
            summary.append({
                "object": object_label,
                "method": method,
                "repeated_success_counts": counts,
                "mean_success_count": statistics.mean(counts),
                "mean_success_rate": statistics.mean(counts) / total,
                "success_count_population_std": statistics.pstdev(counts),
                "mean_lift_m": statistics.mean(lifts),
            })
    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "purpose": "Paired repeatability check before locking focused policy.",
        "official_success_definition_changed": False,
        "repeats": cli.repeats,
        "rows": rows,
        "summary": summary,
    }
    summary_path = OUTPUT / "summary.yaml"
    with summary_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(report, handle, allow_unicode=True, sort_keys=False)
    print("RETRIEVAL_PAIRED_REPEATS=COMPLETE", flush=True)
    print("SUMMARY={}".format(summary_path), flush=True)


if __name__ == "__main__":
    main()
