"""Record report cases and keep only outcomes verified in the actual replay.

Candidate trajectories are ranked by success frequency in the three locked
final evaluations. A success video is accepted only when the same single-env
replay used for recording reaches the official success state and remains
visibly lifted at the end. Rejected attempts are retained for auditing.
"""

import argparse
from collections import defaultdict
from pathlib import Path
import shutil
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
FINAL_RESULTS = ROOT / "custom_tools/results/taskid_locked_final_holdout_v1"
LOCK = ROOT / "custom_tools/configs/taskid_final_model_lock_v1.yaml"
TRAJECTORY_ROOT = (
    ROOT / "dexgrasp/dataset/scaled_category_final_v1_preprocessed")
RESIDUAL_CONFIG = ROOT / "custom_tools/configs/residual_ppo_stage1.yaml"
CATEGORIES = ("bottle", "mug", "bowl", "camera")


def load_yaml(path):
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def candidates():
    scores = defaultdict(int)
    counts = {}
    for seed in (2025, 2026, 2027):
        result = load_yaml(FINAL_RESULTS / ("seed{}".format(seed))
                           / "temporal3.yaml")["checkpoint_results"][0]
        for item in result["objects"]:
            object_id = item["object_id"]
            counts[object_id] = int(item["trajectory_count"])
            for index in item["official_peak_success_local_indices"]:
                scores[(object_id, int(index))] += 1

    by_category = {}
    for category in CATEGORIES:
        category_objects = sorted(
            object_id for object_id in counts
            if object_id.split("-", 2)[1] == category)
        ranked = []
        for object_id in category_objects:
            for index in range(counts[object_id]):
                ranked.append({
                    "object_id": object_id,
                    "trajectory_index": index,
                    "batch_success_seeds": scores[(object_id, index)],
                })
        ranked.sort(key=lambda row: (
            -row["batch_success_seeds"], row["object_id"],
            row["trajectory_index"]))
        by_category[category] = ranked
    return by_category


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="custom_tools/results/taskid_verified_report_renders_v2")
    parser.add_argument("--successes-per-category", type=int, default=2)
    parser.add_argument("--failures-per-category", type=int, default=1)
    parser.add_argument("--max-attempts-per-category", type=int, default=20)
    parser.add_argument("--min-maximum-lift-m", type=float, default=0.10)
    parser.add_argument("--min-final-lift-m", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--capture-stride", type=int, default=2)
    parser.add_argument("--min-free-vram-mb", type=int, default=4500)
    cli = parser.parse_args()

    lock = load_yaml(LOCK)
    model = lock["models"]["primary"]
    checkpoint = ROOT / model["checkpoint"]
    bc_config = ROOT / model["config"]
    output = (ROOT / cli.output_dir).resolve()
    attempts_dir = output / "attempts"
    accepted_dir = output / "accepted"
    attempts_dir.mkdir(parents=True, exist_ok=True)
    accepted_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "policy": "locked_temporal3",
        "acceptance": {
            "official_success_required": True,
            "minimum_maximum_lift_m": cli.min_maximum_lift_m,
            "minimum_final_lift_m": cli.min_final_lift_m,
        },
        "cases": [],
    }
    for category, rows in candidates().items():
        success_count = 0
        failure_count = 0
        for attempt_number, row in enumerate(
                rows[:cli.max_attempts_per_category], 1):
            if (success_count >= cli.successes_per_category
                    and failure_count >= cli.failures_per_category):
                break
            tag = "{}_{:02d}_seedhits{}_traj{}".format(
                category, attempt_number, row["batch_success_seeds"],
                row["trajectory_index"])
            capture_dir = attempts_dir / tag
            result_path = attempts_dir / (tag + ".yaml")
            command = [
                sys.executable, "-u",
                str(ROOT / "custom_tools/evaluate_residual_ppo.py"),
                "--object-id", row["object_id"],
                "--trajectory-root", str(TRAJECTORY_ROOT),
                "--trajectory-indices", str(row["trajectory_index"]),
                "--residual-config", str(RESIDUAL_CONFIG),
                "--bc-checkpoint", str(checkpoint),
                "--bc-config", str(bc_config),
                "--zero-residual",
                "--seed", str(cli.seed),
                "--capture-dir", str(capture_dir),
                "--capture-stride", str(cli.capture_stride),
                "--min-free-vram-mb", str(cli.min_free_vram_mb),
                "--output", str(result_path),
            ]
            print("[{}] attempt {}/{}: {}".format(
                category, attempt_number, cli.max_attempts_per_category, tag),
                flush=True)
            subprocess.run(command, cwd=str(ROOT), check=True)
            item = load_yaml(result_path)["objects"][0]
            official = int(item["official_peak_success_count"]) == 1
            maximum_lift = float(
                item["diagnostic_maximum_lift_m_by_trajectory"][0])
            final_lift = float(
                item["diagnostic_final_lift_m_by_trajectory"][0])
            verified_success = (
                official
                and maximum_lift >= cli.min_maximum_lift_m
                and final_lift >= cli.min_final_lift_m)
            clear_failure = (not official and maximum_lift < 0.03)
            accepted_as = None
            if verified_success and success_count < cli.successes_per_category:
                success_count += 1
                accepted_as = "success_{:02d}".format(success_count)
            elif clear_failure and failure_count < cli.failures_per_category:
                failure_count += 1
                accepted_as = "failure_{:02d}".format(failure_count)

            record = dict(row)
            record.update({
                "category": category,
                "actual_official_success": official,
                "maximum_lift_m": maximum_lift,
                "final_lift_m": final_lift,
                "accepted_as": accepted_as,
                "attempt_video": str(capture_dir / "env000.mp4"),
            })
            if accepted_as:
                destination = accepted_dir / (
                    "{}_{}_{}".format(category, accepted_as, tag))
                shutil.copytree(capture_dir, destination)
                shutil.copy2(result_path, destination.with_suffix(".yaml"))
                record["accepted_video"] = str(destination / "env000.mp4")
                print("  ACCEPT {} max_lift={:.3f} final_lift={:.3f}".format(
                    accepted_as, maximum_lift, final_lift), flush=True)
            else:
                print("  reject official={} max_lift={:.3f} final_lift={:.3f}"
                      .format(official, maximum_lift, final_lift), flush=True)
            summary["cases"].append(record)

    summary["complete"] = all(
        sum(row["accepted_as"] is not None
            and row["accepted_as"].startswith("success")
            for row in summary["cases"] if row["category"] == category)
        >= cli.successes_per_category
        for category in CATEGORIES)
    with (output / "summary.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(summary, handle, sort_keys=False)
    print("VERIFIED_RECORDING_COMPLETE={} output={}".format(
        summary["complete"], output), flush=True)


if __name__ == "__main__":
    main()
