"""Evaluate an action blend with one fresh Isaac Gym process per object."""

import argparse
import collections
from datetime import datetime
from pathlib import Path
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]


def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--online-checkpoint", required=True)
    parser.add_argument("--online-config", required=True)
    parser.add_argument("--temporal-checkpoint", required=True)
    parser.add_argument("--temporal-config", required=True)
    parser.add_argument("--temporal-weight", type=float, required=True)
    parser.add_argument("--residual-config", required=True)
    parser.add_argument("--trajectory-root", required=True)
    parser.add_argument("--object-selection", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--min-free-vram-mb", type=int, default=4500)
    parser.add_argument("--max-attempts", type=int, default=5)
    return parser.parse_args()


def absolute(path):
    return Path(path).expanduser().resolve()


def load_yaml(path):
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def verify_worker(worker, cli):
    result = worker.get("blend_result", {})
    if float(result.get("temporal_weight", -1.0)) != cli.temporal_weight:
        raise RuntimeError("Existing worker has a different blend weight")
    if Path(result["online_checkpoint"]).resolve() != absolute(
            cli.online_checkpoint):
        raise RuntimeError("Existing worker has a different Online-R1 model")
    if Path(result["temporal_checkpoint"]).resolve() != absolute(
            cli.temporal_checkpoint):
        raise RuntimeError("Existing worker has a different Temporal3 model")
    return result


def main():
    cli = parse_cli()
    if cli.max_attempts < 1:
        raise ValueError("--max-attempts must be positive")
    if not 0.0 <= cli.temporal_weight <= 1.0:
        raise ValueError("--temporal-weight must be in [0, 1]")
    output = absolute(cli.output)
    if output.exists():
        raise FileExistsError(output)

    selection_path = absolute(cli.object_selection)
    selection = load_yaml(selection_path)
    object_ids = list(selection["object_ids"])
    per_object_dir = output.parent / (output.stem + "_objects")
    per_object_dir.mkdir(parents=True, exist_ok=True)
    object_results = []

    for index, object_id in enumerate(object_ids, 1):
        object_output = per_object_dir / (object_id + ".yaml")
        if object_output.exists():
            worker = load_yaml(object_output)
            result = verify_worker(worker, cli)
            if result["objects"][0]["object_id"] != object_id:
                raise RuntimeError(
                    "Existing worker object mismatch: {}".format(
                        object_output))
            print("action blend {}/{}: {} (reuse)".format(
                index, len(object_ids), object_id), flush=True)
            object_results.append(result["objects"][0])
            continue

        command = [
            sys.executable, "-u",
            str(ROOT / "custom_tools/evaluate_action_blend_batched.py"),
            "--online-checkpoint", str(absolute(cli.online_checkpoint)),
            "--online-config", str(absolute(cli.online_config)),
            "--temporal-checkpoint", str(
                absolute(cli.temporal_checkpoint)),
            "--temporal-config", str(absolute(cli.temporal_config)),
            "--temporal-weight", str(cli.temporal_weight),
            "--residual-config", str(absolute(cli.residual_config)),
            "--trajectory-root", str(absolute(cli.trajectory_root)),
            "--object-selection", str(selection_path),
            "--object-id", object_id,
            "--output", str(object_output),
            "--seed", str(cli.seed),
            "--min-free-vram-mb", str(cli.min_free_vram_mb),
        ]
        print("action blend {}/{}: {}".format(
            index, len(object_ids), object_id), flush=True)
        for attempt in range(1, cli.max_attempts + 1):
            completed = subprocess.run(
                command, cwd=str(ROOT), check=False)
            if completed.returncode == 0:
                break
            print("{} attempt {}/{} failed".format(
                object_id, attempt, cli.max_attempts), flush=True)
        else:
            raise RuntimeError("Worker failed: {}".format(object_id))
        result = verify_worker(load_yaml(object_output), cli)
        object_results.append(result["objects"][0])

    total_success = sum(
        item["official_peak_success_count"] for item in object_results)
    total_trajectories = sum(
        item["trajectory_count"] for item in object_results)
    category_rates = collections.defaultdict(list)
    for item in object_results:
        category_rates[item["category"]].append(
            item["official_peak_success_rate"])
    blend_result = {
        "temporal_weight": float(cli.temporal_weight),
        "online_checkpoint": str(absolute(cli.online_checkpoint)),
        "temporal_checkpoint": str(absolute(cli.temporal_checkpoint)),
        "total_success_count": total_success,
        "total_trajectory_count": total_trajectories,
        "overall_official_peak_success_rate": (
            total_success / total_trajectories),
        "macro_official_peak_success_rate": (
            sum(item["official_peak_success_rate"]
                for item in object_results) / len(object_results)),
        "macro_mean_maximum_lift_m": (
            sum(item["mean_maximum_lift_m"]
                for item in object_results) / len(object_results)),
        "macro_failure_rate": (
            sum(item["failure_rate"]
                for item in object_results) / len(object_results)),
        "category_macro_success_rates": {
            category: sum(values) / len(values)
            for category, values in sorted(category_rates.items())
        },
        "objects": object_results,
    }
    aggregate = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "evaluation_mode": "fresh_process_per_object_action_blend",
        "success_metric": "official_peak_per_object",
        "official_success_definition_changed": False,
        "formal_final_holdout_result": False,
        "final_holdout_accessed": False,
        "seed": int(cli.seed),
        "trajectory_root": str(absolute(cli.trajectory_root)),
        "object_ids": object_ids,
        "blend_result": blend_result,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            aggregate, handle, allow_unicode=True, sort_keys=False)
    print("ISOLATED_ACTION_BLEND_EVALUATION=COMPLETE", flush=True)


if __name__ == "__main__":
    main()
