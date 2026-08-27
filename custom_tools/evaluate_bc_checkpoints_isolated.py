"""Evaluate BC checkpoints with one fresh Isaac Gym process per object."""

import argparse
import collections
from datetime import datetime
from pathlib import Path
import subprocess
import sys

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--ensemble-checkpoint", default="")
    parser.add_argument("--ensemble-second-weight", type=float, default=0.5)
    parser.add_argument("--bc-config", required=True)
    parser.add_argument("--residual-config", required=True)
    parser.add_argument("--trajectory-root", required=True)
    parser.add_argument("--object-selection", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--min-free-vram-mb", type=int, default=4500)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--meshdata-root", default="")
    parser.add_argument("--strict-lift-threshold", type=float, default=0.30)
    parser.add_argument("--strict-hold-steps", type=int, default=20)
    parser.add_argument("--strict-min-contact-count", type=int, default=2)
    parser.add_argument("--strict-max-terminal-drop", type=float, default=0.03)
    parser.add_argument("--policy-motion-steps", type=int, default=70)
    parser.add_argument("--temporal-ensemble-decay", type=float, default=None)
    parser.add_argument("--diffusion-residual-scale", type=float, default=None)
    parser.add_argument(
        "--full-observation-residual-scale", type=float, default=None)
    parser.add_argument("--dynamic-candidate-routing", action="store_true")
    parser.add_argument("--late-lift-z-boost", type=float, default=0.0)
    parser.add_argument("--late-lift-start-step", type=int, default=40)
    parser.add_argument("--late-lift-contact-gate", type=int, default=0)
    parser.add_argument("--hold-grip-scale", type=float, default=0.0)
    parser.add_argument("--hold-grip-reference-step", type=int, default=40)
    parser.add_argument("--max-trajectories-per-object", type=int, default=0)
    parser.add_argument("--use-expert-actions", action="store_true")
    parser.add_argument(
        "--allow-stateful-multicheckpoint", action="store_true",
        help="Diagnostic only: repeated resets are not selection-grade deterministic.")
    return parser.parse_args()


def absolute(path):
    return Path(path).expanduser().resolve()


def aggregate_checkpoint(checkpoint, object_results):
    objects = [item["objects"][0] for item in object_results]
    total_success = sum(x["official_peak_success_count"] for x in objects)
    total_strict = sum(x["strict_terminal_success_count"] for x in objects)
    total_stable = sum(x["stable_official_success_count"] for x in objects)
    total_goal_center = sum(
        x["goal_center_30cm_success_count"] for x in objects)
    total_trajectories = sum(x["trajectory_count"] for x in objects)
    category_rates = collections.defaultdict(list)
    for item in objects:
        category_rates[item["category"]].append(
            item["official_peak_success_rate"])
    return {
        "checkpoint": checkpoint,
        "checkpoint_epoch": object_results[0]["checkpoint_epoch"],
        "total_success_count": total_success,
        "total_strict_terminal_success_count": total_strict,
        "total_stable_official_success_count": total_stable,
        "total_goal_center_30cm_success_count": total_goal_center,
        "total_trajectory_count": total_trajectories,
        "overall_official_peak_success_rate": total_success / total_trajectories,
        "overall_strict_terminal_success_rate": total_strict / total_trajectories,
        "overall_stable_official_success_rate": (
            total_stable / total_trajectories),
        "overall_goal_center_30cm_success_rate": (
            total_goal_center / total_trajectories),
        "macro_official_peak_success_rate": sum(
            x["official_peak_success_rate"] for x in objects) / len(objects),
        "macro_strict_terminal_success_rate": sum(
            x["strict_terminal_success_rate"] for x in objects) / len(objects),
        "strict_terminal_definition": object_results[0][
            "strict_terminal_definition"],
        "goal_center_30cm_diagnostic": object_results[0][
            "goal_center_30cm_diagnostic"],
        "action_source": object_results[0]["action_source"],
        "macro_mean_maximum_lift_m": sum(
            x["mean_maximum_lift_m"] for x in objects) / len(objects),
        "macro_failure_rate": sum(
            x["failure_rate"] for x in objects) / len(objects),
        "category_macro_success_rates": {
            key: sum(values) / len(values)
            for key, values in sorted(category_rates.items())},
        "objects": objects,
    }


def main():
    cli = parse_cli()
    if cli.max_attempts < 1:
        raise ValueError("--max-attempts must be positive")
    output = absolute(cli.output)
    if output.exists():
        raise FileExistsError(output)
    checkpoints = [str(absolute(path)) for path in cli.checkpoint]
    ensemble_checkpoint = (
        str(absolute(cli.ensemble_checkpoint))
        if cli.ensemble_checkpoint else "")
    if ensemble_checkpoint and len(checkpoints) != 1:
        raise ValueError("Policy ensemble evaluation accepts one primary checkpoint")
    result_checkpoint_labels = list(checkpoints)
    if ensemble_checkpoint:
        result_checkpoint_labels = ["{}+{}@{:.2f}".format(
            checkpoints[0], ensemble_checkpoint,
            float(cli.ensemble_second_weight))]
    if len(checkpoints) > 1 and not cli.allow_stateful_multicheckpoint:
        raise ValueError(
            "Multiple checkpoints in one persistent simulator are diagnostic only; "
            "use one checkpoint per run or pass --allow-stateful-multicheckpoint.")
    selection_path = absolute(cli.object_selection)
    with selection_path.open(encoding="utf-8") as handle:
        selection = yaml.safe_load(handle)
    object_ids = list(selection["object_ids"])
    per_object_dir = output.parent / (output.stem + "_objects")
    per_object_dir.mkdir(parents=True, exist_ok=True)
    worker_outputs = []
    for index, object_id in enumerate(object_ids, 1):
        object_output = per_object_dir / (object_id + ".yaml")
        if object_output.exists():
            with object_output.open(encoding="utf-8") as handle:
                worker = yaml.safe_load(handle)
            actual = [item["checkpoint"] for item in worker["checkpoint_results"]]
            if actual != result_checkpoint_labels:
                raise RuntimeError(
                    "Existing worker has a different checkpoint list: {}".format(
                        object_output))
            print("isolated checkpoint evaluation {}/{}: {} (reuse)".format(
                index, len(object_ids), object_id), flush=True)
            worker_outputs.append(worker)
            continue
        command = [
            sys.executable,
            str(REPO_ROOT / "custom_tools/evaluate_bc_checkpoints_batched.py"),
            "--bc-config", str(absolute(cli.bc_config)),
            "--residual-config", str(absolute(cli.residual_config)),
            "--trajectory-root", str(absolute(cli.trajectory_root)),
            "--object-selection", str(selection_path),
            "--object-id", object_id,
            "--seed", str(cli.seed),
            "--min-free-vram-mb", str(cli.min_free_vram_mb),
            "--output", str(object_output),
            "--strict-lift-threshold", str(cli.strict_lift_threshold),
            "--strict-hold-steps", str(cli.strict_hold_steps),
            "--strict-min-contact-count", str(cli.strict_min_contact_count),
            "--strict-max-terminal-drop", str(cli.strict_max_terminal_drop),
            "--policy-motion-steps", str(cli.policy_motion_steps),
            "--late-lift-z-boost", str(cli.late_lift_z_boost),
            "--late-lift-start-step", str(cli.late_lift_start_step),
            "--late-lift-contact-gate", str(cli.late_lift_contact_gate),
            "--hold-grip-scale", str(cli.hold_grip_scale),
            "--hold-grip-reference-step", str(cli.hold_grip_reference_step),
            "--max-trajectories-per-object",
            str(cli.max_trajectories_per_object),
        ]
        if ensemble_checkpoint:
            command.extend([
                "--ensemble-checkpoint", ensemble_checkpoint,
                "--ensemble-second-weight", str(cli.ensemble_second_weight),
            ])
        if cli.meshdata_root:
            command.extend([
                "--meshdata-root", str(absolute(cli.meshdata_root))])
        if cli.temporal_ensemble_decay is not None:
            command.extend([
                "--temporal-ensemble-decay",
                str(cli.temporal_ensemble_decay)])
        if cli.diffusion_residual_scale is not None:
            command.extend([
                "--diffusion-residual-scale",
                str(cli.diffusion_residual_scale)])
        if cli.full_observation_residual_scale is not None:
            command.extend([
                "--full-observation-residual-scale",
                str(cli.full_observation_residual_scale)])
        if cli.dynamic_candidate_routing:
            command.append("--dynamic-candidate-routing")
        if cli.use_expert_actions:
            command.append("--use-expert-actions")
        for checkpoint in checkpoints:
            command.extend(["--checkpoint", checkpoint])
        print("isolated checkpoint evaluation {}/{}: {}".format(
            index, len(object_ids), object_id), flush=True)
        for attempt in range(1, cli.max_attempts + 1):
            completed = subprocess.run(
                command, cwd=str(REPO_ROOT), check=False)
            if completed.returncode == 0:
                break
            print("{} attempt {}/{} failed".format(
                object_id, attempt, cli.max_attempts), flush=True)
        else:
            raise RuntimeError("Worker failed: {}".format(object_id))
        with object_output.open(encoding="utf-8") as handle:
            worker_outputs.append(yaml.safe_load(handle))

    checkpoint_results = []
    for checkpoint_index, checkpoint in enumerate(result_checkpoint_labels):
        checkpoint_results.append(aggregate_checkpoint(
            checkpoint,
            [worker["checkpoint_results"][checkpoint_index]
             for worker in worker_outputs]))
    aggregate = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "evaluation_mode": "fresh_process_per_object_multi_checkpoint",
        "success_metric": "stable_official_success",
        "official_peak_metric_retained": True,
        "official_success_definition_changed": False,
        "seed": cli.seed,
        "trajectory_root": str(absolute(cli.trajectory_root)),
        "object_ids": object_ids,
        "checkpoint_results": checkpoint_results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(aggregate, handle, allow_unicode=True, sort_keys=False)
    print("ISOLATED_MULTI_CHECKPOINT_EVALUATION=COMPLETE", flush=True)


if __name__ == "__main__":
    main()
