"""Freeze a balanced Temporal3 residual-PPO curriculum from training rollouts."""

import argparse
from pathlib import Path

import yaml


def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def closest(candidates, target):
    return min(candidates, key=lambda item: (abs(item[1] - target), item[0]))


def select_object(item):
    source_indices = [int(value) for value in item["trajectory_indices"]]
    lifts = [float(value) for value in item[
        "diagnostic_maximum_lift_m_by_trajectory"]]
    if len(source_indices) != len(lifts):
        raise ValueError("{} index/lift lengths differ".format(item["object_id"]))
    lift_by_source = dict(zip(source_indices, lifts))

    successful_local = [
        int(value) for value in item["diagnostic_ever_success_indices"]]
    successes = {
        source_indices[local_index] for local_index in successful_local}
    successful = [
        (index, lift_by_source[index]) for index in sorted(successes)]
    if len(successful) < 2:
        raise ValueError("{} has fewer than two genuine successes".format(
            item["object_id"]))

    moderate = closest(successful, 0.30)
    strong = closest(
        [value for value in successful if value != moderate], 0.50)
    failures = sorted(
        [(index, lift) for index, lift in lift_by_source.items()
         if index not in successes and 0.0 <= lift <= 0.30],
        key=lambda value: (value[1], value[0]))
    if len(failures) < 2:
        raise ValueError("{} has fewer than two valid failures".format(
            item["object_id"]))
    near_miss = failures[-1]
    remaining = failures[:-1]
    median_failure = remaining[(len(remaining) - 1) // 2]
    selected = [moderate, strong, near_miss, median_failure]
    if len({value[0] for value in selected}) != 4:
        raise RuntimeError("{} did not produce four unique samples".format(
            item["object_id"]))
    return {
        "indices": [value[0] for value in selected],
        "anchor_indices": [moderate[0], strong[0]],
        "anchor_flags": [True, True, False, False],
        "roles": {
            "success_anchor_moderate": moderate[0],
            "success_anchor_strong": strong[0],
            "failure_near_miss": near_miss[0],
            "failure_median": median_failure[0],
        },
        "maximum_lift_m": {
            str(index): float(lift) for index, lift in selected},
    }


def main():
    cli = parse_cli()
    audit_path = Path(cli.audit).expanduser().resolve()
    output_path = Path(cli.output).expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(output_path)
    with audit_path.open("r", encoding="utf-8") as handle:
        audit = yaml.safe_load(handle)
    selections = {
        item["object_id"]: select_object(item) for item in audit["objects"]}
    object_ids = [item["object_id"] for item in audit["objects"]]
    result = {
        "status": "frozen_stage1_selection",
        "stage": "temporal3_behavior_anchored_gated_residual",
        "trajectory_root": audit["trajectory_root"],
        "source_audit": str(audit_path),
        "uses_unseen_test_objects": False,
        "selection_rule": (
            "Exactly four training trajectories per object: two trajectories "
            "that genuinely reached the official success condition, one "
            "highest-lift physically valid failure, and one median-lift "
            "failure. Never mark a failed trajectory as an anchor."),
        "object_ids": object_ids,
        "trajectory_indices_by_object": {
            key: value["indices"] for key, value in selections.items()},
        "anchor_indices_by_object": {
            key: value["anchor_indices"] for key, value in selections.items()},
        "anchor_flags_by_object": {
            key: value["anchor_flags"] for key, value in selections.items()},
        "selection_details": selections,
        "total_environments": 4 * len(object_ids),
        "total_anchor_environments": 2 * len(object_ids),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(result, handle, allow_unicode=True, sort_keys=False)
    print("TEMPORAL3_CURRICULUM=FROZEN")
    print("objects={} environments={} anchors={}".format(
        len(object_ids), result["total_environments"],
        result["total_anchor_environments"]))
    print("Saved: {}".format(output_path))


if __name__ == "__main__":
    main()
