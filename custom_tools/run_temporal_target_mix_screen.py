"""Successive-halving screen for three Temporal3 supervision mixtures.

Training always uses the complete frozen dataset.  Isaac Gym evaluation is
staged over three disjoint four-object development subsets to save time.
"""

import argparse
import collections
import json
from pathlib import Path
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT / "custom_tools/configs/"
    / "unified_student_taskid_temporal3_targetmix_v1.yaml"
)
INIT = (
    ROOT / "custom_tools/runs/bc/"
    / "unified_student_taskid_online_r1_frac025_seed2025_e10_v1/"
    / "epoch=001-step=2232.ckpt"
)
TRAJECTORY_ROOT = (
    ROOT / "dexgrasp/dataset/scaled_category_final_v1_preprocessed"
)
RESIDUAL_CONFIG = ROOT / "custom_tools/configs/residual_ppo_stage1.yaml"
PROTOCOL = (
    ROOT / "custom_tools/configs/scaled_evaluation_protocol_v1.json"
)
SELECTIONS = (
    ROOT / "custom_tools/configs/temporal_target_screen_round1_4.yaml",
    ROOT / "custom_tools/configs/temporal_target_screen_round2_4.yaml",
    ROOT / "custom_tools/configs/temporal_target_screen_round3_4.yaml",
)
CONTROL_T100 = (
    ROOT / "custom_tools/results/taskid_temporal3_development_v1/"
    / "temporal3_epoch04.yaml"
)
CONTROL_D30 = (
    ROOT / "custom_tools/results/"
    / "taskid_temporal3_demo30_development_v1/demo30_epoch04.yaml"
)
OUTPUT_ROOT = (
    ROOT / "custom_tools/results/temporal_target_mix_screen_v1"
)
REPEAT_MARGIN = 0.02

CANDIDATES = (
    {
        "label": "teacher50_demo50",
        "teacher_weight": 0.50,
        "run_name": (
            "unified_student_taskid_temporal3_t50_d50_seed2025_e4_v1"),
    },
    {
        "label": "teacher30_demo70",
        "teacher_weight": 0.30,
        "run_name": (
            "unified_student_taskid_temporal3_t30_d70_seed2025_e4_v1"),
    },
    {
        "label": "teacher00_demo100",
        "teacher_weight": 0.00,
        "run_name": (
            "unified_student_taskid_temporal3_t00_d100_seed2025_e4_v1"),
    },
)


def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-free-vram-mb", type=int, default=4500)
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_yaml(path):
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def checkpoint(candidate):
    run_dir = ROOT / "custom_tools/runs/bc" / candidate["run_name"]
    matches = list(run_dir.glob("epoch=003-step=*.ckpt"))
    if len(matches) != 1:
        raise RuntimeError(
            "Expected one epoch-4 checkpoint in {}".format(run_dir))
    return matches[0].resolve()


def train(cli, candidate):
    run_dir = ROOT / "custom_tools/runs/bc" / candidate["run_name"]
    last = run_dir / "last.ckpt"
    resource = run_dir / "resource_summary.yaml"
    if last.is_file() and resource.is_file():
        checkpoint(candidate)
        print("[REUSE] training {}".format(candidate["label"]), flush=True)
        return
    if list(run_dir.glob("*.ckpt")):
        raise RuntimeError(
            "Partial target-mixture run needs inspection: {}".format(
                run_dir))
    command = [
        sys.executable, "-u",
        str(ROOT / "custom_tools/train_bc.py"),
        "--config", str(CONFIG),
        "--run-name", candidate["run_name"],
        "--seed", "2025",
        "--num-epochs", "4",
        "--learning-rate", "2e-5",
        "--teacher-weight", str(candidate["teacher_weight"]),
        "--online-sample-fraction", "0.25",
        "--init-checkpoint", str(INIT),
        "--min-free-vram-mb", str(cli.min_free_vram_mb),
    ]
    print("RUN {}".format(" ".join(command)), flush=True)
    if not cli.dry_run:
        subprocess.run(command, cwd=str(ROOT), check=True)
        checkpoint(candidate)


def evaluation_result(path, expected_checkpoint=None):
    data = load_yaml(path)
    rows = data.get("checkpoint_results", [])
    if len(rows) != 1:
        raise RuntimeError("Expected one result in {}".format(path))
    if (
        expected_checkpoint is not None
        and Path(rows[0]["checkpoint"]).resolve()
        != expected_checkpoint.resolve()
    ):
        raise RuntimeError("Checkpoint mismatch in {}".format(path))
    return rows[0]


def evaluate(cli, candidate, round_index):
    model = checkpoint(candidate)
    output = (
        OUTPUT_ROOT / candidate["label"]
        / "round{}.yaml".format(round_index))
    if output.is_file():
        evaluation_result(output, model)
        print("[REUSE] {} round{}".format(
            candidate["label"], round_index), flush=True)
        return output
    command = [
        sys.executable, "-u",
        str(ROOT / "custom_tools/evaluate_bc_checkpoints_isolated.py"),
        "--checkpoint", str(model),
        "--bc-config", str(CONFIG),
        "--residual-config", str(RESIDUAL_CONFIG),
        "--trajectory-root", str(TRAJECTORY_ROOT),
        "--object-selection", str(SELECTIONS[round_index - 1]),
        "--output", str(output),
        "--seed", "2025",
        "--min-free-vram-mb", str(cli.min_free_vram_mb),
        "--max-attempts", str(cli.max_attempts),
    ]
    print("RUN {}".format(" ".join(command)), flush=True)
    subprocess.run(command, cwd=str(ROOT), check=True)
    evaluation_result(output, model)
    return output


def aggregate_objects(objects):
    if not objects:
        raise ValueError("Cannot aggregate an empty object list")
    object_ids = [item["object_id"] for item in objects]
    if len(object_ids) != len(set(object_ids)):
        raise RuntimeError("Duplicate objects in cumulative screen")
    category_rates = collections.defaultdict(list)
    for item in objects:
        category_rates[item["category"]].append(
            float(item["official_peak_success_rate"]))
    success = sum(
        int(item["official_peak_success_count"]) for item in objects)
    trajectories = sum(int(item["trajectory_count"]) for item in objects)
    return {
        "object_count": len(objects),
        "object_ids": object_ids,
        "success_count": success,
        "trajectory_count": trajectories,
        "overall_success_rate": success / trajectories,
        "macro_success_rate": sum(
            float(item["official_peak_success_rate"])
            for item in objects) / len(objects),
        "mean_maximum_lift_m": sum(
            float(item["mean_maximum_lift_m"])
            for item in objects) / len(objects),
        "failure_rate": sum(
            float(item["failure_rate"])
            for item in objects) / len(objects),
        "category_macro_success_rates": {
            category: sum(values) / len(values)
            for category, values in sorted(category_rates.items())
        },
    }


def control_objects(path, allowed_ids):
    result = evaluation_result(path)
    by_id = {
        item["object_id"]: item for item in result["objects"]
    }
    missing = set(allowed_ids) - set(by_id)
    if missing:
        raise RuntimeError(
            "Control result lacks objects: {}".format(sorted(missing)))
    return [by_id[object_id] for object_id in allowed_ids]


def candidate_objects(candidate, rounds):
    objects = []
    paths = []
    for round_index in rounds:
        path = (
            OUTPUT_ROOT / candidate["label"]
            / "round{}.yaml".format(round_index))
        result = evaluation_result(path, checkpoint(candidate))
        objects.extend(result["objects"])
        paths.append(str(path))
    return objects, paths


def row(candidate, rounds):
    objects, paths = candidate_objects(candidate, rounds)
    values = aggregate_objects(objects)
    return {
        "label": candidate["label"],
        "teacher_weight": candidate["teacher_weight"],
        "demo_weight": 1.0 - candidate["teacher_weight"],
        **values,
        "outputs": paths,
    }


def control_row(label, path, object_ids):
    values = aggregate_objects(control_objects(path, object_ids))
    return {
        "label": label,
        "teacher_weight": 1.0 if label == "teacher100_control" else 0.7,
        "demo_weight": 0.0 if label == "teacher100_control" else 0.3,
        **values,
        "outputs": [str(path)],
    }


def ranking_key(item):
    return (
        item["macro_success_rate"],
        item["mean_maximum_lift_m"],
        -item["failure_rate"],
    )


def validate_protocol():
    selections = [load_yaml(path)["object_ids"] for path in SELECTIONS]
    flat = [object_id for group in selections for object_id in group]
    if len(flat) != 12 or len(set(flat)) != 12:
        raise RuntimeError("Screen rounds must contain 12 unique objects")
    with PROTOCOL.open(encoding="utf-8") as handle:
        protocol = json.load(handle)
    development = {
        object_id
        for category in protocol["categories"].values()
        for object_id in category["development"]
    }
    final = {
        object_id
        for category in protocol["categories"].values()
        for object_id in category["final_holdout"]
    }
    if set(flat) != development or set(flat) & final:
        raise RuntimeError(
            "Target-mixture screen does not exactly partition development")
    for group in selections:
        categories = {
            object_id.split("-", 2)[1] for object_id in group}
        if categories != {"bottle", "mug", "bowl", "camera"}:
            raise RuntimeError(
                "Every round must contain exactly one object per category")
    return selections


def write_summary(summary):
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_ROOT / "summary.yaml").open(
        "w", encoding="utf-8"
    ) as handle:
        yaml.safe_dump(
            summary, handle, allow_unicode=True, sort_keys=False)


def print_rows(title, rows):
    print(title, flush=True)
    for rank, item in enumerate(sorted(
            rows, key=ranking_key, reverse=True), 1):
        print(
            "#{:02d} {} objects={} success={}/{} macro={:.2f}% "
            "lift={:.3f}m failure={:.2f}%".format(
                rank, item["label"], item["object_count"],
                item["success_count"], item["trajectory_count"],
                100 * item["macro_success_rate"],
                item["mean_maximum_lift_m"],
                100 * item["failure_rate"]),
            flush=True,
        )


def main():
    cli = parse_cli()
    if cli.max_attempts < 1:
        raise ValueError("--max-attempts must be positive")
    required = (
        CONFIG, INIT, TRAJECTORY_ROOT, RESIDUAL_CONFIG, PROTOCOL,
        CONTROL_T100, CONTROL_D30, *SELECTIONS,
        ROOT / "custom_tools/data/distillation/"
        / "online_taskid_scaled20_r1_r2_aggregated.npz",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing target-mixture inputs: {}".format(missing))
    selections = validate_protocol()

    for candidate in CANDIDATES:
        train(cli, candidate)
    if cli.dry_run:
        print(
            "DRY_RUN: full-data training for three candidates, then "
            "4->8->12-object successive-halving evaluation.",
            flush=True,
        )
        return

    # Round 1: all three new mixtures on one object per category.
    for candidate in CANDIDATES:
        evaluate(cli, candidate, 1)
    round1_rows = [row(candidate, (1,)) for candidate in CANDIDATES]
    round1_rows.sort(key=ranking_key, reverse=True)
    survivors = [
        next(item for item in CANDIDATES if item["label"] == result["label"])
        for result in round1_rows[:2]
    ]

    # Round 2: the two best new mixtures on a second object per category.
    for candidate in survivors:
        evaluate(cli, candidate, 2)
    round2_rows = [row(candidate, (1, 2)) for candidate in survivors]
    round2_rows.sort(key=ranking_key, reverse=True)
    best = survivors[
        next(index for index, candidate in enumerate(survivors)
             if candidate["label"] == round2_rows[0]["label"])
    ]
    first_eight = selections[0] + selections[1]
    control8 = [
        control_row("teacher100_control", CONTROL_T100, first_eight),
        control_row("teacher70_demo30_control", CONTROL_D30, first_eight),
    ]
    best_control8 = max(control8, key=ranking_key)
    advance_to_full = (
        round2_rows[0]["macro_success_rate"]
        >= best_control8["macro_success_rate"]
    )

    summary = {
        "status": (
            "round2_complete" if not advance_to_full
            else "round3_pending"),
        "stage": "Temporal3 offline supervision target mixture",
        "formal_final_holdout_result": False,
        "final_holdout_accessed": False,
        "training_data_reduced": False,
        "training_trajectories": 1726,
        "online_sample_fraction": 0.25,
        "fixed_epoch": 4,
        "fixed_epoch_reason": (
            "epoch 4 was best for both teacher100 and teacher70_demo30 "
            "controls; intermediate checkpoints are not re-screened"),
        "screen_protocol": (
            "all 3 candidates on round1; top 2 cumulative on round2; "
            "best candidate reaches round3 only if its 8-object macro "
            "success is not below the better existing control"),
        "round1_ranking": round1_rows,
        "round2_ranking": round2_rows,
        "round2_controls": control8,
        "advance_to_full": advance_to_full,
        "full_repeat_margin": REPEAT_MARGIN,
    }

    if not advance_to_full:
        summary["status"] = "complete_stopped_after_round2"
        summary["repeat_recommended"] = False
        write_summary(summary)
        print_rows("ROUND1", round1_rows)
        print_rows("ROUND2_CUMULATIVE", round2_rows + control8)
        print("ADVANCE_TO_FULL=False", flush=True)
        print("TEMPORAL_TARGET_MIX_SCREEN=COMPLETE", flush=True)
        return

    # Round 3: only the cumulative-eight winner reaches the last four objects.
    evaluate(cli, best, 3)
    full_candidate = row(best, (1, 2, 3))
    all_ids = selections[0] + selections[1] + selections[2]
    full_controls = [
        control_row("teacher100_control", CONTROL_T100, all_ids),
        control_row("teacher70_demo30_control", CONTROL_D30, all_ids),
    ]
    locked_control = next(
        item for item in full_controls
        if item["label"] == "teacher100_control")
    improvement = (
        full_candidate["macro_success_rate"]
        - locked_control["macro_success_rate"]
    )
    repeat_recommended = improvement >= REPEAT_MARGIN
    summary.update({
        "status": "complete",
        "full_candidate": full_candidate,
        "full_controls": full_controls,
        "full_candidate_minus_locked_temporal3": improvement,
        "repeat_threshold": (
            locked_control["macro_success_rate"] + REPEAT_MARGIN),
        "repeat_recommended": repeat_recommended,
    })
    write_summary(summary)
    print_rows("ROUND1", round1_rows)
    print_rows("ROUND2_CUMULATIVE", round2_rows + control8)
    print_rows("FULL12", [full_candidate] + full_controls)
    print("ADVANCE_TO_FULL=True", flush=True)
    print("REPEAT_RECOMMENDED={}".format(
        repeat_recommended), flush=True)
    print("TEMPORAL_TARGET_MIX_SCREEN=COMPLETE", flush=True)


if __name__ == "__main__":
    main()
