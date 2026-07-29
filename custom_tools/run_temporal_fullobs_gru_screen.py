"""Paired screen of Temporal3 versus a full-observation three-frame GRU."""

import argparse
import collections
import json
from pathlib import Path
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONTROL_CONFIG = (
    ROOT / "custom_tools/configs/"
    / "unified_student_taskid_temporal3_v1.yaml"
)
GRU_CONFIG = (
    ROOT / "custom_tools/configs/"
    / "unified_student_taskid_temporal3_fullobs_gru_v1.yaml"
)
INIT = (
    ROOT / "custom_tools/runs/bc/"
    / "unified_student_taskid_temporal3_seed2025_e4_v1/"
    / "epoch=003-step=5152.ckpt"
)
RUNS = (
    {
        "label": "paired_temporal3",
        "config": CONTROL_CONFIG,
        "run_name": "temporal3_paired_freshadam_seed2025_e2_v1",
    },
    {
        "label": "fullobs_gru",
        "config": GRU_CONFIG,
        "run_name": "temporal3_fullobs_gru_seed2025_e2_v1",
    },
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
LOCKED_RESULT = (
    ROOT / "custom_tools/results/taskid_temporal3_development_v1/"
    / "temporal3_epoch04.yaml"
)
OUTPUT_ROOT = (
    ROOT / "custom_tools/results/temporal_fullobs_gru_screen_v1"
)
REPEAT_MARGIN = 0.02


def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-free-vram-mb", type=int, default=4500)
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_yaml(path):
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def run_dir(candidate):
    return ROOT / "custom_tools/runs/bc" / candidate["run_name"]


def checkpoint(candidate):
    matches = list(run_dir(candidate).glob("epoch=001-step=*.ckpt"))
    if len(matches) != 1:
        raise RuntimeError(
            "Expected one epoch-2 checkpoint in {}".format(
                run_dir(candidate)))
    return matches[0].resolve()


def train(cli, candidate):
    directory = run_dir(candidate)
    if (
        (directory / "last.ckpt").is_file()
        and (directory / "resource_summary.yaml").is_file()
    ):
        checkpoint(candidate)
        print("[REUSE] training {}".format(candidate["label"]), flush=True)
        return
    if list(directory.glob("*.ckpt")):
        raise RuntimeError(
            "Partial paired run needs inspection: {}".format(directory))
    command = [
        sys.executable, "-u",
        str(ROOT / "custom_tools/train_bc.py"),
        "--config", str(candidate["config"]),
        "--run-name", candidate["run_name"],
        "--seed", "2025",
        "--num-epochs", "2",
        "--batch-size", "64",
        "--learning-rate", "2e-5",
        "--teacher-weight", "1.0",
        "--online-sample-fraction", "0.25",
        "--init-checkpoint", str(INIT),
        "--min-free-vram-mb", str(cli.min_free_vram_mb),
    ]
    print("RUN {}".format(" ".join(command)), flush=True)
    if not cli.dry_run:
        subprocess.run(command, cwd=str(ROOT), check=True)
        checkpoint(candidate)


def read_result(path, expected_checkpoint=None):
    rows = load_yaml(path).get("checkpoint_results", [])
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
        read_result(output, model)
        print("[REUSE] {} round{}".format(
            candidate["label"], round_index), flush=True)
        return
    command = [
        sys.executable, "-u",
        str(ROOT / "custom_tools/evaluate_bc_checkpoints_isolated.py"),
        "--checkpoint", str(model),
        "--bc-config", str(candidate["config"]),
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
    read_result(output, model)


def aggregate(objects):
    ids = [item["object_id"] for item in objects]
    if not ids or len(ids) != len(set(ids)):
        raise RuntimeError("Invalid cumulative object set")
    category_rates = collections.defaultdict(list)
    for item in objects:
        category_rates[item["category"]].append(
            float(item["official_peak_success_rate"]))
    success = sum(
        int(item["official_peak_success_count"]) for item in objects)
    count = sum(int(item["trajectory_count"]) for item in objects)
    return {
        "object_count": len(objects),
        "object_ids": ids,
        "success_count": success,
        "trajectory_count": count,
        "overall_success_rate": success / count,
        "macro_success_rate": sum(
            float(item["official_peak_success_rate"])
            for item in objects) / len(objects),
        "mean_maximum_lift_m": sum(
            float(item["mean_maximum_lift_m"])
            for item in objects) / len(objects),
        "failure_rate": sum(
            float(item["failure_rate"]) for item in objects) / len(objects),
        "category_macro_success_rates": {
            category: sum(values) / len(values)
            for category, values in sorted(category_rates.items())
        },
    }


def candidate_row(candidate, rounds):
    objects = []
    outputs = []
    for round_index in rounds:
        path = (
            OUTPUT_ROOT / candidate["label"]
            / "round{}.yaml".format(round_index))
        objects.extend(
            read_result(path, checkpoint(candidate))["objects"])
        outputs.append(str(path))
    return {
        "label": candidate["label"],
        **aggregate(objects),
        "outputs": outputs,
    }


def locked_row(object_ids):
    result = read_result(LOCKED_RESULT)
    by_id = {item["object_id"]: item for item in result["objects"]}
    missing = set(object_ids) - set(by_id)
    if missing:
        raise RuntimeError(
            "Locked Temporal3 lacks objects: {}".format(sorted(missing)))
    return {
        "label": "locked_temporal3_epoch04",
        **aggregate([by_id[object_id] for object_id in object_ids]),
        "outputs": [str(LOCKED_RESULT)],
    }


def ranking_key(row):
    return (
        row["macro_success_rate"],
        row["mean_maximum_lift_m"],
        -row["failure_rate"],
    )


def print_rows(title, rows):
    print(title, flush=True)
    for rank, row in enumerate(
            sorted(rows, key=ranking_key, reverse=True), 1):
        print(
            "#{:02d} {} objects={} success={}/{} macro={:.2f}% "
            "lift={:.3f}m failure={:.2f}%".format(
                rank, row["label"], row["object_count"],
                row["success_count"], row["trajectory_count"],
                100 * row["macro_success_rate"],
                row["mean_maximum_lift_m"],
                100 * row["failure_rate"]),
            flush=True,
        )


def validate_protocol():
    selections = [load_yaml(path)["object_ids"] for path in SELECTIONS]
    flat = [object_id for group in selections for object_id in group]
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
    if len(flat) != 12 or set(flat) != development or set(flat) & final:
        raise RuntimeError("GRU screen selections violate protocol")
    return selections


def write_summary(summary):
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_ROOT / "summary.yaml").open(
        "w", encoding="utf-8"
    ) as handle:
        yaml.safe_dump(
            summary, handle, allow_unicode=True, sort_keys=False)


def main():
    cli = parse_cli()
    if cli.max_attempts < 1:
        raise ValueError("--max-attempts must be positive")
    required = (
        CONTROL_CONFIG, GRU_CONFIG, INIT, TRAJECTORY_ROOT,
        RESIDUAL_CONFIG, PROTOCOL, LOCKED_RESULT, *SELECTIONS,
        ROOT / "custom_tools/data/distillation/"
        / "online_taskid_scaled20_r1_r2_aggregated.npz",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing full-observation GRU inputs: {}".format(missing))
    selections = validate_protocol()

    for candidate in RUNS:
        train(cli, candidate)
    if cli.dry_run:
        print(
            "DRY_RUN: two full-data paired trainings; both evaluated on "
            "eight development objects; round3 only if GRU wins.",
            flush=True,
        )
        return

    # Both models always receive the same first eight development objects.
    for round_index in (1, 2):
        for candidate in RUNS:
            evaluate(cli, candidate, round_index)
    first_eight = selections[0] + selections[1]
    rows8 = [
        candidate_row(candidate, (1, 2)) for candidate in RUNS
    ]
    locked8 = locked_row(first_eight)
    gru8 = next(row for row in rows8 if row["label"] == "fullobs_gru")
    paired8 = next(
        row for row in rows8 if row["label"] == "paired_temporal3")
    advance = (
        ranking_key(gru8) > ranking_key(paired8)
        and ranking_key(gru8) > ranking_key(locked8)
    )
    summary = {
        "status": (
            "round3_pending" if advance
            else "complete_stopped_after_round2"),
        "stage": (
            "Temporal3 versus shared full-observation encoder plus GRU"),
        "formal_final_holdout_result": False,
        "final_holdout_accessed": False,
        "training_data_reduced": False,
        "training_trajectories": 1726,
        "online_sample_fraction": 0.25,
        "paired_controls": (
            "both candidates start from the same locked Temporal3 weights, "
            "use fresh Adam, batch 64, lr 2e-5, and train for two epochs"),
        "architecture_change": (
            "shared DexRep encoding of observations t-2,t-1,t; one-layer "
            "GRU hidden 128; zero-initialized 28-D residual action head"),
        "round2_cumulative": rows8 + [locked8],
        "advance_to_full": advance,
        "advance_rule": (
            "GRU must rank above both its paired continued-Temporal3 "
            "control and the locked Temporal3 on the same first 8 objects"),
        "full_repeat_margin": REPEAT_MARGIN,
    }
    if not advance:
        summary["repeat_recommended"] = False
        write_summary(summary)
        print_rows("CUMULATIVE8", rows8 + [locked8])
        print("ADVANCE_TO_FULL=False", flush=True)
        print("TEMPORAL_FULLOBS_GRU_SCREEN=COMPLETE", flush=True)
        return

    for candidate in RUNS:
        evaluate(cli, candidate, 3)
    rows12 = [
        candidate_row(candidate, (1, 2, 3)) for candidate in RUNS
    ]
    all_ids = first_eight + selections[2]
    locked12 = locked_row(all_ids)
    gru12 = next(row for row in rows12 if row["label"] == "fullobs_gru")
    improvement = (
        gru12["macro_success_rate"]
        - locked12["macro_success_rate"])
    summary.update({
        "status": "complete",
        "full12": rows12 + [locked12],
        "full_gru_minus_locked_temporal3": improvement,
        "repeat_threshold": (
            locked12["macro_success_rate"] + REPEAT_MARGIN),
        "repeat_recommended": improvement >= REPEAT_MARGIN,
    })
    write_summary(summary)
    print_rows("CUMULATIVE8", rows8 + [locked8])
    print_rows("FULL12", rows12 + [locked12])
    print("ADVANCE_TO_FULL=True", flush=True)
    print("REPEAT_RECOMMENDED={}".format(
        summary["repeat_recommended"]), flush=True)
    print("TEMPORAL_FULLOBS_GRU_SCREEN=COMPLETE", flush=True)


if __name__ == "__main__":
    main()
