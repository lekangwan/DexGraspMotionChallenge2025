"""Screen continued Temporal3 training on Temporal3-visited states."""

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
    / "unified_student_taskid_temporal3_onpolicy_v1.yaml"
)
SOURCE_CHECKPOINT = (
    ROOT / "custom_tools/runs/bc/"
    / "unified_student_taskid_temporal3_seed2025_e4_v1/"
    / "epoch=003-step=5152.ckpt"
)
RUN_NAME = "unified_student_taskid_temporal3_onpolicy_seed2025_e8_v1"
RUN_DIR = ROOT / "custom_tools/runs/bc" / RUN_NAME
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
ORIGINAL_RESULT = (
    ROOT / "custom_tools/results/taskid_temporal3_development_v1/"
    / "temporal3_epoch04.yaml"
)
CONTINUATION_ROOT = (
    ROOT / "custom_tools/results/taskid_temporal3_continue8_development_v1"
)
OUTPUT_ROOT = (
    ROOT / "custom_tools/results/temporal_onpolicy_screen_v1"
)
EPOCHS = (5, 6, 7, 8)
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


def checkpoint(epoch):
    matches = list(RUN_DIR.glob(
        "epoch={:03d}-step=*.ckpt".format(epoch - 1)))
    if len(matches) != 1:
        raise RuntimeError(
            "Expected one epoch {} checkpoint in {}".format(epoch, RUN_DIR))
    return matches[0].resolve()


def train(cli):
    last = RUN_DIR / "last.ckpt"
    resource = RUN_DIR / "resource_summary.yaml"
    if last.is_file() and resource.is_file():
        for epoch in EPOCHS:
            checkpoint(epoch)
        print("[REUSE] completed on-policy continuation", flush=True)
        return
    existing = list(RUN_DIR.glob("epoch=*-step=*.ckpt"))
    resume = SOURCE_CHECKPOINT
    if existing:
        resume = max(
            existing,
            key=lambda path: int(
                path.name.split("=", 1)[1].split("-", 1)[0]))
    command = [
        sys.executable, "-u",
        str(ROOT / "custom_tools/train_bc.py"),
        "--config", str(CONFIG),
        "--run-name", RUN_NAME,
        "--seed", "2025",
        "--num-epochs", "8",
        "--learning-rate", "2e-5",
        "--teacher-weight", "1.0",
        "--online-sample-fraction", "0.25",
        "--resume-checkpoint", str(resume),
        "--min-free-vram-mb", str(cli.min_free_vram_mb),
    ]
    print("RUN {}".format(" ".join(command)), flush=True)
    if not cli.dry_run:
        subprocess.run(command, cwd=str(ROOT), check=True)
        for epoch in EPOCHS:
            checkpoint(epoch)


def eval_result(path, expected_checkpoint=None):
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


def evaluate(cli, epoch, round_index):
    model = checkpoint(epoch)
    output = (
        OUTPUT_ROOT / "epoch{:02d}".format(epoch)
        / "round{}.yaml".format(round_index))
    if output.is_file():
        eval_result(output, model)
        print("[REUSE] epoch{} round{}".format(
            epoch, round_index), flush=True)
        return
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
    eval_result(output, model)


def aggregate(objects):
    ids = [item["object_id"] for item in objects]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Duplicate cumulative screen objects")
    categories = collections.defaultdict(list)
    for item in objects:
        categories[item["category"]].append(
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
            float(item["failure_rate"])
            for item in objects) / len(objects),
        "category_macro_success_rates": {
            key: sum(values) / len(values)
            for key, values in sorted(categories.items())
        },
    }


def candidate_row(epoch, rounds):
    objects = []
    outputs = []
    for round_index in rounds:
        path = (
            OUTPUT_ROOT / "epoch{:02d}".format(epoch)
            / "round{}.yaml".format(round_index))
        objects.extend(eval_result(path, checkpoint(epoch))["objects"])
        outputs.append(str(path))
    return {
        "label": "temporal_onpolicy_epoch{:02d}".format(epoch),
        "epoch": epoch,
        **aggregate(objects),
        "outputs": outputs,
    }


def subset_control(label, path, object_ids):
    result = eval_result(path)
    by_id = {item["object_id"]: item for item in result["objects"]}
    missing = set(object_ids) - set(by_id)
    if missing:
        raise RuntimeError(
            "Control lacks objects: {}".format(sorted(missing)))
    return {
        "label": label,
        **aggregate([by_id[object_id] for object_id in object_ids]),
        "outputs": [str(path)],
    }


def ranking_key(row):
    return (
        row["macro_success_rate"],
        row["mean_maximum_lift_m"],
        -row["failure_rate"],
    )


def continuation_path(epoch):
    return CONTINUATION_ROOT / "temporal3_epoch{:02d}.yaml".format(epoch)


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
        raise RuntimeError("On-policy screen selections violate protocol")
    return selections


def controls(object_ids):
    rows = [
        subset_control(
            "locked_temporal3_epoch04", ORIGINAL_RESULT, object_ids)
    ]
    rows.extend(
        subset_control(
            "old_online_continuation_epoch{:02d}".format(epoch),
            continuation_path(epoch), object_ids)
        for epoch in EPOCHS
    )
    return rows


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
        CONFIG, SOURCE_CHECKPOINT, TRAJECTORY_ROOT, RESIDUAL_CONFIG,
        PROTOCOL, ORIGINAL_RESULT, *SELECTIONS,
        ROOT / "custom_tools/data/distillation/"
        / "online_taskid_temporal3_r1_train4_offset4.npz",
        *(continuation_path(epoch) for epoch in EPOCHS),
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing on-policy screen inputs: {}".format(missing))
    selections = validate_protocol()
    train(cli)
    if cli.dry_run:
        print(
            "DRY_RUN: resume exact epoch-4 model/Adam on Temporal3-visited "
            "states; screen epochs 5-8 over 4->8->12 objects.",
            flush=True)
        return

    # Round 1: all four continued checkpoints.
    for epoch in EPOCHS:
        evaluate(cli, epoch, 1)
    round1 = [candidate_row(epoch, (1,)) for epoch in EPOCHS]
    round1.sort(key=ranking_key, reverse=True)
    surviving_epochs = [row["epoch"] for row in round1[:2]]

    # Round 2: the two best checkpoints, cumulative eight objects.
    for epoch in surviving_epochs:
        evaluate(cli, epoch, 2)
    round2 = [
        candidate_row(epoch, (1, 2)) for epoch in surviving_epochs]
    round2.sort(key=ranking_key, reverse=True)
    best_epoch = round2[0]["epoch"]
    first_eight = selections[0] + selections[1]
    control8 = controls(first_eight)
    best_control8 = max(control8, key=ranking_key)
    advance = (
        round2[0]["macro_success_rate"]
        >= best_control8["macro_success_rate"]
    )
    summary = {
        "status": "round3_pending" if advance else "complete_stopped_round2",
        "stage": "Temporal3 on-policy online imitation",
        "formal_final_holdout_result": False,
        "final_holdout_accessed": False,
        "source_checkpoint": str(SOURCE_CHECKPOINT),
        "optimizer_restored": True,
        "controlled_difference": (
            "same full offline data, teacher100 target, sampling, optimizer, "
            "learning rate, seed, and continuation epochs; online states "
            "change from R1/R2-visited to Temporal3-visited only"),
        "onpolicy_samples": 22080,
        "online_sample_fraction": 0.25,
        "round1_ranking": round1,
        "round2_ranking": round2,
        "round2_controls": control8,
        "advance_to_full": advance,
        "full_repeat_margin": REPEAT_MARGIN,
    }
    if not advance:
        summary["repeat_recommended"] = False
        write_summary(summary)
        print_rows("ROUND1", round1)
        print_rows("ROUND2_CUMULATIVE", round2 + control8)
        print("ADVANCE_TO_FULL=False", flush=True)
        print("TEMPORAL_ONPOLICY_SCREEN=COMPLETE", flush=True)
        return

    evaluate(cli, best_epoch, 3)
    full_candidate = candidate_row(best_epoch, (1, 2, 3))
    all_ids = selections[0] + selections[1] + selections[2]
    full_controls = controls(all_ids)
    locked = next(
        row for row in full_controls
        if row["label"] == "locked_temporal3_epoch04")
    improvement = (
        full_candidate["macro_success_rate"]
        - locked["macro_success_rate"]
    )
    summary.update({
        "status": "complete",
        "full_candidate": full_candidate,
        "full_controls": full_controls,
        "full_candidate_minus_locked_temporal3": improvement,
        "repeat_threshold": (
            locked["macro_success_rate"] + REPEAT_MARGIN),
        "repeat_recommended": improvement >= REPEAT_MARGIN,
    })
    write_summary(summary)
    print_rows("ROUND1", round1)
    print_rows("ROUND2_CUMULATIVE", round2 + control8)
    print_rows("FULL12", [full_candidate] + full_controls)
    print("ADVANCE_TO_FULL=True", flush=True)
    print("REPEAT_RECOMMENDED={}".format(
        summary["repeat_recommended"]), flush=True)
    print("TEMPORAL_ONPOLICY_SCREEN=COMPLETE", flush=True)


if __name__ == "__main__":
    main()
