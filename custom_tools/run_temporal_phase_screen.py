"""Progressive paired screen for explicit rollout-progress conditioning."""

import argparse
import collections
import json
from pathlib import Path
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
PHASE_CONFIG = (
    ROOT / "custom_tools/configs/"
    / "unified_student_taskid_temporal3_phase_v1.yaml")
CONTROL_CONFIG = (
    ROOT / "custom_tools/configs/"
    / "unified_student_taskid_temporal3_v1.yaml")
INIT = (
    ROOT / "custom_tools/runs/bc/"
    / "unified_student_taskid_temporal3_seed2025_e4_v1/"
    / "epoch=003-step=5152.ckpt")
RUN_NAME = "temporal3_phase_seed2025_e2_v1"
RUN_DIR = ROOT / "custom_tools/runs/bc" / RUN_NAME
OUTPUT_ROOT = ROOT / "custom_tools/results/temporal_phase_screen_v1"
TRAJECTORY_ROOT = (
    ROOT / "dexgrasp/dataset/scaled_category_final_v1_preprocessed")
RESIDUAL_CONFIG = ROOT / "custom_tools/configs/residual_ppo_stage1.yaml"
PROTOCOL = (
    ROOT / "custom_tools/configs/scaled_evaluation_protocol_v1.json")
SELECTIONS = (
    ROOT / "custom_tools/configs/temporal_target_screen_round1_4.yaml",
    ROOT / "custom_tools/configs/temporal_target_screen_round2_4.yaml",
    ROOT / "custom_tools/configs/temporal_target_screen_round3_4.yaml",
)
LOCKED_RESULT = (
    ROOT / "custom_tools/results/taskid_temporal3_development_v1/"
    / "temporal3_epoch04.yaml")
PAIRED = {
    1: {
        "checkpoint": (
            ROOT / "custom_tools/runs/bc/"
            / "temporal3_paired_freshadam_seed2025_e2_v1/"
            / "epoch=000-step=2577.ckpt"),
        "rounds": {
            1: (
                ROOT / "custom_tools/results/"
                / "temporal_fullobs_gru_epoch1_audit_v1/"
                / "paired_temporal3_epoch01/round1.yaml"),
        },
    },
    2: {
        "checkpoint": (
            ROOT / "custom_tools/runs/bc/"
            / "temporal3_paired_freshadam_seed2025_e2_v1/"
            / "epoch=001-step=5154.ckpt"),
        "rounds": {
            1: (
                ROOT / "custom_tools/results/"
                / "temporal_fullobs_gru_screen_v1/"
                / "paired_temporal3/round1.yaml"),
            2: (
                ROOT / "custom_tools/results/"
                / "temporal_fullobs_gru_screen_v1/"
                / "paired_temporal3/round2.yaml"),
        },
    },
}
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


def phase_checkpoint(epoch):
    matches = list(RUN_DIR.glob(
        "epoch={:03d}-step=*.ckpt".format(epoch - 1)))
    if len(matches) != 1:
        raise RuntimeError(
            "Expected one phase epoch {} checkpoint".format(epoch))
    return matches[0].resolve()


def train(cli):
    if (
        (RUN_DIR / "last.ckpt").is_file()
        and (RUN_DIR / "resource_summary.yaml").is_file()
    ):
        phase_checkpoint(1)
        phase_checkpoint(2)
        print("[REUSE] phase training", flush=True)
        return
    if list(RUN_DIR.glob("*.ckpt")):
        raise RuntimeError(
            "Partial phase run needs inspection: {}".format(RUN_DIR))
    command = [
        sys.executable, "-u", str(ROOT / "custom_tools/train_bc.py"),
        "--config", str(PHASE_CONFIG),
        "--run-name", RUN_NAME,
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
        phase_checkpoint(1)
        phase_checkpoint(2)


def read_result(path, checkpoint=None):
    rows = load_yaml(path).get("checkpoint_results", [])
    if len(rows) != 1:
        raise RuntimeError("Expected one result in {}".format(path))
    if (
        checkpoint is not None
        and Path(rows[0]["checkpoint"]).resolve() != checkpoint.resolve()
    ):
        raise RuntimeError("Checkpoint mismatch in {}".format(path))
    return rows[0]


def evaluate(
        cli, label, checkpoint, config, round_index, output):
    if output.is_file():
        read_result(output, checkpoint.resolve())
        print("[REUSE] {} round{}".format(
            label, round_index), flush=True)
        return
    command = [
        sys.executable, "-u",
        str(ROOT / "custom_tools/evaluate_bc_checkpoints_isolated.py"),
        "--checkpoint", str(checkpoint),
        "--bc-config", str(config),
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
    read_result(output, checkpoint.resolve())


def phase_result_path(epoch, round_index):
    return (
        OUTPUT_ROOT / "phase_epoch{:02d}".format(epoch)
        / "round{}.yaml".format(round_index))


def paired_result_path(epoch, round_index):
    existing = PAIRED[epoch]["rounds"].get(round_index)
    if existing is not None:
        return existing
    return (
        OUTPUT_ROOT / "paired_epoch{:02d}".format(epoch)
        / "round{}.yaml".format(round_index))


def ensure_paired(cli, epoch, round_index):
    path = paired_result_path(epoch, round_index)
    evaluate(
        cli, "paired_epoch{:02d}".format(epoch),
        PAIRED[epoch]["checkpoint"], CONTROL_CONFIG,
        round_index, path)
    return path


def aggregate(objects):
    categories = collections.defaultdict(list)
    for item in objects:
        categories[item["category"]].append(
            float(item["official_peak_success_rate"]))
    return {
        "object_count": len(objects),
        "object_ids": [item["object_id"] for item in objects],
        "success_count": sum(
            int(item["official_peak_success_count"]) for item in objects),
        "trajectory_count": sum(
            int(item["trajectory_count"]) for item in objects),
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
            for category, values in sorted(categories.items())
        },
    }


def result_row(label, checkpoint, paths):
    objects = []
    for path in paths:
        objects.extend(read_result(path, checkpoint)["objects"])
    return {
        "label": label,
        **aggregate(objects),
        "outputs": [str(path) for path in paths],
    }


def locked_row(object_ids):
    by_id = {
        item["object_id"]: item
        for item in read_result(LOCKED_RESULT)["objects"]}
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
            flush=True)


def save(summary):
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_ROOT / "summary.yaml").open(
        "w", encoding="utf-8"
    ) as handle:
        yaml.safe_dump(
            summary, handle, allow_unicode=True, sort_keys=False)


def validate_protocol():
    groups = [load_yaml(path)["object_ids"] for path in SELECTIONS]
    flat = [object_id for group in groups for object_id in group]
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
        raise RuntimeError("Phase screen violates evaluation protocol")
    return groups


def main():
    cli = parse_cli()
    required = (
        PHASE_CONFIG, CONTROL_CONFIG, INIT, TRAJECTORY_ROOT,
        RESIDUAL_CONFIG, PROTOCOL, LOCKED_RESULT, *SELECTIONS,
        *(item["checkpoint"] for item in PAIRED.values()),
        *(path for item in PAIRED.values()
          for path in item["rounds"].values()),
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(missing)
    groups = validate_protocol()
    train(cli)
    if cli.dry_run:
        print(
            "DRY_RUN: train two phase epochs; screen both on 4 objects; "
            "best phase epoch advances to 8; 12 only if it beats controls.",
            flush=True)
        return

    for epoch in (1, 2):
        evaluate(
            cli, "phase_epoch{:02d}".format(epoch),
            phase_checkpoint(epoch), PHASE_CONFIG, 1,
            phase_result_path(epoch, 1))
    phase4 = [
        result_row(
            "phase_epoch{:02d}".format(epoch),
            phase_checkpoint(epoch),
            [phase_result_path(epoch, 1)])
        for epoch in (1, 2)
    ]
    phase4.sort(key=ranking_key, reverse=True)
    best_epoch = int(phase4[0]["label"].rsplit("epoch", 1)[1])

    evaluate(
        cli, "phase_epoch{:02d}".format(best_epoch),
        phase_checkpoint(best_epoch), PHASE_CONFIG, 2,
        phase_result_path(best_epoch, 2))
    paired_paths8 = [
        ensure_paired(cli, best_epoch, round_index)
        for round_index in (1, 2)]
    phase8 = result_row(
        "phase_epoch{:02d}".format(best_epoch),
        phase_checkpoint(best_epoch),
        [phase_result_path(best_epoch, 1),
         phase_result_path(best_epoch, 2)])
    paired8 = result_row(
        "paired_epoch{:02d}".format(best_epoch),
        PAIRED[best_epoch]["checkpoint"].resolve(),
        paired_paths8)
    ids8 = groups[0] + groups[1]
    locked8 = locked_row(ids8)
    advance = (
        ranking_key(phase8) > ranking_key(paired8)
        and ranking_key(phase8) > ranking_key(locked8))
    summary = {
        "status": (
            "round3_pending" if advance
            else "complete_stopped_after_round2"),
        "stage": "Temporal3 plus normalized rollout progress",
        "formal_final_holdout_result": False,
        "final_holdout_accessed": False,
        "training_data_reduced": False,
        "phase_feature": (
            "one scalar, 2*min(step,69)/69-1; no future state"),
        "phase_round1_ranking": phase4,
        "selected_epoch": best_epoch,
        "cumulative8": [phase8, paired8, locked8],
        "advance_to_full": advance,
        "repeat_margin": REPEAT_MARGIN,
    }
    if not advance:
        summary["repeat_recommended"] = False
        save(summary)
        print_rows("PHASE_ROUND1", phase4)
        print_rows("PHASE_CUMULATIVE8", [phase8, paired8, locked8])
        print("ADVANCE_TO_FULL=False", flush=True)
        print("TEMPORAL_PHASE_SCREEN=COMPLETE", flush=True)
        return

    evaluate(
        cli, "phase_epoch{:02d}".format(best_epoch),
        phase_checkpoint(best_epoch), PHASE_CONFIG, 3,
        phase_result_path(best_epoch, 3))
    paired3 = ensure_paired(cli, best_epoch, 3)
    phase12 = result_row(
        "phase_epoch{:02d}".format(best_epoch),
        phase_checkpoint(best_epoch),
        [phase_result_path(best_epoch, i) for i in (1, 2, 3)])
    paired12 = result_row(
        "paired_epoch{:02d}".format(best_epoch),
        PAIRED[best_epoch]["checkpoint"].resolve(),
        paired_paths8 + [paired3])
    locked12 = locked_row(ids8 + groups[2])
    improvement = (
        phase12["macro_success_rate"]
        - locked12["macro_success_rate"])
    summary.update({
        "status": "complete",
        "full12": [phase12, paired12, locked12],
        "phase_minus_locked": improvement,
        "repeat_recommended": improvement >= REPEAT_MARGIN,
    })
    save(summary)
    print_rows("PHASE_ROUND1", phase4)
    print_rows("PHASE_CUMULATIVE8", [phase8, paired8, locked8])
    print_rows("PHASE_FULL12", [phase12, paired12, locked12])
    print("ADVANCE_TO_FULL=True", flush=True)
    print("REPEAT_RECOMMENDED={}".format(
        summary["repeat_recommended"]), flush=True)
    print("TEMPORAL_PHASE_SCREEN=COMPLETE", flush=True)


if __name__ == "__main__":
    main()
