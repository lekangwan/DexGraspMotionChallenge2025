"""Audit epoch 1 before rejecting the full-observation GRU experiment."""

import argparse
import collections
from pathlib import Path
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = (
    ROOT / "custom_tools/results/temporal_fullobs_gru_epoch1_audit_v1")
TRAJECTORY_ROOT = (
    ROOT / "dexgrasp/dataset/scaled_category_final_v1_preprocessed")
RESIDUAL_CONFIG = ROOT / "custom_tools/configs/residual_ppo_stage1.yaml"
SELECTIONS = (
    ROOT / "custom_tools/configs/temporal_target_screen_round1_4.yaml",
    ROOT / "custom_tools/configs/temporal_target_screen_round2_4.yaml",
)
LOCKED_RESULT = (
    ROOT / "custom_tools/results/taskid_temporal3_development_v1/"
    / "temporal3_epoch04.yaml")
CANDIDATES = (
    {
        "label": "paired_temporal3_epoch01",
        "config": (
            ROOT / "custom_tools/configs/"
            / "unified_student_taskid_temporal3_v1.yaml"),
        "checkpoint": (
            ROOT / "custom_tools/runs/bc/"
            / "temporal3_paired_freshadam_seed2025_e2_v1/"
            / "epoch=000-step=2577.ckpt"),
    },
    {
        "label": "fullobs_gru_epoch01",
        "config": (
            ROOT / "custom_tools/configs/"
            / "unified_student_taskid_temporal3_fullobs_gru_v1.yaml"),
        "checkpoint": (
            ROOT / "custom_tools/runs/bc/"
            / "temporal3_fullobs_gru_seed2025_e2_v1/"
            / "epoch=000-step=2577.ckpt"),
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


def read_result(path, expected=None):
    rows = load_yaml(path).get("checkpoint_results", [])
    if len(rows) != 1:
        raise RuntimeError("Expected one result in {}".format(path))
    if (
        expected is not None
        and Path(rows[0]["checkpoint"]).resolve() != expected.resolve()
    ):
        raise RuntimeError("Checkpoint mismatch in {}".format(path))
    return rows[0]


def evaluate(cli, candidate, round_index):
    output = (
        OUTPUT_ROOT / candidate["label"]
        / "round{}.yaml".format(round_index))
    if output.is_file():
        read_result(output, candidate["checkpoint"].resolve())
        print("[REUSE] {} round{}".format(
            candidate["label"], round_index), flush=True)
        return
    command = [
        sys.executable, "-u",
        str(ROOT / "custom_tools/evaluate_bc_checkpoints_isolated.py"),
        "--checkpoint", str(candidate["checkpoint"]),
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
    if not cli.dry_run:
        subprocess.run(command, cwd=str(ROOT), check=True)
        read_result(output, candidate["checkpoint"].resolve())


def aggregate(objects):
    categories = collections.defaultdict(list)
    for item in objects:
        categories[item["category"]].append(
            float(item["official_peak_success_rate"]))
    return {
        "object_count": len(objects),
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


def candidate_row(candidate, rounds):
    objects = []
    outputs = []
    for round_index in rounds:
        path = (
            OUTPUT_ROOT / candidate["label"]
            / "round{}.yaml".format(round_index))
        objects.extend(
            read_result(
                path, candidate["checkpoint"].resolve())["objects"])
        outputs.append(str(path))
    return {
        "label": candidate["label"],
        **aggregate(objects),
        "outputs": outputs,
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


def main():
    cli = parse_cli()
    required = (
        TRAJECTORY_ROOT, RESIDUAL_CONFIG, LOCKED_RESULT,
        *SELECTIONS,
        *(item["config"] for item in CANDIDATES),
        *(item["checkpoint"] for item in CANDIDATES),
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(missing)
    for candidate in CANDIDATES:
        evaluate(cli, candidate, 1)
    if cli.dry_run:
        print(
            "DRY_RUN: epoch-1 pair on four objects; expand to eight only "
            "if GRU beats both controls.", flush=True)
        return

    ids4 = load_yaml(SELECTIONS[0])["object_ids"]
    rows4 = [candidate_row(item, (1,)) for item in CANDIDATES]
    locked4 = locked_row(ids4)
    gru4 = next(
        row for row in rows4 if row["label"] == "fullobs_gru_epoch01")
    paired4 = next(
        row for row in rows4
        if row["label"] == "paired_temporal3_epoch01")
    advance = (
        ranking_key(gru4) > ranking_key(paired4)
        and ranking_key(gru4) > ranking_key(locked4))
    summary = {
        "status": (
            "round2_pending" if advance
            else "complete_stopped_after_round1"),
        "formal_final_holdout_result": False,
        "final_holdout_accessed": False,
        "purpose": (
            "check whether the zero-start GRU peaked after one epoch"),
        "round1": rows4 + [locked4],
        "advance_to_eight": advance,
    }
    if not advance:
        summary["gru_retained"] = False
        save(summary)
        print_rows("EPOCH1_ROUND1", rows4 + [locked4])
        print("ADVANCE_TO_EIGHT=False", flush=True)
        print("TEMPORAL_FULLOBS_GRU_EPOCH1_AUDIT=COMPLETE", flush=True)
        return

    for candidate in CANDIDATES:
        evaluate(cli, candidate, 2)
    ids8 = ids4 + load_yaml(SELECTIONS[1])["object_ids"]
    rows8 = [candidate_row(item, (1, 2)) for item in CANDIDATES]
    locked8 = locked_row(ids8)
    gru8 = next(
        row for row in rows8 if row["label"] == "fullobs_gru_epoch01")
    paired8 = next(
        row for row in rows8
        if row["label"] == "paired_temporal3_epoch01")
    retained = (
        ranking_key(gru8) > ranking_key(paired8)
        and ranking_key(gru8) > ranking_key(locked8))
    summary.update({
        "status": "complete",
        "round2_cumulative": rows8 + [locked8],
        "gru_retained": retained,
    })
    save(summary)
    print_rows("EPOCH1_ROUND1", rows4 + [locked4])
    print_rows("EPOCH1_CUMULATIVE8", rows8 + [locked8])
    print("ADVANCE_TO_EIGHT=True", flush=True)
    print("GRU_RETAINED={}".format(retained), flush=True)
    print("TEMPORAL_FULLOBS_GRU_EPOCH1_AUDIT=COMPLETE", flush=True)


if __name__ == "__main__":
    main()
