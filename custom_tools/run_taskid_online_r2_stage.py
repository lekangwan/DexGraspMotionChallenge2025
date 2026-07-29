"""Aggregate, train, and screen the second Task-ID online-imitation round."""

import argparse
from pathlib import Path
import subprocess
import sys

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT / "custom_tools/configs/"
    / "unified_student_taskid_online_r2_scaled20_v1.yaml"
)
INIT = (
    ROOT / "custom_tools/runs/bc/"
    / "unified_student_taskid_online_r1_frac025_seed2025_e10_v1/"
    / "epoch=001-step=2232.ckpt"
)
AGGREGATED = (
    ROOT / "custom_tools/data/distillation/"
    / "online_taskid_scaled20_r1_r2_aggregated.npz"
)
RUN_NAME = (
    "unified_student_taskid_online_r2_agg12_frac025_seed2025_e4_v1"
)
RUN_DIR = ROOT / "custom_tools/runs/bc" / RUN_NAME
SELECTION = ROOT / "custom_tools/configs/scaled_development_all12.yaml"
TRAJECTORY_ROOT = (
    ROOT / "dexgrasp/dataset/scaled_category_final_v1_preprocessed"
)
RESIDUAL_CONFIG = ROOT / "custom_tools/configs/residual_ppo_stage1.yaml"
OUTPUT_ROOT = ROOT / "custom_tools/results/taskid_online_r2_development_v1"
R1_RESULT = (
    ROOT / "custom_tools/results/taskid_online_r1_development_v1/"
    / "online_25pct_epoch02.yaml"
)
EPOCHS = (1, 2, 3, 4)


def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-free-vram-mb", type=int, default=4500)
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run(command, dry_run=False):
    print("RUN: {}".format(" ".join(str(item) for item in command)), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=str(ROOT), check=True)


def load_yaml(path):
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def checkpoint(epoch):
    matches = list(RUN_DIR.glob(
        "epoch={:03d}-step=*.ckpt".format(epoch - 1)
    ))
    if len(matches) != 1:
        raise RuntimeError(
            "Expected one epoch {} checkpoint in {}".format(epoch, RUN_DIR)
        )
    return matches[0].resolve()


def result(path, expected_checkpoint):
    aggregate = load_yaml(path)
    rows = aggregate.get("checkpoint_results", [])
    if len(rows) != 1:
        raise RuntimeError("Expected one checkpoint result in {}".format(path))
    if Path(rows[0]["checkpoint"]).resolve() != expected_checkpoint.resolve():
        raise RuntimeError("Checkpoint mismatch in {}".format(path))
    return rows[0]


def evaluate(cli, epoch):
    model = checkpoint(epoch)
    output = OUTPUT_ROOT / "online_r2_epoch{:02d}.yaml".format(epoch)
    if output.is_file():
        result(output, model)
        print("[REUSE] online R2 epoch {}".format(epoch), flush=True)
        return output
    command = [
        sys.executable,
        "-u",
        str(ROOT / "custom_tools/evaluate_bc_checkpoints_isolated.py"),
        "--checkpoint",
        str(model),
        "--bc-config",
        str(CONFIG),
        "--residual-config",
        str(RESIDUAL_CONFIG),
        "--trajectory-root",
        str(TRAJECTORY_ROOT),
        "--object-selection",
        str(SELECTION),
        "--output",
        str(output),
        "--seed",
        "2025",
        "--min-free-vram-mb",
        str(cli.min_free_vram_mb),
        "--max-attempts",
        str(cli.max_attempts),
    ]
    run(command)
    result(output, model)
    return output


def compact(label, method, epoch, model, output):
    item = result(output, model)
    return {
        "label": label,
        "method": method,
        "epoch": epoch,
        "checkpoint": str(model),
        "success_count": int(item["total_success_count"]),
        "trajectory_count": int(item["total_trajectory_count"]),
        "overall_success_rate": float(
            item["overall_official_peak_success_rate"]
        ),
        "macro_success_rate": float(
            item["macro_official_peak_success_rate"]
        ),
        "mean_maximum_lift_m": float(
            item["macro_mean_maximum_lift_m"]
        ),
        "failure_rate": float(item["macro_failure_rate"]),
        "category_macro_success_rates": item[
            "category_macro_success_rates"
        ],
        "result": str(output),
    }


def main():
    cli = parse_cli()
    if cli.max_attempts < 1:
        raise ValueError("--max-attempts must be positive")
    required = [
        CONFIG,
        INIT,
        SELECTION,
        TRAJECTORY_ROOT,
        RESIDUAL_CONFIG,
        R1_RESULT,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing online R2 inputs: {}".format(missing))

    if not AGGREGATED.is_file():
        run([
            sys.executable,
            str(ROOT / "custom_tools/merge_taskid_online_rounds.py"),
        ], dry_run=cli.dry_run)
    if cli.dry_run and not AGGREGATED.is_file():
        print("DRY_RUN: training follows successful aggregation.", flush=True)
        return
    data = np.load(AGGREGATED, allow_pickle=False)
    try:
        if data["observations"].shape != (44160, 2460):
            raise RuntimeError("Unexpected aggregated observation shape")
        if np.bincount(
            data["category_indices"].astype(np.int64), minlength=4
        ).tolist() != [11040, 11040, 11040, 11040]:
            raise RuntimeError("Aggregated categories are not balanced")
    finally:
        data.close()

    last = RUN_DIR / "last.ckpt"
    resource = RUN_DIR / "resource_summary.yaml"
    if last.is_file() and resource.is_file():
        print("[REUSE] completed R2 training: {}".format(RUN_DIR), flush=True)
    else:
        existing = list(RUN_DIR.glob("*.ckpt"))
        if existing:
            raise RuntimeError(
                "Partial online R2 training needs inspection: {}".format(
                    RUN_DIR
                )
            )
        run([
            sys.executable,
            "-u",
            str(ROOT / "custom_tools/train_bc.py"),
            "--config",
            str(CONFIG),
            "--run-name",
            RUN_NAME,
            "--seed",
            "2025",
            "--num-epochs",
            "4",
            "--learning-rate",
            "2e-5",
            "--teacher-weight",
            "1.0",
            "--online-sample-fraction",
            "0.25",
            "--init-checkpoint",
            str(INIT),
            "--min-free-vram-mb",
            str(cli.min_free_vram_mb),
        ], dry_run=cli.dry_run)
    if cli.dry_run:
        print(
            "DRY_RUN: four R2 checkpoints will be evaluated on the frozen "
            "12-object development split.",
            flush=True,
        )
        return

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    outputs = {epoch: evaluate(cli, epoch) for epoch in EPOCHS}
    r1_checkpoint = INIT
    rows = [
        compact(
            "online_r1_control",
            "online_r1",
            2,
            r1_checkpoint,
            R1_RESULT,
        )
    ]
    rows.extend(
        compact(
            "online_r2_epoch{:02d}".format(epoch),
            "online_r2_aggregated",
            epoch,
            checkpoint(epoch),
            outputs[epoch],
        )
        for epoch in EPOCHS
    )
    rows.sort(
        key=lambda row: (
            row["macro_success_rate"],
            row["mean_maximum_lift_m"],
            -row["failure_rate"],
        ),
        reverse=True,
    )
    summary = {
        "status": "complete",
        "stage": "Task-ID online imitation round 2",
        "formal_final_holdout_result": False,
        "final_holdout_accessed": False,
        "selection_metric": (
            "object-macro official peak success; lift/failure tie-break only"
        ),
        "r1_r2_trajectory_pair_overlap": 0,
        "aggregated_online_samples": 44160,
        "online_sampling_fraction": 0.25,
        "ranking": rows,
    }
    with (OUTPUT_ROOT / "summary.yaml").open(
        "w", encoding="utf-8"
    ) as handle:
        yaml.safe_dump(summary, handle, allow_unicode=True, sort_keys=False)
    print("TASKID_ONLINE_R2_STAGE=COMPLETE", flush=True)
    for rank, row in enumerate(rows, 1):
        print(
            "#{:02d} {} success={}/{} macro={:.2f}% lift={:.3f}m "
            "failure={:.2f}%".format(
                rank,
                row["label"],
                row["success_count"],
                row["trajectory_count"],
                100 * row["macro_success_rate"],
                row["mean_maximum_lift_m"],
                100 * row["failure_rate"],
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
