"""Train and strictly screen two controlled Task-ID online-data weights."""

import argparse
from pathlib import Path
import subprocess
import sys

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    REPO_ROOT
    / "custom_tools/configs/unified_student_taskid_online_r1_scaled20_v1.yaml"
)
OFFLINE = (
    REPO_ROOT / "custom_tools/runs/bc/"
    / "unified_student_taskid_scaled20_t100_seed2025_e20_v1/"
    / "epoch=014-step=14145.ckpt"
)
ONLINE_DATA = (
    REPO_ROOT / "custom_tools/data/distillation/"
    / "online_taskid_scaled20_r1_train4.npz"
)
SELECTION = (
    REPO_ROOT / "custom_tools/configs/scaled_development_all12.yaml"
)
TRAJECTORY_ROOT = (
    REPO_ROOT / "dexgrasp/dataset/scaled_category_final_v1_preprocessed"
)
RESIDUAL_CONFIG = (
    REPO_ROOT / "custom_tools/configs/residual_ppo_stage1.yaml"
)
OUTPUT_ROOT = (
    REPO_ROOT / "custom_tools/results/taskid_online_r1_development_v1"
)
RUNS = {
    "online_natural": {
        "run_name": (
            "unified_student_taskid_online_r1_natural_seed2025_e10_v1"),
        "online_fraction": None,
    },
    "online_25pct": {
        "run_name": (
            "unified_student_taskid_online_r1_frac025_seed2025_e10_v1"),
        "online_fraction": 0.25,
    },
}
EPOCHS = (2, 4, 6, 8, 10)


def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-free-vram-mb", type=int, default=4500)
    parser.add_argument(
        "--max-attempts", type=int, default=5,
        help=(
            "Fresh-process retries for occasional PhysX mesh initialization "
            "failures (default: 5)."))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run(command, dry_run=False):
    print("RUN: {}".format(" ".join(str(item) for item in command)), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=str(REPO_ROOT), check=True)


def verify_online_data():
    data = np.load(ONLINE_DATA, allow_pickle=False)
    if data["observations"].shape != (22080, 2460):
        raise RuntimeError("Unexpected online observation shape")
    if data["teacher_actions"].shape != (22080, 28):
        raise RuntimeError("Unexpected online action shape")
    if not np.isfinite(data["observations"]).all():
        raise RuntimeError("Online observations contain non-finite values")
    counts = np.bincount(
        data["category_indices"].astype(np.int64), minlength=4)
    if counts.tolist() != [5520, 5520, 5520, 5520]:
        raise RuntimeError("Online category counts changed: {}".format(counts))


def train_command(cli, settings):
    command = [
        sys.executable,
        "-u",
        str(REPO_ROOT / "custom_tools/train_bc.py"),
        "--config",
        str(CONFIG),
        "--run-name",
        settings["run_name"],
        "--seed",
        "2025",
        "--num-epochs",
        "10",
        "--learning-rate",
        "2e-5",
        "--teacher-weight",
        "1.0",
        "--init-checkpoint",
        str(OFFLINE),
        "--min-free-vram-mb",
        str(cli.min_free_vram_mb),
    ]
    if settings["online_fraction"] is not None:
        command.extend([
            "--online-sample-fraction",
            str(settings["online_fraction"]),
        ])
    return command


def checkpoint(run_dir, epoch):
    matches = list(run_dir.glob(
        "epoch={:03d}-step=*.ckpt".format(epoch - 1)))
    if len(matches) != 1:
        raise RuntimeError(
            "Expected one epoch {} checkpoint in {}".format(epoch, run_dir))
    return matches[0].resolve()


def read_result(path, expected_checkpoint):
    with path.open(encoding="utf-8") as handle:
        aggregate = yaml.safe_load(handle)
    rows = aggregate.get("checkpoint_results", [])
    if len(rows) != 1:
        raise RuntimeError("Expected one checkpoint result in {}".format(path))
    if Path(rows[0]["checkpoint"]).resolve() != expected_checkpoint.resolve():
        raise RuntimeError("Checkpoint mismatch in {}".format(path))
    return rows[0]


def evaluate(cli, label, checkpoint_path):
    output = OUTPUT_ROOT / (label + ".yaml")
    if output.is_file():
        read_result(output, checkpoint_path)
        print("REUSE: {}".format(output), flush=True)
        return output
    command = [
        sys.executable,
        "-u",
        str(REPO_ROOT / "custom_tools/evaluate_bc_checkpoints_isolated.py"),
        "--checkpoint",
        str(checkpoint_path),
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
    read_result(output, checkpoint_path)
    return output


def main():
    cli = parse_cli()
    required = [
        CONFIG, OFFLINE, ONLINE_DATA, SELECTION, TRAJECTORY_ROOT,
        RESIDUAL_CONFIG]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing online stage inputs: {}".format(
            missing))
    verify_online_data()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    run_dirs = {}
    for method, settings in RUNS.items():
        run_dir = (
            REPO_ROOT / "custom_tools/runs/bc" / settings["run_name"])
        run_dirs[method] = run_dir
        last = run_dir / "last.ckpt"
        resource = run_dir / "resource_summary.yaml"
        if last.is_file() and resource.is_file():
            print("REUSE completed training: {}".format(run_dir), flush=True)
            continue
        existing = list(run_dir.glob("*.ckpt"))
        if existing:
            raise RuntimeError(
                "Partial online training needs inspection: {}".format(
                    run_dir))
        run(train_command(cli, settings), dry_run=cli.dry_run)

    if cli.dry_run:
        print(
            "DRY_RUN: after training, 10 checkpoints will be evaluated on "
            "the frozen 12-object development split.")
        return

    outputs = {}
    for method, run_dir in run_dirs.items():
        for epoch in EPOCHS:
            label = "{}_epoch{:02d}".format(method, epoch)
            path = checkpoint(run_dir, epoch)
            outputs[label] = evaluate(cli, label, path)

    rows = []
    for label, output in outputs.items():
        method, epoch_text = label.rsplit("_epoch", 1)
        path = checkpoint(run_dirs[method], int(epoch_text))
        result = read_result(output, path)
        rows.append({
            "label": label,
            "method": method,
            "epoch": int(epoch_text),
            "checkpoint": str(path),
            "success_count": int(result["total_success_count"]),
            "trajectory_count": int(result["total_trajectory_count"]),
            "overall_success_rate": float(
                result["overall_official_peak_success_rate"]),
            "macro_success_rate": float(
                result["macro_official_peak_success_rate"]),
            "mean_maximum_lift_m": float(
                result["macro_mean_maximum_lift_m"]),
            "failure_rate": float(result["macro_failure_rate"]),
            "category_macro_success_rates": result[
                "category_macro_success_rates"],
            "result": str(output),
        })
    rows.sort(
        key=lambda row: (
            row["macro_success_rate"],
            row["mean_maximum_lift_m"],
            -row["failure_rate"]),
        reverse=True)
    summary = {
        "status": "complete",
        "stage": "Task-ID online imitation round 1",
        "formal_final_holdout_result": False,
        "final_holdout_accessed": False,
        "offline_control_checkpoint": str(OFFLINE),
        "online_samples": 22080,
        "candidate_count": len(rows),
        "selection_metric": (
            "object-macro official peak success; lift/failure tie-break only"),
        "ranking": rows,
    }
    summary_path = OUTPUT_ROOT / "summary.yaml"
    with summary_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(summary, handle, allow_unicode=True, sort_keys=False)
    print("TASKID_ONLINE_R1_STAGE=COMPLETE")
    for rank, row in enumerate(rows, 1):
        print(
            "#{:02d} {} success={}/{} macro={:.2f}% lift={:.3f}m "
            "failure={:.2f}%".format(
                rank, row["label"], row["success_count"],
                row["trajectory_count"], 100 * row["macro_success_rate"],
                row["mean_maximum_lift_m"], 100 * row["failure_rate"]))


if __name__ == "__main__":
    main()
