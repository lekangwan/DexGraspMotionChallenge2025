"""Train and screen a frozen-Temporal3 attention residual."""

import argparse
from pathlib import Path
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT / "custom_tools/configs/"
    / "unified_student_taskid_temporal3_attention_v1.yaml"
)
INIT = (
    ROOT / "custom_tools/runs/bc/"
    / "unified_student_taskid_temporal3_seed2025_e4_v1/"
    / "epoch=003-step=5152.ckpt"
)
RUN_NAME = "unified_student_taskid_temporal3_attnres_seed2025_e4_v1"
RUN_DIR = ROOT / "custom_tools/runs/bc" / RUN_NAME
SELECTION = ROOT / "custom_tools/configs/scaled_development_all12.yaml"
TRAJECTORY_ROOT = (
    ROOT / "dexgrasp/dataset/scaled_category_final_v1_preprocessed"
)
RESIDUAL_CONFIG = ROOT / "custom_tools/configs/residual_ppo_stage1.yaml"
OUTPUT_ROOT = (
    ROOT / "custom_tools/results/taskid_temporal_attention_development_v1"
)
CONTROL_RESULT = (
    ROOT / "custom_tools/results/taskid_temporal3_development_v1/"
    / "temporal3_epoch04.yaml"
)
EPOCHS = (1, 2, 3, 4)
REPEAT_MARGIN = 0.02


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
            "Expected one epoch {} checkpoint in {}".format(epoch, RUN_DIR))
    return matches[0].resolve()


def read_result(path, expected_checkpoint):
    aggregate = load_yaml(path)
    rows = aggregate.get("checkpoint_results", [])
    if len(rows) != 1:
        raise RuntimeError("Expected one checkpoint result in {}".format(path))
    if Path(rows[0]["checkpoint"]).resolve() != expected_checkpoint.resolve():
        raise RuntimeError("Checkpoint mismatch in {}".format(path))
    return rows[0]


def evaluate(cli, epoch):
    model = checkpoint(epoch)
    output = OUTPUT_ROOT / "attention_epoch{:02d}.yaml".format(epoch)
    if output.is_file():
        read_result(output, model)
        print("[REUSE] attention epoch {}".format(epoch), flush=True)
        return output
    run([
        sys.executable,
        "-u",
        str(ROOT / "custom_tools/evaluate_bc_checkpoints_isolated.py"),
        "--checkpoint", str(model),
        "--bc-config", str(CONFIG),
        "--residual-config", str(RESIDUAL_CONFIG),
        "--trajectory-root", str(TRAJECTORY_ROOT),
        "--object-selection", str(SELECTION),
        "--output", str(output),
        "--seed", "2025",
        "--min-free-vram-mb", str(cli.min_free_vram_mb),
        "--max-attempts", str(cli.max_attempts),
    ])
    read_result(output, model)
    return output


def compact(label, method, epoch, model, output):
    item = read_result(output, model)
    return {
        "label": label,
        "method": method,
        "epoch": epoch,
        "checkpoint": str(model),
        "success_count": int(item["total_success_count"]),
        "trajectory_count": int(item["total_trajectory_count"]),
        "overall_success_rate": float(
            item["overall_official_peak_success_rate"]),
        "macro_success_rate": float(
            item["macro_official_peak_success_rate"]),
        "mean_maximum_lift_m": float(
            item["macro_mean_maximum_lift_m"]),
        "failure_rate": float(item["macro_failure_rate"]),
        "category_macro_success_rates": item[
            "category_macro_success_rates"],
        "result": str(output),
    }


def main():
    cli = parse_cli()
    if cli.max_attempts < 1:
        raise ValueError("--max-attempts must be positive")
    required = [
        CONFIG, INIT, SELECTION, TRAJECTORY_ROOT, RESIDUAL_CONFIG,
        CONTROL_RESULT,
        ROOT / "custom_tools/data/distillation/"
        / "online_taskid_scaled20_r1_r2_aggregated.npz",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing attention-stage inputs: {}".format(missing))

    last = RUN_DIR / "last.ckpt"
    resource = RUN_DIR / "resource_summary.yaml"
    if last.is_file() and resource.is_file():
        print("[REUSE] completed attention training: {}".format(
            RUN_DIR), flush=True)
    else:
        existing = list(RUN_DIR.glob("*.ckpt"))
        if existing:
            raise RuntimeError(
                "Partial attention training needs inspection: {}".format(
                    RUN_DIR))
        run([
            sys.executable,
            "-u",
            str(ROOT / "custom_tools/train_bc.py"),
            "--config", str(CONFIG),
            "--run-name", RUN_NAME,
            "--seed", "2025",
            "--num-epochs", "4",
            "--learning-rate", "1e-4",
            "--teacher-weight", "1.0",
            "--online-sample-fraction", "0.25",
            "--init-checkpoint", str(INIT),
            "--min-free-vram-mb", str(cli.min_free_vram_mb),
        ], dry_run=cli.dry_run)
    if cli.dry_run:
        print(
            "DRY_RUN: four attention checkpoints will be evaluated on the "
            "frozen 12-object development split.",
            flush=True)
        return

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    outputs = {epoch: evaluate(cli, epoch) for epoch in EPOCHS}
    control = compact(
        "temporal3_control", "temporal3", 4, INIT, CONTROL_RESULT)
    candidates = [
        compact(
            "attention_epoch{:02d}".format(epoch),
            "temporal3_attention_residual",
            epoch,
            checkpoint(epoch),
            outputs[epoch])
        for epoch in EPOCHS
    ]
    candidates.sort(
        key=lambda row: (
            row["macro_success_rate"],
            row["mean_maximum_lift_m"],
            -row["failure_rate"],
        ),
        reverse=True)
    best = candidates[0]
    margin = best["macro_success_rate"] - control["macro_success_rate"]
    summary = {
        "status": "complete",
        "stage": "frozen Temporal3 plus three-token attention residual",
        "formal_final_holdout_result": False,
        "final_holdout_accessed": False,
        "control": control,
        "attention_ranking": candidates,
        "architecture": {
            "tokens": (
                "three compact tokens: (p[t-2],a[t-2]), "
                "(p[t-1],a[t-1]), and (p[t],zero-placeholder)"),
            "transformer": (
                "2 encoder layers, 4 heads, model_dim=128, ff_dim=256"),
            "output": "zero-initialized 28-D residual added to Temporal3",
            "frozen": "entire pre-trained Temporal3 base",
        },
        "selection_metric": (
            "object-macro official peak success; lift/failure tie-break only"),
        "repeat_rule": (
            "repeat only if best attention macro is at least 2 percentage "
            "points above the seed-2025 Temporal3 control"),
        "repeat_threshold": (
            control["macro_success_rate"] + REPEAT_MARGIN),
        "best_attention_minus_control": margin,
        "repeat_recommended": margin >= REPEAT_MARGIN,
    }
    with (OUTPUT_ROOT / "summary.yaml").open(
        "w", encoding="utf-8"
    ) as handle:
        yaml.safe_dump(summary, handle, allow_unicode=True, sort_keys=False)
    print("TASKID_TEMPORAL_ATTENTION_STAGE=COMPLETE", flush=True)
    print(
        "control macro={:.2f}% threshold={:.2f}%".format(
            100 * control["macro_success_rate"],
            100 * summary["repeat_threshold"]),
        flush=True)
    for rank, row in enumerate(candidates, 1):
        print(
            "#{:02d} {} success={}/{} macro={:.2f}% lift={:.3f}m "
            "failure={:.2f}%".format(
                rank,
                row["label"],
                row["success_count"],
                row["trajectory_count"],
                100 * row["macro_success_rate"],
                row["mean_maximum_lift_m"],
                100 * row["failure_rate"]),
            flush=True)
    print("REPEAT_RECOMMENDED={}".format(
        summary["repeat_recommended"]), flush=True)


if __name__ == "__main__":
    main()
