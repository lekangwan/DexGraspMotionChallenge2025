"""Strictly screen all offline Task-ID student checkpoints on frozen dev data."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
BC_CONFIG = (
    REPO_ROOT
    / "custom_tools/configs/unified_student_taskid_scaled20_v1.yaml"
)
RESIDUAL_CONFIG = (
    REPO_ROOT / "custom_tools/configs/residual_ppo_stage1.yaml"
)
TRAJECTORY_ROOT = (
    REPO_ROOT / "dexgrasp/dataset/scaled_category_final_v1_preprocessed"
)
SELECTION = (
    REPO_ROOT / "custom_tools/configs/scaled_development_all12.yaml"
)
PROTOCOL = (
    REPO_ROOT / "custom_tools/configs/scaled_evaluation_protocol_v1.json"
)
OUTPUT_ROOT = (
    REPO_ROOT / "custom_tools/results/taskid_offline_development_v1"
)
RUNS = {
    "t100": (
        REPO_ROOT / "custom_tools/runs/bc/"
        / "unified_student_taskid_scaled20_t100_seed2025_e20_v1"
    ),
    "t70_demo30": (
        REPO_ROOT / "custom_tools/runs/bc/"
        / "unified_student_taskid_scaled20_t70_demo30_seed2025_e20_v1"
    ),
}
EPOCHS = (5, 10, 15, 20)


def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--min-free-vram-mb", type=int, default=4500)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_yaml(path):
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def validate_selection():
    selection = load_yaml(SELECTION)
    with PROTOCOL.open(encoding="utf-8") as handle:
        protocol = json.load(handle)
    expected_dev = {
        object_id
        for item in protocol["categories"].values()
        for object_id in item["development"]
    }
    holdout = {
        object_id
        for item in protocol["categories"].values()
        for object_id in item["final_holdout"]
    }
    actual = set(selection["object_ids"])
    if actual != expected_dev or actual & holdout or len(actual) != 12:
        raise RuntimeError("Development selection no longer matches protocol")
    if selection.get("final_holdout_accessed") is not False:
        raise RuntimeError("Selection must explicitly keep holdout inaccessible")


def checkpoint(run_dir, epoch):
    matches = list(run_dir.glob(
        "epoch={:03d}-step=*.ckpt".format(epoch - 1)))
    if len(matches) != 1:
        raise RuntimeError(
            "Expected one epoch {} checkpoint in {}".format(epoch, run_dir))
    return matches[0].resolve()


def candidates():
    result = []
    for method, run_dir in RUNS.items():
        for epoch in EPOCHS:
            result.append({
                "method": method,
                "epoch": epoch,
                "label": "{}_epoch{:02d}".format(method, epoch),
                "checkpoint": checkpoint(run_dir, epoch),
            })
    return result


def validate_result(path, expected_checkpoint):
    result = load_yaml(path)
    rows = result.get("checkpoint_results", [])
    if len(rows) != 1:
        raise RuntimeError("Expected one checkpoint result in {}".format(path))
    if Path(rows[0]["checkpoint"]).resolve() != expected_checkpoint.resolve():
        raise RuntimeError("Checkpoint mismatch in {}".format(path))
    return rows[0]


def main():
    cli = parse_cli()
    validate_selection()
    stage_candidates = candidates()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(
        "Frozen development screening: candidates={} objects=12 seed={}"
        .format(len(stage_candidates), cli.seed),
        flush=True)
    for index, candidate in enumerate(stage_candidates, 1):
        output = (
            OUTPUT_ROOT / "seed{}".format(cli.seed)
            / (candidate["label"] + ".yaml")
        )
        if output.is_file():
            validate_result(output, candidate["checkpoint"])
            print(
                "[{}/{}] REUSE {}".format(
                    index, len(stage_candidates), candidate["label"]),
                flush=True)
            continue
        command = [
            sys.executable,
            "-u",
            str(REPO_ROOT / "custom_tools/evaluate_bc_checkpoints_isolated.py"),
            "--checkpoint",
            str(candidate["checkpoint"]),
            "--bc-config",
            str(BC_CONFIG),
            "--residual-config",
            str(RESIDUAL_CONFIG),
            "--trajectory-root",
            str(TRAJECTORY_ROOT),
            "--object-selection",
            str(SELECTION),
            "--output",
            str(output),
            "--seed",
            str(cli.seed),
            "--min-free-vram-mb",
            str(cli.min_free_vram_mb),
            "--max-attempts",
            str(cli.max_attempts),
        ]
        print(
            "[{}/{}] RUN {}".format(
                index, len(stage_candidates), candidate["label"]),
            flush=True)
        if cli.dry_run:
            print(" ".join(command), flush=True)
            continue
        subprocess.run(command, cwd=str(REPO_ROOT), check=True)
        validate_result(output, candidate["checkpoint"])

    if cli.dry_run:
        return
    rows = []
    for candidate in stage_candidates:
        output = (
            OUTPUT_ROOT / "seed{}".format(cli.seed)
            / (candidate["label"] + ".yaml")
        )
        result = validate_result(output, candidate["checkpoint"])
        rows.append({
            **{key: candidate[key] for key in ("method", "epoch", "label")},
            "checkpoint": str(candidate["checkpoint"]),
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
        key=lambda item: (
            item["macro_success_rate"],
            item["mean_maximum_lift_m"],
            -item["failure_rate"]),
        reverse=True)
    summary = {
        "status": "complete",
        "formal_final_holdout_result": False,
        "final_holdout_accessed": False,
        "seed": cli.seed,
        "selection": str(SELECTION),
        "candidate_count": len(rows),
        "selection_metric": (
            "object-macro official peak success; lift/failure tie-break only"),
        "ranking": rows,
    }
    summary_path = OUTPUT_ROOT / "summary_seed{}.yaml".format(cli.seed)
    with summary_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(summary, handle, allow_unicode=True, sort_keys=False)
    print("TASKID_OFFLINE_EVALUATION=COMPLETE")
    for rank, row in enumerate(rows, 1):
        print(
            "#{:02d} {} success={}/{} macro={:.2f}% lift={:.3f}m "
            "failure={:.2f}%".format(
                rank, row["label"], row["success_count"],
                row["trajectory_count"], 100 * row["macro_success_rate"],
                row["mean_maximum_lift_m"], 100 * row["failure_rate"]))


if __name__ == "__main__":
    main()
