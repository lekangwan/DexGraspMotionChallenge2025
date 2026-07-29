"""Screen two global Online-R1/Temporal3 action blends on development only."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
ONLINE_CHECKPOINT = (
    ROOT / "custom_tools/runs/bc/"
    / "unified_student_taskid_online_r1_frac025_seed2025_e10_v1/"
    / "epoch=001-step=2232.ckpt"
)
ONLINE_CONFIG = (
    ROOT / "custom_tools/configs/"
    / "unified_student_taskid_online_r1_scaled20_v1.yaml"
)
TEMPORAL_CHECKPOINT = (
    ROOT / "custom_tools/runs/bc/"
    / "unified_student_taskid_temporal3_seed2025_e4_v1/"
    / "epoch=003-step=5152.ckpt"
)
TEMPORAL_CONFIG = (
    ROOT / "custom_tools/configs/unified_student_taskid_temporal3_v1.yaml"
)
RESIDUAL_CONFIG = (
    ROOT / "custom_tools/configs/residual_ppo_stage1.yaml"
)
TRAJECTORY_ROOT = (
    ROOT / "dexgrasp/dataset/scaled_category_final_v1_preprocessed"
)
SELECTION = (
    ROOT / "custom_tools/configs/scaled_development_all12.yaml"
)
PROTOCOL = (
    ROOT / "custom_tools/configs/scaled_evaluation_protocol_v1.json"
)
ONLINE_RESULT = (
    ROOT / "custom_tools/results/taskid_online_r1_development_v1/"
    / "online_25pct_epoch02.yaml"
)
TEMPORAL_RESULT = (
    ROOT / "custom_tools/results/taskid_temporal3_development_v1/"
    / "temporal3_epoch04.yaml"
)
OUTPUT_ROOT = (
    ROOT / "custom_tools/results/taskid_action_blend_development_v1"
)
WEIGHTS = (0.50, 0.75)
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


def endpoint_result(path, expected_checkpoint):
    rows = load_yaml(path).get("checkpoint_results", [])
    if len(rows) != 1:
        raise RuntimeError(
            "Expected one endpoint result in {}".format(path))
    if Path(rows[0]["checkpoint"]).resolve() != expected_checkpoint.resolve():
        raise RuntimeError("Endpoint checkpoint mismatch: {}".format(path))
    return rows[0]


def verify_no_holdout_access():
    development = set(load_yaml(SELECTION)["object_ids"])
    with PROTOCOL.open(encoding="utf-8") as handle:
        protocol = json.load(handle)
    final_holdout = {
        object_id
        for category in protocol["categories"].values()
        for object_id in category["final_holdout"]
    }
    overlap = development & final_holdout
    if overlap:
        raise RuntimeError(
            "Development selection overlaps final holdout: {}".format(
                sorted(overlap)))


def evaluate(cli, weight):
    label = "lambda{:03d}_seed2025".format(round(100 * weight))
    output = OUTPUT_ROOT / (label + ".yaml")
    if output.exists():
        result = load_yaml(output)["blend_result"]
        if float(result["temporal_weight"]) != weight:
            raise RuntimeError("Blend result weight mismatch: {}".format(
                output))
        print("[REUSE] {}".format(label), flush=True)
        return output
    command = [
        sys.executable, "-u",
        str(ROOT / "custom_tools/evaluate_action_blend_isolated.py"),
        "--online-checkpoint", str(ONLINE_CHECKPOINT),
        "--online-config", str(ONLINE_CONFIG),
        "--temporal-checkpoint", str(TEMPORAL_CHECKPOINT),
        "--temporal-config", str(TEMPORAL_CONFIG),
        "--temporal-weight", str(weight),
        "--residual-config", str(RESIDUAL_CONFIG),
        "--trajectory-root", str(TRAJECTORY_ROOT),
        "--object-selection", str(SELECTION),
        "--output", str(output),
        "--seed", "2025",
        "--min-free-vram-mb", str(cli.min_free_vram_mb),
        "--max-attempts", str(cli.max_attempts),
    ]
    print("RUN {}".format(" ".join(command)), flush=True)
    if not cli.dry_run:
        subprocess.run(command, cwd=str(ROOT), check=True)
    return output


def compact(label, method, weight, result, path):
    return {
        "label": label,
        "method": method,
        "temporal_weight": weight,
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
        "result": str(path),
    }


def main():
    cli = parse_cli()
    required = (
        ONLINE_CHECKPOINT, ONLINE_CONFIG,
        TEMPORAL_CHECKPOINT, TEMPORAL_CONFIG,
        RESIDUAL_CONFIG, TRAJECTORY_ROOT, SELECTION, PROTOCOL,
        ONLINE_RESULT, TEMPORAL_RESULT,
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing action-blend inputs: {}".format(missing))
    verify_no_holdout_access()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    outputs = {
        weight: evaluate(cli, weight)
        for weight in WEIGHTS
    }
    if cli.dry_run:
        print(
            "DRY_RUN: two development-only blends would be evaluated.",
            flush=True)
        return

    online = endpoint_result(ONLINE_RESULT, ONLINE_CHECKPOINT)
    temporal = endpoint_result(TEMPORAL_RESULT, TEMPORAL_CHECKPOINT)
    rows = [
        compact(
            "online_r1_control", "endpoint_control", 0.0,
            online, ONLINE_RESULT),
        compact(
            "temporal3_control", "endpoint_control", 1.0,
            temporal, TEMPORAL_RESULT),
    ]
    for weight, path in outputs.items():
        rows.append(compact(
            "global_blend_lambda_{:.2f}".format(weight),
            "global_action_blend", weight,
            load_yaml(path)["blend_result"], path))
    rows.sort(
        key=lambda row: (
            row["macro_success_rate"],
            row["mean_maximum_lift_m"],
            -row["failure_rate"],
        ),
        reverse=True)

    best_control = max(
        row["macro_success_rate"]
        for row in rows if row["method"] == "endpoint_control")
    blends = [
        row for row in rows if row["method"] == "global_action_blend"]
    best_blend = max(blends, key=lambda row: row["macro_success_rate"])
    improvement = best_blend["macro_success_rate"] - best_control
    repeat_recommended = improvement >= REPEAT_MARGIN
    summary = {
        "status": "complete",
        "stage": "global Online-R1/Temporal3 action shrinkage",
        "formal_final_holdout_result": False,
        "final_holdout_accessed": False,
        "training_performed": False,
        "seed": 2025,
        "candidate_weights": list(WEIGHTS),
        "action_formula": (
            "(1 - temporal_weight) * Online-R1 + "
            "temporal_weight * Temporal3"),
        "selection_metric": (
            "object-macro official peak success; lift/failure tie-break only"),
        "predeclared_repeat_rule": (
            "repeat seeds 2026/2027 only if the best blend exceeds the "
            "better endpoint by at least 2.00 percentage points"),
        "best_endpoint_macro_success": best_control,
        "best_blend_label": best_blend["label"],
        "best_blend_improvement": improvement,
        "repeat_recommended": repeat_recommended,
        "ranking": rows,
    }
    summary_path = OUTPUT_ROOT / "summary.yaml"
    with summary_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            summary, handle, allow_unicode=True, sort_keys=False)
    print("ACTION_BLEND_STAGE=COMPLETE", flush=True)
    for rank, row in enumerate(rows, 1):
        print(
            "#{:02d} {} success={}/{} macro={:.2f}% lift={:.3f}m "
            "failure={:.2f}%".format(
                rank, row["label"], row["success_count"],
                row["trajectory_count"],
                100 * row["macro_success_rate"],
                row["mean_maximum_lift_m"],
                100 * row["failure_rate"],
            ),
            flush=True,
        )
    print(
        "REPEAT_RECOMMENDED={}".format(repeat_recommended),
        flush=True)


if __name__ == "__main__":
    main()
