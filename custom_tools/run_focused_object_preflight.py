"""Replay demonstrations and current experts on three focused train objects."""

import argparse
from datetime import datetime
from pathlib import Path
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "custom_tools/results/focused_object_preflight"
BC_CONFIG = ROOT / "custom_tools/configs/multicategory_bc_noise005.yaml"
OBJECTS = {
    "bottle_dc005": {
        "selection": ROOT / "custom_tools/configs/focused_bottle_dc005_train_all.yaml",
        "checkpoint": ROOT / "custom_tools/runs/bc/category_expert_bottle_noise005_soup_seed2025_e40_v1/epoch=039-step=2560.ckpt",
    },
    "mug_b4ae": {
        "selection": ROOT / "custom_tools/configs/focused_mug_b4ae_train_all.yaml",
        "checkpoint": ROOT / "custom_tools/runs/bc/model_soups/noise005_s2025_s2026_weighted2to1.ckpt",
    },
    "camera_82819": {
        "selection": ROOT / "custom_tools/configs/focused_camera_82819_train_all.yaml",
        "checkpoint": ROOT / "custom_tools/runs/bc/category_expert_camera_noise005_soup_seed2025_e40_v1/epoch=009-step=500.ckpt",
    },
}


def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-free-vram-mb", type=int, default=4500)
    return parser.parse_args()


def main():
    cli = parse_cli()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for label, item in OBJECTS.items():
        directory = OUTPUT / label
        summary = directory / "summary.yaml"
        if not summary.is_file():
            command = [
                sys.executable,
                str(ROOT / "custom_tools/diagnose_bc_closed_loop.py"),
                "--trajectory-selection", str(item["selection"]),
                "--trajectory-root", str(
                    ROOT / "dexgrasp/dataset/bc_multicategory_train"),
                "--bc-checkpoint", str(item["checkpoint"]),
                "--bc-config", str(BC_CONFIG),
                "--output-dir", str(directory),
                "--seed", "2025",
                "--horizon", "122",
                "--min-free-vram-mb", str(cli.min_free_vram_mb),
            ]
            print("RUN {}: {}".format(label, " ".join(command)), flush=True)
            subprocess.run(command, cwd=str(ROOT), check=True)
        else:
            print("REUSE {}: {}".format(label, summary), flush=True)
        with summary.open(encoding="utf-8") as handle:
            result = yaml.safe_load(handle)
        expert = result["expert_action_replay"]
        policy = result["bc_closed_loop_rollout"]
        error = result["overall_error"]
        rows.append({
            "label": label,
            "object_id": result["trajectories"][0]["object_id"],
            "trajectory_count": result["trajectory_count"],
            "expert_success_count": expert["official_peak_success_count"],
            "expert_success_rate": expert["official_peak_success_rate"],
            "expert_mean_lift_m": expert["mean_maximum_lift_m"],
            "current_policy_success_count": policy[
                "official_peak_success_count"],
            "current_policy_success_rate": policy[
                "official_peak_success_rate"],
            "current_policy_mean_lift_m": policy["mean_maximum_lift_m"],
            "teacher_forced_action_mae": error["teacher_all_mae"],
            "closed_loop_action_mae": error["closed_loop_all_mae"],
            "error_amplification": error[
                "closed_to_teacher_action_error_ratio"],
            "checkpoint": str(item["checkpoint"]),
            "diagnostic": str(summary),
        })
    ranking = sorted(rows, key=lambda row: (
        -row["expert_success_rate"],
        -row["current_policy_success_rate"],
        -row["trajectory_count"],
    ))
    result = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "purpose": "Select focused train objects by replay ceiling and policy gap.",
        "final_unseen_v2_used": False,
        "ranking": ranking,
    }
    path = OUTPUT / "focused_preflight_summary.yaml"
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(result, handle, allow_unicode=True, sort_keys=False)
    print("FOCUSED_PREFLIGHT=COMPLETE summary={}".format(path), flush=True)


if __name__ == "__main__":
    main()
