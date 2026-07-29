"""Evaluate frozen controls on the geometry-confirmation set."""

import argparse
from datetime import datetime
from pathlib import Path
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "custom_tools/results/evaluations/geometry_confirmation_controls"
TRAJECTORIES = ROOT / "dexgrasp/dataset/unseen_v2_candidates_preprocessed"
SELECTION = ROOT / "custom_tools/configs/hparam_geometry_confirmation_v1.yaml"
MANIFEST = ROOT / "custom_tools/configs/hparam_geometry_confirmation_manifest.json"
BC_CONFIG = ROOT / "custom_tools/configs/unified_student_online_round1.yaml"
RESIDUAL_CONFIG = ROOT / "custom_tools/configs/residual_ppo_soup_anchored_gated.yaml"
ROBUST_SUMMARY = (ROOT / "custom_tools/results/evaluations/robust_oa_search/"
                  "robust_search_summary.yaml")
CONTROLS = {
    "noisefix_t70_seed2025_e05": (
        ROOT / "custom_tools/runs/bc/"
        "unified_student_online_r1_noisefix_seed2025_e20_v1/"
        "epoch=004-step=2140.ckpt"),
    "previous_t85_seed2025_e05": (
        ROOT / "custom_tools/runs/bc/"
        "unified_hparam_teacher85_seed2025_e20_v1/"
        "epoch=004-step=2140.ckpt"),
}
SOUP = (ROOT / "custom_tools/runs/bc/model_soups/"
        "noise005_s2025_s2026_weighted2to1.ckpt")
TEACHERS = {
    "bottle": ROOT / "custom_tools/runs/bc/category_expert_bottle_noise005_soup_seed2025_e40_v1/epoch=039-step=2560.ckpt",
    "mug": SOUP,
    "bowl": ROOT / "custom_tools/runs/bc/category_expert_bowl_noise005_soup_seed2025_e40_v1/epoch=039-step=1680.ckpt",
    "camera": ROOT / "custom_tools/runs/bc/category_expert_camera_noise005_soup_seed2025_e40_v1/epoch=009-step=500.ckpt",
}


def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-free-vram-mb", type=int, default=4500)
    parser.add_argument("--max-attempts", type=int, default=2)
    return parser.parse_args()


def run(command):
    print("RUN: {}".format(" ".join(str(item) for item in command)), flush=True)
    subprocess.run(command, cwd=str(ROOT), check=True)


def main():
    cli = parse_cli()
    for path in list(CONTROLS.values()) + list(TEACHERS.values()):
        if not path.is_file():
            raise FileNotFoundError(path)
    OUTPUT.mkdir(parents=True, exist_ok=True)

    student_dir = OUTPUT / "single_students"
    command = [sys.executable, str(ROOT / "custom_tools/screen_bc_candidates.py")]
    for label, checkpoint in CONTROLS.items():
        command.extend(["--candidate", "{}={}".format(label, checkpoint)])
    command.extend([
        "--bc-config", str(BC_CONFIG),
        "--residual-config", str(RESIDUAL_CONFIG),
        "--trajectory-root", str(TRAJECTORIES),
        "--object-selection", str(SELECTION),
        "--output-dir", str(student_dir), "--seed", "2025",
        "--min-free-vram-mb", str(cli.min_free_vram_mb),
        "--max-attempts", str(cli.max_attempts),
    ])
    run(command)

    teacher_dir = OUTPUT / "routed_teacher_pool"
    command = [sys.executable,
               str(ROOT / "custom_tools/evaluate_routed_bc_repeats.py")]
    for category, checkpoint in TEACHERS.items():
        command.extend(["--teacher", "{}={}".format(category, checkpoint)])
    command.extend([
        "--repeats", "1", "--manifest", str(MANIFEST),
        "--manifest-split", "train", "--bc-config", str(BC_CONFIG),
        "--residual-config", str(RESIDUAL_CONFIG),
        "--trajectory-root", str(TRAJECTORIES),
        "--output-dir", str(teacher_dir), "--seed", "2025",
        "--min-free-vram-mb", str(cli.min_free_vram_mb),
        "--max-attempts", str(cli.max_attempts),
    ])
    run(command)

    with (student_dir / "screen_summary.yaml").open(encoding="utf-8") as handle:
        students = yaml.safe_load(handle)["ranking"]
    with (teacher_dir / "routed_repeats_summary.yaml").open(
            encoding="utf-8") as handle:
        teachers = yaml.safe_load(handle)
    with ROBUST_SUMMARY.open(encoding="utf-8") as handle:
        robust = yaml.safe_load(handle)

    selected_row = int(robust["selected_robust_row"])
    selected = [
        item for key, item in robust["geometry_confirmation_results"].items()
        if key.startswith("r{:02d}_seed".format(selected_row))
    ]
    selected.sort(key=lambda item: item["label"])
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "frozen_geometry_confirmation_controls_complete",
        "final_unseen_v2_used": False,
        "trajectory_count": 293,
        "locked_robust_students": selected,
        "single_student_controls": students,
        "routed_teacher_pool": teachers["repeats"][0],
        "interpretation_rule": (
            "Compare official success count first, then macro success, lift, "
            "and failure; controls do not reopen hyperparameter selection."),
    }
    path = OUTPUT / "geometry_controls_summary.yaml"
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(summary, handle, allow_unicode=True, sort_keys=False)
    print("GEOMETRY_CONTROLS=COMPLETE summary={}".format(path), flush=True)


if __name__ == "__main__":
    main()
