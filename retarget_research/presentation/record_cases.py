from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "retarget_research/presentation/assets/videos"
ISAAC_STATE = ROOT / "retarget_research/scripts/render_isaac_state_replay.py"
LINKER_REPLAY = ROOT / "retarget_research/retargeting/evaluate/replay_linker_isaac.py"


def linker_command(report_path: Path, output_stem: str):
    report = json.loads(report_path.read_text())
    command = [
        sys.executable, str(LINKER_REPLAY),
        "--source", report["source"],
        "--target", report["target"],
        "--object-dir", str(Path(report["source"]).parents[1] / "meshdata" / report["object_name"]),
        "--object-name", report["object_name"],
        "--source-index", str(report["source_trajectory_index"]),
        "--target-index", str(report["target_trajectory_index"]),
        "--finger-stiffness", str(report.get("finger_stiffness", 120.0)),
        "--finger-damping", str(report.get("finger_damping", 5.0)),
        "--mimic-stiffness", str(report.get("mimic_stiffness", 120.0)),
        "--mimic-damping", str(report.get("mimic_damping", 5.0)),
        "--output", str(OUT / f"{output_stem}.json"),
        "--video-output", str(OUT / f"{output_stem}.mp4"),
    ]
    return command


def state_command(report_path: Path, output_stem: str):
    return [
        sys.executable, str(ISAAC_STATE),
        "--state", str(report_path),
        "--output", str(OUT / f"{output_stem}.mp4"),
    ]


def commands():
    baseline = ROOT / "retarget_research/outputs/formal_1000/linker_o6_optimized_v2_evaluation/sem-WineBottle-f331ad8d0e6654ef8f992b1fe7075c8f/source_10_physics.json"
    functional = ROOT / "retarget_research/outputs/formal_1000/linker_vector_v2alpha_c3g8_v1_evaluation/sem-WineBottle-f331ad8d0e6654ef8f992b1fe7075c8f/source_10_physics.json"
    supervised = ROOT / "retarget_research/advanced_policy/runs/formal_final_online3/xhand_official_phase_residual_v1_policy_eval_expert/sem-Battery-62733b55e76a3b718c9d9ab13336021b/source_12.json"
    ppo = ROOT / "retarget_research/advanced_policy/runs/residual_rl_general/xhand_official_policy_eval_test/sem-Battery-62733b55e76a3b718c9d9ab13336021b/source_12.json"
    return [
        linker_command(baseline, "linker_pose_baseline"),
        linker_command(functional, "linker_function_vector"),
        state_command(supervised, "xhand_supervised_battery12"),
        state_command(ppo, "xhand_residual_ppo_battery12"),
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for command in commands():
        print(" ".join(command), flush=True)
        if args.execute:
            subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
