#!/usr/bin/env python3
"""Run phase-basis CEM on every entry of a frozen calibration manifest."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[3]
REFINE = Path(__file__).with_name("physics_cem_refine.py")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hand", choices=("linker", "xhand", "wuji"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--population", type=int, default=8)
    parser.add_argument("--elite", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--selection-margin", type=float, default=1.0)
    parser.add_argument("--parameterization", choices=("global", "phase", "cradle"),
                        default="phase")
    parser.add_argument("--confirm-single-env", action="store_true")
    parser.add_argument("--parameter-scale", type=float, default=1.0)
    parser.add_argument("--score-mode", choices=("standard", "grasp_quality"),
                        default="standard")
    parser.add_argument("--accumulate-object-trajectories", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    number = 0
    total = sum(len(entry["trajectory_indices"]) for entry in manifest["entries"])
    for entry in manifest["entries"]:
        source_target = args.target_dir / f"{entry['object_name']}.npy"
        indices = ([int(index) for index in entry["trajectory_indices"]]
                   if args.accumulate_object_trajectories
                   else [int(entry["trajectory_indices"][0])])
        accumulated = args.output_dir / "objects" / f"{entry['object_name']}.npy"
        if args.accumulate_object_trajectories and not accumulated.exists():
            accumulated.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_target, accumulated)
        for source_index in indices:
            number += 1
            target = accumulated if args.accumulate_object_trajectories else source_target
            case_dir = args.output_dir / entry["object_name"]
            output = accumulated if args.accumulate_object_trajectories else (
                case_dir / f"source_{source_index}.npy")
            report = case_dir / f"source_{source_index}.json"
            if not report.exists():
                command = [
                    sys.executable, "-u", str(REFINE),
                    "--hand", args.hand,
                    "--source", entry["source_path"],
                    "--target", str(target),
                    "--source-index", str(source_index),
                    "--object-dir", entry["object_asset_path"],
                    "--output", str(output),
                    "--report-output", str(report),
                    "--population", str(args.population),
                    "--elite", str(args.elite),
                    "--iterations", str(args.iterations),
                    "--seed", str(args.seed + number),
                    "--device", args.device,
                    "--lift-threshold", "0.15",
                    "--parameterization", args.parameterization,
                    "--selection-margin", str(args.selection_margin),
                    "--parameter-scale", str(args.parameter_scale),
                    "--score-mode", args.score_mode,
                ]
                if args.confirm_single_env:
                    command.append("--confirm-single-env")
                subprocess.run(command, cwd=ROOT, check=True)
            result = json.loads(report.read_text(encoding="utf-8"))
            results.append({
                "category": entry["category"],
                "object_name": entry["object_name"],
                "source_trajectory_index": source_index,
                "output": str(output.resolve()),
                "baseline_metric": result["baseline_metric"],
                "refined_metric": result["metric"],
                "parameters": result["parameters"],
            })
            print(f"[{args.hand}] {number}/{total}", flush=True)

    summary = {
        "schema_version": 1,
        "hand": args.hand,
        "parameterization": args.parameterization,
        "population": args.population,
        "elite": args.elite,
        "iterations": args.iterations,
        "trajectory_count": len(results),
        "results": results,
    }
    (args.output_dir / "screen_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
