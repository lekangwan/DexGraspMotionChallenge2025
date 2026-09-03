#!/usr/bin/env python3
"""Run target-hand grasp-pose synthesis on a frozen manifest."""

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[3]
SYNTHESIZE = Path(__file__).with_name("synthesize_target_grasp_pose.py")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hand", choices=("linker", "xhand", "wuji"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--max-joint-residual", type=float, default=0.40)
    parser.add_argument("--max-wrist-translation", type=float, default=0.03)
    parser.add_argument("--max-wrist-rotation", type=float, default=0.20)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for number, entry in enumerate(manifest["entries"], 1):
        source_index = int(entry["trajectory_indices"][0])
        output = args.output_dir / f"{entry['object_name']}.npy"
        report = output.with_suffix(".json")
        if not (args.resume and output.exists() and report.exists()):
            subprocess.run([
                sys.executable, "-u", str(SYNTHESIZE),
                "--hand", args.hand,
                "--source", entry["source_path"],
                "--target", str(args.target_dir / f"{entry['object_name']}.npy"),
                "--source-index", str(source_index),
                "--object-dir", entry["object_asset_path"],
                "--output", str(output),
                "--iterations", str(args.iterations),
                "--max-joint-residual", str(args.max_joint_residual),
                "--max-wrist-translation", str(args.max_wrist_translation),
                "--max-wrist-rotation", str(args.max_wrist_rotation),
            ], cwd=ROOT, check=True)
        result = json.loads(report.read_text(encoding="utf-8"))
        results.append({
            "category": entry["category"],
            "object_name": entry["object_name"],
            "source_trajectory_index": source_index,
            "output": str(output.resolve()),
            "selected_fingers": result["selected_fingers"],
            "final_loss": result["history"][-1],
        })
        print(f"[{args.hand}] {number}/{len(manifest['entries'])}", flush=True)

    (args.output_dir / "screen_summary.json").write_text(
        json.dumps({
            "schema_version": 1,
            "method": "target_grasp_pose_synthesis_v1",
            "hand": args.hand,
            "trajectory_count": len(results),
            "results": results,
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
