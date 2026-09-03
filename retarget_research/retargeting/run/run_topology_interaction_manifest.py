#!/usr/bin/env python3
"""Apply bounded hand-object interaction refinement to a frozen manifest."""

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[3]
REFINE = Path(__file__).with_name("refine_topology_interaction.py")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hand", choices=("linker", "xhand", "wuji"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=150)
    parser.add_argument("--learning-rate", type=float, default=0.005)
    parser.add_argument("--baseline-weight", type=float, default=5.0)
    parser.add_argument("--velocity-weight", type=float, default=1.0)
    parser.add_argument("--acceleration-weight", type=float, default=0.5)
    parser.add_argument("--max-joint-residual", type=float, default=0.10)
    parser.add_argument("--surface-neighbors", type=int, default=1)
    parser.add_argument("--interaction-threshold", type=float, default=0.025)
    parser.add_argument("--optimize-from", choices=("close", "grasp"), default="grasp")
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
                sys.executable, "-u", str(REFINE),
                "--hand", args.hand,
                "--source", entry["source_path"],
                "--target", str(args.target_dir / f"{entry['object_name']}.npy"),
                "--source-index", str(source_index),
                "--object-dir", entry["object_asset_path"],
                "--output", str(output),
                "--iterations", str(args.iterations),
                "--learning-rate", str(args.learning_rate),
                "--baseline-weight", str(args.baseline_weight),
                "--velocity-weight", str(args.velocity_weight),
                "--acceleration-weight", str(args.acceleration_weight),
                "--max-joint-residual", str(args.max_joint_residual),
                "--surface-neighbors", str(args.surface_neighbors),
                "--interaction-threshold", str(args.interaction_threshold),
                "--optimize-from", args.optimize_from,
            ], cwd=ROOT, check=True)
        result = json.loads(report.read_text(encoding="utf-8"))
        results.append({
            "category": entry["category"],
            "object_name": entry["object_name"],
            "source_trajectory_index": source_index,
            "output": str(output.resolve()),
            "edge_count": result["edge_count"],
            "final_loss": result["history"][-1],
        })
        print(f"[{args.hand}] {number}/{len(manifest['entries'])}", flush=True)

    summary = {
        "schema_version": 1,
        "method": "topology_interaction_v2_bounded",
        "hand": args.hand,
        "trajectory_count": len(results),
        "results": results,
    }
    (args.output_dir / "screen_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
