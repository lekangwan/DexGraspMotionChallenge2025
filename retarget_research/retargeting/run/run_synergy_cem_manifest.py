#!/usr/bin/env python3
"""Build a joint-synergy basis and run the same low-rank CEM on every case."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
REFINE = Path(__file__).with_name("physics_cem_refine.py")
sys.path.insert(0, str(REFINE.parent))
from physics_cem_refine import phase_frames  # noqa: E402


def target_row(target, source_index):
    indices = np.asarray(target["source_trajectory_indices"], dtype=np.int64)
    rows = np.flatnonzero(indices == int(source_index))
    if len(rows) != 1:
        raise ValueError(f"source index {source_index} is not unique")
    return int(rows[0])


def build_basis(entries, target_dir, rank):
    """Extract normalized close/lift joint patterns and return top SVD directions."""
    patterns = []
    for entry in entries:
        source_index = int(entry["trajectory_indices"][0])
        target = np.load(
            Path(target_dir) / f"{entry['object_name']}.npy", allow_pickle=True).item()
        frames = np.asarray(
            target["grasp_seqs"][target_row(target, source_index)], dtype=np.float32)
        close, grasp = phase_frames(frames)
        for vector in (frames[grasp, 6:] - frames[close, 6:],
                       frames[-1, 6:] - frames[grasp, 6:]):
            norm = float(np.linalg.norm(vector))
            if norm >= 1e-4:
                patterns.append(vector / norm)
    matrix = np.stack(patterns)
    _, _, right = np.linalg.svd(matrix, full_matrices=False)
    return right[:min(rank, len(right))].astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hand", choices=("linker", "xhand", "wuji"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=3)
    parser.add_argument("--population", type=int, default=8)
    parser.add_argument("--elite", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--selection-margin", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--accumulate-object-trajectories", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    basis_path = args.output_dir / "synergy_basis.npy"
    if not basis_path.exists():
        np.save(basis_path, build_basis(
            manifest["entries"], args.target_dir, args.rank))
    basis = np.load(basis_path)
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
            output = accumulated if args.accumulate_object_trajectories else (
                args.output_dir / entry["object_name"] / f"source_{source_index}.npy")
            report = args.output_dir / entry["object_name"] / f"source_{source_index}.json"
            if not report.exists():
                subprocess.run([
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
                    "--selection-margin", str(args.selection_margin),
                    "--seed", str(args.seed + number),
                    "--device", args.device,
                    "--lift-threshold", "0.15",
                    "--parameterization", "synergy",
                    "--synergy-basis", str(basis_path),
                ], cwd=ROOT, check=True)
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

    (args.output_dir / "screen_summary.json").write_text(
        json.dumps({
            "schema_version": 1,
            "hand": args.hand,
            "parameterization": "joint_synergy_phase_basis",
            "rank": int(len(basis)),
            "basis": str(basis_path.resolve()),
            "population": args.population,
            "elite": args.elite,
            "iterations": args.iterations,
            "trajectory_count": len(results),
            "results": results,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
