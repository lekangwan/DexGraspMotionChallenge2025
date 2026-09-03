#!/usr/bin/env python3
"""Blend synthesized grasp trajectories with their original warm start."""

import argparse
import json
from pathlib import Path

import numpy as np


def row_for(data, source_index):
    rows = np.flatnonzero(
        np.asarray(data["source_trajectory_indices"], dtype=np.int64)
        == int(source_index)
    )
    if len(rows) != 1:
        raise ValueError(f"source index {source_index} matched {len(rows)} rows")
    return int(rows[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scale", type=float, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for entry in manifest["entries"]:
        name = entry["object_name"]
        source_index = int(entry["trajectory_indices"][0])
        baseline = np.load(
            args.baseline_dir / f"{name}.npy", allow_pickle=True
        ).item()
        candidate = np.load(
            args.candidate_dir / f"{name}.npy", allow_pickle=True
        ).item()
        base_row = row_for(baseline, source_index)
        candidate_row = row_for(candidate, source_index)
        base_frames = np.asarray(baseline["grasp_seqs"][base_row])
        candidate_frames = np.asarray(candidate["grasp_seqs"][candidate_row])
        result = dict(candidate)
        sequences = np.asarray(candidate["grasp_seqs"]).copy()
        sequences[candidate_row] = base_frames + args.scale * (
            candidate_frames - base_frames
        )
        result["grasp_seqs"] = sequences.astype(np.float32)
        result["retarget_method"] = "target_grasp_pose_synthesis_blend_v2"
        result["target_grasp_pose_blend"] = {
            "schema": "target_grasp_pose_synthesis_blend_v2",
            "scale": args.scale,
            "baseline": str((args.baseline_dir / f"{name}.npy").resolve()),
            "candidate": str((args.candidate_dir / f"{name}.npy").resolve()),
        }
        np.save(args.output_dir / f"{name}.npy", result, allow_pickle=True)


if __name__ == "__main__":
    main()
