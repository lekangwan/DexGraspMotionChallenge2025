#!/usr/bin/env python3
"""Confirm CEM candidates in isolated single-case physics before saving them."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from physics_cem_refine import PhysicsCase, make_env_args, physics_score, target_row


def evaluate(case, target, device):
    from train_residual_ppo_general import GeneralResidualEnv, rollout
    env = GeneralResidualEnv([case], target, make_env_args(device, 0.15))
    try:
        return rollout(env)[0]
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hand", choices=("linker", "xhand", "wuji"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--baseline-target-dir", type=Path, required=True)
    parser.add_argument("--candidate-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selection-margin", type=float, default=1.0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    candidate_summary = json.loads(args.candidate_summary.read_text(encoding="utf-8"))
    candidates = {
        (row["object_name"], int(row["source_trajectory_index"])): row
        for row in candidate_summary["results"]
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for number, entry in enumerate(manifest["entries"], 1):
        source_index = int(entry["trajectory_indices"][0])
        key = (entry["object_name"], source_index)
        candidate_row = candidates[key]
        source = np.load(entry["source_path"], allow_pickle=True).item()
        baseline = np.load(
            args.baseline_target_dir / f"{entry['object_name']}.npy",
            allow_pickle=True,
        ).item()
        candidate = np.load(candidate_row["output"], allow_pickle=True).item()
        baseline_index = target_row(baseline, source_index)
        candidate_index = target_row(candidate, source_index)
        baseline_frames = np.asarray(baseline["grasp_seqs"][baseline_index])
        candidate_frames = np.asarray(candidate["grasp_seqs"][candidate_index])

        def make_case(frames):
            return PhysicsCase(
                args.hand,
                entry["category"],
                entry["object_name"],
                source_index,
                frames,
                Path(entry["object_asset_path"]) / "coacd" / "decomposed.obj",
                float(np.asarray(source["obj_scale"])[source_index]),
                np.asarray(source["obj_rotmat"])[source_index],
            )

        baseline_metric = evaluate(make_case(baseline_frames), baseline, args.device)
        if np.array_equal(baseline_frames, candidate_frames):
            candidate_metric = baseline_metric
        else:
            candidate_metric = evaluate(make_case(candidate_frames), baseline, args.device)
        baseline_score = physics_score(baseline_metric, 0.15)
        candidate_score = physics_score(candidate_metric, 0.15)
        accepted = candidate_score > baseline_score + args.selection_margin

        refined = dict(candidate if accepted else baseline)
        sequences = np.asarray(refined["grasp_seqs"]).copy()
        source_row = candidate_index if accepted else baseline_index
        sequences[source_row] = candidate_frames if accepted else baseline_frames
        refined["grasp_seqs"] = sequences
        refined["isolated_physics_confirmation"] = {
            "accepted": bool(accepted),
            "selection_margin": args.selection_margin,
            "baseline_score": float(baseline_score),
            "candidate_score": float(candidate_score),
            "baseline_metric": baseline_metric,
            "candidate_metric": candidate_metric,
        }
        output = args.output_dir / entry["object_name"] / f"source_{source_index}.npy"
        output.parent.mkdir(parents=True, exist_ok=True)
        np.save(output, refined, allow_pickle=True)
        results.append({
            "category": entry["category"],
            "object_name": entry["object_name"],
            "source_trajectory_index": source_index,
            "output": str(output.resolve()),
            "baseline_metric": baseline_metric,
            "refined_metric": candidate_metric if accepted else baseline_metric,
            "parameters": candidate_row["parameters"] if accepted else [0.0] * 5,
            "candidate_accepted": bool(accepted),
        })
        print(f"[{args.hand}] {number}/{len(manifest['entries'])} accepted={accepted}", flush=True)

    (args.output_dir / "screen_summary.json").write_text(
        json.dumps({
            "schema_version": 1,
            "hand": args.hand,
            "parameterization": "lift_cradle_isolated_confirmation_v2",
            "trajectory_count": len(results),
            "accepted_count": sum(row["candidate_accepted"] for row in results),
            "results": results,
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
