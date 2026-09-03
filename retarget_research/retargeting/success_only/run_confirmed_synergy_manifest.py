#!/usr/bin/env python3
"""只对失败子manifest运行带单环境确认的Rank-k协同CEM。"""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
RUN_DIR = Path(__file__).resolve().parent.parent / "run"
REFINE = RUN_DIR / "physics_cem_refine.py"
sys.path.insert(0, str(RUN_DIR))
from run_synergy_cem_manifest import build_basis  # noqa: E402
from physics_cem_refine import PhysicsCase, physics_score, target_row  # noqa: E402
from confirm_physics_cem_manifest import evaluate  # noqa: E402


METRIC_FIELDS = (
    "max_lift_m",
    "final_lift_m",
    "peak_to_final_drop_m",
    "terminal_lift_range_m",
    "terminal_contact_ratio",
    "hand_object_contact_steps",
    "success",
    "transport_stability_success",
    "max_palm_relative_translation_change_m",
    "max_palm_relative_rotation_change_deg",
)


def compact_metric(metric):
    """只保存确认决策需要的标量，避免重复写入整段位置与接触序列。"""
    return {field: metric.get(field) for field in METRIC_FIELDS}


def robust_decision(baseline_metrics, candidate_metrics, margin):
    """候选每次都稳定运输且平均分显著更高时才允许替换基线。"""
    if not baseline_metrics or len(baseline_metrics) != len(candidate_metrics):
        raise ValueError("基线与候选确认次数必须相同且大于零")
    baseline_scores = [physics_score(item, 0.15) for item in baseline_metrics]
    candidate_scores = [physics_score(item, 0.15) for item in candidate_metrics]
    baseline_quality = [
        bool(item["success"] and item.get("transport_stability_success", False))
        for item in baseline_metrics
    ]
    candidate_quality = [
        bool(item["success"] and item.get("transport_stability_success", False))
        for item in candidate_metrics
    ]
    baseline_mean = float(np.mean(baseline_scores))
    candidate_mean = float(np.mean(candidate_scores))
    accepted = bool(
        all(candidate_quality)
        and candidate_mean > baseline_mean + margin
    )
    return {
        "accepted": accepted,
        "selection_margin": float(margin),
        "baseline_quality_passes": int(sum(baseline_quality)),
        "candidate_quality_passes": int(sum(candidate_quality)),
        "confirmation_repeats": len(baseline_metrics),
        "baseline_mean_score": baseline_mean,
        "candidate_mean_score": candidate_mean,
        "baseline_metrics": [compact_metric(item) for item in baseline_metrics],
        "candidate_metrics": [compact_metric(item) for item in candidate_metrics],
    }


def make_case(hand, entry, source_index, frames, source):
    """把一条目标手轨迹及物体尺度、旋转和mesh封装成独立确认案例。"""
    return PhysicsCase(
        hand=hand,
        category=entry["category"],
        object_name=entry["object_name"],
        source_index=source_index,
        target_frames=np.asarray(frames, dtype=np.float32),
        mesh_path=Path(entry["object_asset_path"]) / "coacd" / "decomposed.obj",
        scale=float(np.asarray(source["obj_scale"])[source_index]),
        rotation=np.asarray(source["obj_rotmat"])[source_index],
    )


def repeated_confirmation(hand, entry, source_index, baseline_data,
                          candidate_data, repeats, margin, device):
    """分别重建单案例环境重复评估基线和候选，输出稳健接受决策。"""
    source = np.load(entry["source_path"], allow_pickle=True).item()
    baseline_row = target_row(baseline_data, source_index)
    candidate_row = target_row(candidate_data, source_index)
    baseline_case = make_case(
        hand, entry, source_index,
        baseline_data["grasp_seqs"][baseline_row], source,
    )
    candidate_case = make_case(
        hand, entry, source_index,
        candidate_data["grasp_seqs"][candidate_row], source,
    )
    baseline_metrics = []
    candidate_metrics = []
    for _ in range(repeats):
        baseline_metrics.append(evaluate(baseline_case, baseline_data, device))
        candidate_metrics.append(evaluate(candidate_case, baseline_data, device))
    return robust_decision(baseline_metrics, candidate_metrics, margin)


def main():
    """读取失败子集，用确认重放保护原基线，并保留完整物体NPY。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--hand", choices=("linker", "xhand", "wuji"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--synergy-basis", type=Path)
    parser.add_argument("--rank", type=int, default=5)
    parser.add_argument("--population", type=int, default=8)
    parser.add_argument("--elite", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--selection-margin", type=float, default=1.0)
    parser.add_argument("--confirmation-repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.confirmation_repeats < 1:
        parser.error("--confirmation-repeats必须至少为1")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    basis_path = args.output_dir / "synergy_basis.npy"
    if not basis_path.exists():
        if args.synergy_basis:
            shutil.copy2(args.synergy_basis, basis_path)
        else:
            np.save(
                basis_path,
                build_basis(manifest["entries"], args.target_dir, args.rank),
            )

    results = []
    number = 0
    total = sum(len(entry["trajectory_indices"]) for entry in manifest["entries"])
    for entry in manifest["entries"]:
        source_target = args.target_dir / f"{entry['object_name']}.npy"
        accumulated = args.output_dir / "objects" / f"{entry['object_name']}.npy"
        if not accumulated.exists():
            accumulated.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_target, accumulated)
        for source_index in map(int, entry["trajectory_indices"]):
            number += 1
            report = (
                args.output_dir / entry["object_name"] / f"source_{source_index}.json"
            )
            confirmation_path = report.with_name(
                f"source_{source_index}_robust_confirmation.json"
            )
            if not report.exists():
                subprocess.run([
                    sys.executable,
                    "-u",
                    str(REFINE),
                    "--hand", args.hand,
                    "--source", entry["source_path"],
                    "--target", str(accumulated),
                    "--source-index", str(source_index),
                    "--object-dir", entry["object_asset_path"],
                    "--output", str(accumulated),
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
                    "--confirm-single-env",
                ], cwd=ROOT, check=True)
            result = json.loads(report.read_text(encoding="utf-8"))
            single_confirmation = result.get("single_env_confirmation")
            if confirmation_path.exists():
                confirmation = json.loads(
                    confirmation_path.read_text(encoding="utf-8")
                )
            elif single_confirmation and single_confirmation["accepted"]:
                baseline_data = np.load(source_target, allow_pickle=True).item()
                candidate_data = np.load(accumulated, allow_pickle=True).item()
                confirmation = repeated_confirmation(
                    args.hand,
                    entry,
                    source_index,
                    baseline_data,
                    candidate_data,
                    args.confirmation_repeats,
                    args.selection_margin,
                    args.device,
                )
                confirmation["single_env_prefilter_accepted"] = True
                if not confirmation["accepted"]:
                    baseline_row = target_row(baseline_data, source_index)
                    candidate_row = target_row(candidate_data, source_index)
                    sequences = np.asarray(candidate_data["grasp_seqs"]).copy()
                    sequences[candidate_row] = baseline_data["grasp_seqs"][baseline_row]
                    candidate_data["grasp_seqs"] = sequences
                    np.save(accumulated, candidate_data, allow_pickle=True)
                confirmation_path.parent.mkdir(parents=True, exist_ok=True)
                confirmation_path.write_text(
                    json.dumps(confirmation, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
            else:
                confirmation = {
                    "accepted": False,
                    "single_env_prefilter_accepted": False,
                    "confirmation_repeats": 0,
                    "reason": "CEM候选未通过第一次单环境确认或未修改基线",
                }
                confirmation_path.parent.mkdir(parents=True, exist_ok=True)
                confirmation_path.write_text(
                    json.dumps(confirmation, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
            accepted = bool(confirmation["accepted"])
            selected_metric = (
                confirmation["candidate_metrics"][-1]
                if accepted and confirmation.get("candidate_metrics")
                else confirmation["baseline_metrics"][-1]
                if confirmation.get("baseline_metrics")
                else result["baseline_metric"]
            )
            results.append({
                "category": entry["category"],
                "object_name": entry["object_name"],
                "source_trajectory_index": source_index,
                "output": str(accumulated.resolve()),
                "baseline_metric": result["baseline_metric"],
                "refined_metric": selected_metric,
                "parameters": (
                    result["parameters"]
                    if accepted else [0.0] * len(result["parameters"])
                ),
                "single_env_confirmation": single_confirmation,
                "robust_confirmation": confirmation,
            })
            print(f"[{args.hand}] {number}/{total} confirmed={accepted}", flush=True)

    summary = {
        "schema_version": 1,
        "hand": args.hand,
        "parameterization": "confirmed_joint_synergy_phase_basis",
        "rank": int(len(np.load(basis_path))),
        "population": args.population,
        "elite": args.elite,
        "iterations": args.iterations,
        "confirmation_repeats": args.confirmation_repeats,
        "trajectory_count": len(results),
        "confirmed_accept_count": sum(bool(
            row["robust_confirmation"]["accepted"]
        ) for row in results),
        "results": results,
    }
    (args.output_dir / "screen_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
