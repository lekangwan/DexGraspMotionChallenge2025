#!/usr/bin/env python3
"""Refine one retargeted trajectory with physics-scored CEM sampling."""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np


RESEARCH_ROOT = Path(__file__).resolve().parents[2]
ADVANCED = RESEARCH_ROOT / "advanced_policy"
EVALUATE = RESEARCH_ROOT / "retargeting" / "evaluate"
for path in (ADVANCED, EVALUATE):
    sys.path.insert(0, str(path))


@dataclass
class PhysicsCase:
    hand: str
    category: str
    object_name: str
    source_index: int
    target_frames: np.ndarray
    mesh_path: Path
    scale: float
    rotation: np.ndarray

    def __post_init__(self):
        from observations import build_object_shape_descriptor
        self.shape = build_object_shape_descriptor(self.mesh_path, self.scale)


def smoothstep(values):
    values = np.clip(values, 0.0, 1.0)
    return values * values * (3.0 - 2.0 * values)


def apply_trajectory_parameters(frames, parameters):
    """Apply low-dimensional wrist, finger and lift edits to a trajectory."""
    result = np.asarray(frames, dtype=np.float32).copy()
    parameters = np.asarray(parameters, dtype=np.float32)
    length, dimensions = result.shape
    if dimensions <= 6:
        raise ValueError("target trajectory must contain wrist and finger DOFs")
    time = np.linspace(0.0, 1.0, length, dtype=np.float32)
    approach_gate = smoothstep((time - 0.15) / 0.35)[:, None]
    closure_gate = smoothstep((time - 0.35) / 0.30)[:, None]
    lift_gate = smoothstep((time - 0.65) / 0.25)
    result[:, :3] += approach_gate * parameters[:3]
    result[:, 3:6] += approach_gate * parameters[3:6]
    finger_offsets = parameters[6:-1]
    finger_indices = np.array_split(
        np.arange(6, dimensions), len(finger_offsets))
    for offset, indices in zip(finger_offsets, finger_indices):
        if len(indices):
            result[:, indices] += closure_gate * float(offset)
    result[:, 2] += lift_gate * float(parameters[-1])
    return result


def phase_frames(frames):
    """Estimate closure start and the final pre-lift grasp frame."""
    fingers = np.asarray(frames[:, 6:], dtype=np.float32)
    movement = np.linalg.norm(fingers - fingers[0], axis=1)
    candidates = np.flatnonzero(movement >= max(0.2 * float(movement.max()), 1e-3))
    close = int(candidates[0]) if len(candidates) else 20
    wrist_z = np.asarray(frames[:, 2], dtype=np.float32)
    base_z = float(wrist_z[close:].min())
    lifts = np.flatnonzero(
        (np.arange(len(frames)) > close) & (wrist_z >= base_z + 0.03))
    lift = int(lifts[0]) if len(lifts) else min(55, len(frames) - 1)
    return close, max(close + 1, lift - 1)


def apply_phase_parameters(frames, parameters):
    """Apply separate five-finger residuals in closure and lift phases."""
    result = np.asarray(frames, dtype=np.float32).copy()
    parameters = np.asarray(parameters, dtype=np.float32)
    if len(parameters) != 10:
        raise ValueError("phase parameterization expects 10 values")
    close, grasp = phase_frames(result)
    indices = np.arange(len(result), dtype=np.float32)
    close_gate = smoothstep(
        (indices - close) / float(max(1, grasp - close)))
    lift_gate = smoothstep((indices - grasp) / 10.0)
    finger_indices = np.array_split(np.arange(6, result.shape[1]), 5)
    for finger, group in enumerate(finger_indices):
        result[:, group] += (
            close_gate[:, None] * parameters[finger]
            + lift_gate[:, None] * parameters[5 + finger])
    return result


def apply_synergy_parameters(frames, parameters, basis):
    """Apply close/lift residuals in a morphology-specific joint-synergy basis."""
    result = np.asarray(frames, dtype=np.float32).copy()
    basis = np.asarray(basis, dtype=np.float32)
    parameters = np.asarray(parameters, dtype=np.float32)
    rank, finger_dimension = basis.shape
    if finger_dimension != result.shape[1] - 6 or len(parameters) != 2 * rank:
        raise ValueError("synergy basis or parameter dimension mismatch")
    close, grasp = phase_frames(result)
    indices = np.arange(len(result), dtype=np.float32)
    close_gate = smoothstep(
        (indices - close) / float(max(1, grasp - close)))
    lift_gate = smoothstep((indices - grasp) / 10.0)
    close_residual = parameters[:rank] @ basis
    lift_residual = parameters[rank:] @ basis
    result[:, 6:] += (
        close_gate[:, None] * close_residual[None]
        + lift_gate[:, None] * lift_residual[None])
    return result


def apply_cradle_parameters(frames, parameters):
    """Apply lift-only lateral wrist translation and wrist tilt."""
    result = np.asarray(frames, dtype=np.float32).copy()
    parameters = np.asarray(parameters, dtype=np.float32)
    if len(parameters) != 5:
        raise ValueError("cradle parameterization expects 5 values")
    _, grasp = phase_frames(result)
    indices = np.arange(len(result), dtype=np.float32)
    lift_gate = smoothstep((indices - grasp) / 10.0)
    result[:, :2] += lift_gate[:, None] * parameters[None, :2]
    result[:, 3:6] += lift_gate[:, None] * parameters[None, 2:]
    return result


def physics_score(metric, lift_threshold=0.15, grasp_quality=False):
    """Rank stable transport first, then lift and persistent contact."""
    score = (
        100.0 * float(metric["success"])
        + 25.0 * np.clip(metric["final_lift_m"] / lift_threshold, -1.0, 1.2)
        + 10.0 * np.clip(metric["max_lift_m"] / lift_threshold, 0.0, 1.2)
        + 5.0 * float(metric["terminal_contact_ratio"])
        + 2.0 * np.clip(metric["hand_object_contact_steps"] / 150.0, 0.0, 1.0)
        - 10.0 * max(0.0, metric["peak_to_final_drop_m"] - 0.03)
        - 5.0 * max(0.0, metric["max_xy_drift_m"] - 0.25)
    )
    if metric.get("transport_stability_success"):
        score += 40.0
    translation = metric.get("max_palm_relative_translation_change_m")
    rotation = metric.get("max_palm_relative_rotation_change_deg")
    if translation is not None:
        score -= 20.0 * max(0.0, float(translation) - 0.03) / 0.03
    if rotation is not None:
        score -= 10.0 * max(0.0, float(rotation) - 30.0) / 30.0
    if grasp_quality:
        score += 4.0 * np.clip(
            float(metric.get("terminal_loaded_finger_mean", 0.0)) / 5.0, 0.0, 1.0)
        score += 8.0 * float(metric.get("terminal_thumb_opposition_ratio", 0.0))
        score += 4.0 * float(metric.get("lift_thumb_opposition_ratio", 0.0))
    return score


def make_env_args(device, lift_threshold=0.15):
    args = SimpleNamespace()
    args.device = device
    args.dt = 1.0 / 60.0
    args.substeps = 2
    args.finger_stiffness = 120.0
    args.finger_damping = 5.0
    args.mimic_stiffness = 120.0
    args.mimic_damping = 5.0
    args.clearance = 0.005
    args.object_friction = 1.0
    args.settle_steps = 30
    args.steps_per_frame = 3
    args.hold_steps = 30
    args.lift_threshold = lift_threshold
    return args


def target_row(target_data, source_index):
    indices = np.asarray(target_data.get(
        "source_trajectory_indices",
        np.arange(len(target_data["grasp_seqs"]))), dtype=np.int64)
    matches = np.flatnonzero(indices == int(source_index))
    if not len(matches):
        raise ValueError("source trajectory is absent from target file")
    return int(matches[0])


def evaluate_population(hand, base_case, target_data, population, device,
                        lift_threshold, parameterization="global", basis=None,
                        grasp_quality=False):
    from train_residual_ppo_general import GeneralResidualEnv, rollout
    cases = [PhysicsCase(
        hand, base_case.category, base_case.object_name,
        base_case.source_index,
        (apply_phase_parameters(base_case.target_frames, parameters)
         if parameterization == "phase"
         else apply_cradle_parameters(base_case.target_frames, parameters)
         if parameterization == "cradle"
         else apply_synergy_parameters(base_case.target_frames, parameters, basis)
         if parameterization == "synergy"
         else apply_trajectory_parameters(base_case.target_frames, parameters)),
        base_case.mesh_path, base_case.scale, base_case.rotation,
    ) for parameters in population]
    env = GeneralResidualEnv(
        cases, target_data, make_env_args(device, lift_threshold))
    try:
        metrics = rollout(env)
    finally:
        env.close()
    return metrics, np.asarray([
        physics_score(item, lift_threshold, grasp_quality) for item in metrics
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hand", choices=("linker", "xhand", "wuji"), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--source-index", type=int, required=True)
    parser.add_argument("--object-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path)
    parser.add_argument("--population", type=int, default=12)
    parser.add_argument("--elite", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lift-threshold", type=float, default=0.15)
    parser.add_argument("--parameterization", choices=("global", "phase", "synergy", "cradle"),
                        default="global")
    parser.add_argument("--selection-margin", type=float, default=0.0)
    parser.add_argument("--synergy-basis", type=Path)
    parser.add_argument("--confirm-single-env", action="store_true")
    parser.add_argument("--parameter-scale", type=float, default=1.0)
    parser.add_argument("--score-mode", choices=("standard", "grasp_quality"),
                        default="standard")
    args = parser.parse_args()
    if not 1 <= args.elite <= args.population:
        parser.error("elite must be within population")

    source = np.load(args.source, allow_pickle=True).item()
    target = np.load(args.target, allow_pickle=True).item()
    row = target_row(target, args.source_index)
    frames = np.asarray(target["grasp_seqs"][row], dtype=np.float32)
    object_name = args.source.stem
    base_case = PhysicsCase(
        args.hand, object_name.split("-", 2)[1], object_name,
        args.source_index, frames,
        args.object_dir / "coacd" / "decomposed.obj",
        float(np.asarray(source["obj_scale"])[args.source_index]),
        np.asarray(source["obj_rotmat"])[args.source_index],
    )

    rng = np.random.default_rng(args.seed)
    basis = None
    if args.parameterization == "synergy":
        if args.synergy_basis is None:
            parser.error("--synergy-basis is required for synergy parameterization")
        basis = np.asarray(np.load(args.synergy_basis), dtype=np.float32)
        rank = len(basis)
        std = np.asarray([0.08] * rank + [0.10] * rank, dtype=np.float32)
        bounds = np.asarray([0.20] * rank + [0.25] * rank, dtype=np.float32)
    elif args.parameterization == "phase":
        std = np.asarray([0.05] * 5 + [0.07] * 5, dtype=np.float32)
        bounds = np.asarray([0.12] * 5 + [0.18] * 5, dtype=np.float32)
    elif args.parameterization == "cradle":
        std = np.asarray([0.006] * 2 + [0.06] * 3, dtype=np.float32)
        bounds = np.asarray([0.015] * 2 + [0.15] * 3, dtype=np.float32)
    else:
        linker_coupled = args.hand == "linker" and frames.shape[1] == 12
        if linker_coupled:
            finger_std = [0.08, 0.12] + [0.15] * 4
            finger_bounds = [0.20, 0.30] + [0.35] * 4
        else:
            finger_std = [0.12] * 5
            finger_bounds = [0.30] * 5
        std = np.asarray(
            [0.010] * 3 + [0.10] * 3 + finger_std + [0.020],
            dtype=np.float32)
        bounds = np.asarray(
            [0.025] * 3 + [0.25] * 3 + finger_bounds + [0.050],
            dtype=np.float32)
    if args.parameter_scale <= 0.0:
        parser.error("parameter-scale must be positive")
    std *= args.parameter_scale
    bounds *= args.parameter_scale
    mean = np.zeros(len(std), dtype=np.float32)
    best_parameters = mean.copy()
    best_score = -np.inf
    best_metric = None
    baseline_score = None
    baseline_metric = None
    history = []
    for iteration in range(1, args.iterations + 1):
        population = rng.normal(
            mean, std, size=(args.population, len(mean))).astype(np.float32)
        population = np.clip(population, -bounds, bounds)
        population[0] = 0.0
        metrics, scores = evaluate_population(
            args.hand, base_case, target, population, args.device,
            args.lift_threshold, args.parameterization, basis,
            args.score_mode == "grasp_quality")
        if baseline_metric is None:
            baseline_metric = metrics[0]
            baseline_score = float(scores[0])
            best_metric = baseline_metric
            best_score = baseline_score
            best_parameters = np.zeros_like(mean)
        ranking = np.argsort(scores)[::-1]
        elite = population[ranking[:args.elite]]
        mean = elite.mean(axis=0)
        std = np.maximum(elite.std(axis=0), bounds * 0.05)
        winner = int(ranking[0])
        if scores[winner] > best_score + args.selection_margin:
            best_score = float(scores[winner])
            best_parameters = population[winner].copy()
            best_metric = metrics[winner]
        history.append({
            "iteration": iteration,
            "best_score": float(scores[winner]),
            "best_success": bool(metrics[winner]["success"]),
            "best_final_lift_m": float(metrics[winner]["final_lift_m"]),
            "baseline_score": baseline_score,
            "mean": mean.tolist(),
            "std": std.tolist(),
        })
        print(json.dumps(history[-1], ensure_ascii=False))

    confirmation = None
    if args.confirm_single_env and np.any(best_parameters != 0.0):
        zero = np.zeros_like(best_parameters)[None]
        candidate = best_parameters[None]
        confirmed_baseline_metrics, confirmed_baseline_scores = evaluate_population(
            args.hand, base_case, target, zero, args.device,
            args.lift_threshold, args.parameterization, basis,
            args.score_mode == "grasp_quality")
        confirmed_candidate_metrics, confirmed_candidate_scores = evaluate_population(
            args.hand, base_case, target, candidate, args.device,
            args.lift_threshold, args.parameterization, basis,
            args.score_mode == "grasp_quality")
        confirmed_baseline = confirmed_baseline_metrics[0]
        confirmed_candidate = confirmed_candidate_metrics[0]
        preserves_success = (
            not confirmed_baseline["success"] or confirmed_candidate["success"])
        preserves_transport = (
            not confirmed_baseline.get("transport_stability_success", False)
            or confirmed_candidate.get("transport_stability_success", False))
        accepted = bool(
            confirmed_candidate_scores[0]
            > confirmed_baseline_scores[0] + args.selection_margin
            and preserves_success
            and preserves_transport)
        confirmation = {
            "accepted": accepted,
            "baseline_score": float(confirmed_baseline_scores[0]),
            "candidate_score": float(confirmed_candidate_scores[0]),
            "preserves_success": bool(preserves_success),
            "preserves_transport": bool(preserves_transport),
        }
        baseline_metric = confirmed_baseline
        baseline_score = float(confirmed_baseline_scores[0])
        if accepted:
            best_metric = confirmed_candidate
            best_score = float(confirmed_candidate_scores[0])
        else:
            best_parameters = np.zeros_like(best_parameters)
            best_metric = confirmed_baseline
            best_score = float(confirmed_baseline_scores[0])

    refined = dict(target)
    sequences = np.asarray(target["grasp_seqs"]).copy()
    sequences[row] = (
        apply_phase_parameters(frames, best_parameters)
        if args.parameterization == "phase"
        else apply_cradle_parameters(frames, best_parameters)
        if args.parameterization == "cradle"
        else apply_synergy_parameters(frames, best_parameters, basis)
        if args.parameterization == "synergy"
        else apply_trajectory_parameters(frames, best_parameters))
    refined["grasp_seqs"] = sequences
    refined["physics_cem"] = {
        "schema": ("physics_cem_phase_basis_v1"
                   if args.parameterization == "phase"
                   else "physics_cem_lift_cradle_v1"
                   if args.parameterization == "cradle"
                   else "physics_cem_synergy_basis_v1"
                   if args.parameterization == "synergy"
                   else "physics_cem_lowdim_v1"),
        "parameterization": args.parameterization,
        "source_index": int(args.source_index),
        "parameters": best_parameters.tolist(),
        "baseline_score": baseline_score,
        "baseline_metric": baseline_metric,
        "score": best_score,
        "metric": best_metric,
        "history": history,
        "single_env_confirmation": confirmation,
        "score_mode": args.score_mode,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, refined, allow_pickle=True)
    report_output = args.report_output or args.output.with_suffix(".json")
    report_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.write_text(
        json.dumps(refined["physics_cem"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")


if __name__ == "__main__":
    main()
