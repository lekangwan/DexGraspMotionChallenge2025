#!/usr/bin/env python3
"""Bake a lift-time slip feedback controller into an open-loop trajectory."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from isaacgym import gymapi
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[3]
for path in (
    ROOT / "retarget_research" / "advanced_policy",
    ROOT / "retarget_research" / "retargeting" / "evaluate",
    ROOT / "retarget_research" / "retargeting" / "run",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from geort_exact_fk import anatomical_limits  # noqa: E402
from physics_cem_refine import PhysicsCase, make_env_args, physics_score  # noqa: E402
from physics_contact_stop_closure import (  # noqa: E402
    FINGER_GROUPS,
    TIP_LINKS,
    body_maps,
    closure_directions,
    contact_loads,
    load_case,
    phase_frames,
    step_commands,
)
from retarget_research.minimal_impl.kinematics import build_target_model  # noqa: E402
from train_residual_ppo_general import (  # noqa: E402
    GeneralResidualEnv,
    interpolated_policy_command,
    rollout,
)
from isaac_replay_common import read_object_state  # noqa: E402


def palm_local_object(env, index):
    """Return object position expressed in the current physical wrist frame."""
    dof = env.gym.get_actor_dof_states(
        env.envs[index], env.hands[index], gymapi.STATE_ALL)["pos"]
    obj = read_object_state(env.gym, env.envs[index], env.objects[index])
    rotation = Rotation.from_euler("xyz", np.asarray(dof[3:6], dtype=np.float64))
    return rotation.inv().apply(
        np.asarray(obj["object_position"], dtype=np.float64)
        - np.asarray(dof[:3], dtype=np.float64))


def collect_feedback_profiles(cases, target_data, slip_threshold,
                              min_contact_impulse, tighten_per_frame,
                              max_residual, max_recoverable_slip,
                              require_existing_contact):
    """Run the incumbent and tighten unloaded fingers when palm-relative slip appears."""
    env = GeneralResidualEnv(cases, target_data, make_env_args("cpu", 0.15))
    try:
        env.reset()
        maps = body_maps(env)
        groups = FINGER_GROUPS[cases[0].hand]
        tips = TIP_LINKS[cases[0].hand]
        phases = [phase_frames(case.target_frames) for case in cases]
        directions = [closure_directions(case.target_frames, *phase, groups)
                      for case, phase in zip(cases, phases)]
        lift_steps = [(grasp + 1) * env.steps_per_frame for _, grasp in phases]
        residuals = [np.zeros(case.target_frames.shape[1] - 6, dtype=np.float32)
                     for case in cases]
        profiles = [np.zeros((len(case.target_frames), len(residual)), dtype=np.float32)
                    for case, residual in zip(cases, residuals)]
        references = [None] * len(cases)
        had_support = np.zeros(len(cases), dtype=bool)
        trigger_steps = np.zeros(len(cases), dtype=np.int32)
        max_slips = np.zeros(len(cases), dtype=np.float64)
        horizon = max(len(case.target_frames) for case in cases) * env.steps_per_frame
        increment = float(tighten_per_frame) / env.steps_per_frame

        for step in range(horizon):
            commands = []
            for i, case in enumerate(cases):
                command = interpolated_policy_command(
                    case.target_frames, env.policy_open_commands[i], step,
                    env.steps_per_frame)
                if step >= lift_steps[i]:
                    local = palm_local_object(env, i)
                    if references[i] is None:
                        references[i] = local.copy()
                    slip = float(np.linalg.norm(local - references[i]))
                    max_slips[i] = max(max_slips[i], slip)
                    contacts = env.gym.get_env_rigid_contacts(env.envs[i])
                    loads = contact_loads(contacts, *maps[i])
                    loaded = np.asarray([
                        loads.get(name, 0.0) >= min_contact_impulse for name in tips
                    ], dtype=bool)
                    support = int(loaded.sum())
                    had_support[i] = bool(had_support[i] or support >= 2)
                    elapsed = step - lift_steps[i]
                    if require_existing_contact:
                        trigger = bool(
                            support >= 1
                            and slip_threshold <= slip <= max_recoverable_slip
                        )
                    else:
                        trigger = bool(
                            slip >= slip_threshold
                            or (had_support[i] and support < 2)
                            or (elapsed >= 3 and not had_support[i])
                        )
                    if trigger:
                        active = np.flatnonzero(~loaded)
                        if not len(active):
                            active = np.arange(5)
                        for finger in active:
                            group = np.asarray(groups[int(finger)], dtype=np.int64)
                            current = residuals[i][group]
                            proposed = current + directions[i][int(finger)] * increment
                            norm = float(np.linalg.norm(proposed))
                            if norm > max_residual:
                                proposed *= max_residual / norm
                            residuals[i][group] = proposed
                        trigger_steps[i] += 1
                    command[6:] += residuals[i]
                profiles[i][min(step // env.steps_per_frame, len(profiles[i]) - 1)] = residuals[i]
                commands.append(command)
            step_commands(env, commands)
        return profiles, [{
            "feedback_trigger_steps": int(trigger_steps[i]),
            "max_detected_palm_relative_slip_m": float(max_slips[i]),
            "final_residual_l2_rad": float(np.linalg.norm(residuals[i])),
        } for i in range(len(cases))]
    finally:
        env.close()


def apply_profile(frames, profile, lower, upper):
    """Add the time-varying finger residual and clip to target-hand limits."""
    result = np.asarray(frames, dtype=np.float32).copy()
    result[:, 6:] = np.clip(result[:, 6:] + profile, lower, upper)
    return result


def select_candidates(cases, candidates, target_data, reports, score_margin):
    """Use full physics replay to reject regressions and insignificant score changes."""
    env = GeneralResidualEnv(cases + candidates, target_data, make_env_args("cpu", 0.15))
    try:
        metrics = rollout(env)
    finally:
        env.close()
    count = len(cases)
    selected = []
    for i in range(count):
        baseline, candidate = metrics[i], metrics[count + i]
        base_score = float(physics_score(baseline, 0.15))
        candidate_score = float(physics_score(candidate, 0.15))
        base_quality = bool(baseline["success"] and baseline["transport_stability_success"])
        candidate_quality = bool(
            candidate["success"] and candidate["transport_stability_success"])
        keep = bool(
            candidate_score >= base_score + score_margin
            and (not baseline["success"] or candidate["success"])
            and (not base_quality or candidate_quality)
        )
        reports[i].update({
            "physics_selected": keep,
            "baseline_physics_score": base_score,
            "candidate_physics_score": candidate_score,
            "baseline_success": bool(baseline["success"]),
            "candidate_success": bool(candidate["success"]),
            "baseline_transport": bool(baseline["transport_stability_success"]),
            "candidate_transport": bool(candidate["transport_stability_success"]),
        })
        selected.append(keep)
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hand", choices=("linker", "xhand", "wuji"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--slip-threshold", type=float, default=0.005)
    parser.add_argument("--min-contact-impulse", type=float, default=0.02)
    parser.add_argument("--tighten-per-frame", type=float, default=0.015)
    parser.add_argument("--max-residual", type=float, default=0.25)
    parser.add_argument("--score-margin", type=float, default=1.0)
    parser.add_argument("--max-recoverable-slip", type=float, default=float("inf"))
    parser.add_argument("--require-existing-contact", action="store_true")
    args = parser.parse_args()

    model = build_target_model(args.hand, device="cpu")
    lower, upper = anatomical_limits(model, args.hand)
    lower, upper = lower.cpu().numpy(), upper.cpu().numpy()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for start in range(0, len(manifest["entries"]), args.batch_size):
        loaded = [load_case(args.hand, entry, args.target_dir)
                  for entry in manifest["entries"][start:start + args.batch_size]]
        cases = [item[0] for item in loaded]
        profiles, reports = collect_feedback_profiles(
            cases, loaded[0][1], args.slip_threshold, args.min_contact_impulse,
            args.tighten_per_frame, args.max_residual,
            args.max_recoverable_slip, args.require_existing_contact)
        candidate_frames = [apply_profile(case.target_frames, profile, lower, upper)
                            for case, profile in zip(cases, profiles)]
        candidates = [PhysicsCase(
            case.hand, case.category, case.object_name, case.source_index, frames,
            case.mesh_path, case.scale, case.rotation)
            for case, frames in zip(cases, candidate_frames)]
        selected = select_candidates(
            cases, candidates, loaded[0][1], reports, args.score_margin)
        for (case, target, row), profile, candidate, keep, report in zip(
                loaded, profiles, candidate_frames, selected, reports):
            output = dict(target)
            sequences = np.asarray(target["grasp_seqs"]).copy()
            sequences[row] = candidate if keep else case.target_frames
            output["grasp_seqs"] = sequences
            report.update({
                "schema": "physics_lift_slip_feedback_v1",
                "slip_threshold_m": args.slip_threshold,
                "min_contact_impulse": args.min_contact_impulse,
                "tighten_per_frame_rad": args.tighten_per_frame,
                "max_per_finger_residual_rad": args.max_residual,
                "max_recoverable_slip_m": args.max_recoverable_slip,
                "require_existing_contact": args.require_existing_contact,
                "score_margin": args.score_margin,
            })
            output["physics_slip_feedback"] = report
            np.save(args.output_dir / f"{case.object_name}.npy", output, allow_pickle=True)
            summaries.append({"object_name": case.object_name,
                              "source_trajectory_index": case.source_index, **report})
            print(json.dumps(summaries[-1], ensure_ascii=False))

    output_manifest = dict(manifest)
    output_manifest["purpose"] = "physics lift-slip feedback calibration50 candidates"
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(
        json.dumps(output_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir.parent / f"{args.hand}_slip_feedback_summary.json").write_text(
        json.dumps({"hand": args.hand, "results": summaries}, ensure_ascii=False, indent=2)
        + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
