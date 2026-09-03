#!/usr/bin/env python3
"""Use real PhysX fingertip contacts to acquire a small grasp residual."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from isaacgym import gymapi


ROOT = Path(__file__).resolve().parents[3]
ADVANCED = ROOT / "retarget_research" / "advanced_policy"
EVALUATE = ROOT / "retarget_research" / "retargeting" / "evaluate"
RUN = ROOT / "retarget_research" / "retargeting" / "run"
for path in (ADVANCED, EVALUATE, RUN):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from isaac_replay_common import (  # noqa: E402
    actor_body_indices,
    count_contacts_by_hand_body,
    read_object_state,
)
from physics_cem_refine import PhysicsCase, make_env_args, physics_score  # noqa: E402
from train_residual_ppo_general import (  # noqa: E402
    GeneralResidualEnv,
    interpolated_policy_command,
    rollout,
)


FINGER_GROUPS = {
    "linker": ((0, 1), (2,), (3,), (4,), (5,)),
    "xhand": ((0, 1, 2), (3, 4, 5), (6, 7), (8, 9), (10, 11)),
    "wuji": ((0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 10, 11),
             (12, 13, 14, 15), (16, 17, 18, 19)),
}
TIP_LINKS = {
    "linker": ("rh_thumb_distal", "rh_index_distal", "rh_middle_distal",
               "rh_ring_distal", "rh_pinky_distal"),
    "xhand": ("right_hand_thumb_rota_link2", "right_hand_index_rota_link2",
              "right_hand_mid_link2", "right_hand_ring_link2",
              "right_hand_pinky_link2"),
    "wuji": ("finger1_tip_link", "finger2_tip_link", "finger3_tip_link",
             "finger4_tip_link", "finger5_tip_link"),
}


def load_case(hand, entry, target_dir):
    """Load one incumbent CEM trajectory and its object initialization."""
    target_path = Path(target_dir) / f"{entry['object_name']}.npy"
    target = np.load(target_path, allow_pickle=True).item()
    source_index = int(entry["trajectory_indices"][0])
    indices = np.asarray(target["source_trajectory_indices"], dtype=np.int64)
    rows = np.flatnonzero(indices == source_index)
    if len(rows) != 1:
        raise ValueError(f"{target_path}: source index {source_index} is not unique")
    source = np.load(entry["source_path"], allow_pickle=True).item()
    frames = np.asarray(target["grasp_seqs"][int(rows[0])], dtype=np.float32)
    case = PhysicsCase(
        hand=hand,
        category=entry["category"],
        object_name=entry["object_name"],
        source_index=source_index,
        target_frames=frames,
        mesh_path=Path(entry["object_asset_path"]) / "coacd" / "decomposed.obj",
        scale=float(np.asarray(source["obj_scale"])[source_index]),
        rotation=np.asarray(source["obj_rotmat"])[source_index],
    )
    return case, target, int(rows[0])


def phase_frames(frames):
    """Estimate closure start and the last pre-lift grasp frame from the trajectory."""
    fingers = np.asarray(frames[:, 6:], dtype=np.float32)
    movement = np.linalg.norm(fingers - fingers[0], axis=1)
    threshold = 0.2 * float(movement.max())
    close_candidates = np.flatnonzero(movement >= max(threshold, 1e-3))
    close_frame = int(close_candidates[0]) if len(close_candidates) else 20
    wrist_z = np.asarray(frames[:, 2], dtype=np.float32)
    base_z = float(wrist_z[close_frame:].min())
    lift_candidates = np.flatnonzero(
        (np.arange(len(frames)) > close_frame) & (wrist_z >= base_z + 0.03)
    )
    lift_frame = int(lift_candidates[0]) if len(lift_candidates) else min(55, len(frames) - 1)
    return close_frame, max(close_frame + 1, lift_frame - 1)


def closure_directions(frames, close_frame, grasp_frame, groups):
    """Derive each finger's closing direction from the incumbent motion itself."""
    start = frames[close_frame, 6:]
    end = frames[grasp_frame, 6:]
    fallback = frames[-1, 6:]
    directions = []
    for group in groups:
        group = np.asarray(group, dtype=np.int64)
        direction = end[group] - start[group]
        norm = float(np.linalg.norm(direction))
        if norm < 1e-4:
            direction = fallback[group] - start[group]
            norm = float(np.linalg.norm(direction))
        directions.append(direction / norm if norm >= 1e-4 else np.zeros_like(direction))
    return directions


def body_maps(env):
    """Build per-environment rigid-body name maps used for fingertip contacts."""
    maps = []
    for gym_env, hand_actor, object_actor in zip(env.envs, env.hands, env.objects):
        indices = actor_body_indices(env.gym, gym_env, hand_actor)
        names = env.gym.get_actor_rigid_body_names(gym_env, hand_actor)
        maps.append((dict(zip(indices, names)), actor_body_indices(
            env.gym, gym_env, object_actor)))
    return maps


def contact_loads(contacts, hand_index_to_name, object_body_indices):
    """Sum positive normal impulses for each hand body in hand-object contacts."""
    objects = set(object_body_indices)
    loads = {name: 0.0 for name in hand_index_to_name.values()}
    for contact in contacts:
        body0, body1 = int(contact["body0"]), int(contact["body1"])
        if body0 in hand_index_to_name and body1 in objects:
            loads[hand_index_to_name[body0]] += max(0.0, float(contact["lambda"]))
        elif body1 in hand_index_to_name and body0 in objects:
            loads[hand_index_to_name[body1]] += max(0.0, float(contact["lambda"]))
    return loads


def step_commands(env, commands):
    """Map policy commands to physical DOFs, clip them, and advance one PhysX step."""
    for i, command in enumerate(commands):
        physical = np.asarray(env.mappers[i](command), dtype=np.float32)
        env.gym.set_actor_dof_position_targets(
            env.envs[i], env.hands[i], np.clip(physical, env.lower, env.upper))
    env.gym.simulate(env.sim)
    env.gym.fetch_results(env.sim, True)


def acquire_batch(cases, target_data, max_extra, extra_steps, stable_steps,
                  min_contact_fingers, max_object_drift, min_contact_impulse,
                  require_thumb):
    """Replay to grasp, close unfixed fingers, and stop each on stable contact."""
    env = GeneralResidualEnv(cases, target_data, make_env_args("cpu", 0.15))
    try:
        env.reset()
        maps = body_maps(env)
        groups = FINGER_GROUPS[cases[0].hand]
        tip_names = TIP_LINKS[cases[0].hand]
        phases = [phase_frames(case.target_frames) for case in cases]
        grasp_steps = [(grasp + 1) * env.steps_per_frame - 1 for _, grasp in phases]
        max_grasp_step = max(grasp_steps)
        last_commands = [command.copy() for command in env.policy_open_commands]
        for step in range(max_grasp_step + 1):
            commands = []
            for i, case in enumerate(cases):
                local_step = min(step, grasp_steps[i])
                command = interpolated_policy_command(
                    case.target_frames, env.policy_open_commands[i], local_step,
                    env.steps_per_frame)
                last_commands[i] = command.copy()
                commands.append(command)
            step_commands(env, commands)

        start_positions = np.stack([
            read_object_state(env.gym, gym_env, actor)["object_position"]
            for gym_env, actor in zip(env.envs, env.objects)
        ])
        directions = [closure_directions(case.target_frames, *phase, groups)
                      for case, phase in zip(cases, phases)]
        frozen = np.zeros((len(cases), 5), dtype=bool)
        streak = np.zeros((len(cases), 5), dtype=np.int32)
        contact_steps = np.full((len(cases), 5), -1, dtype=np.int32)
        acquired = [command.copy() for command in last_commands]
        for step in range(extra_steps):
            fraction = (step + 1) / float(extra_steps)
            commands = []
            for i in range(len(cases)):
                command = acquired[i].copy()
                for finger, group in enumerate(groups):
                    if not frozen[i, finger]:
                        command[6 + np.asarray(group)] = (
                            last_commands[i][6 + np.asarray(group)]
                            + directions[i][finger] * max_extra * fraction)
                commands.append(command)
            step_commands(env, commands)
            for i, gym_env in enumerate(env.envs):
                contacts = env.gym.get_env_rigid_contacts(gym_env)
                loads = contact_loads(contacts, *maps[i])
                for finger, tip in enumerate(tip_names):
                    loaded = loads.get(tip, 0.0) >= min_contact_impulse
                    streak[i, finger] = streak[i, finger] + 1 if loaded else 0
                    if not frozen[i, finger] and streak[i, finger] >= stable_steps:
                        frozen[i, finger] = True
                        contact_steps[i, finger] = step
                        acquired[i][6 + np.asarray(groups[finger])] = commands[i][
                            6 + np.asarray(groups[finger])]

        reports = []
        for i, case in enumerate(cases):
            position = read_object_state(
                env.gym, env.envs[i], env.objects[i])["object_position"]
            drift = float(np.linalg.norm(position[:2] - start_positions[i, :2]))
            contact_fingers = int(frozen[i].sum())
            opposition = bool(frozen[i, 0] and frozen[i, 1:].any())
            accepted = (
                contact_fingers >= min_contact_fingers
                and (opposition or not require_thumb)
                and drift <= max_object_drift
            )
            residual = acquired[i][6:] - last_commands[i][6:] if accepted else np.zeros(
                case.target_frames.shape[1] - 6, dtype=np.float32)
            reports.append({
                "residual": residual.astype(np.float32),
                "accepted": bool(accepted),
                "contact_fingers": contact_fingers,
                "contact_steps": contact_steps[i].tolist(),
                "object_xy_drift_m": drift,
                "thumb_opposition": opposition,
                "close_frame": int(phases[i][0]),
                "grasp_frame": int(phases[i][1]),
            })
        return reports
    finally:
        env.close()


def physics_gate(cases, target_data, reports):
    """Keep a contact candidate only when a full replay beats its incumbent safely."""
    candidates = [PhysicsCase(
        case.hand, case.category, case.object_name, case.source_index,
        apply_residual(case.target_frames, report), case.mesh_path, case.scale,
        case.rotation,
    ) for case, report in zip(cases, reports)]
    comparison = cases + candidates
    env = GeneralResidualEnv(comparison, target_data, make_env_args("cpu", 0.15))
    try:
        metrics = rollout(env)
    finally:
        env.close()
    count = len(cases)
    for i, report in enumerate(reports):
        baseline, candidate = metrics[i], metrics[count + i]
        base_score = float(physics_score(baseline, 0.15))
        candidate_score = float(physics_score(candidate, 0.15))
        base_quality = bool(baseline["success"] and baseline["transport_stability_success"])
        candidate_quality = bool(
            candidate["success"] and candidate["transport_stability_success"])
        selected = bool(
            report["accepted"]
            and candidate_score > base_score
            and (not baseline["success"] or candidate["success"])
            and (not base_quality or candidate_quality)
        )
        report.update({
            "physics_selected": selected,
            "baseline_physics_score": base_score,
            "candidate_physics_score": candidate_score,
            "baseline_success": bool(baseline["success"]),
            "candidate_success": bool(candidate["success"]),
            "baseline_transport": bool(baseline["transport_stability_success"]),
            "candidate_transport": bool(candidate["transport_stability_success"]),
        })
        if not selected:
            report["residual"] = np.zeros_like(report["residual"])


def apply_residual(frames, report):
    """Blend the acquired finger residual into closure and retain it during lift."""
    result = np.asarray(frames, dtype=np.float32).copy()
    close_frame, grasp_frame = report["close_frame"], report["grasp_frame"]
    gate = np.zeros(len(result), dtype=np.float32)
    gate[grasp_frame:] = 1.0
    if grasp_frame > close_frame:
        x = np.linspace(0.0, 1.0, grasp_frame - close_frame + 1, dtype=np.float32)
        gate[close_frame:grasp_frame + 1] = x * x * (3.0 - 2.0 * x)
    result[:, 6:] += gate[:, None] * report["residual"][None]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hand", choices=("linker", "xhand", "wuji"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--max-extra", type=float, default=0.30)
    parser.add_argument("--extra-steps", type=int, default=45)
    parser.add_argument("--stable-steps", type=int, default=2)
    parser.add_argument("--min-contact-fingers", type=int, default=2)
    parser.add_argument("--max-object-drift", type=float, default=0.02)
    parser.add_argument("--min-contact-impulse", type=float, default=0.0)
    parser.add_argument("--require-thumb", action="store_true")
    parser.add_argument("--physics-gate", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    entries = manifest["entries"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for start in range(0, len(entries), args.batch_size):
        loaded = [load_case(args.hand, entry, args.target_dir)
                  for entry in entries[start:start + args.batch_size]]
        cases = [item[0] for item in loaded]
        reports = acquire_batch(
            cases, loaded[0][1], args.max_extra, args.extra_steps,
            args.stable_steps, args.min_contact_fingers, args.max_object_drift,
            args.min_contact_impulse, args.require_thumb)
        if args.physics_gate:
            physics_gate(cases, loaded[0][1], reports)
        for (case, target, row), report in zip(loaded, reports):
            output = dict(target)
            sequences = np.asarray(target["grasp_seqs"]).copy()
            sequences[row] = apply_residual(case.target_frames, report)
            output["grasp_seqs"] = sequences
            metadata = {key: value for key, value in report.items() if key != "residual"}
            metadata["residual"] = report["residual"].tolist()
            metadata["schema"] = (
                "physics_loaded_contact_stop_closure_v2"
                if args.physics_gate else "physics_contact_stop_closure_v1")
            output["physics_contact_stop_closure"] = metadata
            np.save(args.output_dir / f"{case.object_name}.npy", output, allow_pickle=True)
            summaries.append({"object_name": case.object_name,
                              "source_trajectory_index": case.source_index, **metadata})
            print(json.dumps(summaries[-1], ensure_ascii=False))

    output_manifest = dict(manifest)
    output_manifest["purpose"] = "physics contact-stop closure calibration50 candidates"
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(
        json.dumps(output_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir.parent / f"{args.hand}_contact_stop_summary.json").write_text(
        json.dumps({"hand": args.hand, "results": summaries}, ensure_ascii=False, indent=2)
        + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
