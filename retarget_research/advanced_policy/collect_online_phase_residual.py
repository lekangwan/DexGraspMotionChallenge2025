#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from evaluate_policy_isaac import prepare_hand, read_policy_pre_action_state
from observations import build_object_shape_descriptor, build_runtime_observation
from runtime import PolicyRunner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hand", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--policy-split", type=Path, required=True)
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-trajectories", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    split = json.loads(args.policy_split.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    by_object = {entry["object_name"]: entry for entry in manifest["entries"]}
    runner = PolicyRunner(
        checkpoint=str(args.checkpoint),
        data_dir=str(args.data_dir),
        device=args.device,
        normalized_action_clip=5.0,
        action_rate_limit_scale=0.0,
    )
    if runner.model_type != "phase_residual":
        raise ValueError("在线采集学生必须是phase_residual")
    motion_steps = runner.motion_steps
    asset, dof_properties, dof_names, policy_open, open_first, mapper, action_order = \
        prepare_hand(args.hand, None)
    gym = asset.gym
    collected = 0
    for record in split["records"]:
        if record["split"] != "train":
            continue
        entry = by_object.get(record["object_name"])
        if entry is None:
            continue
        source_index = int(record["source_trajectory_index"])
        source = np.load(Path(entry["source_path"]), allow_pickle=True).item()
        target = np.load(args.target_dir / f"{record['object_name']}.npy",
                         allow_pickle=True).item()
        target_indices = np.asarray(target["source_trajectory_indices"], dtype=np.int64)
        target_row = int(np.flatnonzero(target_indices == source_index)[0])
        expert_frames = np.asarray(target["grasp_seqs"][target_row], dtype=np.float32)
        env, hand_asset, hand_shape, object_asset, object_shape, dof_props, base_state = \
            prepare_env(gym, asset, dof_properties, dof_names, open_first, mapper,
                        entry, source_index)
        try:
            observations, commands, phases, expert_deltas = [], [], [], []
            for step in range(240):
                if step == 0:
                    gym.set_actor_root_state_tensor_indexed(
                        env, gym.get_actor_root_state_tensor(env), [object_shape],
                        len(gym.get_actor_root_state_tensor(env)) - 1)
                dof_states, object_state, contact_count = read_policy_pre_action_state(
                    gym, env, hand_asset, object_asset)
                observation = build_runtime_observation(
                    dof_states, object_state, initial_position, contact_count,
                    shape_descriptor, 0.30)
                if step == 0:
                    runner.reset(record["category"], observation,
                                 initial_action=policy_open)
                    previous_command = policy_open.copy()
                command = runner.act(observation)
                phase = min(step / float(motion_steps - 1), 1.0)
                expert_step = min(int(round(phase * (len(expert_frames) - 1))),
                                  len(expert_frames) - 1)
                expert_command = expert_frames[expert_step]
                observations.append(observation)
                commands.append(previous_command)
                phases.append(phase)
                expert_deltas.append(expert_command - previous_command)
                physical = np.asarray(mapper(command), dtype=np.float32)
                gym.set_actor_dof_position_targets(env, hand_asset, physical)
                gym.simulate(gym.sim)
                gym.fetch_results(gym.sim, True)
                previous_command = command
            observations = np.stack(observations).astype(np.float32)
            commands = np.stack(commands).astype(np.float32)
            phases = np.asarray(phases, dtype=np.float32)
            expert_deltas = np.stack(expert_deltas).astype(np.float32)
            np.savez(
                args.output_dir / f"online_{record['object_name']}_{source_index}.npz",
                observations=observations,
                previous_commands=commands,
                phase=phases[:, None],
                expert_deltas=expert_deltas,
                category_id=np.full(len(observations), runner.category_id, dtype=np.int64),
                metadata_json=np.array(json.dumps({
                    "hand": args.hand,
                    "object_name": record["object_name"],
                    "source_trajectory_index": source_index,
                    "student_checkpoint": str(args.checkpoint),
                    "alignment": "student_observation_to_phase_expert_delta_v1",
                })),
            )
            collected += 1
            print(f"collected {record['object_name']}[{source_index}]")
        finally:
            gym.destroy_env(env)
        if collected >= args.max_trajectories:
            break
    print(f"COLLECTED={collected}")


if __name__ == "__main__":
    main()
