#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
from isaacgym import gymapi
from scipy.spatial.transform import Rotation

for _p in (str(Path(__file__).resolve().parents[1] / "retargeting" / "evaluate"),
           str(Path(__file__).resolve().parents[1] / "retargeting" / "prepare")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
try:
    from .evaluate_policy_isaac import prepare_hand, set_object_friction  # noqa: E402
    from .observations import build_object_shape_descriptor  # noqa: E402
except ImportError:
    from evaluate_policy_isaac import prepare_hand, set_object_friction  # noqa: E402
    from observations import build_object_shape_descriptor  # noqa: E402
from isaac_replay_common import (  # noqa: E402
    actor_body_indices,
    compute_success_metrics,
    create_cpu_sim,
    load_object_asset,
    object_start_pose,
    read_object_state,
    set_dof_state_and_target,
)

import torch  # noqa: E402


TIP_LINKS = {
    "linker": ("rh_thumb_distal", "rh_index_distal", "rh_middle_distal",
               "rh_ring_distal", "rh_pinky_distal"),
    "xhand": ("right_hand_thumb_rota_link2", "right_hand_index_rota_link2",
              "right_hand_mid_link2", "right_hand_ring_link2",
              "right_hand_pinky_link2"),
    "wuji": ("finger1_tip_link", "finger2_tip_link", "finger3_tip_link",
             "finger4_tip_link", "finger5_tip_link"),
}
try:
    from .residual_ppo import PPOConfig, ResidualActorCritic, RolloutStorage, ppo_update
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from residual_ppo import PPOConfig, ResidualActorCritic, RolloutStorage, ppo_update


class Case:
    def __init__(self, hand, category, object_name, source_index, target_frames,
                 mesh_path, scale, rotation, initial_command=None):
        self.hand = hand
        self.category = category
        self.object_name = object_name
        self.source_index = source_index
        self.target_frames = target_frames
        self.mesh_path = mesh_path
        self.scale = scale
        self.rotation = rotation
        self.initial_command = (
            None if initial_command is None
            else np.asarray(initial_command, dtype=np.float32).copy()
        )
        self.shape = build_object_shape_descriptor(mesh_path, scale)


class DummyArgs:
    pass


def interpolated_policy_command(frames, open_command, physics_step,
                                steps_per_frame):
    """Match the standard replay's frame-to-physics-step interpolation."""
    frames = np.asarray(frames, dtype=np.float32)
    trajectory_steps = len(frames) * int(steps_per_frame)
    if physics_step >= trajectory_steps:
        return frames[-1].copy()
    frame_index = physics_step // int(steps_per_frame)
    substep = physics_step % int(steps_per_frame) + 1
    previous = open_command if frame_index == 0 else frames[frame_index - 1]
    alpha = substep / float(steps_per_frame)
    return previous * (1.0 - alpha) + frames[frame_index] * alpha


class GeneralResidualEnv:
    def __init__(self, cases, target_data, args):
        self.cases = cases
        self.n = len(cases)
        self.device = torch.device(args.device)
        self.gym, self.sim = create_cpu_sim(args.dt, args.substeps)
        self.hand = cases[0].hand
        self.hands, self.objects, self.envs = [], [], []
        self.tip_body_indices, self.hand_body_indices = [], []
        self.object_body_indices = []
        self.open_commands, self.policy_open_commands, self.mappers = [], [], []
        self.lower, self.upper = None, None
        for i, case in enumerate(cases):
            env = self.gym.create_env(self.sim, gymapi.Vec3(-1.0, -1.0, -0.2),
                                      gymapi.Vec3(1.0, 1.0, 1.0), 1)
            dummy = DummyArgs()
            dummy.hand = self.hand
            dummy.finger_stiffness = args.finger_stiffness
            dummy.finger_damping = args.finger_damping
            dummy.mimic_stiffness = args.mimic_stiffness
            dummy.mimic_damping = args.mimic_damping
            dummy.linker_finger_stiffness = args.finger_stiffness
            dummy.linker_finger_damping = args.finger_damping
            dummy.linker_mimic_stiffness = args.mimic_stiffness
            dummy.linker_mimic_damping = args.mimic_damping
            dummy.xhand_finger_stiffness = args.finger_stiffness
            dummy.xhand_finger_damping = args.finger_damping
            dummy.wuji_finger_stiffness = args.finger_stiffness
            dummy.wuji_finger_damping = args.finger_damping
            setup_frames = case.target_frames
            if case.initial_command is not None:
                setup_frames = np.asarray(case.target_frames, dtype=np.float32).copy()
                setup_frames[0] = case.initial_command
            asset, dof_properties, dof_names, policy_open, open_first, mapper, action_order = \
                prepare_hand(self.gym, self.sim, self.hand, target_data,
                             setup_frames, dummy)
            hand_actor = self.gym.create_actor(env, asset, gymapi.Transform(), "hand", 0, 1)
            self.gym.set_actor_dof_properties(env, hand_actor, dof_properties)
            object_asset = load_object_asset(self.gym, self.sim, case.mesh_path.parents[1])
            pose, _ = object_start_pose(case.mesh_path.parents[1], case.scale,
                                        case.rotation, args.clearance)
            object_actor = self.gym.create_actor(env, object_asset, pose, "object", 0, 0)
            self.gym.set_actor_scale(env, object_actor, case.scale)
            set_object_friction(self.gym, env, object_actor, args.object_friction)
            self.hands.append(hand_actor)
            self.objects.append(object_actor)
            self.envs.append(env)
            hand_indices = actor_body_indices(self.gym, env, hand_actor)
            self.hand_body_indices.append(set(hand_indices))
            hand_names = self.gym.get_actor_rigid_body_names(env, hand_actor)
            name_to_index = dict(zip(hand_names, hand_indices))
            self.tip_body_indices.append(tuple(
                name_to_index[name] for name in TIP_LINKS[self.hand]))
            self.object_body_indices.append(set(actor_body_indices(
                self.gym, env, object_actor)))
            self.open_commands.append(np.asarray(mapper(policy_open), dtype=np.float32))
            self.policy_open_commands.append(
                np.asarray(policy_open, dtype=np.float32))
            self.mappers.append(mapper)
            if self.lower is None:
                self.lower = np.asarray(dof_properties["lower"], dtype=np.float32)
                self.upper = np.asarray(dof_properties["upper"], dtype=np.float32)
        self.finger_count = cases[0].target_frames.shape[1] - 6
        self.lift_threshold = float(getattr(args, "lift_threshold", 0.15))
        self.action_dim = self.finger_count
        self.command_dim = cases[0].target_frames.shape[1]
        probe = self.gym.get_actor_dof_states(self.envs[0], self.hands[0], gymapi.STATE_ALL)
        self.dof_count = len(probe["pos"])
        self.obs_dim = self.command_dim + 2 * self.dof_count + 13 + 3 + self.action_dim + 14
        self.prev_lift = np.zeros(self.n, dtype=np.float32)
        self.reward_mode = getattr(args, "reward_mode", "legacy")
        self.previous_effective_residual = np.zeros(
            (self.n, self.action_dim), dtype=np.float32)
        self.dt = args.dt
        self.steps_per_frame = int(args.steps_per_frame)
        trajectory_lengths = {len(case.target_frames) for case in cases}
        if len(trajectory_lengths) != 1:
            raise ValueError("同一并行环境中的基准轨迹长度必须一致")
        self.horizon = args.steps_per_frame * trajectory_lengths.pop() + args.hold_steps
        self.settle_steps = args.settle_steps
        self._capture_rest_states()
        self.positions = [[] for _ in range(self.n)]
        self.contacts = [[] for _ in range(self.n)]
        self.hand_poses = [[] for _ in range(self.n)]
        self.object_rotations = [[] for _ in range(self.n)]
        self.tip_loaded = [[] for _ in range(self.n)]

    def _capture_rest_states(self):
        for i, env in enumerate(self.envs):
            set_dof_state_and_target(self.gym, env, self.hands[i], self.open_commands[i])
        for _ in range(self.settle_steps):
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
        self.object_rest_states = []
        initial_positions = []
        for i, env in enumerate(self.envs):
            state = self.gym.get_actor_rigid_body_states(env, self.objects[i], gymapi.STATE_ALL)
            self.object_rest_states.append(state.copy())
            initial_positions.append(
                read_object_state(self.gym, env, self.objects[i])["object_position"])
        self.initial_positions = np.stack(initial_positions).astype(np.float32)

    def reset(self):
        for i, env in enumerate(self.envs):
            self.gym.set_actor_rigid_body_states(
                env, self.objects[i], self.object_rest_states[i], gymapi.STATE_ALL)
            set_dof_state_and_target(self.gym, env, self.hands[i], self.open_commands[i])
        for _ in range(self.settle_steps):
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
        for i in range(self.n):
            self.positions[i] = []
            self.contacts[i] = []
            self.hand_poses[i] = []
            self.object_rotations[i] = []
            self.tip_loaded[i] = []
        self.prev_lift.fill(0.0)
        self.previous_effective_residual.fill(0.0)

    def observation(self, t, previous_residual):
        values = []
        for i in range(self.n):
            env = self.envs[i]
            dof = self.gym.get_actor_dof_states(env, self.hands[i], gymapi.STATE_ALL)
            obj = read_object_state(self.gym, env, self.objects[i])
            contacts = self.gym.get_env_rigid_contacts(env)
            contact = len(contacts)
            relative = obj["object_position"] - self.initial_positions[i]
            baseline = self.cases[i].target_frames
            base_command = interpolated_policy_command(
                baseline, self.policy_open_commands[i], t,
                self.steps_per_frame)
            values.append(np.concatenate([
                base_command,
                np.asarray(dof["pos"], dtype=np.float32),
                np.asarray(dof["vel"], dtype=np.float32),
                relative.astype(np.float32),
                obj["object_quaternion_xyzw"],
                obj["object_linear_velocity"],
                obj["object_angular_velocity"],
                np.asarray([math.log1p(contact), t / max(1, self.horizon - 1), relative[2]],
                           dtype=np.float32),
                previous_residual[i],
                self.cases[i].shape,
            ]).astype(np.float32))
        result = np.stack(values)
        if result.shape != (self.n, self.obs_dim):
            raise ValueError(f"观测维度错误: {result.shape} vs {(self.n, self.obs_dim)}")
        return torch.as_tensor(result, dtype=torch.float32, device=self.device)

    def step(self, t, residual):
        residual = np.asarray(residual, dtype=np.float32)
        targets = []
        for i, env in enumerate(self.envs):
            baseline = self.cases[i].target_frames
            command = interpolated_policy_command(
                baseline, self.policy_open_commands[i], t,
                self.steps_per_frame)
            command[6:] = command[6:] + residual[i]
            physical = np.asarray(self.mappers[i](command), dtype=np.float32)
            targets.append(np.clip(physical, self.lower, self.upper))
        before = self.prev_lift.copy()
        for env, hand, target in zip(self.envs, self.hands, targets):
            self.gym.set_actor_dof_position_targets(env, hand, target)
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        rewards = np.zeros(self.n, dtype=np.float32)
        for i in range(self.n):
            obj = read_object_state(self.gym, self.envs[i], self.objects[i])
            position = obj["object_position"].astype(np.float64)
            lift = float(position[2] - self.initial_positions[i, 2])
            self.positions[i].append(position)
            all_contacts = self.gym.get_env_rigid_contacts(self.envs[i])
            hand_bodies = self.hand_body_indices[i]
            object_bodies = self.object_body_indices[i]
            contacts = [row for row in all_contacts if (
                (int(row["body0"]) in hand_bodies
                 and int(row["body1"]) in object_bodies)
                or (int(row["body1"]) in hand_bodies
                    and int(row["body0"]) in object_bodies)
            )]
            contact = len(contacts)
            self.contacts[i].append(contact)
            loads = np.zeros(5, dtype=np.float32)
            tip_lookup = {
                int(body): finger for finger, body in enumerate(self.tip_body_indices[i])}
            for row in contacts:
                body0, body1 = int(row["body0"]), int(row["body1"])
                if body0 in tip_lookup and body1 in object_bodies:
                    loads[tip_lookup[body0]] += max(0.0, float(row["lambda"]))
                elif body1 in tip_lookup and body0 in object_bodies:
                    loads[tip_lookup[body1]] += max(0.0, float(row["lambda"]))
            self.tip_loaded[i].append(loads >= 0.02)
            dof = self.gym.get_actor_dof_states(
                self.envs[i], self.hands[i], gymapi.STATE_ALL)
            self.hand_poses[i].append(
                np.asarray(dof["pos"][:6], dtype=np.float64))
            self.object_rotations[i].append(
                np.asarray(obj["object_quaternion_xyzw"], dtype=np.float64))
            delta = lift - float(before[i])
            if self.reward_mode == "pca_contact_transport":
                opposition = bool(loads[0] >= 0.02 and np.any(loads[1:] >= 0.02))
                phase = t / float(max(self.horizon - 1, 1))
                smoothness = np.square(
                    residual[i] - self.previous_effective_residual[i]
                ).mean()
                rewards[i] = 200.0 * max(delta, -0.003)
                rewards[i] += 0.015 * float(np.count_nonzero(loads >= 0.02))
                rewards[i] += 0.10 * float(opposition and phase >= 0.35)
                rewards[i] -= 0.003 * float(np.square(residual[i]).mean())
                rewards[i] -= 0.01 * float(smoothness)
            else:
                rewards[i] = 200.0 * max(delta, -0.002) + 0.02 * float(contact > 0)
                rewards[i] -= 0.002 * float(np.square(residual[i]).mean())
            self.prev_lift[i] = lift
        self.previous_effective_residual = residual.copy()
        return torch.as_tensor(rewards, dtype=torch.float32, device=self.device)

    def metrics(self):
        values = []
        for i in range(self.n):
            metric = compute_success_metrics(
                np.asarray(self.positions[i]), self.initial_positions[i],
                np.asarray(self.contacts[i]), self.dt,
                self.lift_threshold, 0.25, 30,
                terminal_hold_steps=30,
            )
            positions = np.asarray(self.positions[i], dtype=np.float64)
            hand_pose = np.asarray(self.hand_poses[i], dtype=np.float64)
            contacts = np.asarray(self.contacts[i]) > 0
            lifted = positions[:, 2] - self.initial_positions[i, 2]
            starts = np.flatnonzero((lifted >= 0.05) & contacts)
            loaded = np.asarray(self.tip_loaded[i], dtype=bool)
            terminal_loaded = loaded[-30:]
            metric.update({
                "terminal_loaded_finger_mean": float(
                    terminal_loaded.sum(axis=1).mean()),
                "terminal_thumb_opposition_ratio": float(np.mean(
                    terminal_loaded[:, 0] & terminal_loaded[:, 1:].any(axis=1))),
            })
            if len(starts):
                start = int(starts[0])
                metric["lift_thumb_opposition_ratio"] = float(np.mean(
                    loaded[start:, 0] & loaded[start:, 1:].any(axis=1)))
                hand_rotation = Rotation.from_euler("xyz", hand_pose[:, 3:6])
                local_position = hand_rotation.inv().apply(
                    positions - hand_pose[:, :3])
                terminal_position = np.median(local_position[-30:], axis=0)
                translation = float(np.linalg.norm(
                    local_position[start:] - terminal_position, axis=1).max())
                object_rotation = Rotation.from_quat(
                    np.asarray(self.object_rotations[i], dtype=np.float64))
                local_rotation = hand_rotation.inv() * object_rotation
                terminal_rotation = local_rotation[-1]
                rotation = float(np.degrees(
                    (terminal_rotation.inv() * local_rotation[start:]).magnitude()
                ).max())
                metric.update({
                    "max_palm_relative_translation_change_m": translation,
                    "max_palm_relative_rotation_change_deg": rotation,
                    "transport_stability_success": bool(
                        translation <= 0.03 and rotation <= 30.0),
                })
            else:
                metric["lift_thumb_opposition_ratio"] = 0.0
                metric.update({
                    "max_palm_relative_translation_change_m": None,
                    "max_palm_relative_rotation_change_deg": None,
                    "transport_stability_success": False,
                })
            values.append(metric)
        return values

    def close(self):
        self.gym.destroy_sim(self.sim)


def rollout(env, model=None, storage=None, deterministic=False):
    env.reset()
    previous = np.zeros((env.n, env.action_dim), dtype=np.float32)
    for t in range(env.horizon):
        obs = env.observation(t, previous)
        if model is None:
            action = torch.zeros((env.n, env.action_dim), device=env.device)
            log_prob = torch.zeros(env.n, device=env.device)
            value = torch.zeros(env.n, device=env.device)
        else:
            action, log_prob, value = model.act(obs, obs, deterministic=deterministic)
        action_np = action.detach().cpu().numpy()
        reward = env.step(t, action_np)
        done = torch.full((env.n,), t == env.horizon - 1, dtype=torch.bool, device=env.device)
        if storage is not None:
            storage.add(obs, obs, action, log_prob, value, reward, done)
        previous = action_np
    metrics = env.metrics()
    if storage is not None:
        bonus = torch.as_tensor(
            [10.0 if item["success"] else 0.0 for item in metrics],
            dtype=torch.float32, device=env.device,
        )
        storage.rewards[-1] = storage.rewards[-1] + bonus
    return metrics


def summary(metrics):
    return {
        "success_count": int(sum(item["success"] for item in metrics)),
        "trajectory_count": len(metrics),
        "success_rate": float(np.mean([item["success"] for item in metrics])),
        "mean_max_lift_m": float(np.mean([item["max_lift_m"] for item in metrics])),
        "mean_final_lift_m": float(np.mean([item["final_lift_m"] for item in metrics])),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hand", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--policy-split", type=Path, required=True)
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--num-envs", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260819)
    args = parser.parse_args()
    args.device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    split = json.loads(args.policy_split.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    by_object = {entry["object_name"]: entry for entry in manifest["entries"]}
    cases = []
    target_data_by_object = {}
    for record in split["records"]:
        if record["split"] != "train" or record["object_name"] in target_data_by_object:
            continue
        entry = by_object.get(record["object_name"])
        if entry is None:
            continue
        target = np.load(args.target_dir / f"{record['object_name']}.npy",
                         allow_pickle=True).item()
        target_indices = np.asarray(target["source_trajectory_indices"], dtype=np.int64)
        source_index = int(record["source_trajectory_index"])
        if source_index not in target_indices:
            continue
        row = int(np.flatnonzero(target_indices == source_index)[0])
        source = np.load(Path(entry["source_path"]), allow_pickle=True).item()
        scale = float(np.asarray(source["obj_scale"])[source_index])
        cases.append(Case(
            args.hand, record["category"], record["object_name"], source_index,
            np.asarray(target["grasp_seqs"][row], dtype=np.float32),
            Path(entry["object_asset_path"]) / "coacd" / "decomposed.obj", scale,
            np.asarray(source["obj_rotmat"][source_index], dtype=np.float32),
        ))
        target_data_by_object[record["object_name"]] = target
        if len(cases) >= args.num_envs:
            break
    env_args = DummyArgs()
    env_args.device = args.device
    env_args.dt = 1.0 / 60.0
    env_args.substeps = 2
    env_args.physx_threads = 1
    env_args.finger_stiffness = 120.0
    env_args.finger_damping = 5.0
    env_args.mimic_stiffness = 120.0
    env_args.mimic_damping = 5.0
    env_args.linker_finger_stiffness = 120.0
    env_args.linker_finger_damping = 5.0
    env_args.linker_mimic_stiffness = 120.0
    env_args.linker_mimic_damping = 5.0
    env_args.xhand_finger_stiffness = 120.0
    env_args.xhand_finger_damping = 5.0
    env_args.wuji_finger_stiffness = 120.0
    env_args.wuji_finger_damping = 5.0
    env_args.clearance = 0.005
    env_args.object_friction = 1.0
    env_args.settle_steps = 30
    env_args.steps_per_frame = 3
    env_args.hold_steps = 30
    env = GeneralResidualEnv(cases, target_data_by_object[cases[0].object_name], env_args)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    try:
        baseline_metrics = rollout(env)
        print("baseline", json.dumps(summary(baseline_metrics), ensure_ascii=False))
        model = ResidualActorCritic(
            env.obs_dim, env.obs_dim, action_dim=env.action_dim,
            hidden_dims=(128, 128), init_std=0.05,
        ).to(env.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        ppo_config = PPOConfig(learning_rate=3e-4, minibatches=4, update_epochs=4, target_kl=0.5)
        log = []
        for iteration in range(1, args.iterations + 1):
            storage = RolloutStorage()
            metrics = rollout(env, model, storage, deterministic=False)
            batch = storage.finish(torch.zeros(env.n, device=env.device), ppo_config)
            update = ppo_update(model, optimizer, batch, ppo_config)
            item = {"iteration": iteration, **summary(metrics), **update}
            log.append(item)
            print(json.dumps(item, ensure_ascii=False))
            if iteration == 1 or iteration % 50 == 0 or iteration == args.iterations:
                torch.save({
                    "model": model.state_dict(),
                    "config": vars(args),
                    "iteration": iteration,
                    "obs_dim": env.obs_dim,
                    "action_dim": env.action_dim,
                    "command_dim": env.command_dim,
                    "dof_count": env.dof_count,
                }, args.checkpoint_dir / f"residual_ppo_{iteration:04d}.pt")
                (args.checkpoint_dir / "training_log.json").write_text(
                    json.dumps(log, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    finally:
        env.close()


if __name__ == "__main__":
    main()
