#!/usr/bin/env python3
"""在冻结的自主InitialPhase策略上训练Residual PPO。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from isaacgym import gymapi
import numpy as np
import torch

for path in (
    Path(__file__).resolve().parents[1] / "retargeting" / "evaluate",
    Path(__file__).resolve().parents[1] / "retargeting" / "prepare",
):
    sys.path.insert(0, str(path))

from evaluate_policy_isaac import prepare_hand, set_object_friction  # noqa: E402
from isaac_replay_common import (  # noqa: E402
    actor_body_indices,
    compute_success_metrics,
    count_contacts_by_hand_body,
    create_cpu_sim,
    load_object_asset,
    object_start_pose,
    read_object_state,
    read_policy_pre_action_state,
    set_dof_state_and_target,
)
from observations import build_object_shape_descriptor, build_runtime_observation  # noqa: E402
from residual_ppo import PPOConfig, ResidualActorCritic, RolloutStorage, ppo_update  # noqa: E402
from runtime import PolicyRunner  # noqa: E402


class Case:
    def __init__(self, hand, category, object_name, source_index, target_data,
                 target_frames, mesh_path, scale, rotation):
        self.hand = hand
        self.category = category
        self.object_name = object_name
        self.source_index = source_index
        self.target_data = target_data
        self.target_frames = target_frames
        self.mesh_path = mesh_path
        self.scale = scale
        self.rotation = rotation
        self.shape = build_object_shape_descriptor(mesh_path, scale)


class Args:
    pass


class AutonomousResidualEnv:
    """并行Isaac环境；专家轨迹只提供首帧手腕初态。"""

    def __init__(self, cases, base_checkpoint, data_dir, args):
        self.cases = cases
        self.n = len(cases)
        self.device = torch.device(args.device)
        self.gym, self.sim = create_cpu_sim(args.dt, args.substeps)
        self.dt = args.dt
        self.horizon = args.policy_steps
        self.settle_steps = args.settle_steps
        self.residual_scale = float(args.residual_scale)
        self.hands, self.objects, self.envs = [], [], []
        self.mappers, self.open_commands = [], []
        self.hand_body_maps, self.object_indices = [], []
        self.lower = self.upper = None

        hand_args = Args()
        for name, value in {
            "finger_stiffness": 120.0, "finger_damping": 5.0,
            "mimic_stiffness": 120.0, "mimic_damping": 5.0,
            "linker_finger_stiffness": 120.0, "linker_finger_damping": 5.0,
            "linker_mimic_stiffness": 120.0, "linker_mimic_damping": 5.0,
            "xhand_finger_stiffness": 120.0, "xhand_finger_damping": 5.0,
            "wuji_finger_stiffness": 120.0, "wuji_finger_damping": 5.0,
        }.items():
            setattr(hand_args, name, value)
        hand_args.hand = cases[0].hand

        for index, case in enumerate(cases):
            env = self.gym.create_env(
                self.sim, gymapi.Vec3(-1.0, -1.0, -0.2),
                gymapi.Vec3(1.0, 1.0, 1.0), 1,
            )
            asset, properties, _, _, open_command, mapper, _ = prepare_hand(
                self.gym, self.sim, case.hand, case.target_data,
                case.target_frames, hand_args,
            )
            hand_actor = self.gym.create_actor(
                env, asset, gymapi.Transform(), f"hand_{index}", 0, 1)
            self.gym.set_actor_dof_properties(env, hand_actor, properties)
            object_asset = load_object_asset(
                self.gym, self.sim, case.mesh_path.parents[1])
            pose, _ = object_start_pose(
                case.mesh_path.parents[1], case.scale, case.rotation,
                args.clearance,
            )
            object_actor = self.gym.create_actor(
                env, object_asset, pose, f"object_{index}", 0, 0)
            self.gym.set_actor_scale(env, object_actor, case.scale)
            set_object_friction(
                self.gym, env, object_actor, args.object_friction)
            self.envs.append(env)
            self.hands.append(hand_actor)
            self.objects.append(object_actor)
            self.mappers.append(mapper)
            self.open_commands.append(np.asarray(open_command, dtype=np.float32))
            hand_indices = actor_body_indices(self.gym, env, hand_actor)
            hand_names = self.gym.get_actor_rigid_body_names(env, hand_actor)
            self.hand_body_maps.append(dict(zip(hand_indices, hand_names)))
            self.object_indices.append(actor_body_indices(
                self.gym, env, object_actor))
            if self.lower is None:
                self.lower = np.asarray(properties["lower"], dtype=np.float32)
                self.upper = np.asarray(properties["upper"], dtype=np.float32)

        self.base = PolicyRunner(base_checkpoint, data_dir, args.device)
        if self.base.model_type not in {
            "initial_phase_delta", "initial_temporal_feedback"
        }:
            raise ValueError("自主Residual PPO只接受纯参数初态策略")
        for parameter in self.base.model.parameters():
            parameter.requires_grad_(False)
        self.base.model.eval()
        self.command_dim = self.base.dimensions["action_dim"]
        self.action_dim = self.command_dim
        self.category_ids = torch.as_tensor([
            self.base.mappings["category_to_id"][case.category]
            for case in cases
        ], dtype=torch.long, device=self.device)
        self.open_policy = np.stack([
            np.asarray(case.target_frames[0], dtype=np.float32)
            for case in cases
        ])
        self.open_policy[:, 6:] = 0.0
        self._capture_rest_states()
        self.actor_obs_dim = (
            self.base.dimensions["observation_dim"] + 2 * self.action_dim + 1)
        self.positions = [[] for _ in cases]
        self.contacts = [[] for _ in cases]
        self.prev_lift = np.zeros(self.n, dtype=np.float32)

    def _capture_rest_states(self):
        for env, hand, command in zip(
                self.envs, self.hands, self.open_commands):
            set_dof_state_and_target(self.gym, env, hand, command)
        for _ in range(self.settle_steps):
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
        self.object_rest_states = []
        initial = []
        for env, actor in zip(self.envs, self.objects):
            state = self.gym.get_actor_rigid_body_states(
                env, actor, gymapi.STATE_ALL)
            self.object_rest_states.append(state.copy())
            initial.append(read_object_state(
                self.gym, env, actor)["object_position"])
        self.initial_positions = np.stack(initial).astype(np.float32)

    def reset(self):
        for index, env in enumerate(self.envs):
            self.gym.set_actor_rigid_body_states(
                env, self.objects[index], self.object_rest_states[index],
                gymapi.STATE_ALL,
            )
            set_dof_state_and_target(
                self.gym, env, self.hands[index], self.open_commands[index])
        for _ in range(self.settle_steps):
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
        observations = self._raw_observations()
        self.initial_normalized = self._normalize_observations(observations)
        self.normalized_history = np.repeat(
            self.initial_normalized[:, None, :], self.base.history, axis=1)
        self.prev_residual = np.zeros(
            (self.n, self.action_dim), dtype=np.float32)
        self.prev_lift.fill(0.0)
        for index in range(self.n):
            self.positions[index] = []
            self.contacts[index] = []

    def _contact_count(self, index):
        contacts = self.gym.get_env_rigid_contacts(self.envs[index])
        grouped = count_contacts_by_hand_body(
            contacts, self.hand_body_maps[index], self.object_indices[index])
        return int(sum(grouped.values()))

    def _raw_observations(self):
        values = []
        for index, case in enumerate(self.cases):
            dofs, obj, _ = read_policy_pre_action_state(
                self.gym, self.envs[index], self.hands[index],
                self.objects[index], self.hand_body_maps[index],
                self.object_indices[index],
            )
            values.append(build_runtime_observation(
                dofs, obj, self.initial_positions[index],
                self._contact_count(index), case.shape, 0.30,
            ))
        return np.stack(values).astype(np.float32)

    def _normalize_observations(self, observations):
        return (
            observations - self.base.normalization["observation_mean"]
        ) / self.base.normalization["observation_std"]

    def policy_state(self, step):
        observations = self._raw_observations()
        normalized = self._normalize_observations(observations)
        phase_value = min(step / float(self.base.motion_steps - 1), 1.0)
        phase = torch.full(
            (self.n, 1), phase_value, dtype=torch.float32,
            device=self.device,
        )
        if self.base.model_type == "initial_temporal_feedback":
            self.normalized_history = np.concatenate([
                self.normalized_history[:, 1:], normalized[:, None, :]
            ], axis=1)
        with torch.no_grad():
            initial_tensor = torch.as_tensor(
                self.initial_normalized, device=self.device)
            if self.base.model_type == "initial_temporal_feedback":
                base_delta = self.base.model(
                    initial_tensor,
                    torch.as_tensor(self.normalized_history, device=self.device),
                    phase, self.category_ids,
                )
            else:
                base_delta = self.base.model(
                    initial_tensor, phase, self.category_ids)
        actor_obs = torch.cat([
            torch.as_tensor(normalized, device=self.device),
            base_delta,
            phase,
            torch.as_tensor(self.prev_residual, device=self.device),
        ], dim=1)
        return actor_obs, base_delta.detach().cpu().numpy()

    def step(self, base_delta, residual):
        residual = np.asarray(residual, dtype=np.float32)
        normalized_delta = np.clip(
            base_delta + self.residual_scale * residual, -5.0, 5.0)
        delta = (
            normalized_delta * self.base.normalization["initial_delta_std"]
            + self.base.normalization["initial_delta_mean"]
        )
        commands = self.open_policy + delta
        for index, command in enumerate(commands):
            physical = np.asarray(self.mappers[index](command), dtype=np.float32)
            physical = np.clip(physical, self.lower, self.upper)
            self.gym.set_actor_dof_position_targets(
                self.envs[index], self.hands[index], physical)
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        rewards = np.zeros(self.n, dtype=np.float32)
        for index in range(self.n):
            position = read_object_state(
                self.gym, self.envs[index], self.objects[index]
            )["object_position"].astype(np.float64)
            lift = float(position[2] - self.initial_positions[index, 2])
            contact = self._contact_count(index)
            self.positions[index].append(position)
            self.contacts[index].append(contact)
            lift_delta = np.clip(lift - self.prev_lift[index], -0.005, 0.005)
            xy_drift = np.linalg.norm(
                position[:2] - self.initial_positions[index, :2])
            rewards[index] = 150.0 * lift_delta
            rewards[index] += 0.01 * float(contact > 0)
            rewards[index] -= 0.002 * float(xy_drift)
            rewards[index] -= 0.001 * float(np.square(residual[index]).mean())
            self.prev_lift[index] = lift
        self.prev_residual = residual.copy()
        return torch.as_tensor(rewards, device=self.device)

    def metrics(self):
        required_steps = min(30, self.horizon)
        return [compute_success_metrics(
            np.asarray(self.positions[index]), self.initial_positions[index],
            np.asarray(self.contacts[index]), self.dt, 0.30, 0.25,
            required_steps, terminal_hold_steps=required_steps,
        ) for index in range(self.n)]

    def close(self):
        self.gym.destroy_sim(self.sim)


def rollout(env, model, storage=None, deterministic=False):
    env.reset()
    for step in range(env.horizon):
        actor_obs, base_delta = env.policy_state(step)
        if model is None:
            action = torch.zeros(
                (env.n, env.action_dim), device=env.device)
            log_prob = value = torch.zeros(env.n, device=env.device)
        else:
            action, log_prob, value = model.act(
                actor_obs, actor_obs, deterministic=deterministic)
        reward = env.step(base_delta, action.detach().cpu().numpy())
        done = torch.full(
            (env.n,), step == env.horizon - 1,
            dtype=torch.bool, device=env.device,
        )
        if storage is not None:
            storage.add(
                actor_obs, actor_obs, action, log_prob, value, reward, done)
    metrics = env.metrics()
    if storage is not None:
        storage.rewards[-1] += torch.as_tensor([
            10.0 if item["success"] else 0.0 for item in metrics
        ], dtype=torch.float32, device=env.device)
    return metrics


def summarize(metrics):
    return {
        "success_count": int(sum(item["success"] for item in metrics)),
        "trajectory_count": len(metrics),
        "success_rate": float(np.mean([item["success"] for item in metrics])),
        "mean_max_lift_m": float(np.mean([item["max_lift_m"] for item in metrics])),
        "mean_final_lift_m": float(np.mean([item["final_lift_m"] for item in metrics])),
    }


def load_cases(args):
    split = json.loads(args.policy_split.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    entries = {item["object_name"]: item for item in manifest["entries"]}
    cases, used_categories = [], set()
    for record in split["records"]:
        if record["split"] != "train" or record["category"] in used_categories:
            continue
        entry = entries[record["object_name"]]
        target = np.load(
            args.target_dir / f"{record['object_name']}.npy",
            allow_pickle=True,
        ).item()
        indices = np.asarray(target["source_trajectory_indices"], dtype=np.int64)
        matches = np.flatnonzero(
            indices == int(record["source_trajectory_index"]))
        if len(matches) != 1:
            continue
        source = np.load(Path(entry["source_path"]), allow_pickle=True).item()
        source_index = int(record["source_trajectory_index"])
        cases.append(Case(
            args.hand, record["category"], record["object_name"], source_index,
            target, np.asarray(target["grasp_seqs"][matches[0]], dtype=np.float32),
            Path(entry["object_asset_path"]) / "coacd" / "decomposed.obj",
            float(np.asarray(source["obj_scale"])[source_index]),
            np.asarray(source["obj_rotmat"][source_index], dtype=np.float32),
        ))
        used_categories.add(record["category"])
        if len(cases) >= args.num_envs:
            break
    if len(cases) != args.num_envs:
        raise ValueError(f"只找到{len(cases)}个不同训练类别，要求{args.num_envs}个")
    return cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hand", choices=["linker", "xhand", "wuji"], required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--policy-split", type=Path, required=True)
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--num-envs", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--residual-scale", type=float, default=0.10)
    parser.add_argument("--policy-steps", type=int, default=240)
    parser.add_argument("--dt", type=float, default=1.0 / 60.0)
    parser.add_argument("--substeps", type=int, default=2)
    parser.add_argument("--settle-steps", type=int, default=30)
    parser.add_argument("--clearance", type=float, default=0.005)
    parser.add_argument("--object-friction", type=float, default=1.0)
    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cases = load_cases(args)
    env = AutonomousResidualEnv(
        cases, args.base_checkpoint, args.data_dir, args)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    try:
        baseline = summarize(rollout(env, None, deterministic=True))
        print("baseline", json.dumps(baseline, ensure_ascii=False), flush=True)
        model = ResidualActorCritic(
            env.actor_obs_dim, env.actor_obs_dim, env.action_dim,
            hidden_dims=(256, 256), init_std=0.10,
        ).to(env.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        config = PPOConfig(
            learning_rate=1e-4, minibatches=4, update_epochs=4,
            target_kl=0.03, entropy_coef=0.001,
        )
        log = []
        for iteration in range(1, args.iterations + 1):
            storage = RolloutStorage()
            metrics = rollout(env, model, storage, deterministic=False)
            batch = storage.finish(
                torch.zeros(env.n, device=env.device), config)
            update = ppo_update(model, optimizer, batch, config)
            deterministic_metrics = (
                rollout(env, model, deterministic=True)
                if iteration == 1 or iteration % 10 == 0
                else metrics
            )
            item = {
                "iteration": iteration,
                **summarize(deterministic_metrics), **update,
            }
            log.append(item)
            print(json.dumps(item, ensure_ascii=False), flush=True)
            if iteration == 1 or iteration % 10 == 0 or iteration == args.iterations:
                torch.save({
                    "model": model.state_dict(),
                    "schema": "autonomous_initial_phase_residual_ppo_v1",
                    "base_checkpoint": str(args.base_checkpoint.resolve()),
                    "data_dir": str(args.data_dir.resolve()),
                    "residual_scale": args.residual_scale,
                    "iteration": iteration,
                    "obs_dim": env.actor_obs_dim,
                    "action_dim": env.action_dim,
                    "config": vars(args),
                }, args.checkpoint_dir / f"autonomous_residual_ppo_{iteration:04d}.pt")
                (args.checkpoint_dir / "training_log.json").write_text(
                    json.dumps({
                        "baseline": baseline,
                        "categories": [case.category for case in cases],
                        "iterations": log,
                    }, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
    finally:
        env.close()


if __name__ == "__main__":
    main()
