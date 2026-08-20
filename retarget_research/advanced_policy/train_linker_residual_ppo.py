#!/usr/bin/env python3
"""用Isaac Gym物理反馈训练Linker轨迹的12维残差策略。"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from isaacgym import gymapi
import torch

ROOT = Path(__file__).resolve().parents[2]
EVALUATE = ROOT / "retarget_research" / "retargeting" / "evaluate"
sys.path.insert(0, str(EVALUATE))

from isaac_replay_common import (  # noqa: E402
    actor_body_indices,
    compute_success_metrics,
    count_actor_pair_contacts,
    load_object_asset,
    object_start_pose,
    read_object_state,
    set_dof_state_and_target,
)
from linker_replay_utils import expand_linker_frame  # noqa: E402
from replay_linker_isaac import load_linker_asset  # noqa: E402

sys.path.insert(0, str(ROOT.parent))
from residual_ppo import (  # noqa: E402
    PPOConfig,
    ResidualActorCritic,
    RolloutStorage,
    ppo_update,
)

OBS_DIM = 71
ACTION_DIM = 12
RESIDUAL_SCALE = np.asarray(
    [0.004, 0.004, 0.004, 0.04, 0.04, 0.04,
     0.10, 0.10, 0.10, 0.10, 0.10, 0.10],
    dtype=np.float32,
)


def create_residual_sim(dt, substeps, physx_threads):
    """创建残差训练专用CPU PhysX，并固定线程数降低评测抖动。"""
    gym = gymapi.acquire_gym()
    params = gymapi.SimParams()
    params.dt = float(dt)
    params.substeps = int(substeps)
    params.up_axis = gymapi.UP_AXIS_Z
    params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    params.use_gpu_pipeline = False
    params.physx.use_gpu = False
    params.physx.solver_type = 1
    params.physx.num_position_iterations = 8
    params.physx.num_velocity_iterations = 1
    params.physx.num_threads = int(physx_threads)
    params.physx.contact_offset = 0.002
    params.physx.rest_offset = 0.0
    sim = gym.create_sim(0, -1, gymapi.SIM_PHYSX, params)
    if sim is None:
        raise RuntimeError("残差训练CPU PhysX创建失败")
    plane = gymapi.PlaneParams()
    plane.normal = gymapi.Vec3(0.0, 0.0, 1.0)
    gym.add_ground(sim, plane)
    return gym, sim


@dataclass
class Case:
    name: str
    source_index: int
    object_dir: Path
    policy_frames: np.ndarray
    shape: np.ndarray


def shape_descriptor(mesh_path, scale):
    """读取已有14维物体形状描述；训练和部署使用同一实例信息。"""
    from observations import build_object_shape_descriptor

    return build_object_shape_descriptor(mesh_path, scale)


def load_cases(manifest_path, baseline_dir, object_root, limit=None):
    """从manifest和冻结候选目录组成一条环境对应一条轨迹的训练清单。"""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    cases = []
    for entry in manifest["entries"]:
        target_path = Path(baseline_dir) / f"{entry['object_name']}.npy"
        target = np.load(target_path, allow_pickle=True).item()
        source_indices = np.asarray(target["source_trajectory_indices"], dtype=int)
        for source_index in entry["trajectory_indices"]:
            matches = np.flatnonzero(source_indices == int(source_index))
            if len(matches) != 1:
                raise ValueError(f"候选中找不到唯一源轨迹: {entry['object_name']} {source_index}")
            object_dir = Path(entry.get("object_asset_path", Path(object_root) / entry["object_name"]))
            if not object_dir.is_dir():
                raise FileNotFoundError(f"物体资产不存在: {object_dir}")
            scale = float(np.asarray(target["obj_scale"])[matches[0]])
            frames = np.asarray(target["grasp_seqs"][matches[0]], dtype=np.float32)
            if frames.shape != (70, 12):
                raise ValueError(f"Linker残差基线必须是(70,12): {target_path} {frames.shape}")
            cases.append(Case(
                entry["object_name"], int(source_index), object_dir, frames,
                shape_descriptor(object_dir / "coacd" / "decomposed.obj", scale),
            ))
    if limit is not None:
        cases = cases[: int(limit)]
    if not cases:
        raise ValueError("manifest没有可训练轨迹")
    return cases


class LinkerResidualEnv:
    """把多条Linker候选同时放入CPU PhysX，并暴露批量观测和奖励。"""

    def __init__(self, cases, args):
        self.cases = list(cases)
        self.n = len(self.cases)
        self.dt = float(args.dt)
        self.steps_per_frame = int(args.steps_per_frame)
        self.hold_steps = int(args.hold_steps)
        self.horizon = 70 * self.steps_per_frame + self.hold_steps
        self.device = torch.device(args.device)
        self.gym, self.sim = create_residual_sim(self.dt, args.substeps, args.physx_threads)
        self.asset, properties, self.dof_names = load_linker_asset(
            self.gym, self.sim, args.finger_stiffness, args.finger_damping,
            args.mimic_stiffness, args.mimic_damping,
        )
        self.lower = np.asarray(properties["lower"], dtype=np.float32)
        self.upper = np.asarray(properties["upper"], dtype=np.float32)
        self.lower[~np.isfinite(self.lower)] = -10.0
        self.upper[~np.isfinite(self.upper)] = 10.0
        self.envs, self.hands, self.objects = [], [], []
        self.hand_bodies, self.object_bodies = [], []
        side = max(1, math.ceil(math.sqrt(self.n)))
        lower = gymapi.Vec3(-1.0, -1.0, -0.2)
        upper = gymapi.Vec3(1.0, 1.0, 1.0)
        for case in self.cases:
            env = self.gym.create_env(self.sim, lower, upper, side)
            hand = self.gym.create_actor(env, self.asset, gymapi.Transform(), "linker", 0, 1)
            self.gym.set_actor_dof_properties(env, hand, properties)
            target = np.load(
                Path(args.baseline_dir) / f"{case.name}.npy", allow_pickle=True
            ).item()
            source_indices = np.asarray(target["source_trajectory_indices"], dtype=int)
            row = int(np.flatnonzero(source_indices == case.source_index)[0])
            scale = float(np.asarray(target["obj_scale"])[row])
            rotation = np.asarray(target["obj_rotmat"])[row]
            pose, _ = object_start_pose(case.object_dir, scale, rotation, args.clearance)
            object_asset = load_object_asset(self.gym, self.sim, case.object_dir)
            obj = self.gym.create_actor(env, object_asset, pose, "object", 0, 0)
            self.gym.set_actor_scale(env, obj, scale)
            shape_properties = self.gym.get_actor_rigid_shape_properties(env, obj)
            for prop in shape_properties:
                prop.friction = float(args.object_friction)
            self.gym.set_actor_rigid_shape_properties(env, obj, shape_properties)
            self.envs.append(env)
            self.hands.append(hand)
            self.objects.append(obj)
            self.hand_bodies.append(actor_body_indices(self.gym, env, hand))
            self.object_bodies.append(actor_body_indices(self.gym, env, obj))
        self.baselines = np.stack([self._build_policy_schedule(c.policy_frames) for c in self.cases])
        self.physical_baselines = np.stack([
            np.stack([expand_linker_frame(frame, self.dof_names) for frame in schedule])
            for schedule in self.baselines
        ])
        self.open_commands = np.stack([
            self._policy_open(c.policy_frames) for c in self.cases
        ])
        self.open_physical = np.stack([
            expand_linker_frame(frame, self.dof_names) for frame in self.open_commands
        ])
        self.object_rest_states = []
        self.initial_positions = np.zeros((self.n, 3), dtype=np.float64)
        self._settle_steps = int(args.settle_steps)
        self._reset_initial_states()
        self.positions = [[] for _ in range(self.n)]
        self.contacts = [[] for _ in range(self.n)]
        self.prev_lift = np.zeros(self.n, dtype=np.float32)

    @staticmethod
    def _policy_open(frames):
        value = np.asarray(frames[0], dtype=np.float32).copy()
        value[6:] = 0.0
        return value

    def _build_policy_schedule(self, frames):
        previous = self._policy_open(frames)
        values = []
        for frame in frames:
            for substep in range(1, self.steps_per_frame + 1):
                alpha = substep / self.steps_per_frame
                values.append(previous * (1.0 - alpha) + frame * alpha)
            previous = frame
        values.extend([np.asarray(frames[-1], dtype=np.float32)] * self.hold_steps)
        return np.stack(values).astype(np.float32)

    def _reset_initial_states(self):
        for i, env in enumerate(self.envs):
            set_dof_state_and_target(self.gym, env, self.hands[i], self.open_physical[i])
        for _ in range(self._settle_steps):
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
        for i, env in enumerate(self.envs):
            state = self.gym.get_actor_rigid_body_states(env, self.objects[i], gymapi.STATE_ALL)
            self.object_rest_states.append(state.copy())
            self.initial_positions[i] = read_object_state(
                self.gym, env, self.objects[i]
            )["object_position"]

    def reset(self):
        """恢复所有物体和手到相同落稳初态，清空本回合曲线。"""
        for i, env in enumerate(self.envs):
            self.gym.set_actor_rigid_body_states(
                env, self.objects[i], self.object_rest_states[i], gymapi.STATE_ALL
            )
            set_dof_state_and_target(self.gym, env, self.hands[i], self.open_physical[i])
        for _ in range(self._settle_steps):
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
        self.positions = [[] for _ in range(self.n)]
        self.contacts = [[] for _ in range(self.n)]
        self.prev_lift.fill(0.0)

    def _state(self, i):
        env = self.envs[i]
        dof = self.gym.get_actor_dof_states(env, self.hands[i], gymapi.STATE_ALL)
        obj = read_object_state(self.gym, env, self.objects[i])
        contact = count_actor_pair_contacts(
            self.gym.get_env_rigid_contacts(env), self.hand_bodies[i], self.object_bodies[i]
        )
        return dof, obj, int(contact)

    def observation(self, t, previous_residual):
        values = []
        for i in range(self.n):
            dof, obj, contact = self._state(i)
            relative = obj["object_position"] - self.initial_positions[i]
            values.append(np.concatenate([
                self.baselines[i, t],
                np.asarray(dof["pos"], dtype=np.float32),
                relative.astype(np.float32),
                obj["object_quaternion_xyzw"],
                obj["object_linear_velocity"],
                obj["object_angular_velocity"],
                np.asarray([math.log1p(contact), t / max(1, self.horizon - 1), relative[2]], dtype=np.float32),
                previous_residual[i],
                self.cases[i].shape,
            ]).astype(np.float32))
        result = np.stack(values)
        if result.shape != (self.n, OBS_DIM):
            raise ValueError(f"残差观测维度错误: {result.shape}")
        return torch.as_tensor(result, dtype=torch.float32, device=self.device)

    def step(self, t, residual):
        """施加基线加残差的一步物理命令并返回奖励所需状态。"""
        residual = np.asarray(residual, dtype=np.float32)
        targets = []
        for i, env in enumerate(self.envs):
            command = self.baselines[i, t] + residual[i] * RESIDUAL_SCALE
            physical = np.asarray(expand_linker_frame(command, self.dof_names))
            targets.append(np.clip(physical, self.lower, self.upper))
        before = self.prev_lift.copy()
        for env, hand, target in zip(self.envs, self.hands, targets):
            self.gym.set_actor_dof_position_targets(env, hand, target)
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        rewards = np.zeros(self.n, dtype=np.float32)
        for i in range(self.n):
            _, obj, contact = self._state(i)
            position = obj["object_position"].astype(np.float64)
            lift = float(position[2] - self.initial_positions[i, 2])
            self.positions[i].append(position)
            self.contacts[i].append(contact)
            delta = lift - float(before[i])
            rewards[i] = 200.0 * max(delta, -0.002) + 0.02 * float(contact > 0)
            rewards[i] -= 0.002 * float(np.square(residual[i]).mean())
            self.prev_lift[i] = lift
        return torch.as_tensor(rewards, dtype=torch.float32, device=self.device)

    def metrics(self):
        values = []
        for i in range(self.n):
            values.append(compute_success_metrics(
                np.asarray(self.positions[i]), self.initial_positions[i],
                np.asarray(self.contacts[i]), self.dt, 0.30, 0.25, 30,
                terminal_hold_steps=30,
            ))
        return values

    def close(self):
        self.gym.destroy_sim(self.sim)


def rollout(env, model=None, storage=None, deterministic=False):
    """收集一轮固定时长轨迹，并在最后一步加入稳定成功奖励。"""
    env.reset()
    previous = np.zeros((env.n, ACTION_DIM), dtype=np.float32)
    for t in range(env.horizon):
        obs = env.observation(t, previous)
        if model is None:
            action = torch.zeros((env.n, ACTION_DIM), device=env.device)
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
    """压缩一轮指标，便于终端和JSON日志查看。"""
    return {
        "success_count": int(sum(item["success"] for item in metrics)),
        "trajectory_count": len(metrics),
        "success_rate": float(np.mean([item["success"] for item in metrics])),
        "mean_max_lift_m": float(np.mean([item["max_lift_m"] for item in metrics])),
        "mean_final_lift_m": float(np.mean([item["final_lift_m"] for item in metrics])),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--object-root", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--settle-steps", type=int, default=30)
    parser.add_argument("--steps-per-frame", type=int, default=3)
    parser.add_argument("--hold-steps", type=int, default=30)
    parser.add_argument("--substeps", type=int, default=2)
    parser.add_argument("--physx-threads", type=int, default=1)
    parser.add_argument("--dt", type=float, default=1.0 / 60.0)
    parser.add_argument("--clearance", type=float, default=0.005)
    parser.add_argument("--object-friction", type=float, default=1.0)
    parser.add_argument("--finger-stiffness", type=float, default=120.0)
    parser.add_argument("--finger-damping", type=float, default=5.0)
    parser.add_argument("--mimic-stiffness", type=float, default=120.0)
    parser.add_argument("--mimic-damping", type=float, default=5.0)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--init-std", type=float, default=0.05)
    parser.add_argument("--eval-checkpoint", type=Path)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    cases = load_cases(args.manifest, args.baseline_dir, args.object_root, args.num_envs)
    env = LinkerResidualEnv(cases, args)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    try:
        baseline_metrics = rollout(env)
        baseline_report = {"config": vars(args), "summary": summary(baseline_metrics), "metrics": baseline_metrics}
        (args.checkpoint_dir / "baseline_eval.json").write_text(
            json.dumps(baseline_report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
        )
        print("baseline", json.dumps(baseline_report["summary"], ensure_ascii=False))
        model = ResidualActorCritic(
            OBS_DIM, OBS_DIM, action_dim=ACTION_DIM,
            hidden_dims=(args.hidden_dim, args.hidden_dim), init_std=args.init_std,
        ).to(env.device)
        if args.eval_checkpoint is not None:
            checkpoint = torch.load(args.eval_checkpoint, map_location=env.device)
            model.load_state_dict(checkpoint["model"])
        if args.eval_only:
            metrics = rollout(env, model, deterministic=True)
            report = {"config": vars(args), "summary": summary(metrics), "metrics": metrics}
            (args.checkpoint_dir / "policy_eval.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
            )
            print("policy", json.dumps(report["summary"], ensure_ascii=False))
            return
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        ppo_config = PPOConfig(
            learning_rate=args.learning_rate, minibatches=4, update_epochs=4,
            target_kl=0.5)
        log = []
        for iteration in range(1, args.iterations + 1):
            storage = RolloutStorage()
            metrics = rollout(env, model, storage, deterministic=False)
            batch = storage.finish(torch.zeros(env.n, device=env.device), ppo_config)
            update = ppo_update(model, optimizer, batch, ppo_config)
            item = {"iteration": iteration, **summary(metrics), **update}
            log.append(item)
            print(json.dumps(item, ensure_ascii=False))
            if iteration == 1 or iteration % 10 == 0 or iteration == args.iterations:
                torch.save({"model": model.state_dict(), "config": vars(args), "iteration": iteration}, args.checkpoint_dir / f"residual_ppo_{iteration:04d}.pt")
                (args.checkpoint_dir / "training_log.json").write_text(json.dumps(log, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    finally:
        env.close()


if __name__ == "__main__":
    main()
