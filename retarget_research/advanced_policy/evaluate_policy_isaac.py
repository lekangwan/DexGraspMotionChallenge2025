#!/usr/bin/env python3
"""在Isaac Gym CPU PhysX中闭环执行一条目标手策略。

输入：手类型、源/重定向候选、轨迹索引、类别、checkpoint和策略数据目录。
输出：逐步物体/动作曲线与统一抬升成功JSON。
内部逻辑：只从候选首帧取得该抓取方向的张开手腕初态，之后每个60 Hz物理步均由策略读状态出动作。
作用：真正检验BC/Temporal3/Diffusion能否闭环抓取；与离线动作误差严格区分。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from isaacgym import gymapi
import numpy as np
import torch

EVALUATE_DIR = Path(__file__).resolve().parents[1] / "retargeting" / "evaluate"
import sys
sys.path.insert(0, str(EVALUATE_DIR))

from isaac_replay_common import (  # noqa: E402
    IsaacCameraRecorder,
    actor_body_indices,
    compute_success_metrics,
    count_contacts_by_hand_body,
    create_cpu_sim,
    load_object_asset,
    object_start_pose,
    read_object_state,
    read_policy_pre_action_state,
    set_dof_state_and_target,
    summarize_body_contacts,
)
from linker_replay_utils import expand_linker_frame  # noqa: E402
from replay_linker_isaac import load_linker_asset  # noqa: E402
from replay_wuji_isaac import load_wuji_asset  # noqa: E402
from replay_xhand_isaac import load_xhand_asset  # noqa: E402
from wuji_replay_utils import WRIST_NAMES as WUJI_WRIST_NAMES, reorder_wuji_frame  # noqa: E402
from xhand_replay_utils import (  # noqa: E402
    OPTIMIZER_FINGER_NAMES as XHAND_FINGER_NAMES,
    WRIST_NAMES as XHAND_WRIST_NAMES,
    reorder_xhand_frame,
)

try:  # noqa: E402
    from .observations import build_object_shape_descriptor, build_runtime_observation
    from .runtime import PolicyRunner
except ImportError:  # noqa: E402
    from observations import build_object_shape_descriptor, build_runtime_observation
    from runtime import PolicyRunner


WRIST_DIMENSION = 6


def prepare_hand(gym, sim, hand, target_data, policy_frames, args):
    """加载目标手并建立策略动作到Isaac DOF的映射。

    输入：Gym、sim、手类型、候选元数据、候选帧和控制参数。
    输出：asset、属性、DOF名称、张开初态、动作映射函数和策略动作名称顺序。
    内部逻辑：Linker展开mimic，XHand/Wuji按名称重排；仅保留首帧手腕，手指全置零。
    作用：三只结构不同的手共享一个闭环主循环，同时保持各自正确DOF顺序。
    """
    if hand == "linker":
        asset, properties, names = load_linker_asset(
            gym, sim, args.linker_finger_stiffness, args.linker_finger_damping,
            args.linker_mimic_stiffness, args.linker_mimic_damping,
        )
        mapper = lambda action: expand_linker_frame(action, names)
        action_order = [
            "wrist_x", "wrist_y", "wrist_z", "wrist_roll", "wrist_pitch", "wrist_yaw",
            "thumb_yaw", "thumb_pitch", "index_flexion", "middle_flexion",
            "ring_flexion", "little_flexion",
        ]
    elif hand == "xhand":
        asset, properties, names = load_xhand_asset(gym, sim)
        mapper = lambda action: reorder_xhand_frame(action, names)
        action_order = XHAND_WRIST_NAMES + XHAND_FINGER_NAMES
    else:
        optimizer_names = list(target_data.get("wuji_joint_names", []))
        if len(optimizer_names) != 20:
            raise ValueError("Wuji候选缺少20个wuji_joint_names")
        asset, properties, names = load_wuji_asset(gym, sim, optimizer_names)
        mapper = lambda action: reorder_wuji_frame(action, optimizer_names, names)
        action_order = WUJI_WRIST_NAMES + optimizer_names
    policy_open = np.asarray(policy_frames[0], dtype=np.float32).copy()
    policy_open[WRIST_DIMENSION:] = 0.0
    physics_open = np.asarray(mapper(policy_open), dtype=np.float32)
    return asset, properties, names, policy_open, physics_open, mapper, action_order


def set_object_friction(gym, env, actor, friction):
    """把物体全部碰撞形状的摩擦系数设为统一值。"""
    properties = gym.get_actor_rigid_shape_properties(env, actor)
    for prop in properties:
        prop.friction = float(friction)
    gym.set_actor_rigid_shape_properties(env, actor, properties)


def rollout(args):
    """构建单场景并完成策略闭环rollout。

    输入：完整命令行参数。
    输出：可序列化报告和预测动作数组。
    内部逻辑：张开落稳后，反复读取执行前状态、策略推理、映射/限幅、推进一步PhysX并统计接触。
    作用：提供最小但严格的进阶任务最终成功率测量单元。
    """
    np.random.seed(int(args.seed) % (2 ** 32))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    source_data = np.load(args.source, allow_pickle=True).item()
    target_data = np.load(args.target, allow_pickle=True).item()
    policy_frames = np.asarray(target_data["grasp_seqs"][args.target_index], dtype=np.float32)
    expected_dim = {"linker": 12, "xhand": 18, "wuji": 26}[args.hand]
    if policy_frames.shape != (70, expected_dim):
        raise ValueError(f"候选形状错误: {policy_frames.shape}")
    scale = float(np.asarray(source_data["obj_scale"])[args.source_index])
    rotation = np.asarray(source_data["obj_rotmat"])[args.source_index]
    object_dir = args.object_dir.resolve()
    gym, sim = create_cpu_sim(
        args.dt, args.substeps, enable_graphics=args.video_output is not None
    )
    recorder = None
    try:
        hand_asset, dof_properties, dof_names, policy_open, open_first, mapper, action_order = prepare_hand(
            gym, sim, args.hand, target_data, policy_frames, args
        )
        object_asset = load_object_asset(gym, sim, object_dir)
        env = gym.create_env(sim, gymapi.Vec3(-1.0, -1.0, -0.2), gymapi.Vec3(1.0, 1.0, 1.0), 1)
        hand_actor = gym.create_actor(env, hand_asset, gymapi.Transform(), args.hand, 0, 1)
        gym.set_actor_dof_properties(env, hand_actor, dof_properties)
        object_pose, _ = object_start_pose(object_dir, scale, rotation, args.clearance)
        object_actor = gym.create_actor(env, object_asset, object_pose, "object", 0, 0)
        gym.set_actor_scale(env, object_actor, scale)
        set_object_friction(gym, env, object_actor, args.object_friction)
        if args.video_output is not None:
            recorder = IsaacCameraRecorder(
                gym, sim, env, args.video_output, args.video_width,
                args.video_height, args.video_fps, args.steps_per_frame,
            )
        set_dof_state_and_target(gym, env, hand_actor, open_first)
        for _ in range(args.settle_steps):
            gym.simulate(sim)
            gym.fetch_results(sim, True)

        hand_indices = actor_body_indices(gym, env, hand_actor)
        object_indices = actor_body_indices(gym, env, object_actor)
        hand_names = list(gym.get_actor_rigid_body_names(env, hand_actor))
        hand_index_to_name = dict(zip(hand_indices, hand_names))
        initial_position = read_object_state(gym, env, object_actor)["object_position"].copy()
        shape_descriptor = build_object_shape_descriptor(
            object_dir / "coacd" / "decomposed.obj", scale
        )
        residual_rl = None
        rl_action_dim = 0
        if args.residual_rl_checkpoint is not None:
            from residual_ppo import ResidualActorCritic
            payload = torch.load(args.residual_rl_checkpoint, map_location=args.device)
            rl_obs_dim = int(payload.get("obs_dim", 0)) or None
            rl_action_dim = int(payload.get("action_dim", 0)) or None
            if rl_obs_dim is None or rl_action_dim is None:
                state = payload["model"]
                actor_weight = state["actor.1.weight"]
                rl_obs_dim = int(actor_weight.shape[1])
                actor_head = state["actor.5.weight"]
                rl_action_dim = int(actor_head.shape[0])
            residual_rl = ResidualActorCritic(
                rl_obs_dim, rl_obs_dim, action_dim=rl_action_dim,
                hidden_dims=(128, 128), init_std=0.05,
            ).to(args.device)
            residual_rl.load_state_dict(payload["model"])
            residual_rl.eval()
            previous_residual = np.zeros(rl_action_dim, dtype=np.float32)
        runner = PolicyRunner(
            args.checkpoint, args.data_dir, args.device,
            args.diffusion_execute_steps, args.normalized_action_clip,
            args.action_rate_limit_scale,
        )
        phase_expert_teacher = (
            args.teacher_checkpoint is not None
            and str(args.teacher_checkpoint) == "phase_expert"
        )
        teacher = None
        if args.teacher_checkpoint is not None and not phase_expert_teacher:
            teacher = PolicyRunner(
                args.teacher_checkpoint,
                args.data_dir,
                args.device,
                diffusion_execute_steps=1,
                normalized_action_clip=args.normalized_action_clip,
                action_rate_limit_scale=0.0,
            )
            if teacher.model_type not in ("category_teacher", "phase_residual"):
                raise ValueError("在线采集的teacher checkpoint必须是category_teacher或phase_residual")
        if runner.mappings.get("policy_action_order") != action_order:
            raise ValueError("策略训练数据的动作名称顺序与当前目标手候选不一致")
        first_dofs, first_object, first_contacts = read_policy_pre_action_state(
            gym, env, hand_actor, object_actor, hand_index_to_name, object_indices
        )
        first_observation = build_runtime_observation(
            first_dofs,
            first_object,
            initial_position,
            first_contacts,
            shape_descriptor,
            args.lift_threshold,
        )
        runner.reset(args.category, first_observation, initial_action=policy_open)
        if teacher is not None:
            teacher.reset(args.category, first_observation, initial_action=policy_open)
        lower = np.asarray(dof_properties["lower"], dtype=np.float32)
        upper = np.asarray(dof_properties["upper"], dtype=np.float32)
        positions, object_quaternions, contacts, actions, actual_dof_positions = [], [], [], [], []
        online_observations, online_teacher_actions = [], []
        body_contacts = {name: [] for name in hand_names}
        horizon = args.policy_steps if args.policy_steps > 0 else 70 * args.steps_per_frame + args.hold_steps
        for physics_step in range(horizon):
            dof_states, object_state, contact_count = read_policy_pre_action_state(
                gym, env, hand_actor, object_actor, hand_index_to_name, object_indices
            )
            observation = build_runtime_observation(
                dof_states,
                object_state,
                initial_position,
                contact_count,
                shape_descriptor,
                args.lift_threshold,
            )
            if teacher is not None:
                online_observations.append(observation.copy())
                online_teacher_actions.append(teacher.act(observation).copy())
            elif phase_expert_teacher:
                online_observations.append(observation.copy())
                phase_index = min(int(round(physics_step / 209.0 * 69)), 69)
                online_teacher_actions.append(policy_frames[phase_index].copy())
            policy_action = runner.act(observation)
            if residual_rl is not None:
                phase_index = min(physics_step // 3, len(policy_frames) - 1)
                baseline = policy_frames[phase_index]
                dof_pos = np.asarray(dof_states["pos"], dtype=np.float32)
                dof_vel = np.asarray(dof_states["vel"], dtype=np.float32)
                relative = (object_state["object_position"]
                            - initial_position).astype(np.float32)
                rl_obs = np.concatenate([
                    baseline,
                    dof_pos, dof_vel,
                    relative,
                    object_state["object_quaternion_xyzw"],
                    object_state["object_linear_velocity"],
                    object_state["object_angular_velocity"],
                    np.asarray([math.log1p(contact_count),
                                physics_step / max(1, 239.0), relative[2]],
                               dtype=np.float32),
                    previous_residual,
                    shape_descriptor,
                ]).astype(np.float32)
                rl_tensor = torch.from_numpy(rl_obs[None]).to(args.device)
                with torch.no_grad():
                    residual = torch.tanh(
                        residual_rl.distribution(rl_tensor).mean)[0].cpu().numpy()
                policy_action = baseline.copy()
                policy_action[6:] = policy_action[6:] + residual
                previous_residual = residual
            if args.expert_wrist:
                phase_index = min(int(round(physics_step / 209.0 * 69)), 69)
                policy_action = policy_action.copy()
                policy_action[:6] = policy_frames[phase_index][:6]
            physics_action = np.asarray(mapper(policy_action), dtype=np.float32)
            physics_action = np.clip(physics_action, lower, upper)
            gym.set_actor_dof_position_targets(env, hand_actor, physics_action)
            gym.simulate(sim)
            gym.fetch_results(sim, True)
            if recorder is not None:
                recorder.capture(physics_step)
            post_state = read_object_state(gym, env, object_actor)
            post_dofs = gym.get_actor_dof_states(env, hand_actor, gymapi.STATE_ALL)
            step_contacts = gym.get_env_rigid_contacts(env)
            grouped = count_contacts_by_hand_body(step_contacts, hand_index_to_name, object_indices)
            positions.append(post_state["object_position"].copy())
            object_quaternions.append(post_state["object_quaternion_xyzw"].copy())
            contacts.append(sum(grouped.values()))
            actions.append(policy_action.copy())
            actual_dof_positions.append(post_dofs["pos"].copy())
            for name in hand_names:
                body_contacts[name].append(grouped[name])
        positions = np.stack(positions)
        contacts = np.asarray(contacts, dtype=np.int64)
        metrics = compute_success_metrics(
            positions, initial_position, contacts, args.dt,
            args.lift_threshold, args.max_xy_drift, args.sustain_steps,
            terminal_hold_steps=args.hold_steps,
        )
        metrics.update(summarize_body_contacts(body_contacts))
        report = {
            "hand": args.hand,
            "model_type": runner.model_type,
            "category": args.category,
            "object_name": args.object_name,
            "source_trajectory_index": int(args.source_index),
            "target_trajectory_index": int(args.target_index),
            "source": str(args.source.resolve()),
            "target": str(args.target.resolve()),
            "object_dir": str(args.object_dir.resolve()),
            "data_dir": str(args.data_dir.resolve()),
            "initialization_rule": "retargeted_first_frame_wrist_with_all_fingers_open; no_future_expert_actions",
            "evaluation_seed": int(args.seed),
            "diffusion_execute_steps": int(args.diffusion_execute_steps),
            "normalized_action_clip": float(args.normalized_action_clip),
            "action_rate_limit_scale": float(args.action_rate_limit_scale),
            "action_delta_quantile": (
                None
                if args.action_rate_limit_scale <= 0.0
                else float(runner.normalization["action_delta_quantile"])
            ),
            "action_delta_norm_limit": runner.action_delta_norm_limit,
            "policy_steps": int(horizon),
            "dt_s": float(args.dt),
            "physics_dof_names": dof_names,
            "checkpoint": str(args.checkpoint.resolve()),
            "expert_wrist": bool(args.expert_wrist),
            "residual_rl_checkpoint": (
                None
                if args.residual_rl_checkpoint is None
                else str(args.residual_rl_checkpoint.resolve())
            ),
            "teacher_checkpoint": (
                None
                if args.teacher_checkpoint is None
                else str(args.teacher_checkpoint.resolve())
            ),
            **metrics,
            "predicted_policy_actions": np.asarray(actions).tolist(),
            "actual_hand_dof_positions": np.asarray(actual_dof_positions).tolist(),
            "object_quaternions_xyzw": np.asarray(object_quaternions).tolist(),
        }
        if recorder is not None:
            report.update(recorder.close())
        if teacher is not None or phase_expert_teacher:
            if args.online_output is None:
                raise ValueError("提供teacher checkpoint时必须同时提供online output")
            args.online_output.parent.mkdir(parents=True, exist_ok=True)
            metadata = {
                "schema_version": 1,
                "alignment": "student_pre_action_observation_to_category_teacher_action_v1",
                "hand": args.hand,
                "category": args.category,
                "object_name": args.object_name,
                "source_trajectory_index": int(args.source_index),
                "student_checkpoint": str(args.checkpoint.resolve()),
                "teacher_checkpoint": str(args.teacher_checkpoint.resolve()),
            }
            np.savez_compressed(
                args.online_output,
                observations=np.asarray(online_observations, dtype=np.float32),
                teacher_actions=np.asarray(online_teacher_actions, dtype=np.float32),
                executed_actions=np.asarray(actions, dtype=np.float32),
                metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)),
            )
            report["online_output"] = str(args.online_output.resolve())
        return report
    finally:
        if recorder is not None and recorder.writer is not None:
            recorder.close()
        gym.destroy_sim(sim)


def main():
    """解析单轨迹参数、执行闭环并保存统一JSON。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--hand", choices=["linker", "xhand", "wuji"], required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--object-dir", type=Path, required=True)
    parser.add_argument("--object-name", required=True)
    parser.add_argument("--category", required=True)
    parser.add_argument("--expert-wrist", action="store_true")
    parser.add_argument("--residual-rl-checkpoint", type=Path)
    parser.add_argument("--source-index", type=int, required=True)
    parser.add_argument("--target-index", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--teacher-checkpoint", type=Path)
    parser.add_argument("--online-output", type=Path)
    parser.add_argument("--video-output", type=Path, help="可选MP4路径；只为报告案例启用")
    parser.add_argument("--video-width", type=int, default=640)
    parser.add_argument("--video-height", type=int, default=480)
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--diffusion-execute-steps", type=int, default=2)
    parser.add_argument("--normalized-action-clip", type=float, default=5.0)
    parser.add_argument("--action-rate-limit-scale", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--dt", type=float, default=1.0 / 60.0)
    parser.add_argument("--substeps", type=int, default=2)
    parser.add_argument("--steps-per-frame", type=int, default=3)
    parser.add_argument("--settle-steps", type=int, default=30)
    parser.add_argument("--hold-steps", type=int, default=30)
    parser.add_argument("--policy-steps", type=int, default=0)
    parser.add_argument("--clearance", type=float, default=0.005)
    parser.add_argument("--object-friction", type=float, default=1.0)
    parser.add_argument("--lift-threshold", type=float, default=0.30)
    parser.add_argument("--max-xy-drift", type=float, default=0.25)
    parser.add_argument("--sustain-steps", type=int, default=30)
    parser.add_argument("--linker-finger-stiffness", type=float, default=120.0)
    parser.add_argument("--linker-finger-damping", type=float, default=5.0)
    parser.add_argument("--linker-mimic-stiffness", type=float, default=120.0)
    parser.add_argument("--linker-mimic-damping", type=float, default=5.0)
    args = parser.parse_args()
    report = rollout(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"success={report['success']}")
    print(f"max_lift_m={report['max_lift_m']:.6f}")
    print(f"POLICY_ROLLOUT={args.output.resolve()}")


if __name__ == "__main__":
    main()
