#!/usr/bin/env python3
"""在统一Isaac Gym CPU PhysX口径下重放一条XHand候选轨迹。

输入：Shadow源文件、XHand 18维候选文件、轨迹索引和物理参数。
输出：与Linker相同字段的物体抬升、接触和成功JSON报告。
内部逻辑：按名称重排手腕6维和手指12维，再调用共享的插值执行与成功统计。
作用：补齐XHand物理基线，并让其能够与Linker在同一仿真条件下公平比较。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaacgym import gymapi
import numpy as np

try:
    from .isaac_replay_common import (
        IsaacCameraRecorder,
        compute_success_metrics,
        create_cpu_sim,
        load_object_asset,
        object_start_pose,
        replay_position_trajectory,
        save_policy_trace,
        summarize_body_contacts,
        summarize_dof_tracking,
    )
    from .xhand_replay_utils import (
        OPTIMIZER_FINGER_NAMES,
        WRIST_NAMES,
        reorder_xhand_frame,
    )
except ImportError:
    from isaac_replay_common import (
        IsaacCameraRecorder,
        compute_success_metrics,
        create_cpu_sim,
        load_object_asset,
        object_start_pose,
        replay_position_trajectory,
        save_policy_trace,
        summarize_body_contacts,
        summarize_dof_tracking,
    )
    from xhand_replay_utils import (
        OPTIMIZER_FINGER_NAMES,
        WRIST_NAMES,
        reorder_xhand_frame,
    )


RETARGET_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = RETARGET_ROOT.parent
REFERENCE_SCRIPTS = PROJECT_ROOT / "reference" / "HandRetargetTask2026" / "scripts"
XHAND_ASSET_ROOT = REFERENCE_SCRIPTS / "assets" / "xhand"
OBJECT_ROOT = REFERENCE_SCRIPTS / "data" / "sorting" / "object_41"
EXPECTED_DOF_NAMES = set(WRIST_NAMES + OPTIMIZER_FINGER_NAMES)


def load_xhand_asset(gym, sim, finger_stiffness=120.0, finger_damping=5.0):
    """加载18-DOF XHand欧拉手腕URDF并配置位置驱动。

    输入：Gym接口、sim句柄和手指位置环PD刚度/阻尼。
    输出：XHand asset、DOF属性和Isaac实际DOF名称顺序。
    逻辑：固定最外层基座，保留6个虚拟手腕与12个手指关节并分组设PD参数。
    作用：把数学重定向输出转换为可发生真实接触的物理执行器目标。
    """
    options = gymapi.AssetOptions()
    options.fix_base_link = True
    options.disable_gravity = True
    options.collapse_fixed_joints = False
    options.default_dof_drive_mode = int(gymapi.DOF_MODE_POS)
    options.use_mesh_materials = True
    options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
    asset = gym.load_asset(
        sim, str(XHAND_ASSET_ROOT), "xhand_euler_control.urdf", options
    )
    if asset is None:
        raise RuntimeError("XHand 18-DOF URDF加载失败")
    names = list(gym.get_asset_dof_names(asset))
    if len(names) != 18 or set(names) != EXPECTED_DOF_NAMES:
        raise ValueError(f"XHand物理DOF与预期不符: {names}")
    properties = gym.get_asset_dof_properties(asset)
    properties["driveMode"].fill(int(gymapi.DOF_MODE_POS))
    for index, name in enumerate(names):
        is_translation = name in {"virtual_x", "virtual_y", "virtual_z"}
        is_rotation = name in {"virtual_rx", "virtual_ry", "virtual_rz"}
        properties["stiffness"][index] = (
            20000.0
            if is_translation
            else 2000.0
            if is_rotation
            else float(finger_stiffness)
        )
        properties["damping"][index] = (
            500.0
            if is_translation
            else 80.0
            if is_rotation
            else float(finger_damping)
        )
    return asset, properties, names


def replay(args):
    """构建XHand场景、执行一条轨迹并返回统一物理报告。

    输入：包含源/目标文件、索引、时间参数和成功阈值的参数对象。
    输出：可直接序列化为JSON的完整指标字典。
    逻辑：加载同一COACD物体，按名称重排18维轨迹，张开落稳后插值重放。
    作用：完成XHand从已生成候选轨迹到严格物理成功判定的最小闭环。
    """
    source_data = np.load(args.source, allow_pickle=True).item()
    target_data = np.load(args.target, allow_pickle=True).item()
    frames = np.asarray(
        target_data["grasp_seqs"][args.target_index], dtype=np.float32
    )
    if frames.ndim != 2 or frames.shape[1] != 18:
        raise ValueError(f"XHand候选轨迹应为(T,18)，实际为{frames.shape}")
    object_name = args.object_name or args.source.stem
    object_dir = args.object_dir or OBJECT_ROOT / object_name
    if not object_dir.is_dir():
        raise FileNotFoundError(f"找不到物体目录: {object_dir}")
    scale = float(np.asarray(source_data["obj_scale"])[args.source_index])
    rotation = np.asarray(source_data["obj_rotmat"])[args.source_index]

    gym, sim = create_cpu_sim(
        args.dt, args.substeps, enable_graphics=args.video_output is not None
    )
    recorder = None
    try:
        hand_asset, dof_properties, dof_names = load_xhand_asset(
            gym, sim, args.finger_stiffness, args.finger_damping
        )
        object_asset = load_object_asset(gym, sim, object_dir)
        env = gym.create_env(
            sim,
            gymapi.Vec3(-1.0, -1.0, -0.2),
            gymapi.Vec3(1.0, 1.0, 1.0),
            1,
        )
        # filter=1与参考环境一致：禁用手内自碰撞，保留手物碰撞。
        hand_actor = gym.create_actor(
            env, hand_asset, gymapi.Transform(), "xhand", 0, 1
        )
        gym.set_actor_dof_properties(env, hand_actor, dof_properties)
        object_pose, mesh_min_z = object_start_pose(
            object_dir, scale, rotation, args.clearance
        )
        object_actor = gym.create_actor(
            env, object_asset, object_pose, "object", 0, 0
        )
        gym.set_actor_scale(env, object_actor, scale)
        shape_properties = gym.get_actor_rigid_shape_properties(env, object_actor)
        for prop in shape_properties:
            prop.friction = args.object_friction
        gym.set_actor_rigid_shape_properties(env, object_actor, shape_properties)
        if args.video_output is not None:
            recorder = IsaacCameraRecorder(
                gym, sim, env, args.video_output, args.video_width,
                args.video_height, args.video_fps, args.steps_per_frame,
            )

        trajectory = np.stack(
            [reorder_xhand_frame(frame, dof_names) for frame in frames]
        )
        open_first = trajectory[0].copy()
        for index, name in enumerate(dof_names):
            if name not in WRIST_NAMES:
                open_first[index] = 0.0
        contact_samples = [] if getattr(args, "include_contact_samples", False) else None
        policy_trace = {} if args.trace_output is not None else None
        policy_open_first = frames[0].copy()
        policy_open_first[6:] = 0.0
        (
            initial_position,
            positions,
            contact_counts,
            body_contacts,
            actual_dofs,
            commanded_dofs,
        ) = replay_position_trajectory(
            gym,
            sim,
            env,
            hand_actor,
            object_actor,
            trajectory,
            open_first,
            args.settle_steps,
            args.steps_per_frame,
            args.hold_steps,
            contact_sample_sink=contact_samples,
            trace_sink=policy_trace,
            policy_trajectory=frames,
            policy_open_first=policy_open_first,
            video_recorder=recorder,
        )
        metrics = compute_success_metrics(
            positions,
            initial_position,
            contact_counts,
            args.dt,
            args.lift_threshold,
            args.max_xy_drift,
            args.sustain_steps,
            terminal_hold_steps=args.hold_steps,
        )
        metrics.update(summarize_body_contacts(body_contacts))
        metrics.update(summarize_dof_tracking(actual_dofs, commanded_dofs, dof_names))
        report = {
            "hand": "xhand",
            "source": str(args.source.resolve()),
            "target": str(args.target.resolve()),
            "object_name": object_name,
            "source_trajectory_index": args.source_index,
            "target_trajectory_index": args.target_index,
            "frames": int(len(frames)),
            "target_dimensions": int(frames.shape[1]),
            "physics_dofs": len(dof_names),
            "physics_dof_names": dof_names,
            "dt_s": args.dt,
            "substeps": args.substeps,
            "steps_per_frame": args.steps_per_frame,
            "settle_steps": args.settle_steps,
            "hold_steps": args.hold_steps,
            "finger_stiffness": float(args.finger_stiffness),
            "finger_damping": float(args.finger_damping),
            "object_scale": scale,
            "rotated_scaled_mesh_min_z_m": mesh_min_z,
            **metrics,
        }
        if contact_samples is not None:
            report["hand_object_local_contact_samples"] = contact_samples
        if recorder is not None:
            report.update(recorder.close())
        if policy_trace is not None:
            trace_path = save_policy_trace(
                args.trace_output,
                policy_trace,
                {
                    "schema_version": 1,
                    "trace_alignment": "pre_action_state_to_command_v1",
                    "hand": "xhand",
                    "object_name": object_name,
                    "source": str(args.source.resolve()),
                    "target": str(args.target.resolve()),
                    "source_trajectory_index": int(args.source_index),
                    "target_trajectory_index": int(args.target_index),
                    "physics_dof_names": dof_names,
                    "policy_action_order": WRIST_NAMES + OPTIMIZER_FINGER_NAMES,
                    "dt_s": float(args.dt),
                    "steps_per_frame": int(args.steps_per_frame),
                },
            )
            report["policy_trace"] = str(trace_path)
        return report
    finally:
        if recorder is not None and recorder.writer is not None:
            recorder.close()
        gym.destroy_sim(sim)


def main():
    """解析命令行、运行XHand CPU重放并保存报告。

    输入：源/目标/输出路径、轨迹索引及统一物理阈值。
    输出：JSON报告和终端核心摘要。
    逻辑：调用`replay`并创建输出目录，所有默认值与Linker严格口径一致。
    作用：作为evaluate分区中XHand物理基线的标准命令入口。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--trace-output",
        type=Path,
        help="可选NPZ路径；保存进阶策略所需的逐物理步观测和专家动作",
    )
    parser.add_argument("--video-output", type=Path, help="可选MP4路径；只为报告案例启用")
    parser.add_argument("--video-width", type=int, default=640)
    parser.add_argument("--video-height", type=int, default=480)
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--source-index", type=int, default=0)
    parser.add_argument("--target-index", type=int, default=0)
    parser.add_argument("--object-name")
    parser.add_argument(
        "--object-dir",
        type=Path,
        help="物体碰撞资产的完整目录；正式manifest优先使用该路径",
    )
    parser.add_argument("--dt", type=float, default=1.0 / 60.0)
    parser.add_argument("--substeps", type=int, default=2)
    parser.add_argument("--steps-per-frame", type=int, default=3)
    parser.add_argument("--settle-steps", type=int, default=30)
    parser.add_argument("--hold-steps", type=int, default=30)
    parser.add_argument("--clearance", type=float, default=0.005)
    parser.add_argument("--object-friction", type=float, default=1.0)
    parser.add_argument("--finger-stiffness", type=float, default=120.0)
    parser.add_argument("--finger-damping", type=float, default=5.0)
    parser.add_argument("--lift-threshold", type=float, default=0.30)
    parser.add_argument("--max-xy-drift", type=float, default=0.25)
    parser.add_argument("--sustain-steps", type=int, default=30)
    parser.add_argument(
        "--include-contact-samples",
        action="store_true",
        help="在JSON中保存每个接触的手部局部坐标，仅用于指腹校准",
    )
    args = parser.parse_args()
    report = replay(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"frames={report['frames']}")
    print(f"physics_dofs={report['physics_dofs']}")
    print(f"max_lift_m={report['max_lift_m']:.6f}")
    print(f"final_lift_m={report['final_lift_m']:.6f}")
    print(f"longest_sustained_lift_steps={report['longest_sustained_lift_steps']}")
    print(f"success={report['success']}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
