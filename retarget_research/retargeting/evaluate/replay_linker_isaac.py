#!/usr/bin/env python3
"""在Isaac Gym CPU PhysX中重放Linker O6或11轴解耦候选轨迹。

输入：Shadow源文件、Linker候选文件、轨迹索引和物体资产目录。
输出：物体逐帧高度、抬升高度、连续抬升步数和成功标志JSON。
内部逻辑：加载17-DOF虚拟手腕URDF；12维候选按mimic展开，17维候选直接控制11轴。
作用：把几何误差验证推进到真实接触动力学；当前版本为无渲染CPU单轨迹入口。
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
    from .linker_replay_utils import expand_linker_frame, linker_dof_gains
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
    from linker_replay_utils import expand_linker_frame, linker_dof_gains


RETARGET_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = RETARGET_ROOT.parent
REFERENCE_SCRIPTS = PROJECT_ROOT / "reference" / "HandRetargetTask2026" / "scripts"
LINKER_ASSET_ROOT = REFERENCE_SCRIPTS / "assets" / "linkerhand" / "o6" / "right"
OBJECT_ROOT = REFERENCE_SCRIPTS / "data" / "sorting" / "object_41"

EXPECTED_DOF_NAMES = {
    "virtual_joint_x",
    "virtual_joint_y",
    "virtual_joint_z",
    "virtual_joint_roll",
    "virtual_joint_pitch",
    "virtual_joint_yaw",
    "rh_thumb_cmc_yaw",
    "rh_thumb_cmc_pitch",
    "rh_thumb_ip",
    "rh_index_mcp_pitch",
    "rh_index_dip",
    "rh_middle_mcp_pitch",
    "rh_middle_dip",
    "rh_ring_mcp_pitch",
    "rh_ring_dip",
    "rh_pinky_mcp_pitch",
    "rh_pinky_dip",
}


def load_linker_asset(
    gym,
    sim,
    finger_stiffness=120.0,
    finger_damping=5.0,
    mimic_stiffness=120.0,
    mimic_damping=5.0,
):
    """加载带6维虚拟手腕的Linker URDF并配置位置驱动。

    输入：Gym接口和sim句柄。
    输出：Linker asset、DOF属性和Isaac实际DOF名称顺序。
    逻辑：固定最外层基座、保留17个关节，并分别设置手腕、主动手指和mimic手指增益。
    作用：使候选轨迹的手腕6维和完整手指11维都能作为位置目标执行。
    """
    options = gymapi.AssetOptions()
    options.fix_base_link = True
    options.disable_gravity = True
    options.collapse_fixed_joints = False
    options.default_dof_drive_mode = int(gymapi.DOF_MODE_POS)
    options.use_mesh_materials = True
    options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
    asset = gym.load_asset(
        sim, str(LINKER_ASSET_ROOT), "linkerhand_o6_right6d.urdf", options
    )
    if asset is None:
        raise RuntimeError("Linker O6 6D URDF加载失败")
    names = list(gym.get_asset_dof_names(asset))
    if len(names) != 17 or set(names) != EXPECTED_DOF_NAMES:
        raise ValueError(f"Linker物理DOF与预期不符: {names}")
    properties = gym.get_asset_dof_properties(asset)
    properties["driveMode"].fill(int(gymapi.DOF_MODE_POS))
    stiffness, damping = linker_dof_gains(
        names,
        finger_stiffness,
        finger_damping,
        mimic_stiffness,
        mimic_damping,
    )
    properties["stiffness"][:] = stiffness
    properties["damping"][:] = damping
    return asset, properties, names


def replay(args):
    """构建场景、回放候选轨迹并返回物理指标。

    输入：包含文件、索引、仿真频率和成功阈值的命令行参数。
    输出：可直接序列化为JSON的指标字典。
    逻辑：物体先落稳，手从零指姿态进入70帧轨迹，逐物理步记录物体位置。
    作用：完成Linker候选轨迹从数学输出到接触动力学结果的核心验证。
    """
    source_data = np.load(args.source, allow_pickle=True).item()
    target_data = np.load(args.target, allow_pickle=True).item()
    frames = np.asarray(
        target_data["grasp_seqs"][args.target_index], dtype=np.float32
    )
    if frames.ndim != 2 or frames.shape[1] not in {12, 17}:
        raise ValueError(f"Linker候选轨迹应为(T,12)或(T,17)，实际为{frames.shape}")
    inferred_mode = "coupled6" if frames.shape[1] == 12 else "independent11"
    saved_mode = str(target_data.get("joint_mode", inferred_mode))
    if saved_mode != inferred_mode:
        raise ValueError(
            f"候选元数据joint_mode={saved_mode}与维度{frames.shape[1]}不一致"
        )
    source_index = args.source_index
    object_name = args.object_name or args.source.stem
    object_dir = args.object_dir or OBJECT_ROOT / object_name
    if not object_dir.is_dir():
        raise FileNotFoundError(f"找不到物体目录: {object_dir}")
    scale = float(np.asarray(source_data["obj_scale"])[source_index])
    rotation = np.asarray(source_data["obj_rotmat"])[source_index]

    gym, sim = create_cpu_sim(
        args.dt, args.substeps, enable_graphics=args.video_output is not None
    )
    recorder = None
    try:
        linker_asset, dof_properties, dof_names = load_linker_asset(
            gym,
            sim,
            args.finger_stiffness,
            args.finger_damping,
            args.mimic_stiffness,
            args.mimic_damping,
        )
        object_asset = load_object_asset(gym, sim, object_dir)
        lower = gymapi.Vec3(-1.0, -1.0, -0.2)
        upper = gymapi.Vec3(1.0, 1.0, 1.0)
        env = gym.create_env(sim, lower, upper, 1)
        hand_pose = gymapi.Transform()
        # filter=1与参考环境一致：禁用手内自碰撞，保留手物碰撞。
        hand_actor = gym.create_actor(env, linker_asset, hand_pose, "linker", 0, 1)
        gym.set_actor_dof_properties(env, hand_actor, dof_properties)
        object_pose, mesh_min_z = object_start_pose(
            object_dir, scale, rotation, args.clearance
        )
        # Isaac Gym只有同一collision group中的actor才会互相接触；手和物体都用0。
        object_actor = gym.create_actor(env, object_asset, object_pose, "object", 0, 0)
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

        expanded = np.stack(
            [expand_linker_frame(frame, dof_names) for frame in frames]
        )
        open_first = expanded[0].copy()
        for index, name in enumerate(dof_names):
            if not name.startswith("virtual_joint_"):
                open_first[index] = 0.0
        contact_samples = [] if getattr(args, "include_contact_samples", False) else None
        policy_trace = {} if args.trace_output is not None else None
        policy_open_first = frames[0].copy()
        policy_open_first[6:] = 0.0
        (
            initial_object_position,
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
            expanded,
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
            initial_object_position,
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
            "hand": (
                "linker_o6" if inferred_mode == "coupled6" else "linker_independent11"
            ),
            "joint_mode": inferred_mode,
            "source": str(args.source.resolve()),
            "target": str(args.target.resolve()),
            "object_name": object_name,
            "source_trajectory_index": source_index,
            "target_trajectory_index": args.target_index,
            "frames": int(len(frames)),
            "target_dimensions": int(frames.shape[1]),
            "physics_dofs": len(dof_names),
            "physics_dof_names": dof_names,
            "finger_stiffness": args.finger_stiffness,
            "finger_damping": args.finger_damping,
            "mimic_stiffness": args.mimic_stiffness,
            "mimic_damping": args.mimic_damping,
            "dt_s": args.dt,
            "substeps": args.substeps,
            "steps_per_frame": args.steps_per_frame,
            "settle_steps": args.settle_steps,
            "hold_steps": args.hold_steps,
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
                    "hand": "linker",
                    "joint_mode": inferred_mode,
                    "object_name": object_name,
                    "source": str(args.source.resolve()),
                    "target": str(args.target.resolve()),
                    "source_trajectory_index": int(args.source_index),
                    "target_trajectory_index": int(args.target_index),
                    "physics_dof_names": dof_names,
                    "policy_action_order": (
                        ["wrist_x", "wrist_y", "wrist_z", "wrist_roll", "wrist_pitch", "wrist_yaw"]
                        + [
                            "thumb_yaw",
                            "thumb_pitch",
                            "index_flexion",
                            "middle_flexion",
                            "ring_flexion",
                            "little_flexion",
                        ]
                    ),
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
    """解析参数、执行Linker物理重放并保存报告。

    输入：命令行中的源/目标/输出路径与可复现的物理参数。
    输出：JSON报告和终端核心摘要。
    逻辑：调用`replay`，保证输出目录存在后保存全部物体位置曲线。
    作用：作为evaluate分区的Linker CPU PhysX标准命令入口。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--trace-output",
        type=Path,
        help="可选NPZ路径；保存进阶策略所需的逐物理步观测和12维主动命令",
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
    parser.add_argument(
        "--mimic-stiffness",
        type=float,
        default=120.0,
        help="5个从动IP/DIP轴的位置刚度；降低后允许接触自适应",
    )
    parser.add_argument("--mimic-damping", type=float, default=5.0)
    parser.add_argument("--lift-threshold", type=float, default=0.10)
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
