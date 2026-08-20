#!/usr/bin/env python3
"""分阶段消除Wuji拇指在接近物体时过早折叠的问题。

输入：已通过率较高的15点Wuji候选、源索引和解剖配置。
输出：同形状26维候选、运动阶段、拇指尖偏移及关节自然度审计。
内部逻辑：先用零空间逆解得到抓取姿态；接近阶段固定为首帧自然拇指，闭合
阶段平滑过渡到逆解，抬升阶段完全保留逆解和接触位置。
作用：不靠角度裁剪，直接取消“尚未接触却必须追随旧指尖”的错误目标。
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import json
import sys

import nlopt
import numpy as np
import torch


RUN_DIR = Path(__file__).resolve().parent
RETARGET_ROOT = RUN_DIR.parent
PREPARE_DIR = RETARGET_ROOT / "prepare"
if str(RUN_DIR) not in sys.path:
    sys.path.insert(0, str(RUN_DIR))
if str(PREPARE_DIR) not in sys.path:
    sys.path.insert(0, str(PREPARE_DIR))

from object_geometry import transformed_object_vertices  # noqa: E402
from phase_contact import infer_motion_phases  # noqa: E402
from refine_wuji_pad_contacts import (  # noqa: E402
    SOURCE_TIP_INDICES,
    internal_to_saved,
    saved_to_internal,
    wuji_model_q,
)
from retarget_wuji_keypoints import (  # noqa: E402
    apply_anatomy_profile,
    build_shadow_model,
    build_wuji_model,
    load_anatomy_profile,
    shadow_keypoints,
)


METHOD = "point_baseline_thumb_tip_nullspace_phase_open_v2"
THUMB_DOF_COUNT = 4
THUMB_TIP_INDEX = 5


def full_points(model, internal_frame):
    """输入`[关节20,腕6]`单帧，输出Wuji 26个世界关键点NumPy数组。"""
    frame = np.asarray(internal_frame, dtype=np.float32)
    with torch.no_grad():
        points = model.get_penetraion_keypoints(
            q=wuji_model_q(
                torch.as_tensor(frame[:20]), frame[20:26], model.device
            )
        )[0]
    return points.cpu().numpy()


class ThumbNullspaceObjective:
    """保持拇指尖位置并在4关节逆解零空间中选自然姿态。"""

    def __init__(
        self,
        model,
        fixed_joints,
        wrist,
        target_tip,
        lower,
        upper,
        previous_thumb,
        args,
    ):
        """保存普通指/手腕、拇指尖目标、关节范围和上一帧。"""
        self.model = model
        self.fixed_joints = torch.as_tensor(fixed_joints, dtype=torch.float32)
        self.wrist = np.asarray(wrist, dtype=np.float32)
        self.target_tip = torch.as_tensor(target_tip, dtype=torch.float32)
        self.center = torch.as_tensor((lower + upper) * 0.5, dtype=torch.float32)
        self.half_range = torch.as_tensor(
            np.maximum((upper - lower) * 0.5, 1e-4), dtype=torch.float32
        )
        self.previous = (
            None
            if previous_thumb is None
            else torch.as_tensor(previous_thumb, dtype=torch.float32)
        )
        self.args = args
        self.last_components = {}

    def __call__(self, values, gradient=None):
        """输入4维拇指角，返回指尖保真+归一化中性+时序联合loss及梯度。"""
        thumb = torch.tensor(np.asarray(values, dtype=np.float32), requires_grad=True)
        joints = torch.cat([thumb, self.fixed_joints])
        points = self.model.get_penetraion_keypoints(
            q=wuji_model_q(joints, self.wrist, self.model.device)
        )[0]
        tip = float(self.args.tip_weight) * 1000.0 * torch.sum(
            (points[THUMB_TIP_INDEX] - self.target_tip) ** 2
        )
        normalized = (thumb - self.center) / self.half_range
        neutral = float(self.args.neutral_weight) * torch.mean(normalized**2)
        loss = tip + neutral
        components = {"thumb_tip_position": tip, "normalized_neutral": neutral}
        if self.previous is not None and self.args.temporal_weight > 0:
            temporal = float(self.args.temporal_weight) * torch.mean(
                (thumb - self.previous) ** 2
            )
            loss = loss + temporal
            components["temporal"] = temporal
        self.last_components = {
            name: float(value.detach().cpu()) for name, value in components.items()
        }
        if gradient is not None and len(gradient) > 0:
            loss.backward()
            gradient[:] = thumb.grad.detach().cpu().numpy().astype(np.float64)
        return float(loss.detach().cpu())


def refine_trajectory(frames_saved, model, lower, upper, args):
    """逐帧修正一条轨迹的拇指零空间姿态。

    输入：基线26维轨迹、模型、拇指边界和权重。
    输出：修正轨迹、逐帧loss/分项和指尖偏移。
    内部逻辑：目标指尖每帧由基线正向运动学计算；上一帧修正解热启动。
    作用：只沿拇指内部冗余自由度移动，尽量不改变物体接触位置。
    """
    baseline = saved_to_internal(frames_saved)
    baseline_tips = np.stack(
        [full_points(model, frame)[THUMB_TIP_INDEX] for frame in baseline]
    )
    outputs, losses, components, errors = [], [], [], []
    previous = None
    for frame_index, frame in enumerate(baseline):
        objective = ThumbNullspaceObjective(
            model,
            frame[THUMB_DOF_COUNT:20],
            frame[20:26],
            baseline_tips[frame_index],
            lower,
            upper,
            previous,
            args,
        )
        optimizer = nlopt.opt(nlopt.LD_SLSQP, THUMB_DOF_COUNT)
        optimizer.set_min_objective(objective)
        optimizer.set_lower_bounds(lower.tolist())
        optimizer.set_upper_bounds(upper.tolist())
        optimizer.set_maxeval(int(args.maxeval))
        optimizer.set_xtol_rel(1e-7)
        optimizer.set_ftol_rel(1e-9)
        start = np.clip(
            frame[:THUMB_DOF_COUNT] if previous is None else previous,
            lower,
            upper,
        )
        try:
            thumb = optimizer.optimize(start)
        except (nlopt.RoundoffLimited, RuntimeError):
            thumb = start
        result = frame.copy()
        result[:THUMB_DOF_COUNT] = np.asarray(thumb, dtype=np.float32)
        actual_tip = full_points(model, result)[THUMB_TIP_INDEX]
        outputs.append(result)
        losses.append(objective(result[:THUMB_DOF_COUNT]))
        components.append(objective.last_components.copy())
        errors.append(float(np.linalg.norm(actual_tip - baseline_tips[frame_index])))
        previous = result[:THUMB_DOF_COUNT].copy()
    return (
        internal_to_saved(np.stack(outputs)),
        np.asarray(losses, dtype=np.float32),
        components,
        np.asarray(errors, dtype=np.float32),
    )


def phase_aware_thumb_schedule(optimized_internal, close_start, lift_start):
    """让拇指只在真正闭合阶段弯曲。

    输入：零空间优化后的`(T,26)`内部轨迹，以及闭合/抬升起始帧。
    输出：分阶段轨迹和每帧0到1的混合比例。
    内部逻辑：闭合前复用自然首帧；闭合到抬升间用smoothstep平滑插值；
    抬升后原样使用优化解。
    作用：去掉第1帧突然折到约90度的假动作，同时不修改真正承力阶段。
    """
    optimized = np.asarray(optimized_internal, dtype=np.float32)
    if optimized.ndim != 2 or optimized.shape[1] != 26:
        raise ValueError(f"内部轨迹应为(T,26)，实际为{optimized.shape}")
    if not 0 <= int(close_start) < int(lift_start) < len(optimized):
        raise ValueError(
            f"阶段必须满足0<=close<lift<T，实际为{close_start}, {lift_start}, {len(optimized)}"
        )
    alpha = np.zeros(len(optimized), dtype=np.float32)
    width = float(lift_start - close_start)
    linear = (
        np.arange(close_start, lift_start + 1, dtype=np.float32) - close_start
    ) / width
    alpha[close_start : lift_start + 1] = linear * linear * (3.0 - 2.0 * linear)
    alpha[lift_start:] = 1.0
    result = optimized.copy()
    open_thumb = optimized[0, :THUMB_DOF_COUNT].copy()
    result[:, :THUMB_DOF_COUNT] = (
        (1.0 - alpha[:, None]) * open_thumb[None, :]
        + alpha[:, None] * optimized[:, :THUMB_DOF_COUNT]
    )
    return result, alpha


def thumb_transition_frames(contact_close, lift_start, frame_count, lead_frames, settle_frames):
    """把检测到的接触阶段转换成拇指实际闭合时间窗。

    输入：源手接触帧、抬升帧、轨迹长度，以及提前闭合和夹稳帧数。
    输出：满足`0<=start<end<T`的拇指过渡起止帧。
    内部逻辑：在接触前提前若干帧开始，并在抬升前若干帧结束；极短时间窗
    自动收缩但不越界。
    作用：让PD控制下的拇指在抬升前真正到位，而非抬升同时才发出闭合目标。
    """
    if frame_count < 2 or min(lead_frames, settle_frames) < 0:
        raise ValueError("轨迹至少2帧，提前量和稳定量不能为负")
    start = max(0, int(contact_close) - int(lead_frames))
    end = min(int(frame_count) - 1, int(lift_start) - int(settle_frames))
    if end <= start:
        end = min(int(frame_count) - 1, start + 1)
    if end <= start:
        start, end = max(0, end - 1), end
    return start, end


def infer_source_phase(source_frames, source_data, source_index, object_dir, shadow_model, args):
    """从Shadow专家与物体距离推断接近、闭合和抬升阶段。

    输入：一条源轨迹、物体姿态/网格、Shadow模型和距离阈值。
    输出：含close/lift帧、逐指距离和接触数量的阶段字典。
    内部逻辑：至少两根源指尖进入物体表面阈值时开始闭合，随后腕部上升
    到指定高度时进入抬升；没有严格入阈值时使用多指最近帧。
    作用：按每条轨迹自身动作分段，避免硬编码统一的第28或第35帧。
    """
    frames = np.asarray(source_frames, dtype=np.float32).copy()
    frames[:, 2] += float(args.source_z_offset)
    points = shadow_keypoints(frames, shadow_model)
    source_tips = {
        semantic: points[:, point_index]
        for semantic, point_index in SOURCE_TIP_INDICES.items()
    }
    vertices = transformed_object_vertices(
        object_dir,
        np.asarray(source_data["obj_scale"])[source_index],
        np.asarray(source_data["obj_rotmat"])[source_index],
        args.object_clearance,
    )
    return infer_motion_phases(
        frames,
        source_tips,
        vertices,
        args.contact_threshold,
        args.min_contact_tips,
        args.lift_delta,
        contact_fallback="nearest",
    )


def refine_file(args):
    """读取基线中指定轨迹，执行拇指消歧并保存完整追溯元数据。"""
    source_data = np.load(args.source, allow_pickle=True).item()
    data = np.load(args.initial_target, allow_pickle=True).item()
    indices = [int(value) for value in (args.trajectory_indices or data["source_trajectory_indices"])]
    source_indices = np.asarray(data["source_trajectory_indices"], dtype=np.int64)
    by_index = {
        int(index): frames for index, frames in zip(source_indices, data["grasp_seqs"])
    }
    missing = sorted(set(indices) - set(by_index))
    if missing:
        raise ValueError(f"点法基线缺少源索引: {missing}")
    model = build_wuji_model()
    joint_names = list(model.robot.get_joint_parameter_names())
    urdf_lower = model.revolute_joints_q_lower[0].detach().cpu().numpy()
    urdf_upper = model.revolute_joints_q_upper[0].detach().cpu().numpy()
    anatomy_file = Path(str(data["anatomy_config"]))
    _, _, _, anatomy_path, anatomy_sha = load_anatomy_profile(
        anatomy_file, joint_names, urdf_lower, urdf_upper
    )
    if anatomy_sha != data.get("anatomy_config_sha256"):
        raise ValueError("解剖配置SHA与点法基线不一致")
    profile = json.loads(anatomy_file.read_text(encoding="utf-8"))
    all_lower, all_upper, _ = apply_anatomy_profile(
        joint_names, urdf_lower, urdf_upper, profile
    )
    lower, upper = all_lower[:THUMB_DOF_COUNT], all_upper[:THUMB_DOF_COUNT]
    shadow_model = build_shadow_model()
    outputs, all_losses, all_components, all_errors = [], [], [], []
    all_nullspace_errors, all_phases, all_alpha = [], [], []
    for source_index in indices:
        nullspace_saved, losses, components, nullspace_errors = refine_trajectory(
            by_index[source_index], model, lower, upper, args
        )
        phase = infer_source_phase(
            source_data["grasp_seqs"][source_index],
            source_data,
            source_index,
            args.object_dir,
            shadow_model,
            args,
        )
        nullspace_internal = saved_to_internal(nullspace_saved)
        transition_start, transition_end = thumb_transition_frames(
            phase["close_start_frame"],
            phase["lift_start_frame"],
            len(nullspace_internal),
            args.close_lead_frames,
            args.grasp_settle_frames,
        )
        result_internal, alpha = phase_aware_thumb_schedule(
            nullspace_internal,
            transition_start,
            transition_end,
        )
        baseline_internal = saved_to_internal(by_index[source_index])
        baseline_tips = np.stack(
            [full_points(model, frame)[THUMB_TIP_INDEX] for frame in baseline_internal]
        )
        result_tips = np.stack(
            [full_points(model, frame)[THUMB_TIP_INDEX] for frame in result_internal]
        )
        errors = np.linalg.norm(result_tips - baseline_tips, axis=1)
        outputs.append(internal_to_saved(result_internal))
        all_losses.append(losses)
        all_components.append(components)
        all_errors.append(errors)
        all_nullspace_errors.append(nullspace_errors)
        all_alpha.append(alpha)
        all_phases.append({
            "close_start_frame": int(phase["close_start_frame"]),
            "lift_start_frame": int(phase["lift_start_frame"]),
            "thumb_transition_start_frame": int(transition_start),
            "thumb_transition_end_frame": int(transition_end),
            "grasp_frame": int(phase["grasp_frame"]),
            "close_detection": phase["close_detection"],
            "contact_fallback_used": bool(phase["contact_fallback_used"]),
            "close_contact_order_distance_m": float(
                phase["close_contact_order_distance_m"]
            ),
        })
    positions = [int(np.flatnonzero(source_indices == index)[0]) for index in indices]
    output = {
        "grasp_seqs": np.stack(outputs).astype(np.float32),
        "source_trajectory_indices": np.asarray(indices, dtype=np.int64),
        "obj_rotmat": np.asarray(data["obj_rotmat"])[positions],
        "obj_scale": np.asarray(data["obj_scale"])[positions],
        "mapping_semantics": data["mapping_semantics"],
        "mapping_config": data["mapping_config"],
        "wuji_joint_names": data["wuji_joint_names"],
        "anatomy_config": anatomy_path,
        "anatomy_config_sha256": anatomy_sha,
        "source_z_offset": float(data.get("source_z_offset", 0.4)),
        "retarget_method": METHOD,
        "source": str(args.source.resolve()),
        "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "object_dir": str(args.object_dir.resolve()),
        "initial_target": str(args.initial_target.resolve()),
        "initial_target_sha256": hashlib.sha256(args.initial_target.read_bytes()).hexdigest(),
        "optimization_loss_per_frame": np.stack(all_losses),
        "optimization_loss_components_per_frame": all_components,
        "thumb_tip_displacement_m_per_frame": np.stack(all_errors),
        "nullspace_thumb_tip_displacement_m_per_frame": np.stack(all_nullspace_errors),
        "thumb_blend_alpha_per_frame": np.stack(all_alpha),
        "motion_phases": all_phases,
        "maxeval": int(args.maxeval),
        "tip_weight": float(args.tip_weight),
        "neutral_weight": float(args.neutral_weight),
        "temporal_weight": float(args.temporal_weight),
        "contact_threshold": float(args.contact_threshold),
        "min_contact_tips": int(args.min_contact_tips),
        "lift_delta": float(args.lift_delta),
        "object_clearance": float(args.object_clearance),
        "close_lead_frames": int(args.close_lead_frames),
        "grasp_settle_frames": int(args.grasp_settle_frames),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, output, allow_pickle=True)
    joint4_deg = np.degrees(output["grasp_seqs"][:, :, 9])
    print(f"trajectories={len(outputs)}")
    print(f"max_thumb_tip_displacement_mm={1000.0 * float(np.max(output['thumb_tip_displacement_m_per_frame'])):.3f}")
    print(f"thumb_joint4_median_deg={float(np.median(joint4_deg)):.2f}")
    print(f"thumb_joint4_near_90_ratio={float(np.mean((joint4_deg >= 85) & (joint4_deg <= 95))):.4f}")
    print(f"output={args.output}")


def main():
    """解析拇指零空间修正的输入、权重和SLSQP预算并执行。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial-target", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--object-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trajectory-indices", type=int, nargs="*")
    parser.add_argument("--maxeval", type=int, default=80)
    parser.add_argument("--tip-weight", type=float, default=1.0)
    parser.add_argument("--neutral-weight", type=float, default=0.05)
    parser.add_argument("--temporal-weight", type=float, default=0.01)
    parser.add_argument("--source-z-offset", type=float, default=0.4)
    parser.add_argument("--contact-threshold", type=float, default=0.02)
    parser.add_argument("--min-contact-tips", type=int, default=2)
    parser.add_argument("--lift-delta", type=float, default=0.03)
    parser.add_argument("--object-clearance", type=float, default=0.005)
    parser.add_argument("--close-lead-frames", type=int, default=6)
    parser.add_argument("--grasp-settle-frames", type=int, default=3)
    args = parser.parse_args()
    if args.maxeval < 1:
        parser.error("--maxeval必须为正整数")
    if min(args.tip_weight, args.neutral_weight, args.temporal_weight) < 0:
        parser.error("所有权重必须非负")
    if args.contact_threshold <= 0 or args.lift_delta <= 0:
        parser.error("接触距离和抬升量必须为正")
    if not 1 <= args.min_contact_tips <= 5:
        parser.error("--min-contact-tips必须在1到5之间")
    if min(args.close_lead_frames, args.grasp_settle_frames) < 0:
        parser.error("闭合提前量和抬升前稳定量不能为负")
    refine_file(args)


if __name__ == "__main__":
    main()
