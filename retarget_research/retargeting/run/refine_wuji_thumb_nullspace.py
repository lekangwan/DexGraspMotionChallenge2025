#!/usr/bin/env python3
"""在不移动手腕和四根普通指的前提下，消除Wuji拇指冗余折叠解。

输入：已通过率较高的15点Wuji候选、源索引和解剖配置。
输出：同形状26维候选、拇指尖位置误差及关节自然度审计。
内部逻辑：逐帧固定腕部与普通16关节，只优化4个拇指关节；保持基线拇指尖
世界位置，同时以归一化中性姿态和上一帧连续项在冗余逆运动学解中消歧。
作用：保留旧点法已经建立的夹持位置和普通指成功率，去掉导致拇指末节长期
顶到约90度的错误`thumb_middle`一一对应，不做事后角度裁剪。
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
if str(RUN_DIR) not in sys.path:
    sys.path.insert(0, str(RUN_DIR))

from refine_wuji_pad_contacts import internal_to_saved, saved_to_internal, wuji_model_q  # noqa: E402
from retarget_wuji_keypoints import (  # noqa: E402
    apply_anatomy_profile,
    build_wuji_model,
    load_anatomy_profile,
)


METHOD = "point_baseline_thumb_tip_nullspace_v1"
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


def refine_file(args):
    """读取基线中指定轨迹，执行拇指消歧并保存完整追溯元数据。"""
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
    outputs, all_losses, all_components, all_errors = [], [], [], []
    for source_index in indices:
        result, losses, components, errors = refine_trajectory(
            by_index[source_index], model, lower, upper, args
        )
        outputs.append(result)
        all_losses.append(losses)
        all_components.append(components)
        all_errors.append(errors)
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
        "initial_target": str(args.initial_target.resolve()),
        "initial_target_sha256": hashlib.sha256(args.initial_target.read_bytes()).hexdigest(),
        "optimization_loss_per_frame": np.stack(all_losses),
        "optimization_loss_components_per_frame": all_components,
        "thumb_tip_displacement_m_per_frame": np.stack(all_errors),
        "maxeval": int(args.maxeval),
        "tip_weight": float(args.tip_weight),
        "neutral_weight": float(args.neutral_weight),
        "temporal_weight": float(args.temporal_weight),
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trajectory-indices", type=int, nargs="*")
    parser.add_argument("--maxeval", type=int, default=80)
    parser.add_argument("--tip-weight", type=float, default=1.0)
    parser.add_argument("--neutral-weight", type=float, default=0.02)
    parser.add_argument("--temporal-weight", type=float, default=0.01)
    args = parser.parse_args()
    if args.maxeval < 1:
        parser.error("--maxeval必须为正整数")
    if min(args.tip_weight, args.neutral_weight, args.temporal_weight) < 0:
        parser.error("所有权重必须非负")
    refine_file(args)


if __name__ == "__main__":
    main()
