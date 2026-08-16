#!/usr/bin/env python3
"""用DexPilot式掌心/指尖功能向量把Shadow轨迹重定向到Wuji。

输入：Shadow轨迹、12个功能向量配置、Wuji解剖边界和SLSQP预算。
输出：标准`(N,70,26)`Wuji轨迹、逐帧向量损失及完整方法元数据。
内部逻辑：向量按两手零姿态长度比自动缩放；每帧优化掌心位置、功能向量、
中性手型和上一帧连续性，不匹配任何内部指骨绝对位置。
作用：作为当前15个绝对关键点法的底层替代对照，检验跨形态功能关系是否更可靠。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import nlopt
import numpy as np
import torch

from retarget_wuji_keypoints import (
    RETARGET_ROOT,
    apply_anatomy_profile,
    build_shadow_model,
    build_wuji_model,
    clip_start_to_bounds,
    initial_values,
    load_anatomy_profile,
    robust_compute_orth6d_from_eulerXYZ,
    shadow_keypoints,
)


DEFAULT_VECTOR_CONFIG = RETARGET_ROOT / "configs" / "wuji_functional_vectors_v1.json"


def load_vector_config(path):
    """读取并校验12个功能向量及其损失超参数。

    输入：JSON路径。
    输出：解析字典、绝对路径和原始文件SHA-256。
    内部逻辑：检查索引范围、正权重、Huber阈值和两类正则权重。
    作用：让向量定义与数值选择可审查、可安全续跑。
    """
    resolved = Path(path).resolve()
    raw = resolved.read_bytes()
    config = json.loads(raw.decode("utf-8"))
    pairs = config.get("pairs", [])
    if len(pairs) != 12:
        raise ValueError("Wuji功能向量v1必须恰好包含12对")
    semantics = [item["semantic"] for item in pairs]
    if len(set(semantics)) != len(semantics):
        raise ValueError("功能向量语义名称不能重复")
    for item in pairs:
        if not 0 <= int(item["shadow_origin"]) < 21 or not 0 <= int(item["shadow_task"]) < 21:
            raise ValueError(f"Shadow向量索引越界: {item}")
        if not 0 <= int(item["wuji_origin"]) < 26 or not 0 <= int(item["wuji_task"]) < 26:
            raise ValueError(f"Wuji向量索引越界: {item}")
        if not np.isfinite(float(item["weight"])) or float(item["weight"]) <= 0:
            raise ValueError("功能向量权重必须为有限正数")
    for key in ("huber_delta_m", "neutral_joint_weight", "previous_joint_weight"):
        value = float(config[key])
        if not np.isfinite(value) or value < 0 or (key == "huber_delta_m" and value == 0):
            raise ValueError(f"{key}必须是有效的非负值")
    return config, str(resolved), hashlib.sha256(raw).hexdigest()


def wuji_points(model, joints, translation=None, euler=None):
    """输入Wuji关节及可选手腕位姿，输出26个世界关键点张量。"""
    joints = joints.view(1, -1)
    translation = torch.zeros((1, 3), dtype=joints.dtype) if translation is None else translation.view(1, 3)
    euler = torch.zeros((1, 3), dtype=joints.dtype) if euler is None else euler.view(1, 3)
    rotation = robust_compute_orth6d_from_eulerXYZ(euler)
    q = torch.cat([translation, rotation, joints], dim=1)
    return model.get_penetraion_keypoints(q=q)[0]


def zero_pose_scales(shadow_model, wuji_model, pairs):
    """按两只手零姿态对应向量长度比计算12个固定形态尺度。

    输入：两手运动学模型和向量定义。
    输出：形状`(12,)`的目标/源长度比例。
    内部逻辑：只使用零关节姿态，拒绝退化向量，不读取任何成功率或物体。
    作用：补偿手掌和手指尺寸差异，同时避免逐轨迹人工调scale。
    """
    source = shadow_keypoints(np.zeros((1, 28), dtype=np.float32), shadow_model)[0]
    joint_count = len(wuji_model.robot.get_joint_parameter_names())
    with torch.no_grad():
        target = wuji_points(wuji_model, torch.zeros(joint_count, dtype=torch.float32)).cpu().numpy()
    scales = []
    for item in pairs:
        source_length = np.linalg.norm(source[int(item["shadow_task"])] - source[int(item["shadow_origin"])])
        target_length = np.linalg.norm(target[int(item["wuji_task"])] - target[int(item["wuji_origin"])])
        if source_length <= 1e-6 or target_length <= 1e-6:
            raise ValueError(f"零姿态功能向量退化: {item['semantic']}")
        scales.append(target_length / source_length)
    return np.asarray(scales, dtype=np.float64)


def huber_norm(residual, delta):
    """输入`(N,3)`残差和阈值，输出逐向量Huber距离。"""
    distance = torch.linalg.vector_norm(residual, dim=1)
    delta_tensor = distance.new_tensor(float(delta))
    return torch.where(
        distance <= delta_tensor,
        0.5 * distance * distance,
        delta_tensor * (distance - 0.5 * delta_tensor),
    )


class WujiVectorObjective:
    """计算单帧掌心锚定、功能向量、自然手型和连续性损失。"""

    def __init__(
        self,
        model,
        target_palm,
        target_vectors,
        pairs,
        config,
        previous_values=None,
        flexion_couplings=None,
    ):
        """保存一帧目标和固定索引，供NLopt反复求值及自动求导。"""
        self.model = model
        self.target_palm = torch.as_tensor(target_palm, dtype=torch.float32)
        self.target_vectors = torch.as_tensor(target_vectors, dtype=torch.float32)
        self.origin_indices = np.asarray([item["wuji_origin"] for item in pairs], dtype=np.int64)
        self.task_indices = np.asarray([item["wuji_task"] for item in pairs], dtype=np.int64)
        self.weights = torch.as_tensor([item["weight"] for item in pairs], dtype=torch.float32)
        self.huber_delta = float(config["huber_delta_m"])
        self.neutral_weight = float(config["neutral_joint_weight"])
        self.previous_weight = float(config["previous_joint_weight"])
        self.previous = None if previous_values is None else torch.as_tensor(previous_values, dtype=torch.float32)
        self.joint_count = len(model.robot.get_joint_parameter_names())
        self.flexion_couplings = list(flexion_couplings or [])

    def __call__(self, values, gradient=None):
        """输入20关节+6手腕候选，输出可导标量损失并可回填NLopt梯度。"""
        value = torch.tensor(np.asarray(values, dtype=np.float32), requires_grad=True)
        joints = value[: self.joint_count]
        translation = value[self.joint_count : self.joint_count + 3]
        euler = value[self.joint_count + 3 : self.joint_count + 6]
        points = wuji_points(self.model, joints, translation, euler)
        vectors = points[self.task_indices] - points[self.origin_indices]
        vector_loss = torch.mean(self.weights * huber_norm(vectors - self.target_vectors, self.huber_delta)) * 1000.0
        palm_loss = torch.sum((points[0] - self.target_palm) ** 2) * 1000.0
        loss = vector_loss + palm_loss + self.neutral_weight * torch.mean(joints * joints)
        if self.previous is not None:
            loss = loss + self.previous_weight * torch.mean((joints - self.previous[: self.joint_count]) ** 2)
        for coupling in self.flexion_couplings:
            proximal = joints[coupling["proximal_index"]]
            distal = joints[coupling["distal_index"]]
            expected = coupling["ratio"] * proximal + coupling["offset_rad"]
            loss = loss + coupling["weight"] * (distal - expected) ** 2
        if gradient is not None and len(gradient) > 0:
            loss.backward()
            gradient[:] = value.grad.detach().numpy().astype(np.float64)
        return float(loss.detach())


def retarget_vector_trajectory(source_frames, source_points, model, pairs, scales, config, maxeval, translation_bound, anatomy_profile, progress_prefix=None):
    """逐帧优化一条Shadow轨迹并输出内部顺序Wuji轨迹和向量损失。"""
    joint_names = list(model.robot.get_joint_parameter_names())
    urdf_lower = model.revolute_joints_q_lower[0].detach().numpy()
    urdf_upper = model.revolute_joints_q_upper[0].detach().numpy()
    lower_joints, upper_joints, couplings = apply_anatomy_profile(
        joint_names, urdf_lower, urdf_upper, anatomy_profile
    )
    lower = np.concatenate([lower_joints, np.full(3, -translation_bound), np.full(3, -np.pi)])
    upper = np.concatenate([upper_joints, np.full(3, translation_bound), np.full(3, np.pi)])
    source_origin = np.asarray([item["shadow_origin"] for item in pairs], dtype=np.int64)
    source_task = np.asarray([item["shadow_task"] for item in pairs], dtype=np.int64)
    outputs, losses, previous = [], [], None
    for frame_index, frame in enumerate(source_frames):
        target_vectors = (source_points[frame_index, source_task] - source_points[frame_index, source_origin]) * scales[:, None]
        start = initial_values(frame, len(joint_names)) if previous is None else previous.copy()
        start = clip_start_to_bounds(start, lower, upper)
        objective = WujiVectorObjective(
            model, source_points[frame_index, 0], target_vectors, pairs, config,
            previous_values=previous, flexion_couplings=couplings,
        )
        optimizer = nlopt.opt(nlopt.LD_SLSQP, len(start))
        optimizer.set_min_objective(objective)
        optimizer.set_lower_bounds(lower.tolist())
        optimizer.set_upper_bounds(upper.tolist())
        optimizer.set_maxeval(int(maxeval))
        optimizer.set_xtol_rel(1e-6)
        optimizer.set_ftol_rel(1e-8)
        try:
            result = optimizer.optimize(start)
        except (nlopt.RoundoffLimited, RuntimeError):
            result = start
        previous = np.asarray(result, dtype=np.float32)
        outputs.append(previous)
        losses.append(objective(previous))
        if progress_prefix and ((frame_index + 1) % 10 == 0 or frame_index + 1 == len(source_frames)):
            print(f"{progress_prefix}: frames={frame_index + 1}/{len(source_frames)} loss={losses[-1]:.6f}", flush=True)
    return np.stack(outputs), np.asarray(losses, dtype=np.float32)


def retarget_file(args):
    """读取Shadow文件、计算固定形态尺度、运行向量优化并保存标准Wuji候选。"""
    source_data = np.load(args.source, allow_pickle=True).item()
    indices = args.trajectory_indices or [0]
    source_frames = np.asarray(source_data["grasp_seqs"][indices], dtype=np.float32).copy()
    source_frames[:, :, 2] += args.source_z_offset
    shadow_model, wuji_model = build_shadow_model(), build_wuji_model()
    joint_names = list(wuji_model.robot.get_joint_parameter_names())
    vector_config, vector_path, vector_sha = load_vector_config(args.vector_config)
    pairs = vector_config["pairs"]
    scales = zero_pose_scales(shadow_model, wuji_model, pairs)
    urdf_lower = wuji_model.revolute_joints_q_lower[0].detach().numpy()
    urdf_upper = wuji_model.revolute_joints_q_upper[0].detach().numpy()
    _, _, _, anatomy_path, anatomy_sha = load_anatomy_profile(
        args.anatomy_config, joint_names, urdf_lower, urdf_upper
    )
    anatomy_profile = {} if args.anatomy_config is None else json.loads(args.anatomy_config.read_text(encoding="utf-8"))
    outputs, all_losses = [], []
    for trajectory_number, trajectory in enumerate(source_frames):
        points = shadow_keypoints(trajectory, shadow_model)
        internal, losses = retarget_vector_trajectory(
            trajectory, points, wuji_model, pairs, scales, vector_config,
            args.maxeval, args.translation_bound, anatomy_profile,
            progress_prefix=f"trajectory={trajectory_number + 1}/{len(source_frames)} source_index={indices[trajectory_number]}",
        )
        outputs.append(np.concatenate([internal[:, 20:26], internal[:, :20]], axis=1))
        all_losses.append(losses)
    output_frames = np.stack(outputs).astype(np.float32)
    loss_frames = np.stack(all_losses).astype(np.float32)
    output = {
        "grasp_seqs": output_frames,
        "optimization_loss_per_frame": loss_frames,
        "source_trajectory_indices": np.asarray(indices, dtype=np.int64),
        "obj_rotmat": np.asarray(source_data["obj_rotmat"])[indices],
        "obj_scale": np.asarray(source_data["obj_scale"])[indices],
        "retarget_method": "dexpilot_style_functional_vectors_v1",
        "vector_config": vector_path,
        "vector_config_sha256": vector_sha,
        "vector_scales": scales.astype(np.float32),
        "mapping_semantics": [item["semantic"] for item in pairs],
        "wuji_joint_names": joint_names,
        "anatomy_config": anatomy_path,
        "anatomy_config_sha256": anatomy_sha,
        "source_z_offset": float(args.source_z_offset),
        "maxeval": int(args.maxeval),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, output, allow_pickle=True)
    print(f"trajectories={len(output_frames)}")
    print(f"output_shape={output_frames.shape}")
    print(f"mean_vector_loss={loss_frames.mean():.6f}")
    print(f"output={args.output}")


def main():
    """解析单文件向量重定向参数并调用标准保存流程。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trajectory-indices", type=int, nargs="*")
    parser.add_argument("--maxeval", type=int, default=50)
    parser.add_argument("--translation-bound", type=float, default=2.0)
    parser.add_argument("--source-z-offset", type=float, default=0.4)
    parser.add_argument("--vector-config", type=Path, default=DEFAULT_VECTOR_CONFIG)
    parser.add_argument("--anatomy-config", type=Path)
    retarget_file(parser.parse_args())


if __name__ == "__main__":
    main()
