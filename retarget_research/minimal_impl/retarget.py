"""三只手的最小几何重定向入口。

输入：一条GraspM3源NPY、目标手、轨迹索引和输出路径。
输出：70帧目标手动作NPY。
逻辑：XHand/Wuji最小化15点误差；Linker最小化功能向量误差并在闭合期鼓励夹紧。
作用：不依赖原 ``retargeting`` 模块，独立完成基础任务的轨迹生成核心。
"""

import argparse
from pathlib import Path

import nlopt
import numpy as np
from scipy.spatial import cKDTree
import torch
import trimesh

from .config import HANDS, LINKER_GRIP_TARGET, LINKER_VECTORS
from .data import load_npy, save_candidate
from .kinematics import (
    build_shadow_model, build_target_model, initial_pose, joint_names,
    shadow_keypoints, target_keypoints,
)


def huber_distance(residual, delta):
    """输入三维残差，输出Huber距离；逻辑是近处二次、远处线性，作用是降低离群点影响。"""
    distance = torch.linalg.vector_norm(residual, dim=-1)
    return torch.where(distance <= delta, 0.5 * distance.square(), delta * (distance - 0.5 * delta))


class PointObjective:
    """XHand/Wuji单帧15点目标函数。"""

    def __init__(self, model, hand, target, target_indices, previous=None, temporal_weight=0.0):
        """输入目标点和上一帧，输出可重复调用的目标对象；作用是供SLSQP查询损失与梯度。"""
        self.model = model
        self.hand = hand
        self.target = torch.as_tensor(target, dtype=torch.float32)
        self.indices = np.asarray(target_indices, dtype=np.int64)
        self.previous = None if previous is None else torch.as_tensor(previous, dtype=torch.float32)
        self.temporal_weight = float(temporal_weight)
        self.joint_count = len(joint_names(model))

    def __call__(self, values, gradient=None):
        """输入内部姿态，输出15点均方距离；逻辑中用PyTorch反传，作用是给NLopt梯度。"""
        value = torch.tensor(np.asarray(values, dtype=np.float32), requires_grad=True)
        j = self.joint_count
        points = target_keypoints(self.model, self.hand, value[:j], value[j:j + 3], value[j + 3:j + 6])
        loss = torch.mean(torch.sum((points[self.indices] - self.target) ** 2, dim=1)) * 1000.0
        if self.previous is not None:
            loss = loss + self.temporal_weight * torch.mean((value[:j] - self.previous[:j]) ** 2)
        if gradient is not None and len(gradient):
            loss.backward()
            gradient[:] = value.grad.detach().numpy().astype(np.float64)
        return float(loss.detach())


class LinkerObjective:
    """Linker O6单帧功能向量目标函数。"""

    def __init__(self, model, source_points, scales, previous, grip_alpha, contact_anchor=None):
        """输入源点、骨长比例、上一帧、闭合程度和接触锚，输出功能向量目标对象。"""
        self.model = model
        self.source = torch.as_tensor(source_points, dtype=torch.float32)
        self.scales = torch.as_tensor(scales, dtype=torch.float32)
        self.previous = None if previous is None else torch.as_tensor(previous, dtype=torch.float32)
        self.grip_alpha = float(grip_alpha)
        self.contact_anchor = contact_anchor

    def __call__(self, values, gradient=None):
        """输入12维内部姿态并输出损失；逻辑是位置向量、方向、掌心和夹紧项加权求和。"""
        value = torch.tensor(np.asarray(values, dtype=np.float32), requires_grad=True)
        points = target_keypoints(self.model, "linker", value[:6], value[6:9], value[9:12])
        loss = 1000.0 * torch.sum((points[0] - self.source[0]) ** 2)
        for index, (kind, so, st, to, tt, weight) in enumerate(LINKER_VECTORS):
            source_vector = self.source[st] - self.source[so]
            target_vector = points[tt] - points[to]
            if kind == "position":
                alpha = self.grip_alpha if index < 5 else 1.0
                loss = loss + weight * (0.6 + 0.4 * alpha) * huber_distance(
                    target_vector - source_vector * self.scales[index], 0.02)
            else:
                source_unit = source_vector / (torch.linalg.vector_norm(source_vector) + 1e-8)
                target_unit = target_vector / (torch.linalg.vector_norm(target_vector) + 1e-8)
                loss = loss + weight * (0.5 + 0.5 * self.grip_alpha) * huber_distance(
                    target_unit - source_unit, 0.5)
        loss = loss + 0.0025 * torch.mean(value[:6].square())
        if self.previous is not None:
            loss = loss + 0.01 * torch.mean((value[:6] - self.previous[:6]) ** 2)
        if self.contact_anchor is not None:
            tip_indices, anchors = self.contact_anchor
            anchors = torch.as_tensor(anchors, dtype=torch.float32)
            loss = loss + 3.0 * torch.mean(
                torch.sum((points[np.asarray(tip_indices)] - anchors) ** 2, dim=1)
            ) * 1000.0
        grip_target = torch.as_tensor(LINKER_GRIP_TARGET, dtype=torch.float32)
        loss = loss + 8.0 * self.grip_alpha * torch.mean(torch.clamp(grip_target - value[:6], min=0).square())
        if gradient is not None and len(gradient):
            loss.backward()
            gradient[:] = value.grad.detach().numpy().astype(np.float64)
        return float(loss.detach())


def solve(objective, start, lower, upper, maxeval):
    """运行一次有界SLSQP。

    输入：目标函数、初值、上下界和最大求值次数。
    输出：优化后的姿态。
    逻辑：NLopt利用目标函数提供的自动微分梯度迭代。
    作用：把空间关键点误差转换为目标手关节角。
    """
    optimizer = nlopt.opt(nlopt.LD_SLSQP, len(start))
    optimizer.set_min_objective(objective)
    optimizer.set_lower_bounds(lower.tolist())
    optimizer.set_upper_bounds(upper.tolist())
    optimizer.set_maxeval(int(maxeval))
    optimizer.set_xtol_rel(1e-6)
    return np.asarray(optimizer.optimize(np.clip(start, lower + 1e-6, upper - 1e-6)), dtype=np.float32)


def vector_scales(shadow_model, target_model):
    """计算Shadow与Linker零姿态骨段比例。

    输入：两只手的运动学模型。
    输出：15个向量长度比例。
    逻辑：目标零姿态长度除以源零姿态长度。
    作用：避免把Shadow较长手指的绝对长度强加给Linker。
    """
    source = shadow_keypoints(np.zeros((1, 28), dtype=np.float32), shadow_model)[0]
    with torch.no_grad():
        target = target_keypoints(target_model, "linker", torch.zeros(6), torch.zeros(3), torch.zeros(3)).numpy()
    return np.asarray([
        np.linalg.norm(target[tt] - target[to]) / np.linalg.norm(source[st] - source[so])
        for _, so, st, to, tt, _ in LINKER_VECTORS
    ], dtype=np.float32)


def grip_progress(points):
    """估计每帧闭合程度。

    输入：Shadow 21点轨迹。
    输出：0到1的逐帧系数。
    逻辑：用拇指到四指尖平均距离相对首帧和最小值线性归一化。
    作用：张手阶段重视姿态，闭合阶段逐渐加强指尖和夹紧项。
    """
    distance = np.mean(np.linalg.norm(points[:, [4, 8, 12, 16]] - points[:, 20:21], axis=2), axis=1)
    denominator = max(float(distance[0] - distance.min()), 1e-6)
    return np.clip((distance[0] - distance) / denominator, 0.0, 1.0)


def object_vertices(object_dir, scale, rotation):
    """恢复物体世界表面。

    输入：COACD物体目录、缩放和旋转。
    输出：离地5 mm后的世界顶点。
    逻辑：与Isaac初始化相同地旋转、缩放和平移网格。
    作用：让Linker闭合阶段的接触锚与Isaac物体初始姿态一致。
    """
    mesh = trimesh.load_mesh(Path(object_dir) / "coacd/decomposed.obj", process=False)
    vertices = np.asarray(mesh.vertices) @ np.asarray(rotation).T * float(scale)
    vertices[:, 2] += 0.005 - vertices[:, 2].min()
    return vertices.astype(np.float32)


def linker_contact_plan(source_points, source_frames, vertices):
    """生成Linker闭合阶段的物体接触锚。

    输入：Shadow关键点、源动作和物体表面。
    输出：长度70的列表；闭合帧含目标Linker指尖索引与表面点。
    逻辑：首次至少两指距物体≤2 cm为闭合开始，腕部上升3 cm为抬升开始；无阈值帧时取第二近距离最小帧。
    作用：防止功能向量方向正确但五指没有真正贴住物体。
    """
    source_tip_indices = np.asarray([4, 8, 12, 16, 20])
    linker_tip_indices = np.asarray([3, 6, 9, 12, 14])
    tips = source_points[:, source_tip_indices]
    distances, nearest = cKDTree(vertices).query(tips, k=1)
    candidates = np.flatnonzero((distances <= 0.02).sum(1) >= 2)
    close = int(candidates[0]) if len(candidates) else int(np.argmin(np.partition(distances, 1, axis=1)[:, 1]))
    base_z = float(np.min(source_frames[close:, 2]))
    lift_candidates = np.flatnonzero(
        (np.arange(len(source_frames)) >= close) & (source_frames[:, 2] >= base_z + 0.03)
    )
    lift = int(lift_candidates[0]) if len(lift_candidates) else len(source_frames)
    plan = [None] * len(source_frames)
    for frame in range(close, lift):
        active = np.flatnonzero(distances[frame] <= 0.02)
        if frame == close and len(active) < 2:
            active = np.argsort(distances[frame])[:2]
        if len(active):
            plan[frame] = (
                linker_tip_indices[active], vertices[nearest[frame, active]],
            )
    return plan


def retarget_trajectory(
    hand, source_frames, maxeval=50, source_z_offset=0.4, temporal_weight=0.0,
    object_dir=None, scale=None, rotation=None,
):
    """重定向一条完整轨迹。

    输入：手名、``(70,28)``源动作和优化参数。
    输出：``(70,12/18/26)``目标动作及逐帧损失。
    逻辑：每帧独立建SLSQP，以上一帧解热启动；保存时改成手腕在前。
    作用：这是基础任务最核心的“Shadow动作翻译器”。
    """
    spec = HANDS[hand]
    frames = np.asarray(source_frames, dtype=np.float32).copy()
    frames[:, 2] += float(source_z_offset)
    shadow_model = build_shadow_model()
    target_model = build_target_model(hand)
    source_all = shadow_keypoints(frames, shadow_model)
    lower_j = target_model.revolute_joints_q_lower[0].detach().numpy()
    upper_j = target_model.revolute_joints_q_upper[0].detach().numpy()
    lower = np.concatenate([lower_j, np.full(3, -2.0), np.full(3, -np.pi)])
    upper = np.concatenate([upper_j, np.full(3, 2.0), np.full(3, np.pi)])
    scales = vector_scales(shadow_model, target_model) if hand == "linker" else None
    progress = grip_progress(source_all) if hand == "linker" else None
    contact_plan = None
    if hand == "linker" and object_dir is not None:
        contact_plan = linker_contact_plan(
            source_all, frames, object_vertices(object_dir, scale, rotation),
        )
    previous, outputs, losses = None, [], []
    for frame_index, source_frame in enumerate(frames):
        start = initial_pose(source_frame, spec.finger_dim) if previous is None else previous
        if hand == "linker":
            objective = LinkerObjective(
                target_model, source_all[frame_index], scales, previous,
                progress[frame_index], None if contact_plan is None else contact_plan[frame_index],
            )
        else:
            target = source_all[frame_index, np.asarray(spec.source_indices)]
            objective = PointObjective(target_model, hand, target, spec.target_indices, previous, temporal_weight)
        previous = solve(objective, start, lower, upper, maxeval)
        internal = previous
        outputs.append(np.concatenate([internal[spec.finger_dim:], internal[:spec.finger_dim]]))
        losses.append(objective(previous))
        if (frame_index + 1) % 10 == 0:
            print(f"frames={frame_index + 1}/70 loss={losses[-1]:.6f}", flush=True)
    output = np.stack(outputs).astype(np.float32)
    metadata = {"wuji_joint_names": joint_names(target_model)} if hand == "wuji" else {}
    metadata["linker_contact_anchor_weight"] = 3.0 if hand == "linker" and object_dir else 0.0
    return output, np.asarray(losses), metadata


def main():
    """解析单文件命令并保存结果；输入来自CLI，输出为候选NPY和终端进度。"""
    parser = argparse.ArgumentParser(description="三种目标手的最小重定向")
    parser.add_argument("--hand", choices=tuple(HANDS), required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--indices", type=int, nargs="+", default=[0])
    parser.add_argument("--maxeval", type=int, default=50)
    parser.add_argument("--source-z-offset", type=float, default=0.4)
    parser.add_argument("--temporal-weight", type=float, default=0.0)
    parser.add_argument("--object-dir", help="Linker闭合阶段接触锚所需的物体目录")
    args = parser.parse_args()
    source = load_npy(args.source)
    outputs, losses, metadata = [], [], {}
    for index in args.indices:
        frames, frame_losses, metadata = retarget_trajectory(
            args.hand, source["grasp_seqs"][index], args.maxeval,
            args.source_z_offset, args.temporal_weight,
            args.object_dir, source["obj_scale"][index], source["obj_rotmat"][index],
        )
        outputs.append(frames)
        losses.append(frame_losses)
    metadata.update({"hand": args.hand, "optimization_loss": np.stack(losses)})
    save_candidate(args.output, np.stack(outputs), source, args.indices, metadata)
    print(f"output={args.output} shape={np.stack(outputs).shape}")


if __name__ == "__main__":
    main()
