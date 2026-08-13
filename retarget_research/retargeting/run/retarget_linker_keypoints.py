#!/usr/bin/env python3
"""用校准后的物理关键点把一条Shadow轨迹重定向到Linker外形目标手。

输入：包含`grasp_seqs`的Shadow `.npy`、轨迹索引和Linker语义配置。
输出：O6模式为`(N,70,12)`，解耦增强模式为`(N,70,17)`，并保存逐帧loss。
内部逻辑：每帧共同优化手腕6维和6/11个手指关节；O6模式保留真实mimic联动，
解耦模式把5个从动关节作为独立变量，用于验证机械耦合是否是失败主因。
作用：保留真实O6基线，同时建立“相同外形、更高可控自由度”的明确消融上限。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import nlopt
import numpy as np
from scipy.spatial import cKDTree
import torch
import transforms3d


RETARGET_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = RETARGET_ROOT.parent
REFERENCE_SCRIPTS = PROJECT_ROOT / "reference" / "HandRetargetTask2026" / "scripts"
OBJECT_ROOT = REFERENCE_SCRIPTS / "data" / "sorting" / "object_41"
THIRD_PARTY_PK = PROJECT_ROOT / "reference" / "HandRetargetTask2026" / "third_party" / "pytorch_kinematics"
PREPARE_DIR = RETARGET_ROOT / "prepare"
for path in (REFERENCE_SCRIPTS, THIRD_PARTY_PK, PREPARE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from utils.hand_model import HandModel as ShadowHandModel  # noqa: E402
from utils.HandModel_linkerhand import HandModel_Linkerhand  # noqa: E402
from utils.HandModel_xhand import HandModel_xhand  # noqa: E402
from utils.rot6d import robust_compute_orth6d_from_eulerXYZ  # noqa: E402
from object_geometry import (  # noqa: E402
    transformed_object_surface,
    transformed_object_vertices,
)
try:
    from .phase_contact import (  # noqa: E402
        build_phase_contact_plan,
        infer_motion_phases,
        load_pad_config,
        pad_contact_terms,
        world_pad_regions,
    )
except ImportError:
    from phase_contact import (  # noqa: E402
        build_phase_contact_plan,
        infer_motion_phases,
        load_pad_config,
        pad_contact_terms,
        world_pad_regions,
    )


R_ALIGN = np.array(
    [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
    dtype=np.float32,
)
WRIST_OFFSET = np.array([0.003, 0.002, -0.01], dtype=np.float32)
SOURCE_TIP_INDICES = {
    "index": 4,
    "middle": 8,
    "ring": 12,
    "little": 16,
    "thumb": 20,
}


def build_shadow_model():
    """创建CPU上的Shadow运动学模型。

    输入：无显式参数；读取参考MJCF、mesh和关键点文件。
    输出：启用21关键点的Shadow模型。
    逻辑：关闭当前优化不需要的表面采样，仅保留运动学和匹配点。
    作用：把28维源轨迹转换成每帧世界坐标关键点目标。
    """
    base = REFERENCE_SCRIPTS / "assets" / "mjcf_free"
    return ShadowHandModel(
        mjcf_path=str(base / "shadow_hand_vis_new.xml"),
        mesh_path=str(base / "meshes"),
        contact_points_path=str(base / "contact_points.json"),
        penetration_points_path=str(base / "penetration_points.json"),
        n_surface_points=0,
        device="cpu",
        use_joint21=True,
    )


def build_linker_model(joint_mode="coupled6"):
    """按实验模式创建Linker运动学模型。

    输入：`coupled6`或`independent11`，并读取同一套右手URDF和mesh。
    输出：分别暴露6个主动关节或全部11个转动关节的运动学模型。
    逻辑：6维模式使用参考mimic展开；11维模式直接使用通用URDF模型，
    pytorch-kinematics仍按同一Linker连杆树做正向运动学，但不再强制固定倍率。
    作用：只改变可控关节数量，不改变掌形、指长、关键点或物体数据。
    """
    if joint_mode not in {"coupled6", "independent11"}:
        raise ValueError(f"未知Linker关节模式: {joint_mode}")
    asset = REFERENCE_SCRIPTS / "assets" / "linkerhand" / "o6" / "right"
    model_class = (
        HandModel_Linkerhand if joint_mode == "coupled6" else HandModel_xhand
    )
    return model_class(
        robot_name="linkerhand",
        urdf_filename="linkerhand_o6_right.urdf",
        mesh_path="",
        batch_size=1,
        device=torch.device("cpu"),
        mesh_nsp=128,
        hand_scale=1.0,
        asset_dir=str(asset),
        allow_missing_contacts=True,
    )


def model_joint_count(model):
    """返回当前Linker模型向优化器暴露的手指关节数。

    输入：已构建的6维耦合模型或11维解耦模型。
    输出：整数6或11。
    逻辑：直接读取模型关节下界的列数，避免在损失函数中散落硬编码。
    作用：让同一条优化、保存和时序代码安全支持两种实验模式。
    """
    return int(model.revolute_joints_q_lower.shape[1])


def reachable_pad_geometry_from_saved_frame(model, pad_config, saved_frame):
    """计算一帧O6基线姿态下五个真实指腹的世界点和法向。

    输入：Linker运动学模型、校准指腹配置和`[手腕6, 手指J]`保存帧。
    输出：五指到`(P,3)`世界点和单位法向NumPy数组的字典。
    内部逻辑：把欧拉角转为模型所需6D旋转，更新正向运动学，再调用共享指腹变换。
    作用：向接触区域选择器提供目标手真实可达位置，而不是继续使用Shadow指尖位置。
    """
    frame = np.asarray(saved_frame, dtype=np.float32)
    joint_count = model_joint_count(model)
    if frame.shape != (joint_count + 6,):
        raise ValueError(f"基线帧应为{joint_count + 6}维，实际为{frame.shape}")
    translation = torch.as_tensor(frame[:3]).view(1, 3)
    euler = torch.as_tensor(frame[3:6]).view(1, 3)
    rotation = robust_compute_orth6d_from_eulerXYZ(euler)
    joints = torch.as_tensor(frame[6:]).view(1, joint_count)
    model.update_kinematics(torch.cat([translation, rotation, joints], dim=1))
    pads = world_pad_regions(model, pad_config)
    return {
        semantic: (
            points.detach().cpu().numpy(),
            normals.detach().cpu().numpy(),
        )
        for semantic, (points, normals) in pads.items()
    }


def load_pairs(include_thumb_middle: bool, include_finger_middle: bool = False):
    """读取本次优化启用的Shadow↔Linker物理点对。

    输入：是否加入拇指中段，以及是否加入四根普通手指中段。
    输出：参考10点、拇指11点或稠密15点的pair字典列表。
    逻辑：始终保留`use_in_reference_baseline=true`的10对；普通指中段开启时同时强制加入拇指中段。
    作用：在不改变Linker六个主动量的前提下，建立可审计的10/11/15点密度消融。
    """
    config = json.loads(
        (RETARGET_ROOT / "configs" / "linker_o6_keypoint_map.json").read_text()
    )
    include_thumb_middle = include_thumb_middle or include_finger_middle
    return [
        pair
        for pair in config["pairs"]
        if pair["use_in_reference_baseline"]
        or (include_thumb_middle and pair["semantic"] == "thumb_middle")
        or (include_finger_middle and pair.get("use_in_dense15", False))
    ]


def semantic_weights(
    pairs,
    frame_index,
    contact_start_frame,
    late_tip_weight,
    late_thumb_weight,
    late_structure_weight,
):
    """为接触阶段生成按语义区分的关键点权重。

    输入：点对、当前帧、接触起始帧，以及普通指尖/拇指/结构点权重。
    输出：长度与pair数相同的正浮点数组。
    逻辑：接触前全部为1；接触后拇指、其他指尖和掌心/近端分别赋权。
    作用：让闭合抬升阶段优先保住决定抓取的指尖接触，而非只降平均误差。
    """
    if contact_start_frame < 0 or frame_index < contact_start_frame:
        return np.ones(len(pairs), dtype=np.float32)
    weights = []
    for pair in pairs:
        semantic = pair["semantic"]
        if semantic.startswith("thumb"):
            weight = late_thumb_weight
        elif semantic.endswith("_tip"):
            weight = late_tip_weight
        else:
            weight = late_structure_weight
        weights.append(float(weight))
    weights = np.asarray(weights, dtype=np.float32)
    if np.any(weights <= 0):
        raise ValueError("所有关键点权重必须大于0")
    return weights


def source_tip_contact_mask(
    target_points,
    pairs,
    object_vertices,
    distance_threshold,
):
    """判断Shadow专家中哪些指尖在每帧接近物体。

    输入：`(T,M,3)`源关键点、点对、物体顶点和接触距离阈值。
    输出：`(T,M)`布尔数组；只有进入阈值的`*_tip`位置为True。
    逻辑：用物体顶点KD-tree查询每个Shadow指尖逐帧最近距离。
    作用：同时支持源接触点加权和“只借用接触阶段”的目标手表面约束。
    """
    target_points = np.asarray(target_points, dtype=np.float32)
    if target_points.ndim != 3 or target_points.shape[1:] != (len(pairs), 3):
        raise ValueError("target_points必须为(T, pair数量, 3)")
    if distance_threshold < 0:
        raise ValueError("接触距离阈值必须大于等于0")
    tree = cKDTree(np.asarray(object_vertices, dtype=np.float32))
    mask = np.zeros(target_points.shape[:2], dtype=bool)
    for pair_index, pair in enumerate(pairs):
        if not pair["semantic"].endswith("_tip"):
            continue
        distances, _ = tree.query(target_points[:, pair_index, :], k=1)
        mask[:, pair_index] = distances <= distance_threshold
    return mask


def source_contact_point_weights(
    target_points,
    pairs,
    object_vertices,
    distance_threshold,
    contact_point_weight,
):
    """根据Shadow专家指尖是否接近物体，生成逐帧关键点权重。

    输入：`(T,M,3)`源关键点、点对、物体顶点、接触距离阈值和加权值。
    输出：`(T,M)`权重；非指尖或未接近物体的点保持1。
    逻辑：复用源接触掩码，把True位置从1替换为指定权重。
    作用：实现“精确复制源手接触点”的消融，并与目标手自主接触方法区分。
    """
    if contact_point_weight <= 0:
        raise ValueError("接触点权重必须大于0")
    mask = source_tip_contact_mask(
        target_points, pairs, object_vertices, distance_threshold
    )
    weights = np.ones(mask.shape, dtype=np.float32)
    weights[mask] = contact_point_weight
    return weights


def shadow_keypoints(frames, model):
    """批量计算Shadow轨迹的21个世界坐标关键点。

    输入：`(T,28)`源轨迹和Shadow模型。
    输出：`(T,21,3)` NumPy数组。
    逻辑：把欧拉角转换成旋转6D，拼成31维模型参数后做正向运动学。
    作用：为每个优化帧提供已知的几何目标，而不需要目标手关节标签。
    """
    source = torch.as_tensor(frames, dtype=torch.float32)
    q = torch.zeros((len(source), 31), dtype=torch.float32)
    q[:, :3] = source[:, :3]
    q[:, 3:9] = robust_compute_orth6d_from_eulerXYZ(source[:, 3:6])
    q[:, 9:] = source[:, 6:]
    model.set_parameters(q)
    return model.get_penetraion_keypoints().detach().cpu().numpy()


def linker_world_points(model, pairs):
    """取得当前Linker姿态下所有配置点的世界坐标。

    输入：已更新运动学的Linker模型和启用的语义pair列表。
    输出：形状`(M,3)`的PyTorch张量，并保留对关节/手腕的梯度。
    逻辑：逐点应用link局部变换，再应用手腕全局旋转、平移和尺度。
    作用：把物理link标志转换成可与Shadow目标直接比较的坐标。
    """
    points = []
    for pair in pairs:
        local = torch.tensor(
            pair["linker_local_xyz"], dtype=torch.float32, device=model.device
        ).view(1, 1, 3)
        hand_point = model.current_status[pair["linker_link"]].transform_points(local)[0, 0]
        world = hand_point @ model.global_rotation[0].T + model.global_translation[0]
        points.append(world * float(model.scale))
    return torch.stack(points, dim=0)


class LinkerFrameObjective:
    """单帧Linker物理关键点与可选时间连续性目标。"""

    def __init__(
        self,
        model,
        pairs,
        target_world,
        point_weights=None,
        previous_values=None,
        joint_temporal_weight=0.0,
        translation_temporal_weight=0.0,
        rotation_temporal_weight=0.0,
        surface_vertices=None,
        surface_contact_weight=0.0,
        pad_config=None,
        contact_targets=None,
        object_surface=None,
        phase_contact_weight=0.0,
        phase_normal_weight=0.0,
        phase_penetration_weight=0.0,
        contact_offset=-0.001,
        min_signed_distance=-0.003,
        grasp_reference_values=None,
        joint_hold_weight=0.0,
        joint_prior_values=None,
        joint_prior_weight=0.0,
    ):
        """保存单帧优化不变的几何目标和时间连续性设置。

        输入：模型、pair、Shadow目标、点权重、上一帧姿态、连续性、旧表面项，
        以及可选物理指腹/逐指物体区域/穿透/抓形保持设置。
        输出：无返回值；初始化可被NLopt重复调用的目标对象。
        逻辑：把不参与求导的目标与上一帧值转成张量并保存。
        作用：同一实现既能复现逐帧基线，也能实验我们的时序正则改进。
        """
        self.model = model
        self.joint_count = model_joint_count(model)
        self.pairs = pairs
        self.target = torch.as_tensor(target_world, dtype=torch.float32)
        self.point_weights = torch.as_tensor(
            np.ones(len(pairs), dtype=np.float32)
            if point_weights is None
            else point_weights,
            dtype=torch.float32,
        )
        self.previous = (
            None
            if previous_values is None
            else torch.as_tensor(previous_values, dtype=torch.float32)
        )
        self.joint_temporal_weight = float(joint_temporal_weight)
        self.translation_temporal_weight = float(translation_temporal_weight)
        self.rotation_temporal_weight = float(rotation_temporal_weight)
        self.surface_vertices = (
            None
            if surface_vertices is None
            else torch.as_tensor(surface_vertices, dtype=torch.float32)
        )
        self.surface_contact_weight = float(surface_contact_weight)
        self.tip_indices = [
            index
            for index, pair in enumerate(pairs)
            if pair["semantic"].endswith("_tip")
        ]
        self.pad_config = pad_config
        self.contact_targets = {} if contact_targets is None else contact_targets
        self.object_surface = object_surface
        self.phase_contact_weight = float(phase_contact_weight)
        self.phase_normal_weight = float(phase_normal_weight)
        self.phase_penetration_weight = float(phase_penetration_weight)
        self.contact_offset = float(contact_offset)
        self.min_signed_distance = float(min_signed_distance)
        self.grasp_reference = (
            None
            if grasp_reference_values is None
            else torch.as_tensor(grasp_reference_values, dtype=torch.float32)
        )
        self.joint_hold_weight = float(joint_hold_weight)
        self.joint_prior = (
            None
            if joint_prior_values is None
            else torch.as_tensor(joint_prior_values, dtype=torch.float32)
        )
        self.joint_prior_weight = float(joint_prior_weight)
        self.last_components = {}

    def __call__(self, values, gradient=None):
        """计算一组候选姿态的loss及可选梯度。

        输入：`J+6`维`[手指关节J, 平移3, 欧拉角3]`和NLopt梯度缓冲区。
        输出：标量关键点均方平方距离乘1000；需要时写入梯度缓冲区。
        逻辑：计算几何、旧表面、指腹区域距离、相对法向、近似穿透、抓形保持和连续性。
        作用：向SLSQP提供当前姿态好坏和梯度，兼顾源姿态、真实接触、抬升保持与连续性。
        """
        values_tensor = torch.tensor(
            np.asarray(values, dtype=np.float32),
            dtype=torch.float32,
            requires_grad=True,
        )
        joint_end = self.joint_count
        translation_end = joint_end + 3
        rotation_end = joint_end + 6
        joints = values_tensor[:joint_end].view(1, joint_end)
        translation = values_tensor[joint_end:translation_end].view(1, 3)
        euler = values_tensor[translation_end:rotation_end].view(1, 3)
        rotation = robust_compute_orth6d_from_eulerXYZ(euler)
        model_q = torch.cat([translation, rotation, joints], dim=1)
        self.model.update_kinematics(model_q)
        predicted_points = linker_world_points(self.model, self.pairs)
        difference = predicted_points - self.target
        squared_distance = torch.sum(difference * difference, dim=1)
        geometry_loss = (
            torch.sum(self.point_weights * squared_distance)
            / torch.sum(self.point_weights)
            * 1000.0
        )
        loss = geometry_loss
        components = {"geometry": geometry_loss}
        if self.surface_vertices is not None and self.surface_contact_weight > 0:
            tip_points = predicted_points[self.tip_indices]
            surface_difference = (
                tip_points[:, None, :] - self.surface_vertices[None, :, :]
            )
            nearest_squared_distance = torch.min(
                torch.sum(surface_difference * surface_difference, dim=2), dim=1
            ).values
            legacy_surface_loss = self.surface_contact_weight * torch.mean(
                nearest_squared_distance
            ) * 1000.0
            loss = loss + legacy_surface_loss
            components["legacy_surface"] = legacy_surface_loss
        if self.pad_config is not None:
            pads = world_pad_regions(self.model, self.pad_config)
            terms = pad_contact_terms(
                pads,
                self.contact_targets,
                self.object_surface,
                self.contact_offset,
                self.min_signed_distance,
            )
            if "contact" in terms and self.phase_contact_weight > 0:
                phase_contact_loss = (
                    self.phase_contact_weight * terms["contact"] * 1000.0
                )
                loss = loss + phase_contact_loss
                components["phase_contact"] = phase_contact_loss
            if "normal" in terms and self.phase_normal_weight > 0:
                phase_normal_loss = self.phase_normal_weight * terms["normal"]
                loss = loss + phase_normal_loss
                components["phase_normal"] = phase_normal_loss
            if "penetration" in terms and self.phase_penetration_weight > 0:
                phase_penetration_loss = (
                    self.phase_penetration_weight * terms["penetration"] * 1000.0
                )
                loss = loss + phase_penetration_loss
                components["phase_penetration"] = phase_penetration_loss
        if self.grasp_reference is not None and self.joint_hold_weight > 0:
            joint_hold_loss = self.joint_hold_weight * torch.mean(
                (values_tensor[:joint_end] - self.grasp_reference[:joint_end]) ** 2
            )
            loss = loss + joint_hold_loss
            components["joint_hold"] = joint_hold_loss
        if self.joint_prior is not None and self.joint_prior_weight > 0:
            joint_prior_loss = self.joint_prior_weight * torch.mean(
                (values_tensor[:joint_end] - self.joint_prior[:joint_end]) ** 2
            )
            loss = loss + joint_prior_loss
            components["joint_prior"] = joint_prior_loss
        if self.previous is not None:
            previous_rotation = robust_compute_orth6d_from_eulerXYZ(
                self.previous[translation_end:rotation_end].view(1, 3)
            )
            loss = loss + self.joint_temporal_weight * torch.mean(
                (values_tensor[:joint_end] - self.previous[:joint_end]) ** 2
            )
            loss = loss + self.translation_temporal_weight * torch.mean(
                (
                    values_tensor[joint_end:translation_end]
                    - self.previous[joint_end:translation_end]
                )
                ** 2
            )
            loss = loss + self.rotation_temporal_weight * torch.mean(
                (rotation - previous_rotation) ** 2
            )
        self.last_components = {
            name: float(value.detach().cpu().item())
            for name, value in components.items()
        }
        if gradient is not None and len(gradient) > 0:
            loss.backward()
            gradient[:] = values_tensor.grad.detach().cpu().numpy().astype(np.float64)
        return float(loss.detach().cpu().item())


def initial_values(shadow_frame, joint_count=6):
    """从Shadow首帧构造Linker优化初值。

    输入：一帧28维Shadow姿态。
    输出：`joint_count+6`维`[零手指关节, 对齐手腕平移, 对齐手腕欧拉角]`。
    逻辑：将固定坐标旋转左乘Shadow手腕旋转，并添加Linker手腕偏移。
    作用：让首帧从合理朝向开始，降低局部非线性优化失败风险。
    """
    shadow_rotation = transforms3d.euler.euler2mat(
        *[float(value) for value in shadow_frame[3:6]], axes="sxyz"
    )
    aligned_rotation = R_ALIGN @ shadow_rotation
    aligned_euler = np.asarray(
        transforms3d.euler.mat2euler(aligned_rotation, axes="sxyz"),
        dtype=np.float32,
    )
    translation = np.asarray(shadow_frame[:3], dtype=np.float32) + WRIST_OFFSET
    return np.concatenate(
        [np.zeros(joint_count, dtype=np.float32), translation, aligned_euler]
    )


def clip_start_to_bounds(values, lower, upper, epsilon=1e-6):
    """把NLopt初值安全夹到变量边界内侧。

    输入：初值、同形状上下界和距离边界的极小正数。
    输出：位于`[lower+epsilon, upper-epsilon]`内的浮点数组。
    逻辑：先转NumPy数组，再逐维使用`np.clip`消除π附近浮点越界。
    作用：避免合法欧拉角因舍入成略大于π而让整条轨迹提前失败。
    """
    values = np.asarray(values, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    return np.clip(values, lower + epsilon, upper - epsilon)


def intersect_joint_trust_region(lower, upper, baseline_values, joint_count, delta):
    """把全局关节边界与基线附近的小残差范围求交集。

    输入：完整变量上下界、当前帧基线、手指关节数和最大绝对改变量。
    输出：新的完整上下界副本；手腕边界不变。
    内部逻辑：手指前J维分别取全局边界与`baseline±delta`的交集。
    作用：防止接触loss大幅重写原抓形，只允许局部接触修正。
    """
    lower = np.asarray(lower, dtype=np.float64).copy()
    upper = np.asarray(upper, dtype=np.float64).copy()
    baseline = np.asarray(baseline_values, dtype=np.float64)
    if delta <= 0:
        return lower, upper
    lower[:joint_count] = np.maximum(
        lower[:joint_count], baseline[:joint_count] - float(delta)
    )
    upper[:joint_count] = np.minimum(
        upper[:joint_count], baseline[:joint_count] + float(delta)
    )
    return lower, upper


def compose_frozen_lift_values(
    baseline_values,
    grasp_reference,
    baseline_before_lift,
    carry_wrist_residual,
    joint_count=6,
):
    """组合抬升期的固定抓形和连续手腕轨迹。

    输入：当前基线姿态、闭合末帧姿态、基线闭合末帧及是否保留手腕修正。
    输出：固定闭合末帧全部手指关节的内部姿态。
    内部逻辑：手指取优化后的抓形；手腕沿基线运动，并可右乘闭合时形成的局部刚体修正。
    作用：避免闭合优化移动手腕后，在抬升首帧突然跳回基线而立即丢失接触。
    """
    result = np.asarray(baseline_values, dtype=np.float64).copy()
    grasp_reference = np.asarray(grasp_reference, dtype=np.float64)
    translation_start = joint_count
    rotation_start = joint_count + 3
    result[:joint_count] = grasp_reference[:joint_count]
    if carry_wrist_residual:
        if baseline_before_lift is None:
            raise ValueError("保留抬升手腕修正时缺少基线闭合末帧")
        baseline_before_lift = np.asarray(baseline_before_lift, dtype=np.float64)
        base_rotation = transforms3d.euler.euler2mat(
            *baseline_before_lift[rotation_start : rotation_start + 3], axes="sxyz"
        )
        grasp_rotation = transforms3d.euler.euler2mat(
            *grasp_reference[rotation_start : rotation_start + 3], axes="sxyz"
        )
        current_rotation = transforms3d.euler.euler2mat(
            *result[rotation_start : rotation_start + 3], axes="sxyz"
        )
        local_rotation = base_rotation.T @ grasp_rotation
        local_translation = base_rotation.T @ (
            grasp_reference[translation_start : translation_start + 3]
            - baseline_before_lift[translation_start : translation_start + 3]
        )
        result[translation_start : translation_start + 3] += (
            current_rotation @ local_translation
        )
        corrected_rotation = current_rotation @ local_rotation
        result[rotation_start : rotation_start + 3] = transforms3d.euler.mat2euler(
            corrected_rotation, axes="sxyz"
        )
    return result


def compose_lift_joint_residual(
    baseline_values, grasp_reference, baseline_before_lift, joint_count=6
):
    """把闭合末帧学到的小关节残差恒定传播到动态抬升轨迹。

    输入：当前逐帧基线、优化后闭合末帧、原基线闭合末帧和关节数。
    输出：手腕完全沿用当前基线、手指为当前动态基线加固定残差的内部姿态。
    内部逻辑：计算`优化闭合末帧-基线闭合末帧`的前J维差，再叠加到当前帧。
    作用：保持原轨迹的逐帧包覆变化，同时避免抬升期重复优化造成接触目标追逐和抖动。
    """
    result = np.asarray(baseline_values, dtype=np.float64).copy()
    grasp = np.asarray(grasp_reference, dtype=np.float64)
    baseline_grasp = np.asarray(baseline_before_lift, dtype=np.float64)
    result[:joint_count] += grasp[:joint_count] - baseline_grasp[:joint_count]
    return result


def grip_tightening_vector(joint_count, thumb_pitch, fingers):
    """生成与当前关节模式一致的闭合附加量。

    输入：关节数6/11、拇指屈曲增量和四指屈曲增量（弧度）。
    输出：长度为关节数的增量数组。
    逻辑：6维模式只控制主动轴；11维模式也给原从动轴添加按原机构倍率缩放的增量，
    但这些值仍由11维优化器独立维护，不重新施加强制耦合。
    作用：保证旧的可选抓紧参数在两种模式下语义一致。
    """
    if joint_count == 6:
        return np.asarray(
            [0.0, thumb_pitch] + [fingers] * 4, dtype=np.float64
        )
    if joint_count == 11:
        return np.asarray(
            [0.0, thumb_pitch, thumb_pitch * 1.86]
            + [value for _ in range(4) for value in (fingers, fingers * 0.89)],
            dtype=np.float64,
        )
    raise ValueError(f"Linker仅支持6或11个手指关节，实际为{joint_count}")


def retarget_trajectory(
    source_frames,
    model,
    pairs,
    target_points,
    maxeval,
    translation_bound,
    joint_temporal_weight,
    translation_temporal_weight,
    rotation_temporal_weight,
    contact_start_frame,
    late_tip_weight,
    late_thumb_weight,
    late_structure_weight,
    frame_point_weights=None,
    object_vertices=None,
    surface_active_frames=None,
    target_surface_weight=0.0,
    phase_contact_plan=None,
    pad_config=None,
    phase_contact_weight=0.0,
    phase_normal_weight=0.0,
    phase_penetration_weight=0.0,
    phase_joint_hold_weight=0.0,
    phase_contact_offset=-0.001,
    phase_min_signed_distance=-0.003,
    initial_trajectory=None,
    phase_only_refinement=False,
    phase_joint_only=False,
    phase_joint_prior_weight=0.0,
    phase_joint_delta_bound=0.0,
    freeze_lift_grasp=False,
    carry_lift_joint_residual=False,
    carry_lift_wrist_residual=False,
    grip_tighten_thumb_pitch=0.0,
    grip_tighten_fingers=0.0,
):
    """逐帧优化一条70帧Shadow轨迹。

    输入：源帧、模型、点对、Shadow目标、优化限制、旧权重/表面项、新阶段接触计划，
    以及可选的第一阶段语义基线、仅关节细化、关节先验、抬升抓形/手腕残差保持和闭合残差。
    输出：`(T,J+6)`内部目标轨迹、`(T,)`最终loss和逐帧loss分解。
    逻辑：每帧创建带关节边界的SLSQP，首帧用对齐初值，后续用上一帧结果。
    作用：生成可保存和重放的Linker候选动作，同时记录每帧几何质量。
    """
    joint_lower = model.revolute_joints_q_lower[0].detach().cpu().numpy()
    joint_upper = model.revolute_joints_q_upper[0].detach().cpu().numpy()
    joint_count = model_joint_count(model)
    value_count = joint_count + 6
    lower = np.concatenate(
        [joint_lower, np.full(3, -translation_bound), np.full(3, -np.pi)]
    )
    upper = np.concatenate(
        [joint_upper, np.full(3, translation_bound), np.full(3, np.pi)]
    )
    results, losses, component_losses = [], [], []
    previous = None
    grasp_reference = None
    baseline_grasp_reference = None
    if frame_point_weights is not None:
        frame_point_weights = np.asarray(frame_point_weights, dtype=np.float32)
        if frame_point_weights.shape != target_points.shape[:2]:
            raise ValueError("frame_point_weights必须为(T, pair数量)")
    if surface_active_frames is not None:
        surface_active_frames = np.asarray(surface_active_frames, dtype=bool)
        if surface_active_frames.shape != (len(source_frames),):
            raise ValueError("surface_active_frames必须为(T,)")
    if phase_contact_plan is not None and len(phase_contact_plan) != len(source_frames):
        raise ValueError("phase_contact_plan长度必须等于轨迹帧数")
    if initial_trajectory is not None:
        initial_trajectory = np.asarray(initial_trajectory, dtype=np.float32)
        expected_shape = (len(source_frames), value_count)
        if initial_trajectory.shape != expected_shape:
            raise ValueError(
                f"initial_trajectory必须为{expected_shape}内部顺序，"
                f"实际为{initial_trajectory.shape}"
            )
    close_indices = []
    if phase_contact_plan is not None:
        close_indices = [
            index
            for index, frame in enumerate(phase_contact_plan)
            if frame["phase"] == "close"
        ]
    tightening = grip_tightening_vector(
        joint_count, grip_tighten_thumb_pitch, grip_tighten_fingers
    )
    for frame_index, source_frame in enumerate(source_frames):
        baseline_values = (
            None if initial_trajectory is None else initial_trajectory[frame_index]
        )
        start = (
            baseline_values.copy()
            if baseline_values is not None
            else initial_values(source_frame, joint_count)
            if previous is None
            else previous.copy()
        )
        start = clip_start_to_bounds(start, lower, upper)
        point_weights = semantic_weights(
            pairs,
            frame_index,
            contact_start_frame,
            late_tip_weight,
            late_thumb_weight,
            late_structure_weight,
        )
        if frame_point_weights is not None:
            point_weights = point_weights * frame_point_weights[frame_index]
        phase_frame = (
            None if phase_contact_plan is None else phase_contact_plan[frame_index]
        )
        if (
            phase_frame is not None
            and phase_frame["phase"] == "lift"
            and grasp_reference is None
        ):
            grasp_reference = previous.copy() if previous is not None else start.copy()
            baseline_grasp_reference = (
                None
                if initial_trajectory is None or frame_index == 0
                else initial_trajectory[frame_index - 1].copy()
            )
        objective = LinkerFrameObjective(
            model,
            pairs,
            target_points[frame_index],
            point_weights=point_weights,
            previous_values=previous,
            joint_temporal_weight=joint_temporal_weight,
            translation_temporal_weight=translation_temporal_weight,
            rotation_temporal_weight=rotation_temporal_weight,
            surface_vertices=(
                object_vertices
                if surface_active_frames is not None
                and surface_active_frames[frame_index]
                else None
            ),
            surface_contact_weight=target_surface_weight,
            pad_config=pad_config,
            contact_targets=None if phase_frame is None else phase_frame["targets"],
            object_surface=(
                None
                if phase_frame is None
                else (phase_frame["object_vertices"], phase_frame["object_normals"])
            ),
            phase_contact_weight=phase_contact_weight,
            phase_normal_weight=phase_normal_weight,
            phase_penetration_weight=phase_penetration_weight,
            contact_offset=phase_contact_offset,
            min_signed_distance=phase_min_signed_distance,
            grasp_reference_values=(
                grasp_reference
                if phase_frame is not None and phase_frame["phase"] == "lift"
                else None
            ),
            joint_hold_weight=phase_joint_hold_weight,
            joint_prior_values=(
                baseline_values
                if phase_frame is not None and phase_frame["phase"] != "approach"
                else None
            ),
            joint_prior_weight=phase_joint_prior_weight,
        )
        should_copy_baseline = (
            phase_only_refinement
            and baseline_values is not None
            and (phase_frame is None or phase_frame["phase"] == "approach")
        )
        should_freeze_lift = (
            freeze_lift_grasp
            and baseline_values is not None
            and phase_frame is not None
            and phase_frame["phase"] == "lift"
            and grasp_reference is not None
        )
        should_carry_joint_residual = (
            carry_lift_joint_residual
            and baseline_values is not None
            and phase_frame is not None
            and phase_frame["phase"] == "lift"
            and grasp_reference is not None
            and baseline_grasp_reference is not None
        )
        if should_carry_joint_residual:
            result = compose_lift_joint_residual(
                baseline_values,
                grasp_reference,
                baseline_grasp_reference,
                joint_count,
            )
            result = clip_start_to_bounds(result, lower, upper)
        elif should_freeze_lift:
            result = compose_frozen_lift_values(
                baseline_values,
                grasp_reference,
                baseline_grasp_reference,
                carry_lift_wrist_residual,
                joint_count,
            )
            result = clip_start_to_bounds(result, lower, upper)
        elif should_copy_baseline:
            result = start
        else:
            optimizer = nlopt.opt(nlopt.LD_SLSQP, value_count)
            optimizer.set_min_objective(objective)
            frame_lower = lower.copy()
            frame_upper = upper.copy()
            if (
                phase_joint_delta_bound > 0
                and baseline_values is not None
                and phase_frame is not None
                and phase_frame["phase"] != "approach"
            ):
                frame_lower, frame_upper = intersect_joint_trust_region(
                    frame_lower,
                    frame_upper,
                    baseline_values,
                    joint_count,
                    phase_joint_delta_bound,
                )
            if (
                phase_joint_only
                and baseline_values is not None
                and phase_frame is not None
                and phase_frame["phase"] != "approach"
            ):
                # 接触细化只允许手指改变。用极窄而非完全相等的边界兼容NLopt，
                # 同时把初值手腕恢复为逐帧基线，避免闭合优化偷偷移动整只手。
                wrist_slice = slice(joint_count, joint_count + 6)
                start[wrist_slice] = baseline_values[wrist_slice]
                frame_lower[wrist_slice] = baseline_values[wrist_slice] - 1e-10
                frame_upper[wrist_slice] = baseline_values[wrist_slice] + 1e-10
            optimizer.set_lower_bounds(frame_lower.tolist())
            optimizer.set_upper_bounds(frame_upper.tolist())
            optimizer.set_maxeval(maxeval)
            optimizer.set_xtol_rel(1e-6)
            optimizer.set_ftol_rel(1e-8)
            try:
                result = optimizer.optimize(start)
            except (nlopt.RoundoffLimited, RuntimeError):
                result = start
        if phase_frame is not None and phase_frame["phase"] == "close" and close_indices:
            close_rank = close_indices.index(frame_index) + 1
            progress = close_rank / len(close_indices)
            result = np.asarray(result, dtype=np.float64).copy()
            result[:joint_count] += progress * tightening
            result = clip_start_to_bounds(result, lower, upper)
        previous = np.asarray(result, dtype=np.float32)
        results.append(previous)
        losses.append(objective(previous))
        component_losses.append(objective.last_components.copy())
    return (
        np.stack(results),
        np.asarray(losses, dtype=np.float32),
        component_losses,
    )


def retarget_file(args):
    """读取源文件、重定向选定轨迹并保存标准输出字典。

    输入：包含source/output/index/maxeval等字段的命令行参数。
    输出：无Python返回值；写入一个目标`.npy`。
    逻辑：读取并复制源轨迹、加参考Z偏移、计算目标点、运行逐帧优化并重排保存。
    作用：把底层单轨迹函数封装成可复现实验文件入口。
    """
    source_data = np.load(args.source, allow_pickle=True).item()
    indices = args.trajectory_indices or [0]
    source_frames = np.asarray(source_data["grasp_seqs"][indices], dtype=np.float32).copy()
    source_frames[:, :, 2] += args.source_z_offset
    shadow_model = build_shadow_model()
    linker_model = build_linker_model(args.joint_mode)
    joint_count = model_joint_count(linker_model)
    saved_dimension = joint_count + 6
    # 11轴模式给拇指增加了独立IP角，因此自动启用拇指中段点，避免仅凭一个指尖点
    # 同时反推yaw/pitch/IP三个角时出现多解；6轴基线仍保持原来的10点口径。
    include_finger_middle = bool(args.include_finger_middle)
    include_thumb_middle = (
        args.include_thumb_middle
        or include_finger_middle
        or args.joint_mode == "independent11"
    )
    pairs = load_pairs(include_thumb_middle, include_finger_middle)
    pad_config = (
        None
        if args.contact_pad_config is None
        else load_pad_config(args.contact_pad_config, expected_hand="linker")
    )
    initial_by_source_index = {}
    if args.initial_target is not None:
        initial_data = np.load(args.initial_target, allow_pickle=True).item()
        initial_indices = np.asarray(
            initial_data["source_trajectory_indices"], dtype=np.int64
        )
        initial_frames = np.asarray(initial_data["grasp_seqs"], dtype=np.float32)
        expected_initial_shape = (len(initial_indices), 70, saved_dimension)
        if initial_frames.shape != expected_initial_shape:
            raise ValueError(
                f"{args.joint_mode}第一阶段候选应为{expected_initial_shape}，"
                f"实际为{initial_frames.shape}"
            )
        initial_mode = str(initial_data.get("joint_mode", "coupled6"))
        if initial_mode != args.joint_mode:
            raise ValueError(
                f"第一阶段关节模式{initial_mode}与当前{args.joint_mode}不一致"
            )
        initial_by_source_index = {
            int(source_index): frames
            for source_index, frames in zip(initial_indices, initial_frames)
        }
        missing = sorted(set(int(index) for index in indices) - set(initial_by_source_index))
        if missing:
            raise ValueError(f"第一阶段候选缺少源轨迹索引: {missing}")
    outputs, all_losses, all_loss_components = [], [], []
    all_contact_weights, all_surface_active, all_phase_metadata = [], [], []
    for local_index, trajectory in enumerate(source_frames):
        source_index = int(indices[local_index])
        initial_internal = None
        saved_initial = None
        if source_index in initial_by_source_index:
            saved_initial = initial_by_source_index[source_index]
            initial_internal = np.concatenate(
                [saved_initial[:, 6:], saved_initial[:, :6]], axis=1
            )
        all_shadow_points = shadow_keypoints(trajectory, shadow_model)
        source_indices = [pair["shadow_index"] for pair in pairs]
        target_points = all_shadow_points[:, source_indices, :]
        contact_weights = None
        contact_mask = None
        object_vertices = None
        phase_plan = None
        needs_object = args.expert_contact_threshold >= 0 or pad_config is not None
        if needs_object:
            object_name = args.object_name or args.source.stem
            object_vertices, object_normals = transformed_object_surface(
                args.object_root / object_name,
                np.asarray(source_data["obj_scale"])[source_index],
                np.asarray(source_data["obj_rotmat"])[source_index],
                args.object_clearance,
            )
        if args.expert_contact_threshold >= 0:
            contact_mask = source_tip_contact_mask(
                target_points,
                pairs,
                object_vertices,
                args.expert_contact_threshold,
            )
            contact_weights = source_contact_point_weights(
                target_points,
                pairs,
                object_vertices,
                args.expert_contact_threshold,
                args.expert_contact_weight,
            )
        if pad_config is not None:
            source_tip_points = {
                semantic: all_shadow_points[:, point_index, :]
                for semantic, point_index in SOURCE_TIP_INDICES.items()
            }
            reachable_pads = None
            if args.reachable_opposition:
                preliminary_phases = infer_motion_phases(
                    trajectory,
                    source_tip_points,
                    object_vertices,
                    args.phase_contact_threshold,
                    args.phase_min_contact_tips,
                    args.phase_lift_delta,
                )
                reachable_pads = reachable_pad_geometry_from_saved_frame(
                    linker_model,
                    pad_config,
                    saved_initial[preliminary_phases["grasp_frame"]],
                )
            phase_plan = build_phase_contact_plan(
                trajectory,
                source_tip_points,
                object_vertices,
                object_normals,
                contact_threshold=args.phase_contact_threshold,
                min_contact_tips=args.phase_min_contact_tips,
                lift_delta=args.phase_lift_delta,
                region_neighbors=args.phase_region_neighbors,
                opposition_candidate_neighbors=(
                    args.opposition_candidate_neighbors
                ),
                opposition_distance_scale=args.opposition_distance_scale,
                opposition_weight=args.opposition_weight,
                opposition_refine_frames=args.opposition_refine_frames,
                reachable_pads=reachable_pads,
                reachable_pad_alignment_weight=(
                    args.reachable_pad_alignment_weight
                ),
                reachable_min_opposing_fingers=(
                    args.reachable_min_opposing_fingers
                ),
                friction_stability_weight=args.friction_stability_weight,
                friction_coefficient=args.friction_coefficient,
                friction_cone_edges=args.friction_cone_edges,
                max_reachable_distance=args.max_reachable_distance,
            )
        surface_active_frames = (
            None
            if contact_mask is None
            else np.sum(contact_mask, axis=1)
            >= args.surface_activation_min_expert_tips
        )
        internal, losses, loss_components = retarget_trajectory(
            trajectory,
            linker_model,
            pairs,
            target_points,
            args.maxeval,
            args.translation_bound,
            args.joint_temporal_weight,
            args.translation_temporal_weight,
            args.rotation_temporal_weight,
            args.contact_start_frame,
            args.late_tip_weight,
            args.late_thumb_weight,
            args.late_structure_weight,
            contact_weights,
            object_vertices,
            surface_active_frames,
            args.target_surface_weight,
            None if phase_plan is None else phase_plan["frames"],
            pad_config,
            args.phase_contact_weight,
            args.phase_normal_weight,
            args.phase_penetration_weight,
            args.phase_joint_hold_weight,
            args.phase_contact_offset,
            args.phase_min_signed_distance,
            initial_internal,
            args.phase_only_refinement,
            args.phase_joint_only,
            args.phase_joint_prior_weight,
            args.phase_joint_delta_bound,
            args.freeze_lift_grasp,
            args.carry_lift_joint_residual,
            args.carry_lift_wrist_residual,
            args.grip_tighten_thumb_pitch,
            args.grip_tighten_fingers,
        )
        # 内部顺序为关节J+手腕6；保存顺序统一为手腕6+关节J。
        outputs.append(
            np.concatenate(
                [internal[:, joint_count : joint_count + 6], internal[:, :joint_count]],
                axis=1,
            )
        )
        all_losses.append(losses)
        all_loss_components.append(loss_components)
        all_contact_weights.append(
            np.ones(target_points.shape[:2], dtype=np.float32)
            if contact_weights is None
            else contact_weights
        )
        all_surface_active.append(
            np.zeros(len(trajectory), dtype=bool)
            if surface_active_frames is None
            else surface_active_frames
        )
        all_phase_metadata.append(
            None
            if phase_plan is None
            else {
                "close_start_frame": int(phase_plan["close_start_frame"]),
                "lift_start_frame": int(phase_plan["lift_start_frame"]),
                "grasp_frame": int(phase_plan["grasp_frame"]),
                "opposition_start_frame": phase_plan["opposition_start_frame"],
                "opposition_diagnostics": phase_plan["opposition_diagnostics"],
                "source_contact_tip_count": np.asarray(
                    phase_plan["source_contact_tip_count"]
                ).tolist(),
            }
        )
    output_frames = np.stack(outputs).astype(np.float32)
    loss_frames = np.stack(all_losses).astype(np.float32)
    output = {
        "grasp_seqs": output_frames,
        "optimization_loss_per_frame": loss_frames,
        "optimization_loss_components_per_frame": all_loss_components,
        "source_trajectory_indices": np.asarray(indices, dtype=np.int64),
        "obj_rotmat": np.asarray(source_data["obj_rotmat"])[indices],
        "obj_scale": np.asarray(source_data["obj_scale"])[indices],
        "mapping_semantics": [pair["semantic"] for pair in pairs],
        "joint_mode": args.joint_mode,
        "finger_joint_count": joint_count,
        "target_dimension": saved_dimension,
        "include_thumb_middle": bool(include_thumb_middle),
        "include_finger_middle": bool(include_finger_middle),
        "source_z_offset": float(args.source_z_offset),
        "maxeval": int(args.maxeval),
        "joint_temporal_weight": float(args.joint_temporal_weight),
        "translation_temporal_weight": float(args.translation_temporal_weight),
        "rotation_temporal_weight": float(args.rotation_temporal_weight),
        "contact_start_frame": int(args.contact_start_frame),
        "late_tip_weight": float(args.late_tip_weight),
        "late_thumb_weight": float(args.late_thumb_weight),
        "late_structure_weight": float(args.late_structure_weight),
        "expert_contact_threshold": float(args.expert_contact_threshold),
        "expert_contact_weight": float(args.expert_contact_weight),
        "object_clearance": float(args.object_clearance),
        "source_contact_point_weights": np.stack(all_contact_weights).astype(np.float32),
        "target_surface_weight": float(args.target_surface_weight),
        "surface_activation_min_expert_tips": int(
            args.surface_activation_min_expert_tips
        ),
        "surface_active_frames": np.stack(all_surface_active),
        "contact_pad_config": (
            None if args.contact_pad_config is None else str(args.contact_pad_config.resolve())
        ),
        "phase_contact_weight": float(args.phase_contact_weight),
        "phase_normal_weight": float(args.phase_normal_weight),
        "phase_penetration_weight": float(args.phase_penetration_weight),
        "phase_joint_hold_weight": float(args.phase_joint_hold_weight),
        "phase_contact_threshold": float(args.phase_contact_threshold),
        "phase_min_contact_tips": int(args.phase_min_contact_tips),
        "phase_lift_delta": float(args.phase_lift_delta),
        "phase_region_neighbors": int(args.phase_region_neighbors),
        "opposition_candidate_neighbors": int(
            args.opposition_candidate_neighbors
        ),
        "opposition_distance_scale": float(args.opposition_distance_scale),
        "opposition_weight": float(args.opposition_weight),
        "opposition_refine_frames": int(args.opposition_refine_frames),
        "reachable_opposition": bool(args.reachable_opposition),
        "reachable_pad_alignment_weight": float(
            args.reachable_pad_alignment_weight
        ),
        "reachable_min_opposing_fingers": int(
            args.reachable_min_opposing_fingers
        ),
        "friction_stability_weight": float(args.friction_stability_weight),
        "friction_coefficient": float(args.friction_coefficient),
        "friction_cone_edges": int(args.friction_cone_edges),
        "max_reachable_distance": float(args.max_reachable_distance),
        "phase_contact_offset": float(args.phase_contact_offset),
        "phase_min_signed_distance": float(args.phase_min_signed_distance),
        "initial_target": (
            None if args.initial_target is None else str(args.initial_target.resolve())
        ),
        "phase_only_refinement": bool(args.phase_only_refinement),
        "phase_joint_only": bool(args.phase_joint_only),
        "phase_joint_prior_weight": float(args.phase_joint_prior_weight),
        "phase_joint_delta_bound": float(args.phase_joint_delta_bound),
        "freeze_lift_grasp": bool(args.freeze_lift_grasp),
        "carry_lift_joint_residual": bool(args.carry_lift_joint_residual),
        "carry_lift_wrist_residual": bool(args.carry_lift_wrist_residual),
        "grip_tighten_thumb_pitch": float(args.grip_tighten_thumb_pitch),
        "grip_tighten_fingers": float(args.grip_tighten_fingers),
        "phase_metadata": all_phase_metadata,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, output, allow_pickle=True)
    print(f"trajectories={len(output_frames)}")
    print(f"frames={output_frames.shape[1]}")
    print(f"output_shape={output_frames.shape}")
    print(f"mean_final_loss={loss_frames.mean():.6f}")
    print(f"max_final_loss={loss_frames.max():.6f}")
    print(f"output={args.output}")


def main():
    """解析命令行并运行Linker校准关键点基线。

    输入：源/输出、轨迹索引、优化限制、点约束、时序与阶段权重。
    输出：目标npy与终端摘要。
    逻辑：构造参数后调用`retarget_file`。
    作用：作为run分区中第一条自主实现的Linker重定向命令。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trajectory-indices", type=int, nargs="*")
    parser.add_argument("--maxeval", type=int, default=100)
    parser.add_argument("--translation-bound", type=float, default=2.0)
    parser.add_argument("--source-z-offset", type=float, default=0.4)
    parser.add_argument(
        "--joint-mode",
        choices=["coupled6", "independent11"],
        default="coupled6",
        help=(
            "coupled6保持真实O6联动；independent11把5个mimic轴解耦，"
            "仅作为Linker外形高自由度实验"
        ),
    )
    parser.add_argument("--include-thumb-middle", action="store_true")
    parser.add_argument(
        "--include-finger-middle",
        action="store_true",
        help=(
            "加入四根普通指的中段点，并自动加入拇指中段，"
            "从官方10点扩展为稠密15点消融"
        ),
    )
    parser.add_argument("--joint-temporal-weight", type=float, default=0.0)
    parser.add_argument("--translation-temporal-weight", type=float, default=0.0)
    parser.add_argument("--rotation-temporal-weight", type=float, default=0.0)
    parser.add_argument("--contact-start-frame", type=int, default=-1)
    parser.add_argument("--late-tip-weight", type=float, default=1.0)
    parser.add_argument("--late-thumb-weight", type=float, default=1.0)
    parser.add_argument("--late-structure-weight", type=float, default=1.0)
    parser.add_argument(
        "--expert-contact-threshold",
        type=float,
        default=-1.0,
        help="Shadow指尖到物体表面的接触阈值（米）；负数表示关闭",
    )
    parser.add_argument("--expert-contact-weight", type=float, default=1.0)
    parser.add_argument("--object-root", type=Path, default=OBJECT_ROOT)
    parser.add_argument("--object-name")
    parser.add_argument("--object-clearance", type=float, default=0.005)
    parser.add_argument("--target-surface-weight", type=float, default=0.0)
    parser.add_argument(
        "--surface-activation-min-expert-tips", type=int, default=2
    )
    parser.add_argument(
        "--contact-pad-config",
        type=Path,
        help="物理成功重放校准的五指表面区域JSON；不提供则保持旧基线",
    )
    parser.add_argument(
        "--initial-target",
        type=Path,
        help="第一阶段语义基线npy；提供后以对应轨迹为接触细化初值",
    )
    parser.add_argument(
        "--phase-only-refinement",
        action="store_true",
        help="接近阶段原样复制第一阶段基线，只优化闭合和抬升阶段",
    )
    parser.add_argument(
        "--phase-joint-only",
        action="store_true",
        help="闭合和抬升细化时固定第一阶段逐帧手腕，只优化手指关节",
    )
    parser.add_argument(
        "--freeze-lift-grasp",
        action="store_true",
        help="抬升期固定闭合末帧主动关节，仅沿用第一阶段腕部轨迹",
    )
    parser.add_argument(
        "--carry-lift-wrist-residual",
        action="store_true",
        help="抬升期沿基线腕轨迹持续复合闭合末帧学到的局部刚体手腕修正",
    )
    parser.add_argument(
        "--carry-lift-joint-residual",
        action="store_true",
        help="抬升期保留原动态关节轨迹，并恒定叠加闭合末帧学到的关节残差",
    )
    parser.add_argument("--phase-contact-weight", type=float, default=3.0)
    parser.add_argument("--phase-normal-weight", type=float, default=0.02)
    parser.add_argument("--phase-penetration-weight", type=float, default=1.0)
    parser.add_argument("--phase-joint-hold-weight", type=float, default=0.1)
    parser.add_argument("--phase-joint-prior-weight", type=float, default=1.0)
    parser.add_argument(
        "--phase-joint-delta-bound",
        type=float,
        default=0.0,
        help="接触细化相对逐帧基线的单关节最大改变量；0表示不设硬限制",
    )
    parser.add_argument(
        "--grip-tighten-thumb-pitch",
        type=float,
        default=0.0,
        help="闭合阶段给拇指pitch主动关节线性增加的弧度",
    )
    parser.add_argument(
        "--grip-tighten-fingers",
        type=float,
        default=0.0,
        help="闭合阶段给食/中/环/小指主动关节线性增加的弧度",
    )
    parser.add_argument("--phase-contact-threshold", type=float, default=0.02)
    parser.add_argument("--phase-min-contact-tips", type=int, default=2)
    parser.add_argument("--phase-lift-delta", type=float, default=0.03)
    parser.add_argument("--phase-region-neighbors", type=int, default=32)
    parser.add_argument(
        "--opposition-candidate-neighbors",
        type=int,
        default=0,
        help="每根源指尖附近参与对向区域联合选择的候选顶点数；0表示关闭",
    )
    parser.add_argument(
        "--opposition-distance-scale",
        type=float,
        default=0.03,
        help="接触中心偏离Shadow语义位置的归一化尺度（米）",
    )
    parser.add_argument(
        "--opposition-weight",
        type=float,
        default=1.0,
        help="拇指与普通手指表面法向相反的软约束权重",
    )
    parser.add_argument(
        "--opposition-refine-frames",
        type=int,
        default=4,
        help="抬升前提前使用最终对向接触区域的闭合帧数",
    )
    parser.add_argument(
        "--reachable-opposition",
        action="store_true",
        help="用O6基线真实指腹可达位置而非Shadow指尖选择对向表面",
    )
    parser.add_argument(
        "--reachable-pad-alignment-weight",
        type=float,
        default=1.0,
        help="目标手指腹外法向与物体外法向相反的选择权重",
    )
    parser.add_argument(
        "--reachable-min-opposing-fingers",
        type=int,
        default=2,
        help="除拇指外强制优化的最少对侧手指数；O6默认选择最可达的2根",
    )
    parser.add_argument(
        "--friction-stability-weight",
        type=float,
        default=0.0,
        help="候选接触抵消单位重力的摩擦扳手残差权重；0表示关闭",
    )
    parser.add_argument("--friction-coefficient", type=float, default=1.0)
    parser.add_argument("--friction-cone-edges", type=int, default=4)
    parser.add_argument(
        "--max-reachable-distance",
        type=float,
        default=0.0,
        help="候选物体点到当前指腹的硬距离上限（米）；0表示关闭",
    )
    parser.add_argument(
        "--phase-contact-offset",
        type=float,
        default=-0.001,
        help="目标指腹沿物体外法向的偏移；负数表示轻微压入以建立接触力",
    )
    parser.add_argument(
        "--phase-min-signed-distance",
        type=float,
        default=-0.003,
        help="允许的最深近似有符号距离；更深才施加穿透惩罚",
    )
    args = parser.parse_args()
    if args.target_surface_weight < 0:
        parser.error("--target-surface-weight不能为负数")
    if args.target_surface_weight > 0 and args.expert_contact_threshold < 0:
        parser.error("启用目标表面项时必须设置非负--expert-contact-threshold")
    if not 1 <= args.surface_activation_min_expert_tips <= 5:
        parser.error("--surface-activation-min-expert-tips必须在1到5之间")
    phase_nonnegative = {
        "--phase-contact-weight": args.phase_contact_weight,
        "--phase-normal-weight": args.phase_normal_weight,
        "--phase-penetration-weight": args.phase_penetration_weight,
        "--phase-joint-hold-weight": args.phase_joint_hold_weight,
        "--phase-joint-prior-weight": args.phase_joint_prior_weight,
        "--phase-joint-delta-bound": args.phase_joint_delta_bound,
        "--phase-contact-threshold": args.phase_contact_threshold,
        "--phase-lift-delta": args.phase_lift_delta,
        "--opposition-distance-scale": args.opposition_distance_scale,
        "--opposition-weight": args.opposition_weight,
        "--grip-tighten-thumb-pitch": args.grip_tighten_thumb_pitch,
        "--grip-tighten-fingers": args.grip_tighten_fingers,
        "--reachable-pad-alignment-weight": args.reachable_pad_alignment_weight,
        "--friction-stability-weight": args.friction_stability_weight,
        "--friction-coefficient": args.friction_coefficient,
        "--max-reachable-distance": args.max_reachable_distance,
    }
    for name, value in phase_nonnegative.items():
        if value < 0:
            parser.error(f"{name}不能为负数")
    if args.phase_min_signed_distance > args.phase_contact_offset:
        parser.error("--phase-min-signed-distance必须小于等于--phase-contact-offset")
    if args.phase_only_refinement and args.initial_target is None:
        parser.error("--phase-only-refinement必须同时提供--initial-target")
    if args.phase_joint_only and args.initial_target is None:
        parser.error("--phase-joint-only必须同时提供--initial-target")
    if args.freeze_lift_grasp and args.initial_target is None:
        parser.error("--freeze-lift-grasp必须同时提供--initial-target")
    if args.carry_lift_wrist_residual and not args.freeze_lift_grasp:
        parser.error("--carry-lift-wrist-residual必须同时启用--freeze-lift-grasp")
    if args.carry_lift_joint_residual and args.initial_target is None:
        parser.error("--carry-lift-joint-residual必须同时提供--initial-target")
    if args.carry_lift_joint_residual and args.freeze_lift_grasp:
        parser.error("动态关节残差传播与--freeze-lift-grasp不能同时启用")
    if not 1 <= args.phase_min_contact_tips <= 5:
        parser.error("--phase-min-contact-tips必须在1到5之间")
    if args.phase_region_neighbors < 1:
        parser.error("--phase-region-neighbors必须为正整数")
    if args.opposition_candidate_neighbors < 0:
        parser.error("--opposition-candidate-neighbors不能为负数")
    if args.opposition_refine_frames < 1:
        parser.error("--opposition-refine-frames必须为正整数")
    if args.reachable_opposition:
        if args.initial_target is None or args.contact_pad_config is None:
            parser.error("--reachable-opposition需要--initial-target和--contact-pad-config")
        if args.opposition_candidate_neighbors < 1:
            parser.error("--reachable-opposition需要正的--opposition-candidate-neighbors")
    if not 1 <= args.reachable_min_opposing_fingers <= 4:
        parser.error("--reachable-min-opposing-fingers必须在1到4之间")
    if args.friction_cone_edges < 3:
        parser.error("--friction-cone-edges至少为3")
    retarget_file(args)


if __name__ == "__main__":
    main()
