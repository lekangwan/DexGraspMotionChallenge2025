"""定义目标手策略在数据准备和闭环执行中共享的观测构造。

输入：手DOF状态、物体位姿/速度、初始物体位置、接触数和抬升目标。
输出：固定字段顺序的float32特权状态向量。
内部逻辑：补充相对初始位移、剩余抬升量和log接触数后拼接。
作用：让离线专家数据与在线Isaac状态使用同一公式，避免训练—部署字段漂移。
"""

from __future__ import annotations

import numpy as np
import trimesh


OBJECT_SHAPE_DIMENSION = 14


def build_object_shape_descriptor(mesh_path, scale):
    """从COACD表面构造14维确定性实例形状描述。

    输入：`decomposed.obj`路径和该轨迹的物体缩放系数。
    输出：轴向尺寸3、顶点协方差6、表面积、体积、径向分位数3组成的float32向量。
    内部逻辑：顶点先减均值并按scale缩放；协方差取3个对角和3个上三角元素。
    作用：让同类别未见实例策略感知物体大小/长宽比例和粗形状，而非只有xyz与类别ID。
    """
    mesh = trimesh.load_mesh(mesh_path, process=False)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 3:
        raise ValueError(f"物体mesh顶点无效: {mesh_path}")
    centered = (vertices - vertices.mean(axis=0, keepdims=True)) * float(scale)
    extents = np.ptp(centered, axis=0)
    covariance = np.cov(centered, rowvar=False, bias=True)
    covariance_values = covariance[[0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2]]
    area = abs(float(mesh.area)) * float(scale) ** 2
    volume = abs(float(mesh.volume)) * float(scale) ** 3
    radii = np.linalg.norm(centered, axis=1)
    radial_quantiles = np.quantile(radii, [0.25, 0.50, 0.75])
    descriptor = np.concatenate(
        [extents, covariance_values, [area, volume], radial_quantiles]
    ).astype(np.float32)
    if descriptor.shape != (OBJECT_SHAPE_DIMENSION,) or not np.isfinite(descriptor).all():
        raise ValueError(f"物体形状描述无效: {mesh_path}")
    return descriptor


def build_observation_batch(
    hand_position,
    hand_velocity,
    object_position,
    object_quaternion_xyzw,
    object_linear_velocity,
    object_angular_velocity,
    initial_object_position,
    contact_count,
    object_shape_descriptor,
    lift_goal_m=0.30,
):
    """批量构造与`privileged_state_v1`规格一致的观测。

    输入：第一维均为时间/批次的状态、初始位置、接触、实例形状和抬升目标。
    输出：`(N,O)`float32数组。
    内部逻辑：物体位置减初始位置，剩余抬升下限截为0，接触数做log1p。
    作用：离线trace和闭环单步调用通过同一纯函数保持完全相同的字段顺序。
    """
    hand_position = np.asarray(hand_position, dtype=np.float32)
    hand_velocity = np.asarray(hand_velocity, dtype=np.float32)
    object_position = np.asarray(object_position, dtype=np.float32)
    initial = np.asarray(initial_object_position, dtype=np.float32).reshape(1, 3)
    relative = object_position - initial
    remaining = np.maximum(float(lift_goal_m) - relative[:, 2:3], 0.0)
    contact = np.log1p(np.asarray(contact_count, dtype=np.float32).reshape(-1, 1))
    shape = np.asarray(object_shape_descriptor, dtype=np.float32)
    if shape.ndim == 1:
        shape = np.repeat(shape[None], len(hand_position), axis=0)
    if shape.shape != (len(hand_position), OBJECT_SHAPE_DIMENSION):
        raise ValueError(f"物体形状描述维度错误: {shape.shape}")
    arrays = [
        hand_position,
        hand_velocity,
        object_position,
        np.asarray(object_quaternion_xyzw, dtype=np.float32),
        np.asarray(object_linear_velocity, dtype=np.float32),
        np.asarray(object_angular_velocity, dtype=np.float32),
        shape,
        relative,
        remaining.astype(np.float32),
        contact,
    ]
    lengths = {len(value) for value in arrays}
    if len(lengths) != 1:
        raise ValueError(f"观测字段批次长度不一致: {sorted(lengths)}")
    return np.concatenate(arrays, axis=1).astype(np.float32)


def build_runtime_observation(
    dof_states,
    object_state,
    initial_position,
    contact_count,
    object_shape_descriptor,
    lift_goal_m=0.30,
):
    """把Isaac单步状态包装成一维策略观测。

    输入：Isaac DOF结构、物体状态、初始位置、接触数、形状描述和目标高度。
    输出：`(O,)`float32观测。
    内部逻辑：给每个字段增加批次维，调用批量函数后取第0项。
    作用：闭环评测无需复制离线数据准备中的拼接公式。
    """
    return build_observation_batch(
        np.asarray(dof_states["pos"])[None],
        np.asarray(dof_states["vel"])[None],
        object_state["object_position"][None],
        object_state["object_quaternion_xyzw"][None],
        object_state["object_linear_velocity"][None],
        object_state["object_angular_velocity"][None],
        initial_position,
        np.asarray([contact_count]),
        object_shape_descriptor,
        lift_goal_m,
    )[0]
