"""把关节参数转换为空间关键点。

输入：Shadow的28维帧，或目标手的手腕与手指关节角。
输出：世界坐标关键点。
逻辑：复用参考工程的底层URDF/MJCF解析器，但不调用其重定向算法。
作用：为本目录自己的几何损失提供可求梯度的正向运动学。
"""

import sys

import numpy as np
import torch
import transforms3d

from .config import LINKER_POINTS, R_ALIGN, REFERENCE_PK, REFERENCE_SCRIPTS


# 依赖部分：只加入参考工程的模型解析器和其pytorch_kinematics依赖。
for path in (REFERENCE_SCRIPTS, REFERENCE_PK):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from utils.HandModel_linkerhand import HandModel_Linkerhand  # noqa: E402
from utils.HandModel_xhand import HandModel_xhand  # noqa: E402
from utils.hand_model import HandModel as ShadowHandModel  # noqa: E402
from utils.rot6d import robust_compute_orth6d_from_eulerXYZ  # noqa: E402


def build_shadow_model():
    """创建Shadow运动学模型。

    输入：参考工程的MJCF和21点文件。
    输出：CPU ShadowHandModel。
    逻辑：关闭表面采样，只保留正向运动学。
    作用：把每帧28维专家动作变成重定向目标点。
    """
    asset = REFERENCE_SCRIPTS / "assets/mjcf_free"
    return ShadowHandModel(
        mjcf_path=str(asset / "shadow_hand_vis_new.xml"),
        mesh_path=str(asset / "meshes"),
        contact_points_path=str(asset / "contact_points.json"),
        penetration_points_path=str(asset / "penetration_points.json"),
        n_surface_points=0, device="cpu", use_joint21=True,
    )


def build_target_model(hand, device="cpu"):
    """创建一只目标手的可求导模型。

    输入：``linker/xhand/wuji``。
    输出：暴露6/12/20个主动关节的CPU模型。
    逻辑：三只手分别读取题目参考工程提供的URDF和mesh。
    作用：让优化变量真正受目标手骨骼和关节限位约束。
    """
    device = torch.device(device)
    if hand == "linker":
        asset = REFERENCE_SCRIPTS / "assets/linkerhand/o6/right"
        return HandModel_Linkerhand(
            robot_name="linkerhand", urdf_filename="linkerhand_o6_right.urdf",
            mesh_path="", batch_size=1, device=device, mesh_nsp=128,
            hand_scale=1.0, asset_dir=str(asset), allow_missing_contacts=True,
        )
    if hand == "xhand":
        asset = REFERENCE_SCRIPTS / "assets/xhand_right/urdf"
        return HandModel_xhand(
            robot_name="xhand", urdf_filename="xhand_right.urdf", mesh_path="",
            batch_size=1, device=device, mesh_nsp=128,
            hand_scale=1.0, asset_dir=str(asset), allow_missing_contacts=True,
        )
    if hand == "wuji":
        asset = REFERENCE_SCRIPTS / "assets/wujihand_urdf/urdf"
        return HandModel_xhand(
            robot_name="wuji_right", urdf_filename="right.urdf", mesh_path="../",
            batch_size=1, device=device, mesh_nsp=128,
            hand_scale=1.0, asset_dir=str(asset), allow_missing_contacts=True,
        )
    raise ValueError(f"未知目标手: {hand}")


def shadow_keypoints(frames, model):
    """批量计算Shadow关键点。

    输入：``(T,28)``源轨迹和Shadow模型。
    输出：``(T,21,3)``世界坐标数组。
    逻辑：欧拉角转旋转6D，再调用模型正向运动学。
    作用：目标手没有专家关节角时，用这些空间点提供监督。
    """
    source = torch.as_tensor(frames, dtype=torch.float32)
    q = torch.zeros((len(source), 31), dtype=torch.float32)
    q[:, :3] = source[:, :3]
    q[:, 3:9] = robust_compute_orth6d_from_eulerXYZ(source[:, 3:6])
    q[:, 9:] = source[:, 6:]
    model.set_parameters(q)
    return model.get_penetraion_keypoints().detach().cpu().numpy()


def target_keypoints(model, hand, joints, translation, euler):
    """计算目标手全部关键点。

    输入：目标模型、手名、手指角、平移和欧拉角张量。
    输出：保留梯度的 ``(K,3)`` 点集。
    逻辑：XHand/Wuji读取模型自带点；Linker变换15个校准局部点。
    作用：几何误差可反向传播到手腕和关节角。
    """
    joints = joints.reshape(1, -1)
    translation = translation.reshape(1, 3)
    euler = euler.reshape(1, 3)
    q = torch.cat([translation, robust_compute_orth6d_from_eulerXYZ(euler), joints], dim=1)
    if hand != "linker":
        return model.get_penetraion_keypoints(q=q)[0]
    model.update_kinematics(q)
    points = []
    for link_name, xyz in LINKER_POINTS:
        local = torch.tensor(xyz, dtype=torch.float32).reshape(1, 1, 3)
        hand_point = model.current_status[link_name].transform_points(local)[0, 0]
        world = hand_point @ model.global_rotation[0].T + model.global_translation[0]
        points.append(world * float(model.scale))
    return torch.stack(points)


def initial_pose(source_frame, joint_count):
    """构造首帧优化初值。

    输入：一帧Shadow动作和目标手关节数。
    输出：内部顺序 ``[手指, 平移, 欧拉角]``。
    逻辑：手指从零开始，手腕乘固定坐标系对齐旋转。
    作用：避免非线性优化从完全错误的手掌方向开始。
    """
    rotation = transforms3d.euler.euler2mat(*source_frame[3:6], axes="sxyz")
    aligned = np.asarray(R_ALIGN) @ rotation
    euler = transforms3d.euler.mat2euler(aligned, axes="sxyz")
    return np.concatenate([np.zeros(joint_count), source_frame[:3], euler]).astype(np.float64)


def joint_names(model):
    """输入目标模型，输出优化器中的手指关节名；作用是保存Wuji动作顺序。"""
    return list(model.robot.get_joint_parameter_names())
