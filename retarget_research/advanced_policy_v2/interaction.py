"""三只目标手共享的动态手—物几何表征。"""

import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation
import torch


RESEARCH = Path(__file__).resolve().parents[1]
REFERENCE = RESEARCH / "reference/HandRetargetTask2026/scripts"
THIRD_PARTY = RESEARCH / "reference/HandRetargetTask2026/third_party/pytorch_kinematics"
for path in (REFERENCE, THIRD_PARTY):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from utils.HandModel_linkerhand import HandModel_Linkerhand  # noqa: E402
from utils.HandModel_xhand import HandModel_xhand  # noqa: E402
from utils.rot6d import robust_compute_orth6d_from_eulerXYZ  # noqa: E402


PHYSICS_TO_POLICY = {
    "linker": [0, 1, 2, 3, 4, 5, 14, 16, 7, 9, 13, 11],
    "xhand": [0, 1, 2, 3, 4, 5, 15, 16, 17, 6, 7, 8, 9, 10, 13, 14, 11, 12],
    "wuji": list(range(26)),
}


class TargetHandGeometry:
    """把策略动作格式转换为15个语义手点的世界坐标。"""

    def __init__(self, hand):
        self.hand = hand
        assets = REFERENCE / "assets"
        configs = RESEARCH / "retargeting/configs"
        if hand == "linker":
            asset = assets / "linkerhand/o6/right"
            self.model = HandModel_Linkerhand(
                robot_name="linkerhand", urdf_filename="linkerhand_o6_right.urdf",
                mesh_path="", batch_size=1, device=torch.device("cpu"),
                mesh_nsp=128, hand_scale=1.0, asset_dir=str(asset),
                allow_missing_contacts=True,
            )
            self.pairs = json.loads(
                (configs / "linker_o6_keypoint_map.json").read_text(encoding="utf-8")
            )["pairs"]
            self.indices = None
        elif hand == "xhand":
            asset = assets / "xhand_right/urdf"
            self.model = HandModel_xhand(
                robot_name="xhand", urdf_filename="xhand_right.urdf", mesh_path="",
                batch_size=1, device=torch.device("cpu"), mesh_nsp=128,
                hand_scale=1.0, asset_dir=str(asset), allow_missing_contacts=True,
            )
            pairs = json.loads(
                (configs / "xhand_keypoint_map.json").read_text(encoding="utf-8")
            )["pairs"]
            self.indices = [int(item["xhand_index"]) for item in pairs]
        elif hand == "wuji":
            asset = assets / "wujihand_urdf/urdf"
            self.model = HandModel_xhand(
                robot_name="wuji_right", urdf_filename="right.urdf", mesh_path="../",
                batch_size=1, device=torch.device("cpu"), mesh_nsp=128,
                hand_scale=1.0, asset_dir=str(asset), allow_missing_contacts=True,
            )
            pairs = json.loads(
                (configs / "wuji_keypoint_map_v2.json").read_text(encoding="utf-8")
            )["pairs"]
            self.indices = [int(item["wuji_index"]) for item in pairs]
        else:
            raise ValueError(f"未知目标手: {hand}")

    def points_tensor(self, actions):
        """输入动作张量，输出保留关节梯度的15个语义点世界坐标。"""
        actions = torch.as_tensor(actions, dtype=torch.float32)
        if actions.ndim == 1:
            actions = actions[None]
        rotation = robust_compute_orth6d_from_eulerXYZ(actions[:, 3:6])
        model_q = torch.cat([actions[:, :3], rotation, actions[:, 6:]], dim=1)
        if self.hand != "linker":
            return self.model.get_penetraion_keypoints(q=model_q)[:, self.indices]
        self.model.update_kinematics(model_q)
        points = []
        for pair in self.pairs:
            local = torch.as_tensor(pair["linker_local_xyz"], dtype=torch.float32)
            local = local.view(1, 1, 3).expand(len(actions), -1, -1)
            point = self.model.current_status[pair["linker_link"]].transform_points(local)[:, 0]
            world = torch.bmm(
                point[:, None], self.model.global_rotation.transpose(1, 2)
            )[:, 0] + self.model.global_translation
            points.append(world * float(self.model.scale))
        return torch.stack(points, dim=1)

    def points(self, actions):
        """输入`(T,A)`策略动作，输出`(T,15,3)`语义点世界坐标。"""
        with torch.no_grad():
            return self.points_tensor(actions).cpu().numpy()


def policy_pose_from_observations(hand, observations):
    """把观测中的Isaac DOF顺序恢复成策略动作顺序。"""
    observations = np.asarray(observations, dtype=np.float32)
    if observations.ndim == 1:
        observations = observations[None]
    dof_count = (observations.shape[1] - 32) // 2
    positions = observations[:, :dof_count]
    return positions[:, PHYSICS_TO_POLICY[hand]].astype(np.float32)


def moving_object_points(initial_points_wrist, initial_command, observations):
    """由初始点云及逐步物体位姿恢复当前世界点云。"""
    observations = np.asarray(observations, dtype=np.float32)
    dof_count = (observations.shape[1] - 32) // 2
    object_position = observations[:, 2 * dof_count:2 * dof_count + 3]
    object_quaternion = observations[:, 2 * dof_count + 3:2 * dof_count + 7]
    wrist = np.asarray(initial_command, dtype=np.float32)
    wrist_rotation = Rotation.from_euler("xyz", wrist[3:6]).as_matrix().astype(np.float32)
    initial_world = np.asarray(initial_points_wrist, dtype=np.float32) @ wrist_rotation.T + wrist[:3]
    initial_object_rotation = Rotation.from_quat(object_quaternion[0]).as_matrix().astype(np.float32)
    object_local = (initial_world - object_position[0]) @ initial_object_rotation
    current_rotation = Rotation.from_quat(object_quaternion).as_matrix().astype(np.float32)
    return (
        np.einsum("pj,tij->tpi", object_local, current_rotation)
        + object_position[:, None]
    ).astype(np.float32)


def interaction_features(hand_points, object_points, wrist_euler, distance_scale=0.10):
    """编码15个语义点到当前物体表面的最近方向、距离和接近度。"""
    difference = object_points[:, None] - hand_points[:, :, None]
    squared = np.sum(difference * difference, axis=-1)
    nearest_index = np.argmin(squared, axis=-1)
    rows = np.arange(len(hand_points))[:, None]
    points = np.arange(hand_points.shape[1])[None]
    nearest = difference[rows, points, nearest_index]
    wrist_rotation = Rotation.from_euler("xyz", wrist_euler).as_matrix().astype(np.float32)
    local = np.einsum("tpi,tij->tpj", nearest, wrist_rotation)
    distance = np.sqrt(np.maximum(np.min(squared, axis=-1), 0.0))
    scaled_vector = np.clip(local / float(distance_scale), -2.0, 2.0)
    scaled_distance = np.clip(distance / float(distance_scale), 0.0, 2.0)
    proximity = np.exp(-distance / 0.02)
    return np.concatenate(
        [scaled_vector, scaled_distance[..., None], proximity[..., None]], axis=-1
    ).reshape(len(hand_points), -1).astype(np.float32)
