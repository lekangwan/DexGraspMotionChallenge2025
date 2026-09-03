"""构造策略使用的初始手—物几何点云。"""

from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation


def farthest_point_indices(points, count):
    """从mesh顶点确定性选取分布较均匀的点。"""
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError("点集必须是非空(V,3)数组")
    count = int(count)
    center = points.mean(axis=0)
    first = int(np.argmax(np.sum((points - center) ** 2, axis=1)))
    selected = [first]
    distance = np.sum((points - points[first]) ** 2, axis=1)
    for _ in range(1, min(count, len(points))):
        index = int(np.argmax(distance))
        selected.append(index)
        distance = np.minimum(
            distance, np.sum((points - points[index]) ** 2, axis=1)
        )
    if len(selected) < count:
        original = selected.copy()
        selected.extend(original[index % len(original)] for index in range(count - len(selected)))
    return np.asarray(selected, dtype=np.int64)


def object_points_in_initial_wrist(
    object_dir, scale, rotation_matrix, initial_command, point_count=128, clearance=0.005
):
    """把物体表面采样点转换到episode初始手腕坐标系。"""
    mesh_path = Path(object_dir) / "coacd" / "decomposed.obj"
    mesh = trimesh.load_mesh(mesh_path, process=False)
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    rotation_matrix = np.asarray(rotation_matrix, dtype=np.float32)
    world = vertices @ rotation_matrix.T * float(scale)
    world[:, 2] += float(clearance) - float(world[:, 2].min())
    sampled = world[farthest_point_indices(world, point_count)]
    command = np.asarray(initial_command, dtype=np.float32)
    if command.ndim != 1 or len(command) < 6:
        raise ValueError("初始命令必须至少包含6维手腕位姿")
    wrist_rotation = Rotation.from_euler("xyz", command[3:6]).as_matrix().astype(np.float32)
    return ((sampled - command[None, :3]) @ wrist_rotation).astype(np.float32)
