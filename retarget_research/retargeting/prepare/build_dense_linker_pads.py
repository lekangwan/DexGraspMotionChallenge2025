#!/usr/bin/env python3
"""从Linker五个远端网格生成稠密、确定性的接触表面配置。

输入：已有Linker指腹配置、每指采样数、近关节端裁剪比例和输出JSON路径。
输出：保留原body/mesh对应关系、但用稠密网格点覆盖远端连杆表面的新配置。
内部逻辑：读取每个STL，裁掉最靠近关节的一小段，再用最远点采样覆盖剩余表面。
作用：弥补单条Planter成功轨迹只能校准每指6个经验接触点、无法覆盖新物体接触面的不足。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh


def farthest_point_indices(points, count):
    """确定性选择覆盖空间范围的点索引。

    输入：`(N,3)`候选点和期望数量。
    输出：不重复的一维整数索引，数量为`min(N,count)`。
    内部逻辑：从离整体中位数最近的点开始，每次加入离已选集合最远的点。
    作用：比随机采样更可复现，也比按顶点顺序截取更能覆盖指腹正面、侧面和端部。
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError(f"points应为非空(N,3)，实际为{points.shape}")
    count = min(int(count), len(points))
    if count < 1:
        raise ValueError("count必须为正整数")
    center = np.median(points, axis=0)
    selected = [int(np.argmin(np.linalg.norm(points - center, axis=1)))]
    nearest = np.linalg.norm(points - points[selected[0]], axis=1)
    while len(selected) < count:
        nearest[selected] = -1.0
        next_index = int(np.argmax(nearest))
        selected.append(next_index)
        nearest = np.minimum(
            nearest,
            np.linalg.norm(points - points[next_index], axis=1),
        )
    return np.asarray(selected, dtype=np.int64)


def dense_finger_surface(mesh_path, point_count, proximal_trim_fraction):
    """从一个远端指节STL提取均匀覆盖的局部表面点和法向。

    输入：mesh路径、采样数和沿局部Z轴裁掉的近端比例。
    输出：采样点、单位外法向、原顶点数和裁剪后候选数。
    内部逻辑：保留Z坐标不低于指定分位数的顶点，再执行确定性最远点采样。
    作用：排除靠近关节、通常不参与抓持的窄区域，同时保留完整远端接触可能性。
    """
    if not 0 <= proximal_trim_fraction < 0.5:
        raise ValueError("proximal_trim_fraction必须在[0,0.5)范围内")
    mesh = trimesh.load_mesh(mesh_path, process=False)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    normals = np.asarray(mesh.vertex_normals, dtype=np.float64)
    threshold = np.quantile(vertices[:, 2], float(proximal_trim_fraction))
    candidate_indices = np.flatnonzero(vertices[:, 2] >= threshold)
    local_indices = farthest_point_indices(vertices[candidate_indices], point_count)
    selected = candidate_indices[local_indices]
    selected_normals = normals[selected]
    selected_normals /= np.maximum(
        np.linalg.norm(selected_normals, axis=1, keepdims=True), 1e-12
    )
    return vertices[selected], selected_normals, len(vertices), len(candidate_indices)


def build_dense_config(base_config, point_count, proximal_trim_fraction):
    """把五指6点经验配置扩展为网格稠密配置。

    输入：已验证的Linker配置字典、每指点数和近端裁剪比例。
    输出：可由`load_pad_config`直接读取的新字典。
    内部逻辑：复用原配置的body和mesh路径，逐指调用网格采样并记录生成元数据。
    作用：保持运动学link对应不变，只扩大用于接触优化的表面覆盖范围。
    """
    if base_config.get("hand") != "linker":
        raise ValueError("base_config必须属于linker")
    fingers = {}
    for semantic, info in base_config["fingers"].items():
        points, normals, vertex_count, candidate_count = dense_finger_surface(
            Path(info["mesh_path"]), point_count, proximal_trim_fraction
        )
        fingers[semantic] = {
            "body_name": info["body_name"],
            "mesh_path": info["mesh_path"],
            "mesh_vertex_count": int(vertex_count),
            "candidate_vertex_count": int(candidate_count),
            "surface_points": [
                {
                    "local_xyz_m": point.tolist(),
                    "local_outward_normal": normal.tolist(),
                    "source": "distal_mesh_farthest_point_sampling",
                }
                for point, normal in zip(points, normals)
            ],
        }
    return {
        "status": "geometry_dense_surface_development_v1",
        "hand": "linker",
        "calibration_rule": (
            "复用物理校准的远端body/mesh对应；裁掉局部Z最低的近关节区域；"
            "对剩余STL顶点做确定性最远点采样"
        ),
        "points_per_finger": int(point_count),
        "proximal_trim_fraction": float(proximal_trim_fraction),
        "base_config_status": base_config.get("status"),
        "fingers": fingers,
    }


def main():
    """解析参数、生成并保存稠密Linker接触配置。

    输入：命令行`--base-config/--output/--points-per-finger/--trim-fraction`。
    输出：UTF-8 JSON和逐指候选/采样数量摘要。
    内部逻辑：读取基线配置后调用`build_dense_config`，不修改原配置和只读STL。
    作用：作为prepare分区中可重复执行的几何预处理入口。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--points-per-finger", type=int, default=64)
    parser.add_argument("--trim-fraction", type=float, default=0.10)
    args = parser.parse_args()
    base = json.loads(args.base_config.read_text(encoding="utf-8"))
    config = build_dense_config(
        base, args.points_per_finger, args.trim_fraction
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    for semantic, info in config["fingers"].items():
        print(
            f"{semantic}: candidates={info['candidate_vertex_count']} "
            f"selected={len(info['surface_points'])}"
        )
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
