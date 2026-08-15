#!/usr/bin/env python3
"""无需GPU图形上下文，把物理状态轨迹渲染为手骨架+物体网格MP4。

输入：专家trace NPZ或新策略单轨迹JSON、手类型、物体资产目录和输出视频。
输出：固定视角H.264 MP4。
内部逻辑：解析目标手URDF并对每帧DOF做NumPy前向运动学，叠加物体真实网格/位姿后用Matplotlib CPU绘制。
作用：Isaac相机因无CUDA/Vulkan驱动不可用时仍能稳定生成可解释的成功、滑移和失败诊断视频。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from scipy.spatial.transform import Rotation
import trimesh


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_ASSETS = PROJECT_ROOT / "retarget_research/reference/HandRetargetTask2026/scripts/assets"
HAND_URDFS = {
    "linker": REFERENCE_ASSETS / "linkerhand/o6/right/linkerhand_o6_right6d.urdf",
    "xhand": REFERENCE_ASSETS / "xhand/xhand_euler_control.urdf",
    "wuji": REFERENCE_ASSETS / "wujihand_urdf/urdf/right6d.urdf",
}


def parse_vector(text, default):
    """把URDF空格向量属性解析为三维float数组。"""
    return np.asarray(default if text is None else [float(value) for value in text.split()], dtype=np.float64)


def homogeneous(rotation=None, translation=None):
    """由可选3×3旋转和三维平移构造4×4齐次变换。"""
    transform = np.eye(4, dtype=np.float64)
    if rotation is not None:
        transform[:3, :3] = rotation
    if translation is not None:
        transform[:3, 3] = translation
    return transform


def parse_urdf_tree(path):
    """解析URDF关节树为根link和父link到关节列表。

    输入：目标手6D虚拟手腕URDF。
    输出：根link、按父link分组的关节字典和全部边。
    内部逻辑：保存origin xyz/rpy、axis、类型与父子link，不依赖图形/机器人库。
    作用：软件渲染只需关节骨架，无需加载复杂手部mesh材质。
    """
    root = ET.parse(path).getroot()
    links = {node.get("name") for node in root.findall("link")}
    children = set()
    by_parent = {}
    edges = []
    for node in root.findall("joint"):
        parent = node.find("parent").get("link")
        child = node.find("child").get("link")
        children.add(child)
        origin = node.find("origin")
        xyz = parse_vector(None if origin is None else origin.get("xyz"), [0, 0, 0])
        rpy = parse_vector(None if origin is None else origin.get("rpy"), [0, 0, 0])
        axis_node = node.find("axis")
        axis = parse_vector(None if axis_node is None else axis_node.get("xyz"), [1, 0, 0])
        joint = {
            "name": node.get("name"), "type": node.get("type"),
            "parent": parent, "child": child, "xyz": xyz, "rpy": rpy, "axis": axis,
        }
        by_parent.setdefault(parent, []).append(joint)
        edges.append((parent, child))
    roots = sorted(links - children)
    if len(roots) != 1:
        raise ValueError(f"URDF根link数量异常: {roots}")
    return roots[0], by_parent, edges


def joint_motion(joint_type, axis, value):
    """计算一个revolute/continuous/prismatic/fixed关节的局部运动变换。"""
    if joint_type in {"revolute", "continuous"}:
        norm = np.linalg.norm(axis)
        rotation = Rotation.from_rotvec(axis / max(norm, 1e-12) * float(value)).as_matrix()
        return homogeneous(rotation=rotation)
    if joint_type == "prismatic":
        return homogeneous(translation=axis * float(value))
    return np.eye(4, dtype=np.float64)


def forward_link_positions(root_link, by_parent, joint_values):
    """对一帧DOF执行URDF树前向运动学并返回每个link原点。

    输入：根link、关节树和名称到角度/位移的映射。
    输出：link名称到三维世界位置字典。
    内部逻辑：父变换乘origin固定变换，再乘当前关节运动变换并递归。
    作用：用实际物理DOF状态画出目标手腕、掌部和五指骨架。
    """
    transforms = {root_link: np.eye(4, dtype=np.float64)}
    queue = [root_link]
    while queue:
        parent = queue.pop(0)
        for joint in by_parent.get(parent, []):
            origin = homogeneous(
                Rotation.from_euler("xyz", joint["rpy"]).as_matrix(), joint["xyz"]
            )
            motion = joint_motion(
                joint["type"], joint["axis"], joint_values.get(joint["name"], 0.0)
            )
            transforms[joint["child"]] = transforms[parent] @ origin @ motion
            queue.append(joint["child"])
    return {name: transform[:3, 3].copy() for name, transform in transforms.items()}


def load_state_trajectory(path):
    """统一读取专家NPZ或策略JSON中的手/物状态轨迹。

    输入：trace或rollout报告路径。
    输出：手类型、DOF名称/位置、物体位置/四元数、源路径和抽帧间隔。
    内部逻辑：NPZ读取metadata_json；JSON要求新评测器保存实际DOF和物体四元数。
    作用：同一个软件渲染器覆盖重定向专家与学习策略。
    """
    path = Path(path)
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata_json"].item()))
            return {
                "hand": metadata["hand"],
                "dof_names": list(metadata["physics_dof_names"]),
                "dof_positions": archive["hand_dof_position"].copy(),
                "object_positions": archive["object_position"].copy(),
                "object_quaternions": archive["object_quaternion_xyzw"].copy(),
                "source": metadata["source"],
                "source_index": int(metadata["source_trajectory_index"]),
                "capture_every": int(metadata.get("steps_per_frame", 3)),
            }
    report = json.loads(path.read_text(encoding="utf-8"))
    required = {"actual_hand_dof_positions", "object_positions_m", "object_quaternions_xyzw"}
    missing = required - set(report)
    if missing:
        raise ValueError(f"策略报告缺少软件渲染状态{sorted(missing)}，需用新评测器重跑")
    return {
        "hand": report["hand"],
        "dof_names": list(report["physics_dof_names"]),
        "dof_positions": np.asarray(report["actual_hand_dof_positions"], dtype=np.float64),
        "object_positions": np.asarray(report["object_positions_m"], dtype=np.float64),
        "object_quaternions": np.asarray(report["object_quaternions_xyzw"], dtype=np.float64),
        "source": report["source"],
        "source_index": int(report["source_trajectory_index"]),
        "capture_every": 3,
    }


def transformed_object_vertices(vertices, position, quaternion, scale):
    """把物体局部mesh顶点按数据scale和物理根位姿变换到世界坐标。"""
    return Rotation.from_quat(quaternion).apply(vertices * float(scale)) + position


def render_video(state, object_dir, output, fps=20, max_faces=1200, title=""):
    """逐抽帧绘制手骨架、物体网格和地面并编码MP4。

    输入：统一状态、物体目录、输出、帧率、最大面数和标题。
    输出：视频帧数和绝对路径。
    内部逻辑：固定全程坐标范围，物体面过多时确定性等间隔抽样；每帧重做FK。
    作用：完全避开Isaac/Vulkan图形路径，结果适合作为失败机理诊断和报告后备视频。
    """
    hand = state["hand"]
    root_link, by_parent, edges = parse_urdf_tree(HAND_URDFS[hand])
    source = np.load(state["source"], allow_pickle=True).item()
    scale = float(np.asarray(source["obj_scale"])[state["source_index"]])
    mesh = trimesh.load_mesh(Path(object_dir) / "coacd" / "decomposed.obj", process=False)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if len(faces) > max_faces:
        face_indices = np.linspace(0, len(faces) - 1, max_faces, dtype=np.int64)
        faces = faces[face_indices]
    sampled = np.arange(0, len(state["dof_positions"]), state["capture_every"], dtype=np.int64)
    skeletons = []
    all_points = []
    for index in sampled:
        values = dict(zip(state["dof_names"], state["dof_positions"][index]))
        positions = forward_link_positions(root_link, by_parent, values)
        skeletons.append(positions)
        all_points.extend(positions.values())
        world_vertices = transformed_object_vertices(
            vertices, state["object_positions"][index], state["object_quaternions"][index], scale
        )
        all_points.append(world_vertices.min(axis=0))
        all_points.append(world_vertices.max(axis=0))
    all_points = np.asarray(all_points)
    initial_object_z = float(state["object_positions"][0, 2])
    initial_vertices = transformed_object_vertices(
        vertices,
        state["object_positions"][0],
        state["object_quaternions"][0],
        scale,
    )
    table_z = float(initial_vertices[:, 2].min())
    target_z = initial_object_z + 0.30
    guide_xy = (all_points.min(axis=0) + all_points.max(axis=0))[:2] / 2.0
    all_points = np.concatenate(
        [
            all_points,
            np.asarray(
                [
                    [guide_xy[0], guide_xy[1], table_z],
                    [guide_xy[0], guide_xy[1], target_z],
                ]
            ),
        ],
        axis=0,
    )
    minimum, maximum = all_points.min(axis=0), all_points.max(axis=0)
    center = (minimum + maximum) / 2.0
    radius = max(float(np.max(maximum - minimum)) * 0.62, 0.12)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(output, fps=int(fps), codec="libx264", quality=8)
    try:
        for video_index, state_index in enumerate(sampled):
            figure = plt.figure(figsize=(6.4, 4.8), dpi=100)
            axis = figure.add_subplot(111, projection="3d")
            positions = skeletons[video_index]
            for parent, child in edges:
                if parent in positions and child in positions:
                    points = np.stack([positions[parent], positions[child]])
                    is_virtual = "virtual" in parent.lower() or "virtual" in child.lower()
                    axis.plot(
                        points[:, 0], points[:, 1], points[:, 2],
                        color="#7A7A7A" if is_virtual else "#1464A5",
                        linewidth=1.3 if is_virtual else 3.0,
                        linestyle="--" if is_virtual else "-",
                    )
            hand_points = np.stack(list(positions.values()))
            axis.scatter(hand_points[:, 0], hand_points[:, 1], hand_points[:, 2], s=10, color="#0A3157")
            world_vertices = transformed_object_vertices(
                vertices,
                state["object_positions"][state_index],
                state["object_quaternions"][state_index],
                scale,
            )
            collection = Poly3DCollection(
                world_vertices[faces], facecolor="#E58C2B", edgecolor="#8A4B08",
                linewidth=0.08, alpha=0.9,
            )
            axis.add_collection3d(collection)
            plane_extent = radius * 0.82
            plane_x = np.asarray([center[0] - plane_extent, center[0] + plane_extent])
            plane_y = np.asarray([center[1] - plane_extent, center[1] + plane_extent])
            plane_x, plane_y = np.meshgrid(plane_x, plane_y)
            axis.plot_surface(
                plane_x,
                plane_y,
                np.full_like(plane_x, table_z),
                color="#C8B79A",
                alpha=0.22,
                shade=False,
            )
            axis.plot(
                [center[0] - plane_extent, center[0] + plane_extent],
                [center[1], center[1]],
                [initial_object_z, initial_object_z],
                color="#5E6B73",
                linestyle="--",
                linewidth=1.2,
                label="initial object height",
            )
            axis.plot(
                [center[0] - plane_extent, center[0] + plane_extent],
                [center[1], center[1]],
                [target_z, target_z],
                color="#C62828",
                linestyle="--",
                linewidth=2.0,
                label="+30 cm target",
            )
            axis.set_xlim(center[0] - radius, center[0] + radius)
            axis.set_ylim(center[1] - radius, center[1] + radius)
            axis.set_zlim(center[2] - radius, center[2] + radius)
            axis.view_init(elev=22, azim=-48)
            axis.set_box_aspect((1, 1, 1))
            axis.set_xlabel("x / m")
            axis.set_ylabel("y / m")
            axis.set_zlabel("z / m")
            current_lift_cm = 100.0 * (
                float(state["object_positions"][state_index, 2]) - initial_object_z
            )
            prefix = title or hand
            axis.set_title(
                f"{prefix}  frame={video_index + 1}/{len(sampled)}  "
                f"lift={current_lift_cm:+.1f} cm"
            )
            axis.legend(loc="upper left", fontsize=7)
            figure.tight_layout()
            figure.canvas.draw()
            frame = np.asarray(figure.canvas.buffer_rgba())[:, :, :3].copy()
            writer.append_data(frame)
            plt.close(figure)
    finally:
        writer.close()
    return {"video": str(Path(output).resolve()), "video_frame_count": len(sampled)}


def main():
    """解析状态/资产路径并执行CPU软件渲染。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--object-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--max-faces", type=int, default=1200)
    parser.add_argument("--title", default="")
    args = parser.parse_args()
    state = load_state_trajectory(args.state)
    result = render_video(state, args.object_dir, args.output, args.fps, args.max_faces, args.title)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"SOFTWARE_VIDEO={args.output.resolve()}")


if __name__ == "__main__":
    main()
