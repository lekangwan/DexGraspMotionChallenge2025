#!/usr/bin/env python3
"""生成Shadow↔XHand关键点对应的交互HTML、静态图和距离摘要。

输入：XHand语义映射JSON、Shadow模型资产和XHand模型资产。
输出：中性姿态交互HTML、三视图PNG和逐点距离JSON。
内部逻辑：构造两只手中性姿态，通过正向运动学取得关键点并连接语义对。
作用：在运行批量优化前，确认关键点索引、手指语义和坐标系对齐。

本脚本只读取参考资产，绝不覆盖参考仓库中的关键点JSON。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import torch
import trimesh


RETARGET_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = RETARGET_ROOT.parent
REFERENCE_SCRIPTS = (
    PROJECT_ROOT / "reference" / "HandRetargetTask2026" / "scripts"
)
THIRD_PARTY_PK = (
    PROJECT_ROOT
    / "reference"
    / "HandRetargetTask2026"
    / "third_party"
    / "pytorch_kinematics"
)
for path in (REFERENCE_SCRIPTS, THIRD_PARTY_PK):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from utils.hand_model import HandModel as ShadowHandModel  # noqa: E402
from utils.HandModel_xhand import HandModel_xhand as XHandModel  # noqa: E402


def build_models():
    """创建CPU上的Shadow与XHand运动学模型。

    输入：无显式参数；从固定参考资产目录读取MJCF、URDF、mesh和关键点。
    输出：`(shadow, xhand)` 两个可做正向运动学的手模型。
    逻辑：使用参考仓库模型类，并关闭本步骤不需要的表面点采样。
    作用：为后续中性姿态关键点和mesh计算提供统一模型实例。
    """
    assets = REFERENCE_SCRIPTS / "assets"
    shadow_base = assets / "mjcf_free"
    shadow = ShadowHandModel(
        mjcf_path=str(shadow_base / "shadow_hand_vis_new.xml"),
        mesh_path=str(shadow_base / "meshes"),
        contact_points_path=str(shadow_base / "contact_points.json"),
        penetration_points_path=str(shadow_base / "penetration_points.json"),
        n_surface_points=0,
        device="cpu",
        use_joint21=True,
    )

    xhand_asset = assets / "xhand_right" / "urdf"
    xhand = XHandModel(
        robot_name="xhand",
        urdf_filename="xhand_right.urdf",
        mesh_path="",
        batch_size=1,
        device=torch.device("cpu"),
        mesh_nsp=128,
        hand_scale=1.0,
        asset_dir=str(xhand_asset),
        allow_missing_contacts=True,
    )
    return shadow, xhand


def neutral_poses(xhand):
    """构造两只手用于初步对齐检查的中性姿态。

    输入：已创建的XHand模型，用于读取12个关节名称。
    输出：Shadow的31维模型参数和XHand的`9+J`维模型参数。
    逻辑：关节全部置零，并复现参考基线中的XHand手腕旋转与平移对齐。
    作用：让两只手处于可直接比较关键点语义的共同坐标系。
    """
    shadow_q = torch.zeros((1, 31), dtype=torch.float32)
    shadow_q[:, 3:9] = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])

    joint_count = len(xhand.robot.get_joint_parameter_names())
    xhand_q = torch.zeros((1, 9 + joint_count), dtype=torch.float32)
    xhand_q[:, :3] = torch.tensor([0.006, 0.005, -0.01])
    rotation_alignment = torch.tensor(
        [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    xhand_q[:, 3:9] = torch.cat(
        [rotation_alignment[:, 0], rotation_alignment[:, 1]], dim=0
    )
    return shadow_q, xhand_q


def load_geometry():
    """读取语义配置并计算两只手中性姿态的全部几何信息。

    输入：无显式参数；读取XHand映射配置和参考模型资产。
    输出：映射字典、双方全部关键点、双方三角网格。
    逻辑：创建模型、设置中性姿态、执行正向运动学并提取mesh。
    作用：集中完成可视化和距离统计共同依赖的重型数据准备。
    """
    mapping = json.loads(
        (RETARGET_ROOT / "configs" / "xhand_keypoint_map.json").read_text()
    )
    shadow, xhand = build_models()
    shadow_q, xhand_q = neutral_poses(xhand)
    shadow.set_parameters(shadow_q)
    xhand.update_kinematics(xhand_q)

    shadow_points = shadow.get_penetraion_keypoints()[0].detach().cpu().numpy()
    xhand_points = xhand.get_penetraion_keypoints()[0].detach().cpu().numpy()

    shadow_mesh = trimesh.util.concatenate(shadow.get_trimesh_data(0))
    xhand_mesh = xhand.get_meshes_from_q(q=xhand_q, color=[1.0, 0.5, 0.2])
    return mapping, shadow_points, xhand_points, shadow_mesh, xhand_mesh


def selected_points(mapping, shadow_points, xhand_points):
    """根据语义配置选出需要比较的15对关键点。

    输入：映射字典，以及Shadow 21点和XHand 30点的世界坐标。
    输出：语义名称列表、Shadow选中点数组、XHand选中点数组。
    逻辑：按配置中的索引逐项取点并保持相同顺序。
    作用：把底层数字索引转换成可阅读、可比较的语义点对。
    """
    semantics = [pair["semantic"] for pair in mapping["pairs"]]
    source = np.stack(
        [shadow_points[pair["shadow_index"]] for pair in mapping["pairs"]]
    )
    target = np.stack(
        [xhand_points[pair["xhand_index"]] for pair in mapping["pairs"]]
    )
    return semantics, source, target


def write_interactive_html(path, semantics, source, target, shadow_mesh, xhand_mesh):
    """写出可旋转查看的关键点对应HTML。

    输入：输出路径、语义名称、双方匹配点和两只手mesh。
    输出：无返回值；在`path`写入Plotly HTML。
    逻辑：叠加两只手mesh、带语义标签的点和连接每对点的紫色线段。
    作用：支持人工从任意视角检查是否发生错指、错点或坐标系错误。
    """
    figure = go.Figure()
    for mesh, name, color in (
        (shadow_mesh, "Shadow mesh", "lightblue"),
        (xhand_mesh, "XHand mesh", "orange"),
    ):
        figure.add_trace(
            go.Mesh3d(
                x=mesh.vertices[:, 0],
                y=mesh.vertices[:, 1],
                z=mesh.vertices[:, 2],
                i=mesh.faces[:, 0],
                j=mesh.faces[:, 1],
                k=mesh.faces[:, 2],
                color=color,
                opacity=0.38,
                name=name,
            )
        )

    figure.add_trace(
        go.Scatter3d(
            x=source[:, 0],
            y=source[:, 1],
            z=source[:, 2],
            mode="markers+text",
            text=["S:" + name for name in semantics],
            marker={"size": 5, "color": "red"},
            name="Shadow matched points",
        )
    )
    figure.add_trace(
        go.Scatter3d(
            x=target[:, 0],
            y=target[:, 1],
            z=target[:, 2],
            mode="markers+text",
            text=["X:" + name for name in semantics],
            marker={"size": 5, "color": "blue"},
            name="XHand matched points",
        )
    )

    line_x, line_y, line_z = [], [], []
    for source_point, target_point in zip(source, target):
        line_x.extend([source_point[0], target_point[0], None])
        line_y.extend([source_point[1], target_point[1], None])
        line_z.extend([source_point[2], target_point[2], None])
    figure.add_trace(
        go.Scatter3d(
            x=line_x,
            y=line_y,
            z=line_z,
            mode="lines",
            line={"color": "purple", "width": 3},
            name="semantic pairs",
        )
    )
    figure.update_layout(
        title="Shadow ↔ XHand semantic keypoints (neutral pose)",
        scene={"aspectmode": "data"},
    )
    figure.write_html(path)


def _sample_vertices(vertices, maximum=2500):
    """等间隔抽取少量mesh顶点供静态图显示。

    输入：完整顶点数组和最多保留的顶点数。
    输出：不超过`maximum`个顶点的数组。
    逻辑：小数组原样返回，大数组使用等间隔索引下采样。
    作用：降低静态PNG绘制开销，同时保留手部大致轮廓。
    """
    if len(vertices) <= maximum:
        return vertices
    indices = np.linspace(0, len(vertices) - 1, maximum).astype(int)
    return vertices[indices]


def write_static_png(path, semantics, source, target, shadow_mesh, xhand_mesh):
    """写出XY、XZ、YZ三视图静态检查图。

    输入：输出路径、语义点对和两只手mesh。
    输出：无返回值；在`path`写入PNG。
    逻辑：下采样mesh顶点，在三个正交平面绘制双方点和对应连线。
    作用：无需交互浏览器即可快速审查整体对齐与异常点。
    """
    shadow_vertices = _sample_vertices(np.asarray(shadow_mesh.vertices))
    xhand_vertices = _sample_vertices(np.asarray(xhand_mesh.vertices))
    projections = ((0, 1, "X", "Y"), (0, 2, "X", "Z"), (1, 2, "Y", "Z"))
    figure, axes = plt.subplots(1, 3, figsize=(18, 6))
    for axis, (a, b, label_a, label_b) in zip(axes, projections):
        axis.scatter(
            shadow_vertices[:, a], shadow_vertices[:, b], s=1, c="skyblue", alpha=0.10
        )
        axis.scatter(
            xhand_vertices[:, a], xhand_vertices[:, b], s=1, c="orange", alpha=0.10
        )
        axis.scatter(source[:, a], source[:, b], s=28, c="red", label="Shadow")
        axis.scatter(target[:, a], target[:, b], s=28, c="blue", label="XHand")
        for name, source_point, target_point in zip(semantics, source, target):
            axis.plot(
                [source_point[a], target_point[a]],
                [source_point[b], target_point[b]],
                c="purple",
                alpha=0.45,
                linewidth=0.8,
            )
            if name.endswith("tip") or name == "palm":
                axis.annotate(name, (target_point[a], target_point[b]), fontsize=7)
        axis.set_xlabel(label_a + " (m)")
        axis.set_ylabel(label_b + " (m)")
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.2)
    axes[0].legend()
    figure.suptitle("Shadow ↔ XHand semantic mapping, neutral pose")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def write_metrics(path, semantics, source, target):
    """计算并保存中性姿态的逐语义点距离。

    输入：输出路径、语义名称和双方匹配点坐标。
    输出：包含平均、最大和逐点距离的报告字典，同时写入JSON。
    逻辑：对每对点计算欧氏距离并记录双方坐标。
    作用：为人工图像观察提供可量化的几何证据。
    """
    distances = np.linalg.norm(source - target, axis=1)
    report = {
        "pose": "zero_joint_neutral_pose",
        "note": "用于检查语义和坐标系，不代表优化后的最终误差。",
        "pair_count": len(semantics),
        "mean_distance_m": float(distances.mean()),
        "max_distance_m": float(distances.max()),
        "pairs": [
            {
                "semantic": name,
                "distance_m": float(distance),
                "shadow_xyz": source_point.tolist(),
                "xhand_xyz": target_point.tolist(),
            }
            for name, distance, source_point, target_point in zip(
                semantics, distances, source, target
            )
        ],
    }
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


def main():
    """执行XHand中性姿态映射检查命令。

    输入：命令行`--output-dir`，缺省写入项目`outputs/xhand_keypoint_check`。
    输出：HTML、PNG、JSON三个文件，并在终端打印点数和距离摘要。
    逻辑：依次加载几何、选择语义点、写三种产物。
    作用：作为prepare阶段可重复运行的XHand关键点验收入口。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "xhand_keypoint_check",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    mapping, shadow_points, xhand_points, shadow_mesh, xhand_mesh = load_geometry()
    semantics, source, target = selected_points(mapping, shadow_points, xhand_points)

    html_path = args.output_dir / "neutral_mapping.html"
    png_path = args.output_dir / "neutral_mapping.png"
    metrics_path = args.output_dir / "neutral_mapping_metrics.json"
    write_interactive_html(html_path, semantics, source, target, shadow_mesh, xhand_mesh)
    write_static_png(png_path, semantics, source, target, shadow_mesh, xhand_mesh)
    report = write_metrics(metrics_path, semantics, source, target)

    print(f"shadow_keypoints={len(shadow_points)}")
    print(f"xhand_keypoints={len(xhand_points)}")
    print(f"matched_pairs={report['pair_count']}")
    print(f"neutral_mean_distance_m={report['mean_distance_m']:.6f}")
    print(f"neutral_max_distance_m={report['max_distance_m']:.6f}")
    print(f"html={html_path}")
    print(f"png={png_path}")
    print(f"metrics={metrics_path}")


if __name__ == "__main__":
    main()
