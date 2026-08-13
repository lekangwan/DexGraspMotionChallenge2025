"""为不同目标手复用的语义关键点可视化与距离报告函数。

输入：目标手名称、双方匹配点和双方三角mesh。
输出：交互HTML、三视图PNG或逐点距离JSON。
内部逻辑：绘制两只手、语义点和成对连线，并计算对应点欧氏距离。
作用：让XHand、Linker和Wuji使用一致的人工校准产物格式。
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go


def write_mapping_html(
    path: Path,
    target_name: str,
    semantics,
    source,
    target,
    shadow_mesh,
    target_mesh,
):
    """写出可旋转的双方mesh与语义点对应HTML。

    输入：输出路径、目标手名称、语义列表、双方点和双方mesh。
    输出：无返回值；在`path`写入Plotly HTML。
    逻辑：叠加mesh、带标签点和逐对紫色连线。
    作用：从任意视角人工排查错指、错点和坐标系问题。
    """
    figure = go.Figure()
    for mesh, name, color in (
        (shadow_mesh, "Shadow mesh", "lightblue"),
        (target_mesh, f"{target_name} mesh", "lightgreen"),
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
    for points, prefix, color in (
        (source, "S", "red"),
        (target, target_name, "blue"),
    ):
        figure.add_trace(
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode="markers+text",
                text=[f"{prefix}:{name}" for name in semantics],
                marker={"size": 5, "color": color},
                name=f"{prefix} matched points",
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
        title=f"Shadow ↔ {target_name} semantic keypoints",
        scene={"aspectmode": "data"},
    )
    figure.write_html(path)


def _sample_vertices(vertices, maximum=2500):
    """下采样mesh顶点供静态图快速显示。

    输入：完整顶点数组和最大保留数。
    输出：最多`maximum`个顶点。
    逻辑：超过上限时使用等间隔索引。
    作用：控制PNG绘制开销但保留手部轮廓。
    """
    if len(vertices) <= maximum:
        return vertices
    indices = np.linspace(0, len(vertices) - 1, maximum).astype(int)
    return vertices[indices]


def write_mapping_png(
    path: Path,
    target_name: str,
    semantics,
    source,
    target,
    shadow_mesh,
    target_mesh,
):
    """写出关键点对应的XY、XZ和YZ三视图。

    输入：输出路径、目标手名称、语义点对和双方mesh。
    输出：无返回值；在`path`写入PNG。
    逻辑：mesh顶点下采样后投影到三个正交平面，并连接对应点。
    作用：无需交互页面即可快速审查整体对齐和异常点。
    """
    shadow_vertices = _sample_vertices(np.asarray(shadow_mesh.vertices))
    target_vertices = _sample_vertices(np.asarray(target_mesh.vertices))
    projections = ((0, 1, "X", "Y"), (0, 2, "X", "Z"), (1, 2, "Y", "Z"))
    figure, axes = plt.subplots(1, 3, figsize=(18, 6))
    for axis, (a, b, label_a, label_b) in zip(axes, projections):
        axis.scatter(shadow_vertices[:, a], shadow_vertices[:, b], s=1, c="skyblue", alpha=0.10)
        axis.scatter(target_vertices[:, a], target_vertices[:, b], s=1, c="lightgreen", alpha=0.10)
        axis.scatter(source[:, a], source[:, b], s=28, c="red", label="Shadow")
        axis.scatter(target[:, a], target[:, b], s=28, c="blue", label=target_name)
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
    figure.suptitle(f"Shadow ↔ {target_name} semantic mapping")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def write_mapping_metrics(path: Path, target_name: str, semantics, source, target):
    """保存中性姿态逐点距离和总体摘要。

    输入：输出路径、目标手名称、语义名称和双方点坐标。
    输出：报告字典，同时写入JSON。
    逻辑：计算每对点欧氏距离及平均、最大值。
    作用：用数值补充可视化观察，便于校准前后对比。
    """
    distances = np.linalg.norm(source - target, axis=1)
    report = {
        "target_hand": target_name,
        "pair_count": len(semantics),
        "mean_distance_m": float(distances.mean()),
        "max_distance_m": float(distances.max()),
        "pairs": [
            {
                "semantic": name,
                "distance_m": float(distance),
                "shadow_xyz": source_point.tolist(),
                "target_xyz": target_point.tolist(),
            }
            for name, distance, source_point, target_point in zip(
                semantics, distances, source, target
            )
        ],
    }
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report

