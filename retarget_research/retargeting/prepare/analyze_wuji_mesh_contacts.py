#!/usr/bin/env python3
"""用Wuji真实手指mesh而非内部骨架点分析逐帧物体接触几何。

输入：Shadow源文件、Wuji候选、轨迹索引、物体mesh和距离阈值。
输出：每根手指实体最近距离/link、法向分布JSON和最佳帧PNG。
内部逻辑：变换五指全部link mesh顶点，在物体KD-tree中逐帧查询最近表面。
作用：弥合“关键点很近”与“PhysX真实mesh没有接触”的诊断差异。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import cKDTree
import torch


RETARGET_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = RETARGET_ROOT.parent
REFERENCE_SCRIPTS = PROJECT_ROOT / "reference" / "HandRetargetTask2026" / "scripts"
OBJECT_ROOT = REFERENCE_SCRIPTS / "data" / "sorting" / "object_41"
EVALUATE_DIR = RETARGET_ROOT / "evaluate"
for path in (Path(__file__).resolve().parent, EVALUATE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from analyze_target_contact_distribution import (  # noqa: E402
    contact_distribution_metrics,
    write_contact_png,
)
from evaluate_wuji_geometry import build_models, wuji_to_model_q  # noqa: E402
from object_geometry import transformed_object_surface  # noqa: E402


FINGER_NAMES = ["thumb", "index", "middle", "ring", "little"]


def unique_mesh_vertices(model, link_name):
    """取得一个Wuji link去重后的局部mesh顶点。

    输入：已加载Wuji模型和link名称。
    输出：`(V,3)`局部坐标，重复STL三角顶点已删除。
    内部逻辑：将坐标取到1微米精度后调用`np.unique`。
    作用：保持真实表面形状，同时把逐帧KD查询规模缩小约六倍。
    """
    vertices = np.asarray(model.mesh_verts[link_name], dtype=np.float32)
    return np.unique(np.round(vertices, decimals=6), axis=0)


def transform_link_vertices(model, link_name, local_vertices):
    """批量把一个link的局部mesh顶点转换到每帧世界坐标。

    输入：已用整条轨迹更新的Wuji模型、link名和局部顶点。
    输出：`(T,V,3)`世界坐标数组。
    内部逻辑：先应用逐帧link齐次变换，再应用手腕全局旋转和平移。
    作用：为真实手指表面到静态物体表面的最近距离查询提供坐标。
    """
    matrices = model.current_status[link_name].get_matrix().detach().numpy()
    homogeneous = np.concatenate(
        [local_vertices, np.ones((len(local_vertices), 1), dtype=np.float32)],
        axis=1,
    )
    hand = np.einsum("tij,vj->tvi", matrices, homogeneous)[..., :3]
    rotation = model.global_rotation.detach().numpy()
    translation = model.global_translation.detach().numpy()
    return (
        np.einsum("tij,tvj->tvi", rotation, hand)
        + translation[:, None, :]
    ) * float(model.scale)


def closest_finger_mesh_points(model, object_vertices):
    """计算五根完整手指mesh逐帧到物体的最近点。

    输入：已设置整条轨迹的Wuji模型和世界物体顶点。
    输出：距离、手表面点、物体点及最近link，形状以`(T,5)`为主。
    内部逻辑：每指遍历link1–4和tip，对每帧保留所有link中的最小KD距离。
    作用：识别真实接触来自指尖还是近端指节，而不是由语义点猜测。
    """
    frame_count = model.global_translation.shape[0]
    finger_count = len(FINGER_NAMES)
    distances = np.full((frame_count, finger_count), np.inf, dtype=np.float32)
    hand_points = np.zeros((frame_count, finger_count, 3), dtype=np.float32)
    object_points = np.zeros_like(hand_points)
    closest_links = np.empty((frame_count, finger_count), dtype=object)
    tree = cKDTree(object_vertices)
    for finger_index in range(1, 6):
        names = [f"finger{finger_index}_link{part}" for part in range(1, 5)]
        names.append(f"finger{finger_index}_tip_link")
        output_index = finger_index - 1
        for link_name in names:
            local = unique_mesh_vertices(model, link_name)
            world = transform_link_vertices(model, link_name, local)
            for frame in range(frame_count):
                link_distances, object_indices = tree.query(world[frame], k=1)
                best_vertex = int(np.argmin(link_distances))
                best_distance = float(link_distances[best_vertex])
                if best_distance >= distances[frame, output_index]:
                    continue
                distances[frame, output_index] = best_distance
                hand_points[frame, output_index] = world[frame, best_vertex]
                object_points[frame, output_index] = object_vertices[
                    object_indices[best_vertex]
                ]
                closest_links[frame, output_index] = link_name
    return distances, hand_points, object_points, closest_links


def main():
    """运行Wuji真实mesh接触分析并写出报告和图片。

    输入：源/目标、双方索引、物体名、阈值及JSON/PNG路径。
    输出：真实手指表面距离、最近link、法向分布和终端摘要。
    内部逻辑：恢复整条Wuji运动学，查询mesh最近点，再复用统一接触分布统计。
    作用：直接对照PhysX按刚体统计的实际接触，校准后续接触损失表示。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--source-index", type=int, default=0)
    parser.add_argument("--target-index", type=int, default=0)
    parser.add_argument("--object-name")
    parser.add_argument("--clearance", type=float, default=0.005)
    parser.add_argument("--threshold", type=float, default=0.002)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--png", type=Path, required=True)
    args = parser.parse_args()

    source_data = np.load(args.source, allow_pickle=True).item()
    target_data = np.load(args.target, allow_pickle=True).item()
    frames = np.asarray(
        target_data["grasp_seqs"][args.target_index], dtype=np.float32
    )
    object_name = args.object_name or args.source.stem
    object_dir = OBJECT_ROOT / object_name
    scale = float(np.asarray(source_data["obj_scale"])[args.source_index])
    rotation = np.asarray(source_data["obj_rotmat"])[args.source_index]
    vertices, normals = transformed_object_surface(
        object_dir, scale, rotation, args.clearance
    )
    _, model = build_models()
    with torch.no_grad():
        model.update_kinematics(wuji_to_model_q(frames))
    distances, hand_points, object_points, closest_links = (
        closest_finger_mesh_points(model, vertices)
    )
    # 复用统一统计时，hand_points再次查询同一物体；结果应与上面保存的最近点一致。
    report, selected_data = contact_distribution_metrics(
        FINGER_NAMES, hand_points, vertices, normals, args.threshold
    )
    best_frame = report["best_coverage_frame"]
    report.update(
        {
            "hand": "wuji_mesh",
            "source": str(args.source.resolve()),
            "target": str(args.target.resolve()),
            "source_trajectory_index": args.source_index,
            "target_trajectory_index": args.target_index,
            "object_name": object_name,
            "distance_definition": "minimum vertex distance over every link mesh in each finger",
            "best_frame_closest_links": {
                name: str(closest_links[best_frame, index])
                for index, name in enumerate(FINGER_NAMES)
            },
            "per_finger_mesh": {
                name: {
                    "minimum_distance_m": float(distances[:, index].min()),
                    "minimum_distance_frame": int(distances[:, index].argmin()),
                    "closest_link_at_minimum": str(
                        closest_links[distances[:, index].argmin(), index]
                    ),
                    "first_frame_below_threshold": (
                        int(np.flatnonzero(distances[:, index] <= args.threshold)[0])
                        if np.any(distances[:, index] <= args.threshold)
                        else None
                    ),
                }
                for index, name in enumerate(FINGER_NAMES)
            },
        }
    )
    # 使用第一次查询得到的确切最近手/物点，确保图中连线表示真实mesh间隙。
    selected_data["tip_points"] = hand_points[best_frame]
    selected_data["surface_points"] = object_points[best_frame]
    selected_data["distances"] = distances[best_frame]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.png.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_contact_png(
        args.png,
        "wuji_mesh",
        vertices,
        FINGER_NAMES,
        selected_data,
        best_frame,
        args.threshold,
    )
    print(f"best_coverage_frame={best_frame}")
    print(f"contact_finger_count={report['best_frame_metrics']['contact_tip_count']}")
    for name, metrics in report["per_finger_mesh"].items():
        print(
            f"{name}: min={metrics['minimum_distance_m']:.6f}m "
            f"link={metrics['closest_link_at_minimum']}"
        )
    print(f"output={args.output}")
    print(f"png={args.png}")


if __name__ == "__main__":
    main()
