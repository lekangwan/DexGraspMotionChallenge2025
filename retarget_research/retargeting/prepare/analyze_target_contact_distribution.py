#!/usr/bin/env python3
"""比较目标手五个指尖在物体表面的接触分布与法向对抗关系。

输入：手类型、Shadow源文件、目标候选、轨迹索引和物体mesh。
输出：逐帧指尖距离、接触跨度、法向夹角JSON及最佳闭合帧三视图PNG。
内部逻辑：恢复Linker或Wuji指尖，查询最近表面顶点/法向并统计对向接触。
作用：解释“指尖都很近但抓不住”的原因，为法向或力闭合损失提供证据。
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
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

from evaluate_linker_geometry import (  # noqa: E402
    build_models as build_linker_models,
    compute_linker_points,
    linker_to_model_q,
    load_pairs as load_linker_pairs,
)
from evaluate_wuji_geometry import (  # noqa: E402
    build_models as build_wuji_models,
    load_pairs as load_wuji_pairs,
    wuji_to_model_q,
)
from object_geometry import transformed_object_surface  # noqa: E402


def target_tip_points(hand, target_data, target_frames):
    """从保存轨迹恢复Linker或Wuji五个指尖世界坐标。

    输入：`linker/wuji`、候选字典和单条目标帧。
    输出：五个tip语义名称及`(T,5,3)`坐标。
    内部逻辑：按候选映射配置重做目标手正向运动学，再选择`*_tip`。
    作用：为两种结构不同的手提供统一接触分析输入。
    """
    semantics = list(target_data["mapping_semantics"])
    if hand == "linker":
        pairs = load_linker_pairs(semantics)
        _, model = build_linker_models()
        all_points = compute_linker_points(
            model, linker_to_model_q(target_frames), pairs
        )
    elif hand == "wuji":
        mapping_config = Path(
            target_data.get(
                "mapping_config",
                RETARGET_ROOT / "configs" / "wuji_keypoint_map.json",
            )
        )
        pairs = load_wuji_pairs(semantics, mapping_config)
        _, model = build_wuji_models()
        with torch.no_grad():
            full_points = model.get_penetraion_keypoints(
                q=wuji_to_model_q(target_frames)
            ).numpy()
        target_indices = [pair["wuji_index"] for pair in pairs]
        all_points = full_points[:, target_indices]
    else:
        raise ValueError(f"不支持的目标手: {hand}")
    tip_indices = [
        index for index, name in enumerate(semantics) if name.endswith("_tip")
    ]
    tip_names = [semantics[index] for index in tip_indices]
    if len(tip_indices) != 5:
        raise ValueError(f"接触分析需要五个指尖，实际为{tip_names}")
    return tip_names, np.asarray(all_points)[:, tip_indices]


def contact_distribution_metrics(
    tip_names, tip_points, surface_vertices, surface_normals, threshold
):
    """计算逐帧接触覆盖范围和表面法向对抗程度。

    输入：五指名称/坐标、物体顶点/法向和接触距离阈值。
    输出：完整报告字典及最佳覆盖帧的最近点、法向和距离。
    内部逻辑：KD-tree查询最近顶点；对阈值内点两两统计距离和法向夹角。
    作用：区分“多指聚在同侧”与“分布在相对表面形成夹持”的几何状态。
    """
    tree = cKDTree(surface_vertices)
    flat_distances, flat_indices = tree.query(tip_points.reshape(-1, 3), k=1)
    distances = flat_distances.reshape(tip_points.shape[:2])
    nearest_indices = flat_indices.reshape(tip_points.shape[:2])
    nearest_points = surface_vertices[nearest_indices]
    nearest_normals = surface_normals[nearest_indices]
    per_frame = []
    for frame_index in range(len(tip_points)):
        active = np.flatnonzero(distances[frame_index] <= threshold)
        separations, angles = [], []
        for first, second in itertools.combinations(active, 2):
            separations.append(
                np.linalg.norm(
                    nearest_points[frame_index, first]
                    - nearest_points[frame_index, second]
                )
            )
            cosine = np.clip(
                np.dot(
                    nearest_normals[frame_index, first],
                    nearest_normals[frame_index, second],
                ),
                -1.0,
                1.0,
            )
            angles.append(float(np.degrees(np.arccos(cosine))))
        per_frame.append(
            {
                "frame": frame_index,
                "contact_tip_count": int(len(active)),
                "mean_tip_surface_distance_m": float(distances[frame_index].mean()),
                "max_contact_point_separation_m": (
                    float(max(separations)) if separations else 0.0
                ),
                "max_contact_normal_angle_deg": (
                    float(max(angles)) if angles else 0.0
                ),
                "opposing_normal_pair_count_ge_120deg": int(
                    sum(angle >= 120.0 for angle in angles)
                ),
            }
        )
    maximum_count = max(frame["contact_tip_count"] for frame in per_frame)
    candidates = [
        frame["frame"]
        for frame in per_frame
        if frame["contact_tip_count"] == maximum_count
    ]
    best_frame = min(candidates, key=lambda index: distances[index].mean())
    selected = per_frame[best_frame]
    best_active = np.flatnonzero(distances[best_frame] <= threshold)
    best_pairs = []
    for first, second in itertools.combinations(best_active, 2):
        cosine = np.clip(
            np.dot(nearest_normals[best_frame, first], nearest_normals[best_frame, second]),
            -1.0,
            1.0,
        )
        angle = float(np.degrees(np.arccos(cosine)))
        best_pairs.append(
            {
                "first_tip": tip_names[first],
                "second_tip": tip_names[second],
                "surface_point_separation_m": float(
                    np.linalg.norm(
                        nearest_points[best_frame, first]
                        - nearest_points[best_frame, second]
                    )
                ),
                "surface_normal_angle_deg": angle,
                "opposing_ge_120deg": bool(angle >= 120.0),
            }
        )
    report = {
        "threshold_m": float(threshold),
        "tip_names": tip_names,
        "best_coverage_frame": int(best_frame),
        "maximum_contact_tip_count": int(maximum_count),
        "best_frame_metrics": selected,
        "best_frame_tips": {
            name: {
                "distance_m": float(distances[best_frame, index]),
                "inside_threshold": bool(distances[best_frame, index] <= threshold),
                "tip_xyz": tip_points[best_frame, index].tolist(),
                "nearest_surface_xyz": nearest_points[best_frame, index].tolist(),
                "nearest_surface_normal": nearest_normals[best_frame, index].tolist(),
            }
            for index, name in enumerate(tip_names)
        },
        "best_frame_contact_pairs": best_pairs,
        "per_tip": {
            name: {
                "minimum_surface_distance_m": float(distances[:, index].min()),
                "minimum_distance_frame": int(distances[:, index].argmin()),
                "first_frame_below_threshold": (
                    int(np.flatnonzero(distances[:, index] <= threshold)[0])
                    if np.any(distances[:, index] <= threshold)
                    else None
                ),
            }
            for index, name in enumerate(tip_names)
        },
        "per_frame": per_frame,
    }
    selected_data = {
        "tip_points": tip_points[best_frame],
        "surface_points": nearest_points[best_frame],
        "surface_normals": nearest_normals[best_frame],
        "distances": distances[best_frame],
    }
    return report, selected_data


def write_contact_png(
    path,
    hand,
    object_vertices,
    tip_names,
    selected_data,
    frame_index,
    threshold,
):
    """绘制最佳闭合帧的接触位置和物体外法向三视图。

    输入：路径、手名、物体点、五指数据、帧号和接触阈值。
    输出：无返回；写入XY/XZ/YZ三视图PNG。
    内部逻辑：画物体轮廓、指尖到最近点连线，并用箭头显示表面外法向。
    作用：让接触是否集中同侧、法向是否相反能够被直观看见。
    """
    if len(object_vertices) > 3500:
        indices = np.linspace(0, len(object_vertices) - 1, 3500).astype(int)
        object_vertices = object_vertices[indices]
    tips = selected_data["tip_points"]
    contacts = selected_data["surface_points"]
    normals = selected_data["surface_normals"]
    distances = selected_data["distances"]
    projections = ((0, 1, "X", "Y"), (0, 2, "X", "Z"), (1, 2, "Y", "Z"))
    figure, axes = plt.subplots(1, 3, figsize=(18, 6))
    colors = plt.cm.tab10(np.arange(5))
    for axis, (first, second, first_name, second_name) in zip(axes, projections):
        axis.scatter(
            object_vertices[:, first],
            object_vertices[:, second],
            s=2,
            c="lightgray",
            alpha=0.25,
        )
        for index, name in enumerate(tip_names):
            color = colors[index]
            axis.scatter(tips[index, first], tips[index, second], marker="x", s=55, color=color)
            axis.scatter(contacts[index, first], contacts[index, second], s=35, color=color)
            axis.plot(
                [tips[index, first], contacts[index, first]],
                [tips[index, second], contacts[index, second]],
                color=color,
                linewidth=1.2,
            )
            axis.arrow(
                contacts[index, first],
                contacts[index, second],
                normals[index, first] * 0.02,
                normals[index, second] * 0.02,
                color=color,
                width=0.00025,
                head_width=0.002,
                length_includes_head=True,
            )
            axis.annotate(
                f"{name} {distances[index] * 1000:.1f}mm",
                (contacts[index, first], contacts[index, second]),
                fontsize=7,
            )
        axis.set_xlabel(first_name + " (m)")
        axis.set_ylabel(second_name + " (m)")
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.2)
    figure.suptitle(
        f"{hand} contact distribution @ frame {frame_index}, threshold={threshold*1000:.0f} mm"
    )
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main():
    """运行单条目标轨迹的接触分布和法向分析。

    输入：hand、源/目标文件、轨迹索引、阈值、输出JSON/PNG。
    输出：报告、三视图及终端最佳帧摘要。
    内部逻辑：按源物体姿态构建表面，恢复目标指尖并调用纯统计和绘图函数。
    作用：为成功/失败目标手提供可直接横向比较的接触诊断。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--hand", choices=["linker", "wuji"], required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--source-index", type=int, default=0)
    parser.add_argument("--target-index", type=int, default=0)
    parser.add_argument("--object-name")
    parser.add_argument("--clearance", type=float, default=0.005)
    parser.add_argument("--threshold", type=float, default=0.010)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--png", type=Path, required=True)
    args = parser.parse_args()

    source_data = np.load(args.source, allow_pickle=True).item()
    target_data = np.load(args.target, allow_pickle=True).item()
    target_frames = np.asarray(
        target_data["grasp_seqs"][args.target_index], dtype=np.float32
    )
    object_name = args.object_name or args.source.stem
    object_dir = OBJECT_ROOT / object_name
    scale = float(np.asarray(source_data["obj_scale"])[args.source_index])
    rotation = np.asarray(source_data["obj_rotmat"])[args.source_index]
    vertices, normals = transformed_object_surface(
        object_dir, scale, rotation, args.clearance
    )
    tip_names, points = target_tip_points(args.hand, target_data, target_frames)
    report, selected_data = contact_distribution_metrics(
        tip_names, points, vertices, normals, args.threshold
    )
    report.update(
        {
            "hand": args.hand,
            "source": str(args.source.resolve()),
            "target": str(args.target.resolve()),
            "source_trajectory_index": args.source_index,
            "target_trajectory_index": args.target_index,
            "object_name": object_name,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.png.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_contact_png(
        args.png,
        args.hand,
        vertices,
        tip_names,
        selected_data,
        report["best_coverage_frame"],
        args.threshold,
    )
    best = report["best_frame_metrics"]
    print(f"best_coverage_frame={report['best_coverage_frame']}")
    print(f"contact_tip_count={best['contact_tip_count']}")
    print(f"max_contact_point_separation_m={best['max_contact_point_separation_m']:.6f}")
    print(f"max_contact_normal_angle_deg={best['max_contact_normal_angle_deg']:.2f}")
    print(f"opposing_pairs_ge_120deg={best['opposing_normal_pair_count_ge_120deg']}")
    print(f"output={args.output}")
    print(f"png={args.png}")


if __name__ == "__main__":
    main()
