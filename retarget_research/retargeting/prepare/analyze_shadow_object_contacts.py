#!/usr/bin/env python3
"""分析一条Shadow专家轨迹中五个指尖到物体表面的距离。

输入：Shadow源npy、轨迹索引、物体mesh目录和与重定向一致的Z偏移。
输出：逐指尖/逐帧最近表面顶点距离及接触阈值事件JSON。
内部逻辑：恢复Shadow世界关键点，将物体mesh按rotation/scale摆地后用KD-tree查询。
作用：验证接触阶段能否从源轨迹客观识别，为物体感知重定向项提供依据。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import cKDTree


RETARGET_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = RETARGET_ROOT.parent
REFERENCE_SCRIPTS = PROJECT_ROOT / "reference" / "HandRetargetTask2026" / "scripts"
OBJECT_ROOT = REFERENCE_SCRIPTS / "data" / "sorting" / "object_41"
RUN_DIR = RETARGET_ROOT / "run"
if str(RUN_DIR) not in sys.path:
    sys.path.insert(0, str(RUN_DIR))

from retarget_linker_keypoints import (  # noqa: E402
    build_shadow_model,
    shadow_keypoints,
)
from object_geometry import transformed_object_vertices  # noqa: E402


TIP_INDICES = {
    "index_tip": 4,
    "middle_tip": 8,
    "ring_tip": 12,
    "little_tip": 16,
    "thumb_tip": 20,
}


def first_frame_below(distances, threshold):
    """寻找距离第一次低于指定阈值的帧。

    输入：一维逐帧距离和米制阈值。
    输出：首个满足帧的整数；从未满足时为None。
    逻辑：用`np.flatnonzero`查找布尔条件的第一个索引。
    作用：把连续距离曲线压缩为容易比较的接触阶段事件。
    """
    indices = np.flatnonzero(np.asarray(distances) <= threshold)
    return None if len(indices) == 0 else int(indices[0])


def main():
    """计算并保存单条Shadow轨迹的指尖—物体距离报告。

    输入：命令行源文件、轨迹索引、可选物体名、Z偏移和输出路径。
    输出：JSON及终端最小距离/接触首帧摘要。
    逻辑：恢复21个关键点，对五个tip逐帧查询物体顶点KD-tree并统计阈值。
    作用：作为prepare阶段决定物体感知目标形式和接触起始帧的分析入口。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--trajectory-index", type=int, default=0)
    parser.add_argument("--object-name")
    parser.add_argument("--source-z-offset", type=float, default=0.4)
    parser.add_argument("--clearance", type=float, default=0.005)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    data = np.load(args.source, allow_pickle=True).item()
    frames = np.asarray(
        data["grasp_seqs"][args.trajectory_index], dtype=np.float32
    ).copy()
    frames[:, 2] += args.source_z_offset
    scale = float(np.asarray(data["obj_scale"])[args.trajectory_index])
    rotation = np.asarray(data["obj_rotmat"])[args.trajectory_index]
    object_name = args.object_name or args.source.stem
    object_dir = OBJECT_ROOT / object_name
    vertices = transformed_object_vertices(
        object_dir, scale, rotation, args.clearance
    )
    model = build_shadow_model()
    points = shadow_keypoints(frames, model)
    tree = cKDTree(vertices)

    per_tip = {}
    for name, index in TIP_INDICES.items():
        distances, _ = tree.query(points[:, index, :], k=1)
        per_tip[name] = {
            "min_distance_m": float(distances.min()),
            "mean_distance_m": float(distances.mean()),
            "final_distance_m": float(distances[-1]),
            "min_frame": int(distances.argmin()),
            "first_frame_below_5mm": first_frame_below(distances, 0.005),
            "first_frame_below_10mm": first_frame_below(distances, 0.010),
            "first_frame_below_20mm": first_frame_below(distances, 0.020),
            "per_frame_distance_m": distances.tolist(),
        }
    all_distances = np.stack(
        [per_tip[name]["per_frame_distance_m"] for name in TIP_INDICES], axis=1
    )
    report = {
        "source": str(args.source.resolve()),
        "trajectory_index": args.trajectory_index,
        "object_name": object_name,
        "frame_count": int(len(frames)),
        "source_z_offset_m": args.source_z_offset,
        "clearance_m": args.clearance,
        "object_scale": scale,
        "object_vertex_count": int(len(vertices)),
        "object_bounds_m": [vertices.min(axis=0).tolist(), vertices.max(axis=0).tolist()],
        "minimum_any_tip_distance_m": float(all_distances.min()),
        "per_tip": per_tip,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for name, metrics in per_tip.items():
        print(
            f"{name}: min={metrics['min_distance_m']:.6f}m "
            f"first<=10mm={metrics['first_frame_below_10mm']}"
        )
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
