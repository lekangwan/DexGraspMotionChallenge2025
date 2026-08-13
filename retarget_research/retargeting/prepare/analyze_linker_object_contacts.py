#!/usr/bin/env python3
"""分析Linker候选轨迹中指尖到物体表面的距离。

输入：Shadow源npy、Linker候选npy、双方轨迹索引和物体mesh。
输出：Linker各指尖逐帧最近表面顶点距离及接触阈值JSON。
内部逻辑：复用独立Linker正向运动学，在同一摆地物体KD-tree上查询五指距离。
作用：与Shadow专家接触报告对照，定位重定向过程中具体丢失的接触手指。
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
EVALUATE_DIR = RETARGET_ROOT / "evaluate"
PREPARE_DIR = RETARGET_ROOT / "prepare"
for path in (EVALUATE_DIR, PREPARE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from analyze_shadow_object_contacts import (  # noqa: E402
    first_frame_below,
)
from object_geometry import transformed_object_vertices  # noqa: E402
from evaluate_linker_geometry import (  # noqa: E402
    build_models,
    compute_linker_points,
    linker_to_model_q,
    load_pairs,
)


def main():
    """计算并保存一条Linker候选的五指表面距离。

    输入：命令行源/目标文件、双方索引、物体名、间隙和输出路径。
    输出：JSON及逐指尖最小距离、首次10 mm帧终端摘要。
    逻辑：按候选语义恢复校准点，只选择`*_tip`并查询物体顶点KD-tree。
    作用：作为prepare阶段比较专家接触与目标手接触保持质量的入口。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--source-index", type=int, default=0)
    parser.add_argument("--target-index", type=int, default=0)
    parser.add_argument("--object-name")
    parser.add_argument("--clearance", type=float, default=0.005)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source_data = np.load(args.source, allow_pickle=True).item()
    target_data = np.load(args.target, allow_pickle=True).item()
    frames = np.asarray(
        target_data["grasp_seqs"][args.target_index], dtype=np.float32
    )
    semantics = list(target_data["mapping_semantics"])
    pairs = load_pairs(semantics)
    scale = float(np.asarray(source_data["obj_scale"])[args.source_index])
    rotation = np.asarray(source_data["obj_rotmat"])[args.source_index]
    object_name = args.object_name or args.source.stem
    object_dir = OBJECT_ROOT / object_name
    vertices = transformed_object_vertices(
        object_dir, scale, rotation, args.clearance
    )
    _, linker = build_models()
    points = compute_linker_points(linker, linker_to_model_q(frames), pairs)
    tree = cKDTree(vertices)

    per_tip = {}
    for pair_index, semantic in enumerate(semantics):
        if not semantic.endswith("_tip"):
            continue
        distances, _ = tree.query(points[:, pair_index, :], k=1)
        per_tip[semantic] = {
            "min_distance_m": float(distances.min()),
            "mean_distance_m": float(distances.mean()),
            "final_distance_m": float(distances[-1]),
            "min_frame": int(distances.argmin()),
            "first_frame_below_5mm": first_frame_below(distances, 0.005),
            "first_frame_below_10mm": first_frame_below(distances, 0.010),
            "first_frame_below_20mm": first_frame_below(distances, 0.020),
            "per_frame_distance_m": distances.tolist(),
        }
    report = {
        "source": str(args.source.resolve()),
        "target": str(args.target.resolve()),
        "source_trajectory_index": args.source_index,
        "target_trajectory_index": args.target_index,
        "object_name": object_name,
        "frame_count": int(len(frames)),
        "clearance_m": args.clearance,
        "object_scale": scale,
        "object_vertex_count": int(len(vertices)),
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
