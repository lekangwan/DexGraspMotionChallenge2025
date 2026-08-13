#!/usr/bin/env python3
"""检查对向接触区域是否仍位于Shadow专家抓取位置附近。

输入：冻结manifest、物体mesh目录及对向区域选择超参数。
输出：逐轨迹五指中心位移/法向夹角JSON和终端摘要。
内部逻辑：只做Shadow正向运动学、物体表面查询和离散区域选择，不运行重定向优化。
作用：在昂贵轨迹优化前排除“法向对向但接触位置已偏离专家”的错误参数。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


RETARGET_ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = RETARGET_ROOT / "run"
if str(RUN_DIR) not in sys.path:
    sys.path.insert(0, str(RUN_DIR))

from phase_contact import build_phase_contact_plan  # noqa: E402
from retarget_linker_keypoints import (  # noqa: E402
    OBJECT_ROOT,
    SOURCE_TIP_INDICES,
    build_shadow_model,
    shadow_keypoints,
    transformed_object_surface,
)


def inspect_entry(entry, args, shadow_model):
    """计算一个物体全部冻结轨迹的对向区域诊断。

    输入：manifest条目、命令行参数和可复用Shadow模型。
    输出：每条轨迹的阶段帧、中心位移及拇指法向夹角字典。
    内部逻辑：按manifest索引读取轨迹，变换物体mesh，再调用共享阶段计划。
    作用：把物体级输入整理为便于比较的轨迹级结构化结果。
    """
    source_path = Path(entry["source_path"])
    source_data = np.load(source_path, allow_pickle=True).item()
    results = []
    for source_index in entry["trajectory_indices"]:
        trajectory = np.asarray(
            source_data["grasp_seqs"][source_index], dtype=np.float32
        ).copy()
        trajectory[:, 2] += args.source_z_offset
        object_vertices, object_normals = transformed_object_surface(
            args.object_root / entry["object_name"],
            np.asarray(source_data["obj_scale"])[source_index],
            np.asarray(source_data["obj_rotmat"])[source_index],
            args.object_clearance,
        )
        points = shadow_keypoints(trajectory, shadow_model)
        source_tips = {
            semantic: points[:, point_index, :]
            for semantic, point_index in SOURCE_TIP_INDICES.items()
        }
        plan = build_phase_contact_plan(
            trajectory,
            source_tips,
            object_vertices,
            object_normals,
            contact_threshold=args.phase_contact_threshold,
            min_contact_tips=args.phase_min_contact_tips,
            lift_delta=args.phase_lift_delta,
            region_neighbors=args.phase_region_neighbors,
            opposition_candidate_neighbors=args.opposition_candidate_neighbors,
            opposition_distance_scale=args.opposition_distance_scale,
            opposition_weight=args.opposition_weight,
            opposition_refine_frames=args.opposition_refine_frames,
        )
        results.append(
            {
                "source_trajectory_index": int(source_index),
                "close_start_frame": int(plan["close_start_frame"]),
                "lift_start_frame": int(plan["lift_start_frame"]),
                "grasp_frame": int(plan["grasp_frame"]),
                "opposition_diagnostics": plan["opposition_diagnostics"],
            }
        )
    return {"object_name": entry["object_name"], "trajectories": results}


def summarize(results):
    """汇总所有轨迹的语义位移和对向角度。

    输入：`inspect_entry`结果列表。
    输出：轨迹数、位移均值/最大值、法向夹角均值/最小值。
    内部逻辑：收集五指位移与四组拇指夹角后计算简单统计量。
    作用：用少量数字判断候选范围是否过大或对向约束是否过弱。
    """
    displacements, angles = [], []
    for object_result in results:
        for trajectory in object_result["trajectories"]:
            diagnostics = trajectory["opposition_diagnostics"]
            for semantic, values in diagnostics.items():
                if semantic == "shared_normalized_cost":
                    continue
                displacements.append(values["source_tip_to_center_m"])
                if values["thumb_normal_angle_deg"] is not None:
                    angles.append(values["thumb_normal_angle_deg"])
    return {
        "trajectory_count": sum(len(item["trajectories"]) for item in results),
        "mean_source_tip_displacement_m": float(np.mean(displacements)),
        "max_source_tip_displacement_m": float(np.max(displacements)),
        "mean_thumb_normal_angle_deg": float(np.mean(angles)),
        "min_thumb_normal_angle_deg": float(np.min(angles)),
    }


def main():
    """解析参数、运行几何检查并保存报告。

    输入：命令行manifest、输出路径和候选区域超参数。
    输出：包含方法、汇总和逐轨迹诊断的JSON。
    内部逻辑：复用一个Shadow模型顺序处理manifest条目。
    作用：提供对向接触选择的快速、确定性预检查入口。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--object-root", type=Path, default=OBJECT_ROOT)
    parser.add_argument("--source-z-offset", type=float, default=0.4)
    parser.add_argument("--object-clearance", type=float, default=0.005)
    parser.add_argument("--phase-contact-threshold", type=float, default=0.02)
    parser.add_argument("--phase-min-contact-tips", type=int, default=2)
    parser.add_argument("--phase-lift-delta", type=float, default=0.03)
    parser.add_argument("--phase-region-neighbors", type=int, default=32)
    parser.add_argument("--opposition-candidate-neighbors", type=int, default=256)
    parser.add_argument("--opposition-distance-scale", type=float, default=0.03)
    parser.add_argument("--opposition-weight", type=float, default=1.0)
    parser.add_argument("--opposition-refine-frames", type=int, default=4)
    args = parser.parse_args()
    if args.opposition_candidate_neighbors < 1:
        parser.error("--opposition-candidate-neighbors必须为正整数")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    shadow_model = build_shadow_model()
    results = [
        inspect_entry(entry, args, shadow_model) for entry in manifest["entries"]
    ]
    report = {
        "manifest": str(args.manifest.resolve()),
        "method": {
            "opposition_candidate_neighbors": args.opposition_candidate_neighbors,
            "opposition_distance_scale": args.opposition_distance_scale,
            "opposition_weight": args.opposition_weight,
            "phase_region_neighbors": args.phase_region_neighbors,
        },
        "summary": summarize(results),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
