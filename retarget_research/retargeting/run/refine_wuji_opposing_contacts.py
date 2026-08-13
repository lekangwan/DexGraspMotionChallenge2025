#!/usr/bin/env python3
"""用物体表面法向选择对向指尖，对Wuji基线轨迹做二阶段接触精修。

输入：Shadow源文件、已完成的Wuji候选、轨迹索引和接触/优化参数。
输出：保持26维格式的精修候选及逐帧选中接触对和锚点元数据。
内部逻辑：从基线最近表面中选拇指与法向最相反的手指，再局部重优化到外偏锚点。
作用：保留v2整体几何的同时，针对“有潜在对向分布但离表面稍远”的失败补接触。
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import nlopt
import numpy as np
from scipy.spatial import cKDTree
import torch


RETARGET_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = RETARGET_ROOT.parent
REFERENCE_SCRIPTS = PROJECT_ROOT / "reference" / "HandRetargetTask2026" / "scripts"
OBJECT_ROOT = REFERENCE_SCRIPTS / "data" / "sorting" / "object_41"
PREPARE_DIR = RETARGET_ROOT / "prepare"
EVALUATE_DIR = RETARGET_ROOT / "evaluate"
for path in (Path(__file__).resolve().parent, PREPARE_DIR, EVALUATE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from evaluate_wuji_geometry import wuji_to_model_q  # noqa: E402
from object_geometry import transformed_object_surface  # noqa: E402
from retarget_wuji_keypoints import (  # noqa: E402
    WujiFrameObjective,
    build_shadow_model,
    build_wuji_model,
    clip_start_to_bounds,
    load_pairs,
    shadow_keypoints,
)


def select_opposing_contact_anchors(
    tip_names,
    tip_points,
    surface_vertices,
    surface_normals,
    maximum_distance,
    minimum_normal_angle_deg,
    target_surface_offset,
):
    """为每帧选择拇指与法向最相反的一根非拇指并生成锚点。

    输入：五指名称/坐标、物体表面、候选距离、最小夹角和目标外偏距离。
    输出：`(T,5)`选择掩码、`(T,5,3)`锚点和逐帧选择记录。
    内部逻辑：两端都在候选距离内时，取与拇指法向夹角最大的非拇指。
    作用：只补具有对向夹持潜力的一对接触，避免无差别吸附全部指尖。
    """
    tip_names = list(tip_names)
    if "thumb_tip" not in tip_names:
        raise ValueError("五指列表中缺少thumb_tip")
    tree = cKDTree(surface_vertices)
    flat_distance, flat_index = tree.query(tip_points.reshape(-1, 3), k=1)
    distances = flat_distance.reshape(tip_points.shape[:2])
    nearest_indices = flat_index.reshape(tip_points.shape[:2])
    nearest_points = surface_vertices[nearest_indices]
    nearest_normals = surface_normals[nearest_indices]
    mask = np.zeros(tip_points.shape[:2], dtype=bool)
    anchors = np.zeros_like(tip_points, dtype=np.float32)
    records = []
    thumb = tip_names.index("thumb_tip")
    for frame in range(len(tip_points)):
        if distances[frame, thumb] > maximum_distance:
            continue
        best = None
        for other in range(len(tip_names)):
            if other == thumb or distances[frame, other] > maximum_distance:
                continue
            cosine = np.clip(
                np.dot(nearest_normals[frame, thumb], nearest_normals[frame, other]),
                -1.0,
                1.0,
            )
            angle = float(np.degrees(np.arccos(cosine)))
            if best is None or angle > best[0]:
                best = (angle, other)
        if best is None or best[0] < minimum_normal_angle_deg:
            continue
        angle, other = best
        for index in (thumb, other):
            mask[frame, index] = True
            anchors[frame, index] = (
                nearest_points[frame, index]
                + nearest_normals[frame, index] * target_surface_offset
            )
        records.append(
            {
                "frame": frame,
                "thumb_partner": tip_names[other],
                "surface_normal_angle_deg": angle,
                "thumb_initial_distance_m": float(distances[frame, thumb]),
                "partner_initial_distance_m": float(distances[frame, other]),
            }
        )
    return mask, anchors, records


def refine_file(args):
    """读取单条Wuji基线，选择对向锚点并逐帧精修后保存。

    输入：源/基线文件、双方索引、表面选择阈值和损失/参考权重。
    输出：一个`(1,T,26)`精修npy及锚点选择元数据。
    内部逻辑：无锚点帧原样保留；有锚点帧从基线初始化有界SLSQP。
    作用：使方法变化只发生在有明确法向证据的闭合帧，便于归因。
    """
    source_data = np.load(args.source, allow_pickle=True).item()
    initial_data = np.load(args.initial_target, allow_pickle=True).item()
    source_frames = np.asarray(
        source_data["grasp_seqs"][args.source_index], dtype=np.float32
    ).copy()
    source_frames[:, 2] += float(initial_data.get("source_z_offset", 0.4))
    initial_frames = np.asarray(
        initial_data["grasp_seqs"][args.target_index], dtype=np.float32
    )
    if initial_frames.shape != (len(source_frames), 26):
        raise ValueError("Wuji初始候选必须与源轨迹等长且每帧26维")
    mapping_config = Path(
        initial_data.get(
            "mapping_config",
            RETARGET_ROOT / "configs" / "wuji_keypoint_map.json",
        )
    )
    pairs = load_pairs(mapping_config)
    semantics = [pair["semantic"] for pair in pairs]
    source_indices = [pair["shadow_index"] for pair in pairs]
    target_indices = [pair["wuji_index"] for pair in pairs]
    tip_pair_indices = [
        index for index, name in enumerate(semantics) if name.endswith("_tip")
    ]
    tip_names = [semantics[index] for index in tip_pair_indices]
    tip_full_indices = [target_indices[index] for index in tip_pair_indices]

    shadow_model = build_shadow_model()
    wuji_model = build_wuji_model()
    target_points = shadow_keypoints(source_frames, shadow_model)[:, source_indices]
    with torch.no_grad():
        full_initial_points = wuji_model.get_penetraion_keypoints(
            q=wuji_to_model_q(initial_frames)
        ).numpy()
    tip_points = full_initial_points[:, tip_full_indices]
    object_name = args.object_name or args.source.stem
    object_dir = args.object_root / object_name
    scale = float(np.asarray(source_data["obj_scale"])[args.source_index])
    rotation = np.asarray(source_data["obj_rotmat"])[args.source_index]
    vertices, normals = transformed_object_surface(
        object_dir, scale, rotation, args.object_clearance
    )
    anchor_mask, anchors, records = select_opposing_contact_anchors(
        tip_names,
        tip_points,
        vertices,
        normals,
        args.maximum_anchor_distance,
        args.minimum_normal_angle,
        args.target_surface_offset,
    )

    joint_lower = wuji_model.revolute_joints_q_lower[0].detach().numpy()
    joint_upper = wuji_model.revolute_joints_q_upper[0].detach().numpy()
    lower = np.concatenate([joint_lower, np.full(3, -2.0), np.full(3, -np.pi)])
    upper = np.concatenate([joint_upper, np.full(3, 2.0), np.full(3, np.pi)])
    refined_internal, losses = [], []
    for frame in range(len(initial_frames)):
        baseline = np.concatenate(
            [initial_frames[frame, 6:], initial_frames[frame, :6]]
        ).astype(np.float32)
        active_tip_indices = np.flatnonzero(anchor_mask[frame])
        if len(active_tip_indices) == 0:
            refined_internal.append(baseline)
            losses.append(0.0)
            continue
        contact_indices = [tip_full_indices[index] for index in active_tip_indices]
        contact_targets = anchors[frame, active_tip_indices]
        objective = WujiFrameObjective(
            wuji_model,
            target_points[frame],
            target_indices,
            contact_indices=contact_indices,
            contact_targets=contact_targets,
            contact_weight=args.contact_weight,
            reference_values=baseline,
            reference_joint_weight=args.reference_joint_weight,
            reference_translation_weight=args.reference_translation_weight,
            reference_rotation_weight=args.reference_rotation_weight,
        )
        optimizer = nlopt.opt(nlopt.LD_SLSQP, 26)
        optimizer.set_min_objective(objective)
        optimizer.set_lower_bounds(lower.tolist())
        optimizer.set_upper_bounds(upper.tolist())
        optimizer.set_maxeval(args.maxeval)
        optimizer.set_xtol_rel(1e-6)
        optimizer.set_ftol_rel(1e-8)
        start = clip_start_to_bounds(baseline, lower, upper)
        try:
            result = optimizer.optimize(start)
        except (nlopt.RoundoffLimited, RuntimeError):
            result = start
        result = np.asarray(result, dtype=np.float32)
        refined_internal.append(result)
        losses.append(objective(result))
    refined_internal = np.stack(refined_internal)
    refined_frames = np.concatenate(
        [refined_internal[:, 20:26], refined_internal[:, :20]], axis=1
    ).astype(np.float32)
    output = {
        "grasp_seqs": refined_frames[None],
        "source_trajectory_indices": np.asarray([args.source_index]),
        "obj_rotmat": np.asarray(source_data["obj_rotmat"])[[args.source_index]],
        "obj_scale": np.asarray(source_data["obj_scale"])[[args.source_index]],
        "mapping_semantics": semantics,
        "mapping_config": str(mapping_config.resolve()),
        "wuji_joint_names": initial_data["wuji_joint_names"],
        "source_z_offset": float(initial_data.get("source_z_offset", 0.4)),
        "initial_target": str(args.initial_target.resolve()),
        "opposing_anchor_mask": anchor_mask[None],
        "opposing_anchor_targets": anchors[None],
        "opposing_anchor_records": records,
        "refinement_loss_per_frame": np.asarray(losses, dtype=np.float32)[None],
        "maximum_anchor_distance": float(args.maximum_anchor_distance),
        "minimum_normal_angle": float(args.minimum_normal_angle),
        "target_surface_offset": float(args.target_surface_offset),
        "contact_weight": float(args.contact_weight),
        "reference_joint_weight": float(args.reference_joint_weight),
        "reference_translation_weight": float(args.reference_translation_weight),
        "reference_rotation_weight": float(args.reference_rotation_weight),
        "maxeval": int(args.maxeval),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, output, allow_pickle=True)
    print(f"selected_contact_frames={len(records)}")
    print(f"selected_anchor_points={int(anchor_mask.sum())}")
    print(f"output_shape={output['grasp_seqs'].shape}")
    print(f"output={args.output}")


def main():
    """解析Wuji法向对向接触精修命令。

    输入：源/基线/输出路径和锚点选择、接触、参考姿态超参数。
    输出：精修候选npy和终端选中帧数摘要。
    内部逻辑：所有参数显式解析后调用`refine_file`。
    作用：提供可以和v2基线严格对照的二阶段方法入口。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--initial-target", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-index", type=int, default=0)
    parser.add_argument("--target-index", type=int, default=0)
    parser.add_argument("--object-root", type=Path, default=OBJECT_ROOT)
    parser.add_argument("--object-name")
    parser.add_argument("--object-clearance", type=float, default=0.005)
    parser.add_argument("--maximum-anchor-distance", type=float, default=0.020)
    parser.add_argument("--minimum-normal-angle", type=float, default=120.0)
    parser.add_argument("--target-surface-offset", type=float, default=0.005)
    parser.add_argument("--contact-weight", type=float, default=3.0)
    parser.add_argument("--reference-joint-weight", type=float, default=0.1)
    parser.add_argument("--reference-translation-weight", type=float, default=300.0)
    parser.add_argument("--reference-rotation-weight", type=float, default=0.1)
    parser.add_argument("--maxeval", type=int, default=100)
    refine_file(parser.parse_args())


if __name__ == "__main__":
    main()
