#!/usr/bin/env python3
"""按专家接触与目标手指尖距离差，只补偿明显缺接触的手指。

输入：手类型、冻结manifest、当前候选目录、单指最大关节残差和距离阈值。
输出：与manifest对齐的唯一候选npy，以及每指激活、距离和关节残差审计。
内部逻辑：在Shadow专家抓取帧判定哪些指头本应靠近物体；若目标手同指
距离还明显更大，就用数值运动学梯度找到减小距离的小关节残差。残差从闭合帧
渐进加入，抓取帧后保持，同时完整保留原轨迹的手腕和后续关节动态。
作用：比“五指统一继续夹紧”更有针对性地恢复缺失接触，并共享同一规则
覆盖Linker、XHand和Wuji三种不同自由度结构。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Callable

import numpy as np
from scipy.spatial import cKDTree
import torch


RUN_DIR = Path(__file__).resolve().parent
RETARGET_ROOT = RUN_DIR.parent
PREPARE_DIR = RETARGET_ROOT / "prepare"
EVALUATE_DIR = RETARGET_ROOT / "evaluate"
for path in (RUN_DIR, PREPARE_DIR, EVALUATE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import evaluate_linker_geometry as linker_geometry  # noqa: E402
import evaluate_wuji_geometry as wuji_geometry  # noqa: E402
import evaluate_xhand_geometry as xhand_geometry  # noqa: E402
from object_geometry import transformed_object_vertices  # noqa: E402
from refine_linker_squeeze import squeeze_progress  # noqa: E402
from refine_shared_grasp_center import (  # noqa: E402
    SHADOW_TIP_INDICES,
    TIP_SEMANTICS,
    build_hand_models,
    existing_or_inferred_phase,
    shadow_points_and_phases,
    xhand_tip_indices,
)
from slice_manifest_candidates import slice_candidate  # noqa: E402
from wuji_candidate_utils import trajectory_mapping_metadata  # noqa: E402


HAND_DIMENSIONS = {"linker": 12, "xhand": 18, "wuji": 26}
FINGER_JOINT_GROUPS = {
    "xhand": {
        "thumb": [0, 1, 2],
        "index": [3, 4, 5],
        "middle": [6, 7],
        "ring": [8, 9],
        "little": [10, 11],
    },
    "wuji": {
        "thumb": [0, 1, 2, 3],
        "index": [4, 5, 6, 7],
        "middle": [8, 9, 10, 11],
        "ring": [12, 13, 14, 15],
        "little": [16, 17, 18, 19],
    },
    "linker": {
        "thumb": [0, 1],
        "index": [2],
        "middle": [3],
        "ring": [4],
        "little": [5],
    },
}


def joint_limits(target_model) -> tuple[np.ndarray, np.ndarray]:
    """读取目标手优化关节的上下限。

    输入：三手共用的目标运动学模型。
    输出：两个`(J,)` float32数组。
    内部逻辑：兼容模型中`(1,J)`与`(J,)`两种限位形状并拉平。
    作用：所有数值扰动和最终轨迹都不能越过URDF可执行范围。
    """
    lower = np.asarray(
        target_model.revolute_joints_q_lower.detach().cpu().numpy(),
        dtype=np.float32,
    ).reshape(-1)
    upper = np.asarray(
        target_model.revolute_joints_q_upper.detach().cpu().numpy(),
        dtype=np.float32,
    ).reshape(-1)
    if lower.shape != upper.shape or np.any(lower >= upper):
        raise ValueError(f"目标手关节限位无效: {lower.shape}, {upper.shape}")
    return lower, upper


def target_tip_positions(
    hand: str,
    target_model,
    data: dict,
    frame: np.ndarray,
    trajectory_position: int,
) -> dict[str, np.ndarray]:
    """计算一帧目标手的五个语义指尖世界坐标。

    输入：手类型、模型、候选元数据、单帧动作和轨迹行号。
    输出：食/中/环/小/拇指名到三维点的字典。
    内部逻辑：Linker使用实际link局部标定点；XHand/Wuji使用各自
    映射JSON中的指尖索引，并调用与几何评测一致的格式转换。
    作用：隔离三只手的关节和点数差异，为分指距离优化提供同一接口。
    """
    one = np.asarray(frame, dtype=np.float32)[None, :]
    if hand == "linker":
        semantics = [str(value) for value in data.get("mapping_semantics", [])]
        pairs = linker_geometry.load_pairs(semantics)
        points = linker_geometry.compute_linker_points(
            target_model, linker_geometry.linker_to_model_q(one), pairs
        )[0]
        by_name = {
            semantic[: -len("_tip")]: points[semantics.index(semantic)]
            for semantic in TIP_SEMANTICS
        }
    elif hand == "xhand":
        with torch.no_grad():
            points = (
                target_model.get_penetraion_keypoints(
                    q=xhand_geometry.xhand_to_model_q(one)
                )
                .cpu()
                .numpy()[0]
            )
        by_name = {
            semantic[: -len("_tip")]: points[index]
            for semantic, index in zip(TIP_SEMANTICS, xhand_tip_indices())
        }
    elif hand == "wuji":
        mapping_config, semantics = trajectory_mapping_metadata(
            data,
            trajectory_position,
            RETARGET_ROOT / "configs" / "wuji_keypoint_map.json",
        )
        pairs = wuji_geometry.load_pairs(semantics, mapping_config)
        pair_by_name = {pair["semantic"]: pair for pair in pairs}
        with torch.no_grad():
            points = (
                target_model.get_penetraion_keypoints(
                    q=wuji_geometry.wuji_to_model_q(one)
                )
                .cpu()
                .numpy()[0]
            )
        by_name = {
            semantic[: -len("_tip")]: points[
                int(pair_by_name[semantic]["wuji_index"])
            ]
            for semantic in TIP_SEMANTICS
        }
    else:
        raise ValueError(f"未知手类型: {hand}")
    return {name: np.asarray(point, dtype=np.float32) for name, point in by_name.items()}


def bounded_descent_residual(
    joints: np.ndarray,
    group: list[int],
    lower: np.ndarray,
    upper: np.ndarray,
    distance_fn: Callable[[np.ndarray], float],
    epsilon_rad: float,
    max_delta_rad: float,
) -> tuple[np.ndarray, dict]:
    """用中心差分和两点线搜索，求单指的受限下降残差。

    输入：基准关节、单指索引、限位、距离回调、差分步长和残差L2上限。
    输出：全关节残差和基准/最终距离、梯度等审计。
    内部逻辑：每个单指关节在限位内做正负扰动，得到距离对关节的数值梯度；
    沿负梯度只比较半步和整步，若都不改善就返回零。
    作用：不需要人工猜每只手的闭合正负号，也不会为减小距离而无界改关节。
    """
    base = np.asarray(joints, dtype=np.float32)
    lower = np.asarray(lower, dtype=np.float32)
    upper = np.asarray(upper, dtype=np.float32)
    if base.shape != lower.shape or base.shape != upper.shape:
        raise ValueError("关节和限位形状必须一致")
    if epsilon_rad <= 0 or max_delta_rad <= 0 or not group:
        raise ValueError("差分步长、残差上限和单指关节组必须有效")
    baseline_distance = float(distance_fn(base))
    gradient = np.zeros_like(base)
    for index in group:
        plus = base.copy()
        minus = base.copy()
        plus[index] = min(float(upper[index]), float(base[index] + epsilon_rad))
        minus[index] = max(float(lower[index]), float(base[index] - epsilon_rad))
        denominator = float(plus[index] - minus[index])
        if denominator > 1e-9:
            gradient[index] = (float(distance_fn(plus)) - float(distance_fn(minus))) / denominator
    group_gradient = gradient[group]
    norm = float(np.linalg.norm(group_gradient))
    zero = np.zeros_like(base)
    if norm <= 1e-9:
        return zero, {
            "baseline_distance_m": baseline_distance,
            "optimized_distance_m": baseline_distance,
            "gradient_l2_m_per_rad": norm,
            "selected_fraction": 0.0,
            "residual_l2_rad": 0.0,
        }
    direction = np.zeros_like(base)
    direction[group] = -group_gradient / norm
    best_residual = zero
    best_distance = baseline_distance
    best_fraction = 0.0
    for fraction in (0.5, 1.0):
        proposal = np.clip(
            base + direction * float(max_delta_rad) * fraction, lower, upper
        )
        distance = float(distance_fn(proposal))
        if distance < best_distance - 1e-7:
            best_distance = distance
            best_residual = proposal - base
            best_fraction = fraction
    return best_residual.astype(np.float32), {
        "baseline_distance_m": baseline_distance,
        "optimized_distance_m": best_distance,
        "gradient_l2_m_per_rad": norm,
        "selected_fraction": best_fraction,
        "residual_l2_rad": float(np.linalg.norm(best_residual[group])),
    }


def apply_finger_residuals(
    frames: np.ndarray,
    residual: np.ndarray,
    close_start_frame: int,
    grasp_frame: int,
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """把分指关节残差渐进加到原轨迹并裁剪限位。

    输入：`(70,6+J)`轨迹、`(J,)`残差、close/grasp帧及关节限位。
    输出：同形状新轨迹和手腕保持、裁剪数、最大步长审计。
    内部逻辑：调用共享阶段进度从0增至1，只向第6维后的手指加残差；
    抓取帧后保持残差，因此原轨迹后续的动态变化仍逐帧存在。
    作用：避免阶段边界瞬时跳变，并与以前“lift后冻结手指”的失败方法区分。
    """
    original = np.asarray(frames, dtype=np.float32)
    residual = np.asarray(residual, dtype=np.float32)
    if original.shape != (70, 6 + len(residual)):
        raise ValueError(f"轨迹/残差形状错误: {original.shape}, {residual.shape}")
    progress = squeeze_progress(
        len(original), int(close_start_frame), int(grasp_frame)
    )
    output = original.copy()
    effective = residual.copy()
    active_frames = progress > 1e-9
    for joint_index, requested in enumerate(residual):
        if requested > 0:
            available = (
                upper[joint_index] - original[active_frames, 6 + joint_index]
            ) / progress[active_frames]
            effective[joint_index] = min(float(requested), max(0.0, float(np.min(available))))
        elif requested < 0:
            available = (
                lower[joint_index] - original[active_frames, 6 + joint_index]
            ) / progress[active_frames]
            effective[joint_index] = max(float(requested), min(0.0, float(np.max(available))))
    before_clip = output[:, 6:] + progress[:, None] * effective[None, :]
    output[:, 6:] = np.clip(before_clip, lower[None, :], upper[None, :])
    clipped = int(np.count_nonzero(np.abs(output[:, 6:] - before_clip) > 1e-7))
    if not np.array_equal(output[:, :6], original[:, :6]):
        raise AssertionError("分指补偿意外改变了手腕位姿")
    old_steps = np.linalg.norm(np.diff(original[:, 6:], axis=0), axis=1)
    new_steps = np.linalg.norm(np.diff(output[:, 6:], axis=0), axis=1)
    return output.astype(np.float32), {
        "wrist_commands_unchanged": True,
        "joint_limit_clipped_value_count": clipped,
        "requested_residual_l2_rad": float(np.linalg.norm(residual)),
        "effective_residual_l2_rad": float(np.linalg.norm(effective)),
        "effective_joint_residual_rad": effective.astype(float).tolist(),
        "trajectory_bound_reduced_joint_count": int(
            np.count_nonzero(np.abs(effective - residual) > 1e-8)
        ),
        "original_max_joint_step_l2_rad": float(np.max(old_steps)),
        "corrected_max_joint_step_l2_rad": float(np.max(new_steps)),
    }


def refine_manifest(
    hand: str,
    manifest: dict,
    input_dir: Path,
    output_dir: Path,
    max_delta_rad: float,
    source_z_offset: float,
    object_clearance: float,
    contact_threshold: float,
    min_contact_tips: int,
    lift_delta: float,
    mismatch_margin: float,
    epsilon_rad: float,
) -> dict:
    """为一只手的manifest批量生成缺口驱动分指候选。

    输入：手/manifest/目录、残差上限、物体/阶段参数和数值差分步长。
    输出：批次摘要；同时写出每物体候选npy。
    内部逻辑：恢复源手/目标手/物体几何，在专家抓取帧对五指逐一做门控、
    受限数值下降与时序叠加，最后核对轨迹数。
    作用：形成一个全局参数方法，而不是依据单条物理成败挑指头或候选。
    """
    if hand not in HAND_DIMENSIONS:
        raise ValueError(f"未知手类型: {hand}")
    output_dir.mkdir(parents=True, exist_ok=True)
    shadow_model, target_model = build_hand_models(hand)
    lower, upper = joint_limits(target_model)
    expected_joints = HAND_DIMENSIONS[hand] - 6
    if len(lower) != expected_joints:
        raise ValueError(f"{hand}模型关节数{len(lower)}与动作{expected_joints}不一致")
    records = []
    for entry in manifest.get("entries", []):
        object_name = str(entry["object_name"])
        input_path = input_dir / f"{object_name}.npy"
        full = np.load(input_path, allow_pickle=True).item()
        indices = [int(value) for value in entry["trajectory_indices"]]
        data = slice_candidate(full, indices, HAND_DIMENSIONS[hand])
        source = np.load(entry["source_path"], allow_pickle=True).item()
        frames = np.asarray(data["grasp_seqs"], dtype=np.float32)
        outputs, audits, phase_records = [], [], []
        for position, source_index in enumerate(indices):
            source_frames = np.asarray(
                source["grasp_seqs"][source_index], dtype=np.float32
            ).copy()
            source_frames[:, 2] += float(source_z_offset)
            vertices = transformed_object_vertices(
                Path(entry["object_asset_path"]),
                np.asarray(source["obj_scale"])[source_index],
                np.asarray(source["obj_rotmat"])[source_index],
                object_clearance,
            )
            tree = cKDTree(vertices)
            shadow_points, inferred = shadow_points_and_phases(
                shadow_model,
                source_frames,
                vertices,
                contact_threshold,
                min_contact_tips,
                lift_delta,
            )
            phase = existing_or_inferred_phase(data, position, inferred)
            grasp = int(inferred["grasp_frame"])
            close = int(phase["close_start_frame"])
            base_frame = frames[position, grasp]
            base_joints = base_frame[6:].copy()
            target_points = target_tip_positions(
                hand, target_model, data, base_frame, position
            )
            total_residual = np.zeros_like(base_joints)
            finger_audits = {}
            for name in ("index", "middle", "ring", "little", "thumb"):
                source_point = shadow_points[grasp, SHADOW_TIP_INDICES[name]]
                source_distance = float(tree.query(source_point, k=1)[0])
                target_distance = float(tree.query(target_points[name], k=1)[0])
                source_contact = source_distance <= float(contact_threshold)
                gap = target_distance - source_distance
                active = bool(source_contact and gap > float(mismatch_margin))
                audit = {
                    "source_distance_m": source_distance,
                    "target_distance_before_m": target_distance,
                    "target_minus_source_gap_m": gap,
                    "source_contact": source_contact,
                    "active": active,
                }
                if active:
                    group = FINGER_JOINT_GROUPS[hand][name]

                    def distance_fn(candidate_joints, finger=name):
                        """输入候选关节，输出指定指尖到物体顶点的最近距离。"""
                        candidate = base_frame.copy()
                        candidate[6:] = candidate_joints
                        point = target_tip_positions(
                            hand, target_model, data, candidate, position
                        )[finger]
                        return float(tree.query(point, k=1)[0])

                    residual, descent = bounded_descent_residual(
                        base_joints,
                        group,
                        lower,
                        upper,
                        distance_fn,
                        epsilon_rad,
                        max_delta_rad,
                    )
                    total_residual += residual
                    audit.update(descent)
                    audit["joint_indices"] = group
                    audit["joint_residual_rad"] = residual[group].astype(float).tolist()
                else:
                    audit.update(
                        {
                            "optimized_distance_m": target_distance,
                            "residual_l2_rad": 0.0,
                            "joint_indices": FINGER_JOINT_GROUPS[hand][name],
                            "joint_residual_rad": [
                                0.0 for _ in FINGER_JOINT_GROUPS[hand][name]
                            ],
                        }
                    )
                finger_audits[name] = audit
            output, trajectory_audit = apply_finger_residuals(
                frames[position], total_residual, close, grasp, lower, upper
            )
            effective_residual = np.asarray(
                trajectory_audit["effective_joint_residual_rad"], dtype=np.float32
            )
            final_points = target_tip_positions(
                hand, target_model, data, output[grasp], position
            )
            for name, finger_audit in finger_audits.items():
                group = FINGER_JOINT_GROUPS[hand][name]
                finger_audit["effective_joint_residual_rad"] = (
                    effective_residual[group].astype(float).tolist()
                )
                finger_audit["effective_residual_l2_rad"] = float(
                    np.linalg.norm(effective_residual[group])
                )
                finger_audit["optimized_distance_after_trajectory_bound_m"] = float(
                    tree.query(final_points[name], k=1)[0]
                )
            trajectory_audit.update(
                {
                    "source_trajectory_index": source_index,
                    "hand": hand,
                    "close_start_frame": close,
                    "grasp_frame": grasp,
                    "lift_start_frame": int(phase["lift_start_frame"]),
                    "active_finger_count": sum(
                        bool(item["active"]) for item in finger_audits.values()
                    ),
                    "total_residual_l2_rad": float(np.linalg.norm(effective_residual)),
                    "fingers": finger_audits,
                }
            )
            outputs.append(output)
            audits.append(trajectory_audit)
            phase_records.append({"source_trajectory_index": source_index, **phase})
            records.append({"object_name": object_name, **trajectory_audit})
        result = dict(data)
        result.update(
            {
                "grasp_seqs": np.stack(outputs).astype(np.float32),
                "method": f"adaptive_finger_gap_{max_delta_rad:g}rad_v1",
                "adaptive_finger_gap_input": str(input_path.resolve()),
                "adaptive_finger_gap_max_delta_rad": float(max_delta_rad),
                "adaptive_finger_gap_mismatch_margin_m": float(mismatch_margin),
                "adaptive_finger_gap_audit": audits,
                "adaptive_finger_gap_phase_metadata": phase_records,
            }
        )
        np.save(output_dir / input_path.name, result, allow_pickle=True)
    expected = int(manifest.get("trajectory_count", len(records)))
    if len(records) != expected:
        raise ValueError(f"输出轨迹数{len(records)}与manifest声明{expected}不符")
    finger_records = [
        finger
        for record in records
        for finger in record["fingers"].values()
    ]
    active_records = [item for item in finger_records if item["active"]]
    return {
        "method": f"adaptive_finger_gap_{max_delta_rad:g}rad_v1",
        "hand": hand,
        "manifest_purpose": manifest.get("purpose"),
        "input_dir": str(input_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "trajectory_count": len(records),
        "max_delta_rad_per_finger": float(max_delta_rad),
        "mismatch_margin_m": float(mismatch_margin),
        "active_finger_count": len(active_records),
        "trajectory_with_active_finger_count": sum(
            record["active_finger_count"] > 0 for record in records
        ),
        "mean_active_distance_before_m": (
            float(np.mean([item["target_distance_before_m"] for item in active_records]))
            if active_records
            else 0.0
        ),
        "mean_active_distance_after_m": (
            float(
                np.mean(
                    [
                        item["optimized_distance_after_trajectory_bound_m"]
                        for item in active_records
                    ]
                )
            )
            if active_records
            else 0.0
        ),
        "joint_limit_clipped_value_count": sum(
            record["joint_limit_clipped_value_count"] for record in records
        ),
        "records": records,
    }


def main() -> None:
    """解析分指缺口参数，生成候选和静态审计。

    输入：`--hand/--manifest/--input-dir/--output-dir/--max-delta-rad`及阶段阈值。
    输出：候选npy和`adaptive_finger_gap_summary.json`；不启动Isaac Gym。
    内部逻辑：校验正值参数后调用批处理，打印激活指头和优化前后距离。
    作用：为小样本物理筛选提供可复现、可先静态拒绝的单方法入口。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--hand", choices=sorted(HAND_DIMENSIONS), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-delta-rad", type=float, required=True)
    parser.add_argument("--source-z-offset", type=float, default=0.4)
    parser.add_argument("--object-clearance", type=float, default=0.005)
    parser.add_argument("--contact-threshold", type=float, default=0.02)
    parser.add_argument("--min-contact-tips", type=int, default=2)
    parser.add_argument("--lift-delta", type=float, default=0.03)
    parser.add_argument("--mismatch-margin", type=float, default=0.003)
    parser.add_argument("--epsilon-rad", type=float, default=0.01)
    args = parser.parse_args()
    if (
        args.max_delta_rad <= 0
        or args.contact_threshold <= 0
        or args.mismatch_margin < 0
        or args.epsilon_rad <= 0
    ):
        parser.error("残差/接触/差分参数无效")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    summary = refine_manifest(
        args.hand,
        manifest,
        args.input_dir,
        args.output_dir,
        args.max_delta_rad,
        args.source_z_offset,
        args.object_clearance,
        args.contact_threshold,
        args.min_contact_tips,
        args.lift_delta,
        args.mismatch_margin,
        args.epsilon_rad,
    )
    summary_path = args.output_dir / "adaptive_finger_gap_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"hand={summary['hand']} trajectories={summary['trajectory_count']}")
    print(
        f"active_fingers={summary['active_finger_count']} "
        f"active_trajectories={summary['trajectory_with_active_finger_count']}"
    )
    print(
        "mean_active_distance_mm="
        f"{1000 * summary['mean_active_distance_before_m']:.3f}->"
        f"{1000 * summary['mean_active_distance_after_m']:.3f}"
    )
    print(f"joint_limit_clips={summary['joint_limit_clipped_value_count']}")
    print(f"summary={summary_path.resolve()}")


if __name__ == "__main__":
    main()
