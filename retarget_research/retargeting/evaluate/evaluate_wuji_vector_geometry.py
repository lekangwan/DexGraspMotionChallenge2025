#!/usr/bin/env python3
"""独立评估Shadow→Wuji功能向量残差和轨迹连续性。

输入：Shadow源npy、功能向量Wuji候选及双方轨迹索引。
输出：12个向量的平均/最大残差、逐语义误差和关节/手腕跳变JSON。
内部逻辑：重新执行两手正向运动学，核对向量配置SHA和保存的形态scale；
不读取生成阶段保存的优化loss，也不把向量误差冒充绝对关键点距离。
作用：让统一物理评测支持新的底层方法，同时保持几何指标含义可审计。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from evaluate_wuji_geometry import (
    build_models,
    rotation_step_angles,
    shadow_to_model_q,
    wuji_to_model_q,
)


def load_vector_metadata(target_data):
    """核对候选声明的向量配置和SHA，并返回配置、12个scale及语义。"""
    supported_methods = {
        "dexpilot_style_functional_vectors_v1",
        "dexpilot_style_functional_vectors_plus_surface_contact_v1",
    }
    if target_data.get("retarget_method") not in supported_methods:
        raise ValueError("目标文件不是受支持的Wuji功能向量方法")
    config_path = Path(str(target_data["vector_config"])).resolve()
    raw = config_path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != target_data.get("vector_config_sha256"):
        raise ValueError("功能向量配置SHA-256与候选生成时不一致")
    config = json.loads(raw.decode("utf-8"))
    pairs = config["pairs"]
    scales = np.asarray(target_data["vector_scales"], dtype=np.float64)
    semantics = list(target_data["mapping_semantics"])
    if scales.shape != (len(pairs),) or len(pairs) != 12:
        raise ValueError("功能向量scale或pair数量错误")
    if semantics != [item["semantic"] for item in pairs]:
        raise ValueError("候选向量语义顺序与配置不一致")
    return config_path, pairs, scales, semantics


def main():
    """解析一条轨迹、重新计算向量残差和连续性并写入JSON。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--source-index", type=int, default=0)
    parser.add_argument("--target-index", type=int, default=0)
    parser.add_argument("--source-z-offset", type=float)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source_data = np.load(args.source, allow_pickle=True).item()
    target_data = np.load(args.target, allow_pickle=True).item()
    source_frames = np.asarray(source_data["grasp_seqs"][args.source_index], dtype=np.float32).copy()
    target_frames = np.asarray(target_data["grasp_seqs"][args.target_index], dtype=np.float32)
    if source_frames.shape[0] != target_frames.shape[0] or target_frames.shape[1] != 26:
        raise ValueError("Shadow/Wuji向量轨迹帧数或动作维度错误")
    source_z_offset = float(target_data.get("source_z_offset", 0.4)) if args.source_z_offset is None else args.source_z_offset
    source_frames[:, 2] += source_z_offset
    config_path, pairs, scales, semantics = load_vector_metadata(target_data)
    shadow, wuji = build_models()
    with torch.no_grad():
        shadow.set_parameters(shadow_to_model_q(source_frames))
        source_points = shadow.get_penetraion_keypoints().numpy()
        target_points = wuji.get_penetraion_keypoints(q=wuji_to_model_q(target_frames)).numpy()
    source_origin = np.asarray([item["shadow_origin"] for item in pairs])
    source_task = np.asarray([item["shadow_task"] for item in pairs])
    target_origin = np.asarray([item["wuji_origin"] for item in pairs])
    target_task = np.asarray([item["wuji_task"] for item in pairs])
    desired = (source_points[:, source_task] - source_points[:, source_origin]) * scales[None, :, None]
    actual = target_points[:, target_task] - target_points[:, target_origin]
    residuals = np.linalg.norm(actual - desired, axis=2)
    joint_delta = np.diff(target_frames[:, 6:], axis=0)
    wrist_delta = np.diff(target_frames[:, :3], axis=0)
    rotation_delta = rotation_step_angles(target_frames[:, 3:6])
    worst_frame, worst_pair = np.unravel_index(residuals.argmax(), residuals.shape)
    report = {
        "source": str(args.source.resolve()),
        "target": str(args.target.resolve()),
        "source_trajectory_index": args.source_index,
        "target_trajectory_index": args.target_index,
        "frames": int(len(source_frames)),
        "pair_count": len(pairs),
        "source_z_offset_m": source_z_offset,
        "geometry_metric_kind": "functional_vector_residual_m",
        "vector_config": str(config_path),
        "functional_vector_mean_residual_m": float(residuals.mean()),
        "functional_vector_max_residual_m": float(residuals.max()),
        "keypoint_mean_distance_m": float(residuals.mean()),
        "keypoint_max_distance_m": float(residuals.max()),
        "worst_frame": int(worst_frame),
        "worst_semantic": semantics[int(worst_pair)],
        "mean_joint_step_l2_rad": float(np.linalg.norm(joint_delta, axis=1).mean()),
        "max_joint_step_l2_rad": float(np.linalg.norm(joint_delta, axis=1).max()),
        "mean_wrist_translation_step_m": float(np.linalg.norm(wrist_delta, axis=1).mean()),
        "max_wrist_translation_step_m": float(np.linalg.norm(wrist_delta, axis=1).max()),
        "mean_wrist_rotation_step_rad": float(rotation_delta.mean()),
        "max_wrist_rotation_step_rad": float(rotation_delta.max()),
        "per_semantic": {
            name: {
                "mean_vector_residual_m": float(residuals[:, index].mean()),
                "max_vector_residual_m": float(residuals[:, index].max()),
            }
            for index, name in enumerate(semantics)
        },
        "per_frame_mean_vector_residual_m": residuals.mean(axis=1).tolist(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"frames={report['frames']}")
    print(f"pair_count={report['pair_count']}")
    print(f"functional_vector_mean_residual_m={report['functional_vector_mean_residual_m']:.6f}")
    print(f"functional_vector_max_residual_m={report['functional_vector_max_residual_m']:.6f}")
    print(f"worst={report['worst_semantic']}@frame{report['worst_frame']}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
