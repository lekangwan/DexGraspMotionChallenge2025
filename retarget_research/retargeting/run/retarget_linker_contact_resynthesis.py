#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

RUN_DIR = Path(__file__).resolve().parent
RETARGET_ROOT = RUN_DIR.parent
for path in (RUN_DIR, RETARGET_ROOT / "prepare"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from object_geometry import transformed_object_surface
from phase_contact import build_phase_contact_plan, load_pad_config
from retarget_linker_keypoints import (
    build_linker_model,
    build_shadow_model,
    load_pairs,
    model_joint_count,
    retarget_trajectory,
    shadow_keypoints,
)


METHOD = "linker_contact_resynthesis_v1"
PAD_CONFIG = RETARGET_ROOT / "configs" / "linker_contact_pads_v1.json"


def resynthesize_trajectory(source_frames, baseline_frames, object_vertices, object_normals, args):
    shadow_model = build_shadow_model()
    linker_model = build_linker_model("coupled6")
    pairs = load_pairs(False, False)
    source_points = shadow_keypoints(source_frames, shadow_model)
    source_indices = [pair["shadow_index"] for pair in pairs]
    target_points = source_points[:, source_indices, :]
    source_tip_points = {
        semantic: source_points[:, point_index, :]
        for semantic, point_index in {
            "index": 4,
            "middle": 8,
            "ring": 12,
            "little": 16,
            "thumb": 20,
        }.items()
    }
    plan = build_phase_contact_plan(
        source_frames,
        source_tip_points,
        object_vertices,
        object_normals,
        contact_threshold=args.phase_contact_threshold,
        min_contact_tips=args.phase_min_contact_tips,
        lift_delta=args.phase_lift_delta,
        region_neighbors=args.phase_region_neighbors,
        contact_fallback="nearest",
    )
    frame_weights = np.ones(target_points.shape[:2], dtype=np.float32)
    frame_weights[plan["close_start_frame"] :] = args.keypoint_weight
    initial_internal = np.concatenate(
        [baseline_frames[:, 6:], baseline_frames[:, :6]], axis=1
    )
    internal, losses, components = retarget_trajectory(
        source_frames,
        linker_model,
        pairs,
        target_points,
        args.maxeval,
        args.translation_bound,
        args.joint_temporal_weight,
        args.translation_temporal_weight,
        args.rotation_temporal_weight,
        -1,
        1.0,
        1.0,
        1.0,
        frame_point_weights=frame_weights,
        phase_contact_plan=plan["frames"],
        pad_config=load_pad_config(args.pad_config, expected_hand="linker"),
        phase_contact_weight=args.phase_contact_weight,
        phase_normal_weight=args.phase_normal_weight,
        phase_penetration_weight=args.phase_penetration_weight,
        phase_joint_hold_weight=args.phase_joint_hold_weight,
        phase_joint_prior_weight=args.phase_joint_prior_weight,
        initial_trajectory=initial_internal,
        phase_only_refinement=True,
        freeze_lift_grasp=True,
        carry_lift_wrist_residual=True,
        phase_contact_offset=args.phase_contact_offset,
        phase_min_signed_distance=args.phase_min_signed_distance,
    )
    output = np.concatenate(
        [internal[:, model_joint_count(linker_model) :], internal[:, : model_joint_count(linker_model)]],
        axis=1,
    ).astype(np.float32)
    return output, losses, components, plan


def retarget_file(args):
    source_data = np.load(args.source, allow_pickle=True).item()
    baseline_data = np.load(args.baseline, allow_pickle=True).item()
    indices = np.asarray(args.trajectory_indices, dtype=np.int64)
    baseline_indices = np.asarray(baseline_data["source_trajectory_indices"], dtype=np.int64)
    if not np.array_equal(indices, baseline_indices):
        raise ValueError("baseline与manifest轨迹索引不一致")
    outputs, losses, components, phase_metadata = [], [], [], []
    for position, source_index in enumerate(indices):
        source_frames = np.asarray(source_data["grasp_seqs"][source_index], dtype=np.float32).copy()
        source_frames[:, 2] += args.source_z_offset
        object_vertices, object_normals = transformed_object_surface(
            args.object_dir,
            np.asarray(source_data["obj_scale"])[source_index],
            np.asarray(source_data["obj_rotmat"])[source_index],
            args.object_clearance,
        )
        output, loss, detail, plan = resynthesize_trajectory(
            source_frames,
            np.asarray(baseline_data["grasp_seqs"])[position],
            object_vertices,
            object_normals,
            args,
        )
        outputs.append(output)
        losses.append(loss)
        components.append(detail)
        phase_metadata.append(
            {
                "source_trajectory_index": int(source_index),
                "close_start_frame": int(plan["close_start_frame"]),
                "lift_start_frame": int(plan["lift_start_frame"]),
                "grasp_frame": int(plan["grasp_frame"]),
                "contact_fallback_used": bool(plan["contact_fallback_used"]),
            }
        )
    result = dict(baseline_data)
    result.update(
        {
            "grasp_seqs": np.stack(outputs),
            "optimization_loss_per_frame": np.stack(losses),
            "optimization_loss_components_per_frame": components,
            "retarget_method": METHOD,
            "contact_resynthesis_baseline": str(args.baseline.resolve()),
            "contact_pad_config": str(args.pad_config.resolve()),
            "contact_resynthesis_keypoint_weight": args.keypoint_weight,
            "phase_metadata": phase_metadata,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, result, allow_pickle=True)
    print(f"output_shape={result['grasp_seqs'].shape}")
    print(f"output={args.output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--object-dir", type=Path, required=True)
    parser.add_argument("--trajectory-indices", type=int, nargs="+", required=True)
    parser.add_argument("--pad-config", type=Path, default=PAD_CONFIG)
    parser.add_argument("--maxeval", type=int, default=60)
    parser.add_argument("--translation-bound", type=float, default=2.0)
    parser.add_argument("--source-z-offset", type=float, default=0.4)
    parser.add_argument("--object-clearance", type=float, default=0.005)
    parser.add_argument("--phase-contact-threshold", type=float, default=0.02)
    parser.add_argument("--phase-min-contact-tips", type=int, default=2)
    parser.add_argument("--phase-lift-delta", type=float, default=0.03)
    parser.add_argument("--phase-region-neighbors", type=int, default=32)
    parser.add_argument("--keypoint-weight", type=float, default=1e-5)
    parser.add_argument("--phase-contact-weight", type=float, default=5.0)
    parser.add_argument("--phase-normal-weight", type=float, default=0.05)
    parser.add_argument("--phase-penetration-weight", type=float, default=1.0)
    parser.add_argument("--phase-joint-hold-weight", type=float, default=0.05)
    parser.add_argument("--phase-joint-prior-weight", type=float, default=0.01)
    parser.add_argument("--phase-contact-offset", type=float, default=-0.001)
    parser.add_argument("--phase-min-signed-distance", type=float, default=-0.003)
    parser.add_argument("--joint-temporal-weight", type=float, default=0.01)
    parser.add_argument("--translation-temporal-weight", type=float, default=0.0)
    parser.add_argument("--rotation-temporal-weight", type=float, default=0.0)
    args = parser.parse_args()
    retarget_file(args)


if __name__ == "__main__":
    main()
