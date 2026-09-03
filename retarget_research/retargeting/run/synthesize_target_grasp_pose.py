#!/usr/bin/env python3
"""Synthesize one target-hand grasp pose from reachable opposing contacts."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[3]
RUN = Path(__file__).resolve().parent
for path in (RUN, RUN.parent / "prepare"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from object_geometry import transformed_object_surface
from phase_contact import (
    load_pad_config,
    pad_contact_terms,
    select_reachable_opposing_contact_regions,
    world_pad_regions,
)
from refine_topology_interaction import TargetAdapter, load_mapping, phase_frames, target_row
from utils.rot6d import robust_compute_orth6d_from_eulerXYZ


CONFIGS = RUN.parent / "configs"


def update_model(adapter, wrist, joints):
    rotation = robust_compute_orth6d_from_eulerXYZ(wrist[:, 3:6])
    adapter.model.update_kinematics(torch.cat([wrist[:, :3], rotation, joints], dim=1))


def numpy_pads(pads):
    return {
        name: (points.detach().cpu().numpy(), normals.detach().cpu().numpy())
        for name, (points, normals) in pads.items()
    }


def optimize_pose(adapter, wrist, baseline_joints, pad_config, targets,
                  object_surface, args):
    baseline_wrist = torch.as_tensor(wrist[None], dtype=torch.float32)
    wrist_tensor = torch.nn.Parameter(baseline_wrist.clone())
    baseline = torch.as_tensor(baseline_joints[None], dtype=torch.float32)
    joints = torch.nn.Parameter(baseline.clone())
    optimizer = torch.optim.Adam([wrist_tensor, joints], lr=args.learning_rate)
    history = []
    for step in range(args.iterations):
        optimizer.zero_grad()
        update_model(adapter, wrist_tensor, joints)
        terms = pad_contact_terms(
            world_pad_regions(adapter.model, pad_config),
            targets,
            object_surface,
            contact_offset=args.contact_offset,
            min_signed_distance=args.min_signed_distance,
        )
        prior = torch.mean((joints - baseline) ** 2)
        wrist_translation = torch.mean(
            (wrist_tensor[:, :3] - baseline_wrist[:, :3]) ** 2)
        wrist_rotation = torch.mean(
            (wrist_tensor[:, 3:] - baseline_wrist[:, 3:]) ** 2)
        loss = (
            args.contact_weight * terms["contact"]
            + args.normal_weight * terms["normal"]
            + args.penetration_weight * terms["penetration"]
            + args.joint_prior_weight * prior
            + args.wrist_translation_weight * wrist_translation
            + args.wrist_rotation_weight * wrist_rotation
        )
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            joints.clamp_(adapter.lower, adapter.upper)
            if args.max_joint_residual > 0:
                joints.copy_(torch.maximum(
                    torch.minimum(joints, baseline + args.max_joint_residual),
                    baseline - args.max_joint_residual))
            wrist_tensor[:, :3].copy_(torch.maximum(
                torch.minimum(
                    wrist_tensor[:, :3],
                    baseline_wrist[:, :3] + args.max_wrist_translation),
                baseline_wrist[:, :3] - args.max_wrist_translation))
            wrist_tensor[:, 3:].copy_(torch.maximum(
                torch.minimum(
                    wrist_tensor[:, 3:],
                    baseline_wrist[:, 3:] + args.max_wrist_rotation),
                baseline_wrist[:, 3:] - args.max_wrist_rotation))
        if step in (0, args.iterations - 1) or (step + 1) % 25 == 0:
            row = {
                "step": step + 1,
                "loss": float(loss.detach()),
                "contact": float(terms["contact"].detach()),
                "normal": float(terms["normal"].detach()),
                "penetration": float(terms["penetration"].detach()),
                "joint_prior": float(prior.detach()),
                "wrist_translation": float(wrist_translation.detach()),
                "wrist_rotation": float(wrist_rotation.detach()),
            }
            history.append(row)
            print(json.dumps(row), flush=True)
    return (
        wrist_tensor.detach().cpu().numpy()[0],
        joints.detach().cpu().numpy()[0],
        history,
    )


def synthesize_trajectory(baseline, grasp, target_wrist, target_joints):
    result = baseline.copy()
    start = max(0, grasp - 10)
    residual = target_joints - baseline[grasp, 6:]
    wrist_residual = target_wrist - baseline[grasp, :6]
    for frame in range(start, grasp + 1):
        alpha = (frame - start) / max(grasp - start, 1)
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)
        result[frame, 6:] += alpha * residual
        result[frame, :6] += alpha * wrist_residual
    result[grasp + 1:, 6:] += residual
    result[grasp + 1:, :6] += wrist_residual
    return result, start


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hand", choices=("linker", "xhand", "wuji"), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--source-index", type=int, required=True)
    parser.add_argument("--object-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--candidate-neighbors", type=int, default=128)
    parser.add_argument("--region-neighbors", type=int, default=32)
    parser.add_argument("--distance-scale", type=float, default=0.03)
    parser.add_argument("--opposition-weight", type=float, default=3.0)
    parser.add_argument("--pad-alignment-weight", type=float, default=1.0)
    parser.add_argument("--friction-stability-weight", type=float, default=2.0)
    parser.add_argument("--friction-coefficient", type=float, default=0.7)
    parser.add_argument("--max-reachable-distance", type=float, default=0.0)
    parser.add_argument("--contact-weight", type=float, default=1000.0)
    parser.add_argument("--normal-weight", type=float, default=0.1)
    parser.add_argument("--penetration-weight", type=float, default=1000.0)
    parser.add_argument("--joint-prior-weight", type=float, default=1.0)
    parser.add_argument("--wrist-translation-weight", type=float, default=10.0)
    parser.add_argument("--wrist-rotation-weight", type=float, default=1.0)
    parser.add_argument("--max-joint-residual", type=float, default=0.40)
    parser.add_argument("--max-wrist-translation", type=float, default=0.03)
    parser.add_argument("--max-wrist-rotation", type=float, default=0.20)
    parser.add_argument("--contact-offset", type=float, default=-0.001)
    parser.add_argument("--min-signed-distance", type=float, default=-0.003)
    parser.add_argument("--source-z-offset", type=float, default=0.4)
    args = parser.parse_args()

    source_data = np.load(args.source, allow_pickle=True).item()
    target_data = np.load(args.target, allow_pickle=True).item()
    row = target_row(target_data, args.source_index)
    source = np.asarray(source_data["grasp_seqs"][args.source_index], dtype=np.float32).copy()
    source[:, 2] += args.source_z_offset
    baseline = np.asarray(target_data["grasp_seqs"][row], dtype=np.float32)
    _, grasp = phase_frames(source)
    vertices, normals = transformed_object_surface(
        args.object_dir,
        float(np.asarray(source_data["obj_scale"])[args.source_index]),
        np.asarray(source_data["obj_rotmat"])[args.source_index],
    )

    mapping, _, indices = load_mapping(args.hand)
    adapter = TargetAdapter(args.hand, mapping, indices)
    pad_config = load_pad_config(
        CONFIGS / f"{args.hand}_contact_pads_v1.json",
        expected_hand=args.hand,
    )
    wrist = torch.as_tensor(baseline[grasp:grasp + 1, :6], dtype=torch.float32)
    joints = torch.as_tensor(baseline[grasp:grasp + 1, 6:], dtype=torch.float32)
    update_model(adapter, wrist, joints)
    reachable = numpy_pads(world_pad_regions(adapter.model, pad_config))
    targets, diagnostics = select_reachable_opposing_contact_regions(
        reachable,
        vertices,
        normals,
        args.candidate_neighbors,
        args.region_neighbors,
        args.distance_scale,
        args.opposition_weight,
        args.pad_alignment_weight,
        min_opposing_fingers=2,
        friction_stability_weight=args.friction_stability_weight,
        friction_coefficient=args.friction_coefficient,
        max_reachable_distance=args.max_reachable_distance,
    )
    target_wrist, target_joints, history = optimize_pose(
        adapter,
        baseline[grasp, :6],
        baseline[grasp, 6:],
        pad_config,
        targets,
        (vertices, normals),
        args,
    )
    frames, start = synthesize_trajectory(
        baseline, grasp, target_wrist, target_joints)
    result = dict(target_data)
    sequences = np.asarray(target_data["grasp_seqs"]).copy()
    sequences[row] = frames
    result["grasp_seqs"] = sequences
    result["retarget_method"] = "target_grasp_pose_synthesis_v1"
    result["target_grasp_pose_synthesis"] = {
        "schema": "target_grasp_pose_synthesis_v1",
        "source_index": args.source_index,
        "grasp_frame": grasp,
        "blend_start_frame": start,
        "selected_fingers": diagnostics["selected_opposing_fingers"],
        "contact_diagnostics": diagnostics,
        "max_joint_residual": args.max_joint_residual,
        "max_wrist_translation": args.max_wrist_translation,
        "max_wrist_rotation": args.max_wrist_rotation,
        "history": history,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, result, allow_pickle=True)
    args.output.with_suffix(".json").write_text(
        json.dumps(result["target_grasp_pose_synthesis"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
