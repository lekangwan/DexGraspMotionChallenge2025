#!/usr/bin/env python3
"""Jointly refine a full retargeted trajectory with a sparse hand-object graph."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import cKDTree
import torch


ROOT = Path(__file__).resolve().parents[3]
RUN = Path(__file__).resolve().parent
PREPARE = RUN.parent / "prepare"
EVALUATE = RUN.parent / "evaluate"
for path in (RUN, PREPARE, EVALUATE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from evaluate_xhand_geometry import build_models, shadow_to_model_q
from object_geometry import transformed_object_surface
from phase_contact import move_with_wrist
from retarget_linker_keypoints import (
    build_linker_model,
    linker_world_points,
    load_pairs as load_linker_pairs,
)
from retarget_wuji_keypoints import build_wuji_model
from utils.rot6d import robust_compute_orth6d_from_eulerXYZ


CONFIGS = RUN.parent / "configs"


def target_row(data, source_index):
    indices = np.asarray(data["source_trajectory_indices"], dtype=np.int64)
    rows = np.flatnonzero(indices == int(source_index))
    if len(rows) != 1:
        raise ValueError("source trajectory index is absent or duplicated")
    return int(rows[0])


def load_mapping(hand):
    if hand == "linker":
        pairs = load_linker_pairs(True, True)
        return pairs, [int(row["shadow_index"]) for row in pairs], None
    path = CONFIGS / ("xhand_keypoint_map.json" if hand == "xhand"
                      else "wuji_keypoint_map_v2.json")
    pairs = json.loads(path.read_text(encoding="utf-8"))["pairs"]
    target_key = "xhand_index" if hand == "xhand" else "wuji_index"
    return pairs, [int(row["shadow_index"]) for row in pairs], [
        int(row[target_key]) for row in pairs]


def source_points(frames, indices):
    shadow, _ = build_models()
    shadow.set_parameters(shadow_to_model_q(frames))
    points = shadow.get_penetraion_keypoints().detach().cpu().numpy()
    return points[:, indices]


class TargetAdapter:
    def __init__(self, hand, mapping, indices):
        self.hand = hand
        self.mapping = mapping
        self.indices = indices
        if hand == "linker":
            self.model = build_linker_model("coupled6")
        elif hand == "xhand":
            _, self.model = build_models()
        else:
            self.model = build_wuji_model()
        self.lower = self.model.revolute_joints_q_lower[0].detach().cpu()
        self.upper = self.model.revolute_joints_q_upper[0].detach().cpu()

    def points(self, wrist, joints):
        rotation = robust_compute_orth6d_from_eulerXYZ(wrist[:, 3:6])
        model_q = torch.cat([wrist[:, :3], rotation, joints], dim=1)
        if self.hand == "linker":
            self.model.update_kinematics(model_q)
            points = []
            for pair in self.mapping:
                local = torch.as_tensor(
                    pair["linker_local_xyz"], dtype=torch.float32
                ).view(1, 1, 3)
                local = local.expand(len(wrist), -1, -1)
                hand_point = self.model.current_status[
                    pair["linker_link"]].transform_points(local)[:, 0]
                world = torch.bmm(
                    hand_point[:, None, :], self.model.global_rotation.transpose(1, 2)
                )[:, 0] + self.model.global_translation
                points.append(world * float(self.model.scale))
            return torch.stack(points, dim=1)
        return self.model.get_penetraion_keypoints(q=model_q)[:, self.indices]


def phase_frames(frames):
    fingers = np.asarray(frames[:, 6:], dtype=np.float32)
    movement = np.linalg.norm(fingers - fingers[0], axis=1)
    close_rows = np.flatnonzero(movement >= max(0.2 * float(movement.max()), 1e-3))
    close = int(close_rows[0]) if len(close_rows) else 20
    base_z = float(np.asarray(frames[close:, 2]).min())
    lift_rows = np.flatnonzero(
        (np.arange(len(frames)) > close) & (frames[:, 2] >= base_z + 0.03))
    lift = int(lift_rows[0]) if len(lift_rows) else min(55, len(frames) - 1)
    return close, max(close + 1, lift - 1)


def moving_surface(points, normals, frames, grasp):
    moved_points, moved_normals = [], []
    for index, frame in enumerate(frames):
        if index <= grasp:
            moved_points.append(points)
            moved_normals.append(normals)
        else:
            p, n = move_with_wrist(points, normals, frames[grasp], frame)
            moved_points.append(p)
            moved_normals.append(n)
    return np.asarray(moved_points), np.asarray(moved_normals)


def build_graph(source, object_points, close, grasp, neighbors, threshold):
    tree = cKDTree(object_points[grasp])
    edges, distances = [], []
    for hand_index, point in enumerate(source[grasp]):
        d, rows = tree.query(point, k=min(neighbors, len(object_points[grasp])))
        for distance, row in zip(np.atleast_1d(d), np.atleast_1d(rows)):
            if float(distance) <= threshold:
                edges.append((hand_index, int(row)))
                distances.append(float(distance))
    if not edges:
        return np.empty((0, 2), dtype=np.int64), np.empty(0, dtype=np.float32)
    weights = np.exp(-np.square(distances) / (2.0 * (threshold / 2.0) ** 2))
    weights /= max(float(weights.mean()), 1e-8)
    return np.asarray(edges, dtype=np.int64), weights.astype(np.float32)


def nearest_surface_targets(baseline_points, surface_points, surface_normals):
    nearest_points, nearest_normals = [], []
    for frame_points, vertices, normals in zip(
            baseline_points, surface_points, surface_normals):
        rows = cKDTree(vertices).query(frame_points, k=1)[1]
        nearest_points.append(vertices[rows])
        nearest_normals.append(normals[rows])
    return np.asarray(nearest_points), np.asarray(nearest_normals)


def optimize(adapter, baseline, source, source_surface, target_surface,
             target_normals, edges, edge_weights, start, iterations, learning_rate,
             interaction_weight, baseline_weight, velocity_weight,
             acceleration_weight, penetration_weight, max_joint_residual):
    wrist = torch.as_tensor(baseline[:, :6], dtype=torch.float32)
    baseline_joints = torch.as_tensor(baseline[:, 6:], dtype=torch.float32)
    joints = torch.nn.Parameter(baseline_joints.clone())
    source_vectors = source[:, edges[:, 0]] - source_surface[:, edges[:, 1]]
    target_objects = torch.as_tensor(
        target_surface[:, edges[:, 1]], dtype=torch.float32)
    desired = torch.as_tensor(source_vectors, dtype=torch.float32)
    weights = torch.as_tensor(edge_weights, dtype=torch.float32).view(1, -1)
    with torch.no_grad():
        baseline_points = adapter.points(wrist, baseline_joints).detach().cpu().numpy()
    nearest_points, nearest_normals = nearest_surface_targets(
        baseline_points, target_surface, target_normals)
    nearest_points = torch.as_tensor(nearest_points, dtype=torch.float32)
    nearest_normals = torch.as_tensor(nearest_normals, dtype=torch.float32)
    time_weight = torch.zeros(len(baseline), dtype=torch.float32)
    time_weight[start:] = torch.linspace(0.2, 1.0, len(baseline) - start)
    optimizer = torch.optim.Adam([joints], lr=learning_rate)
    history = []
    for step in range(iterations):
        optimizer.zero_grad()
        points = adapter.points(wrist, joints)
        target_vectors = points[:, edges[:, 0]] - target_objects
        vector_error = torch.sum((target_vectors - desired) ** 2, dim=2)
        interaction = torch.sum(
            vector_error * weights * time_weight[:, None]
        ) / torch.clamp(torch.sum(weights * time_weight[:, None]), min=1.0)
        baseline_loss = torch.mean((joints[start:] - baseline_joints[start:]) ** 2)
        velocity_loss = torch.mean(
            ((joints[1:] - joints[:-1])
             - (baseline_joints[1:] - baseline_joints[:-1])) ** 2)
        acceleration = joints[2:] - 2.0 * joints[1:-1] + joints[:-2]
        baseline_acceleration = (
            baseline_joints[2:] - 2.0 * baseline_joints[1:-1]
            + baseline_joints[:-2])
        acceleration_loss = torch.mean((acceleration - baseline_acceleration) ** 2)
        signed_distance = torch.sum(
            (points - nearest_points) * nearest_normals, dim=2)
        penetration = torch.mean(torch.relu(-0.003 - signed_distance[start:]) ** 2)
        loss = (
            interaction_weight * interaction
            + baseline_weight * baseline_loss
            + velocity_weight * velocity_loss
            + acceleration_weight * acceleration_loss
            + penetration_weight * penetration)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            joints[:start] = baseline_joints[:start]
            joints.clamp_(adapter.lower, adapter.upper)
            if max_joint_residual > 0:
                joints.copy_(torch.maximum(
                    torch.minimum(joints, baseline_joints + max_joint_residual),
                    baseline_joints - max_joint_residual))
        if step in (0, iterations - 1) or (step + 1) % 25 == 0:
            history.append({
                "step": step + 1,
                "loss": float(loss.detach()),
                "interaction": float(interaction.detach()),
                "baseline": float(baseline_loss.detach()),
                "velocity": float(velocity_loss.detach()),
                "acceleration": float(acceleration_loss.detach()),
                "penetration": float(penetration.detach()),
            })
            print(json.dumps(history[-1]), flush=True)
    result = baseline.copy()
    result[:, 6:] = joints.detach().cpu().numpy()
    return result, history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hand", choices=("linker", "xhand", "wuji"), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--source-index", type=int, required=True)
    parser.add_argument("--object-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=150)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--surface-neighbors", type=int, default=3)
    parser.add_argument("--interaction-threshold", type=float, default=0.05)
    parser.add_argument("--optimize-from", choices=("close", "grasp"), default="close")
    parser.add_argument("--source-z-offset", type=float, default=0.4)
    parser.add_argument("--interaction-weight", type=float, default=1000.0)
    parser.add_argument("--baseline-weight", type=float, default=1.0)
    parser.add_argument("--velocity-weight", type=float, default=0.2)
    parser.add_argument("--acceleration-weight", type=float, default=0.1)
    parser.add_argument("--penetration-weight", type=float, default=1000.0)
    parser.add_argument("--max-joint-residual", type=float, default=0.35)
    args = parser.parse_args()

    source_data = np.load(args.source, allow_pickle=True).item()
    target_data = np.load(args.target, allow_pickle=True).item()
    row = target_row(target_data, args.source_index)
    source_frames = np.asarray(
        source_data["grasp_seqs"][args.source_index], dtype=np.float32).copy()
    source_frames[:, 2] += args.source_z_offset
    baseline = np.asarray(target_data["grasp_seqs"][row], dtype=np.float32)
    mapping, source_indices, target_indices = load_mapping(args.hand)
    source_kp = source_points(source_frames, source_indices)
    adapter = TargetAdapter(args.hand, mapping, target_indices)
    if baseline.shape[1] - 6 != len(adapter.lower):
        raise ValueError("target action dimension does not match the hand model")
    vertices, normals = transformed_object_surface(
        args.object_dir,
        float(np.asarray(source_data["obj_scale"])[args.source_index]),
        np.asarray(source_data["obj_rotmat"])[args.source_index])
    close, grasp = phase_frames(source_frames)
    source_surface, _ = moving_surface(vertices, normals, source_frames, grasp)
    target_surface, target_normals = moving_surface(vertices, normals, baseline, grasp)
    edges, weights = build_graph(
        source_kp, source_surface, close, grasp,
        args.surface_neighbors, args.interaction_threshold)
    start = close if args.optimize_from == "close" else grasp
    if len(edges):
        refined_frames, history = optimize(
            adapter, baseline, source_kp, source_surface, target_surface,
            target_normals, edges, weights, start, args.iterations,
            args.learning_rate, args.interaction_weight, args.baseline_weight,
            args.velocity_weight, args.acceleration_weight,
            args.penetration_weight, args.max_joint_residual)
    else:
        refined_frames = baseline.copy()
        history = [{"step": 0, "skipped": "no_contact_edge"}]
    refined = dict(target_data)
    sequences = np.asarray(target_data["grasp_seqs"]).copy()
    sequences[row] = refined_frames
    refined["grasp_seqs"] = sequences
    refined["topology_interaction"] = {
        "schema": "topology_interaction_v2_bounded",
        "source_index": args.source_index,
        "close_frame": close,
        "grasp_frame": grasp,
        "optimization_start_frame": start,
        "optimize_from": args.optimize_from,
        "edge_count": len(edges),
        "iterations": args.iterations,
        "learning_rate": args.learning_rate,
        "interaction_threshold": args.interaction_threshold,
        "source_z_offset": args.source_z_offset,
        "interaction_weight": args.interaction_weight,
        "baseline_weight": args.baseline_weight,
        "velocity_weight": args.velocity_weight,
        "acceleration_weight": args.acceleration_weight,
        "penetration_weight": args.penetration_weight,
        "max_joint_residual": args.max_joint_residual,
        "history": history,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, refined, allow_pickle=True)
    args.output.with_suffix(".json").write_text(
        json.dumps(refined["topology_interaction"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")


if __name__ == "__main__":
    main()
