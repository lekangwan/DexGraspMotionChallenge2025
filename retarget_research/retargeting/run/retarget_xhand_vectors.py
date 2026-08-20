#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import nlopt
import numpy as np
import torch

RETARGET_ROOT = Path(__file__).resolve().parents[1]
EVALUATE_DIR = RETARGET_ROOT / "evaluate"
PREPARE_DIR = RETARGET_ROOT / "prepare"
for path in (EVALUATE_DIR, PREPARE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from evaluate_xhand_geometry import build_models, shadow_to_model_q  # noqa: E402
from object_geometry import transformed_object_surface  # noqa: E402
from phase_contact import build_phase_contact_plan  # noqa: E402
from retarget_xhand_contact import internal_to_saved, SOURCE_TIP_INDICES  # noqa: E402
from utils.rot6d import robust_compute_orth6d_from_eulerXYZ  # noqa: E402


METHOD = "anydex_style_segment_vectors_v1"
DEFAULT_VECTOR_CONFIG = RETARGET_ROOT / "configs" / "xhand_anydex_vectors_v1.json"
SHADOW_KEYPOINT_COUNT = 21
XHAND_KEYPOINT_COUNT = 30


def load_vector_config(path):
    resolved = Path(path).resolve()
    raw = resolved.read_bytes()
    config = json.loads(raw.decode("utf-8"))
    pairs = config.get("pairs", [])
    if len(pairs) < 2:
        raise ValueError("向量配置至少需要两对")
    semantics = [item["semantic"] for item in pairs]
    if len(set(semantics)) != len(semantics):
        raise ValueError("向量语义名称不能重复")
    for item in pairs:
        if item["kind"] not in ("position", "direction"):
            raise ValueError(f"未知向量类型: {item['semantic']}")
        if not 0 <= int(item["shadow_origin"]) < SHADOW_KEYPOINT_COUNT:
            raise ValueError(f"Shadow索引越界: {item['semantic']}")
        if not 0 <= int(item["shadow_task"]) < SHADOW_KEYPOINT_COUNT:
            raise ValueError(f"Shadow索引越界: {item['semantic']}")
        if not 0 <= int(item["xhand_origin"]) < XHAND_KEYPOINT_COUNT:
            raise ValueError(f"XHand索引越界: {item['semantic']}")
        if not 0 <= int(item["xhand_task"]) < XHAND_KEYPOINT_COUNT:
            raise ValueError(f"XHand索引越界: {item['semantic']}")
        if not np.isfinite(float(item["weight"])) or float(item["weight"]) <= 0:
            raise ValueError(f"向量权重必须为有限正数: {item['semantic']}")
    for key in ("huber_delta_m", "direction_huber_delta", "neutral_joint_weight",
                "previous_joint_weight", "palm_position_weight"):
        value = float(config[key])
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{key}必须是有效的非负值")
    return config, str(resolved), hashlib.sha256(raw).hexdigest()


def xhand_points(model, joints, translation, euler):
    joints = joints.view(1, -1)
    translation = translation.view(1, 3)
    euler = euler.view(1, 3)
    rotation = robust_compute_orth6d_from_eulerXYZ(euler)
    q = torch.cat([translation, rotation, joints], dim=1).to(model.device)
    return model.get_penetraion_keypoints(q=q)[0]


def zero_pose_points(shadow_model, xhand_model):
    with torch.no_grad():
        shadow_model.set_parameters(shadow_to_model_q(np.zeros((1, 28), dtype=np.float32)))
        shadow = shadow_model.get_penetraion_keypoints().cpu().numpy()[0]
        xhand = xhand_model.get_penetraion_keypoints(
            q=torch.cat([
                torch.zeros((1, 3), dtype=torch.float32),
                robust_compute_orth6d_from_eulerXYZ(torch.zeros((1, 3), dtype=torch.float32)),
                torch.zeros((1, 12), dtype=torch.float32),
            ], dim=1).to(xhand_model.device)
        ).cpu().numpy()[0]
    return shadow, xhand


def zero_pose_scales(shadow_zero, xhand_zero, pairs):
    scales = []
    for item in pairs:
        source_length = np.linalg.norm(
            shadow_zero[int(item["shadow_task"])] - shadow_zero[int(item["shadow_origin"])])
        target_length = np.linalg.norm(
            xhand_zero[int(item["xhand_task"])] - xhand_zero[int(item["xhand_origin"])])
        if source_length <= 1e-6 or target_length <= 1e-6:
            raise ValueError(f"零姿态向量退化: {item['semantic']}")
        scales.append(target_length / source_length)
    return np.asarray(scales, dtype=np.float64)


def huber_norm(residual, delta):
    distance = torch.linalg.vector_norm(residual, dim=1)
    delta_tensor = distance.new_tensor(float(delta))
    return torch.where(
        distance <= delta_tensor,
        0.5 * distance * distance,
        delta_tensor * (distance - 0.5 * delta_tensor),
    )


class XHandVectorObjective:
    def __init__(self, model, target_palm, target_vectors, target_directions,
                 pairs, config, previous_values=None, contact_anchors=None,
                 contact_weight=0.0, alphas=None, grip_flexion_weight=0.0,
                 grip_flexion_targets=None, grip_active=False):
        self.model = model
        self.target_palm = torch.as_tensor(target_palm, dtype=torch.float32)
        self.target_vectors = torch.as_tensor(target_vectors, dtype=torch.float32)
        self.target_directions = torch.as_tensor(target_directions, dtype=torch.float32)
        self.contact_anchors = contact_anchors
        self.contact_weight = float(contact_weight)
        self.alphas = None if alphas is None else torch.as_tensor(alphas, dtype=torch.float32)
        self.alpha_blend = config.get("alpha_blend", {})
        self.grip_flexion_weight = float(grip_flexion_weight)
        self.grip_flexion_targets = (
            None if grip_flexion_targets is None
            else torch.as_tensor(grip_flexion_targets, dtype=torch.float32))
        self.grip_active = bool(grip_active)
        position_mask = np.asarray([item["kind"] == "position" for item in pairs], dtype=bool)
        direction_mask = ~position_mask
        self.pos_origin = np.asarray(
            [item["xhand_origin"] for item, ok in zip(pairs, position_mask) if ok], dtype=np.int64)
        self.pos_task = np.asarray(
            [item["xhand_task"] for item, ok in zip(pairs, position_mask) if ok], dtype=np.int64)
        self.pos_weights = torch.as_tensor(
            [item["weight"] for item, ok in zip(pairs, position_mask) if ok], dtype=torch.float32)
        self.dir_origin = np.asarray(
            [item["xhand_origin"] for item, ok in zip(pairs, direction_mask) if ok], dtype=np.int64)
        self.dir_task = np.asarray(
            [item["xhand_task"] for item, ok in zip(pairs, direction_mask) if ok], dtype=np.int64)
        self.dir_weights = torch.as_tensor(
            [item["weight"] for item, ok in zip(pairs, direction_mask) if ok], dtype=torch.float32)
        if self.alphas is not None and self.alpha_blend:
            tip_open = float(self.alpha_blend.get("tip_open_fraction", 0.3))
            dir_open = float(self.alpha_blend.get("direction_open_fraction", 0.2))
            self.pos_weights[:5] = self.pos_weights[:5] * (
                self.alphas + (1.0 - self.alphas) * tip_open)
            self.dir_weights = self.dir_weights * (
                self.alphas + (1.0 - self.alphas) * dir_open)
        self.huber_delta = float(config["huber_delta_m"])
        self.direction_delta = float(config["direction_huber_delta"])
        self.neutral_weight = float(config["neutral_joint_weight"])
        self.previous_weight = float(config["previous_joint_weight"])
        self.palm_weight = float(config["palm_position_weight"])
        self.previous = None if previous_values is None else torch.as_tensor(
            previous_values, dtype=torch.float32)

    def __call__(self, values, gradient=None):
        value = torch.tensor(np.asarray(values, dtype=np.float32), requires_grad=True)
        joints = value[:12]
        translation = value[12:15]
        euler = value[15:18]
        points = xhand_points(self.model, joints, translation, euler)
        pos_vectors = points[self.pos_task] - points[self.pos_origin]
        pos_loss = torch.mean(
            self.pos_weights * huber_norm(pos_vectors - self.target_vectors, self.huber_delta)
        ) * 1000.0
        dir_vectors = points[self.dir_task] - points[self.dir_origin]
        dir_norms = torch.linalg.vector_norm(dir_vectors, dim=1, keepdim=True)
        dir_unit = dir_vectors / (dir_norms + 1e-8)
        dir_loss = torch.mean(
            self.dir_weights * huber_norm(dir_unit - self.target_directions, self.direction_delta)
        )
        palm_loss = torch.sum((points[0] - self.target_palm) ** 2) * self.palm_weight
        loss = pos_loss + dir_loss + palm_loss + self.neutral_weight * torch.mean(joints * joints)
        if self.contact_anchors is not None and self.contact_weight > 0:
            anchor_points = self.contact_anchors["points"]
            anchor_tip_indices = self.contact_anchors["xhand_tip_indices"]
            tip_positions = points[anchor_tip_indices]
            contact_loss = torch.mean(torch.sum((tip_positions - anchor_points) ** 2, dim=1))
            loss = loss + self.contact_weight * contact_loss * 1000.0
        if self.grip_active and self.grip_flexion_weight > 0 and self.grip_flexion_targets is not None:
            deficit = torch.clamp(self.grip_flexion_targets - joints, min=0.0)
            loss = loss + self.grip_flexion_weight * torch.mean(deficit * deficit)
        if self.previous is not None:
            loss = loss + self.previous_weight * torch.mean((joints - self.previous[:12]) ** 2)
        if gradient is not None and len(gradient) > 0:
            loss.backward()
            gradient[:] = value.grad.detach().numpy().astype(np.float64)
        return float(loss.detach())


XHAND_TIP_INDICES = {"index": 14, "middle": 19, "ring": 24, "little": 29, "thumb": 8}


def frame_contact_anchors(frame_points, contact_plan, frame_index):
    if contact_plan is None:
        return None
    frame_phase = contact_plan["frames"][frame_index]["phase"]
    mask = np.asarray(contact_plan["source_contact_mask"][frame_index], dtype=bool)
    if frame_phase != "close" or not mask.any():
        return None
    semantics = list(SOURCE_TIP_INDICES.keys())
    anchor_points = np.stack([
        frame_points[SOURCE_TIP_INDICES[semantics[i]]]
        for i in np.flatnonzero(mask)
    ]).astype(np.float32)
    anchor_tip_indices = np.asarray(
        [XHAND_TIP_INDICES[semantics[i]] for i in np.flatnonzero(mask)], dtype=np.int64)
    return {
        "points": torch.as_tensor(anchor_points, dtype=torch.float32),
        "xhand_tip_indices": anchor_tip_indices,
    }


def retarget_vector_trajectory(source_points, model, pairs, scales, config,
                               maxeval, translation_bound, joint_lower, joint_upper,
                               contact_plan=None, contact_weight=0.0,
                               grip_flexion_weight=0.0, initial_values=None):
    grip_flexion_targets = np.asarray(config.get("grip_flexion_targets", []), dtype=np.float32)
    lower = np.concatenate([joint_lower, np.full(3, -translation_bound), np.full(3, -np.pi)])
    upper = np.concatenate([joint_upper, np.full(3, translation_bound), np.full(3, np.pi)])
    position_mask = np.asarray([item["kind"] == "position" for item in pairs], dtype=bool)
    direction_mask = ~position_mask
    source_pos_origin = np.asarray(
        [item["shadow_origin"] for item, ok in zip(pairs, position_mask) if ok], dtype=np.int64)
    source_pos_task = np.asarray(
        [item["shadow_task"] for item, ok in zip(pairs, position_mask) if ok], dtype=np.int64)
    source_dir_origin = np.asarray(
        [item["shadow_origin"] for item, ok in zip(pairs, direction_mask) if ok], dtype=np.int64)
    source_dir_task = np.asarray(
        [item["shadow_task"] for item, ok in zip(pairs, direction_mask) if ok], dtype=np.int64)
    position_scales = scales[position_mask]
    alpha_blend = config.get("alpha_blend", {})
    source_tip_indices = [4, 8, 12, 16, 20]
    outputs, losses, previous = [], [], None
    for frame_index, frame_points in enumerate(source_points):
        target_vectors = (
            frame_points[source_pos_task] - frame_points[source_pos_origin]
        ) * position_scales[:, None]
        raw_directions = frame_points[source_dir_task] - frame_points[source_dir_origin]
        target_directions = raw_directions / (
            np.linalg.norm(raw_directions, axis=1, keepdims=True) + 1e-8)
        alphas = None
        if alpha_blend:
            d1 = float(alpha_blend.get("d1_m", 0.025))
            d2 = float(alpha_blend.get("d2_m", 0.05))
            distances = np.linalg.norm(
                frame_points[source_tip_indices] - frame_points[20], axis=1)
            alphas = np.clip((d2 - distances) / (d2 - d1), 0.0, 1.0)
        if previous is None and initial_values is not None:
            start = np.asarray(initial_values, dtype=np.float64).copy()
        elif previous is None:
            start = np.zeros(18, dtype=np.float64)
        else:
            start = previous.copy()
        start = np.clip(start, lower + 1e-6, upper - 1e-6)
        grip_active = bool(
            contact_plan is not None
            and grip_flexion_weight > 0
            and contact_plan["frames"][frame_index]["phase"] in ("close", "lift")
        )
        objective = XHandVectorObjective(
            model, frame_points[0], target_vectors, target_directions, pairs, config,
            previous_values=previous,
            contact_anchors=frame_contact_anchors(frame_points, contact_plan, frame_index),
            contact_weight=contact_weight,
            alphas=alphas,
            grip_flexion_weight=grip_flexion_weight,
            grip_flexion_targets=grip_flexion_targets,
            grip_active=grip_active,
        )
        optimizer = nlopt.opt(nlopt.LD_SLSQP, 18)
        optimizer.set_min_objective(objective)
        optimizer.set_lower_bounds(lower.tolist())
        optimizer.set_upper_bounds(upper.tolist())
        optimizer.set_maxeval(int(maxeval))
        optimizer.set_xtol_rel(1e-6)
        optimizer.set_ftol_rel(1e-8)
        try:
            result = optimizer.optimize(start)
        except (nlopt.RoundoffLimited, RuntimeError):
            result = start
        previous = np.asarray(result, dtype=np.float32)
        outputs.append(previous)
        losses.append(objective(previous))
    return np.stack(outputs), np.asarray(losses, dtype=np.float32)


def retarget_file(args):
    source_data = np.load(args.source, allow_pickle=True).item()
    indices = [int(index) for index in (args.trajectory_indices or [0])]
    vector_config, vector_path, vector_sha = load_vector_config(args.vector_config)
    pairs = vector_config["pairs"]
    shadow_model, xhand_model = build_models()
    shadow_zero, xhand_zero = zero_pose_points(shadow_model, xhand_model)
    scales = zero_pose_scales(shadow_zero, xhand_zero, pairs)
    joint_lower = xhand_model.revolute_joints_q_lower[0].detach().cpu().numpy()
    joint_upper = xhand_model.revolute_joints_q_upper[0].detach().cpu().numpy()
    object_name = args.object_name or args.source.stem
    outputs, all_losses = [], []
    warm_start_by_index = {}
    if args.warm_start_dir is not None:
        warm_data = np.load(
            args.warm_start_dir / f"{args.object_name or args.source.stem}.npy",
            allow_pickle=True).item()
        warm_indices = np.asarray(warm_data["source_trajectory_indices"], dtype=np.int64)
        for warm_row, warm_index in enumerate(warm_indices):
            saved = np.asarray(warm_data["grasp_seqs"][warm_row, 0], dtype=np.float64)
            warm_start_by_index[int(warm_index)] = np.concatenate(
                [saved[6:18], saved[:6]])
    for trajectory_number, source_index in enumerate(indices):
        source_frames = np.asarray(
            source_data["grasp_seqs"][source_index], dtype=np.float32).copy()
        source_frames[:, 2] += args.source_z_offset
        with torch.no_grad():
            shadow_model.set_parameters(shadow_to_model_q(source_frames))
            source_points = shadow_model.get_penetraion_keypoints().cpu().numpy()
        contact_plan = None
        if args.contact_weight > 0 or args.grip_flexion_weight > 0:
            object_vertices, object_normals = transformed_object_surface(
                args.object_root / object_name,
                np.asarray(source_data["obj_scale"])[source_index],
                np.asarray(source_data["obj_rotmat"])[source_index],
                args.object_clearance,
            )
            tip_points = {
                semantic: source_points[:, point_index, :]
                for semantic, point_index in SOURCE_TIP_INDICES.items()
            }
            contact_plan = build_phase_contact_plan(
                source_frames,
                tip_points,
                object_vertices,
                object_normals,
                contact_threshold=args.contact_threshold,
                min_contact_tips=2,
                lift_delta=args.lift_delta,
                contact_fallback=args.contact_fallback,
            )
        internal, losses = retarget_vector_trajectory(
            source_points, xhand_model, pairs, scales, vector_config,
            args.maxeval, args.translation_bound, joint_lower, joint_upper,
            contact_plan=contact_plan, contact_weight=args.contact_weight,
            grip_flexion_weight=args.grip_flexion_weight,
            initial_values=warm_start_by_index.get(source_index),
        )
        outputs.append(internal_to_saved(internal))
        all_losses.append(losses)
    output_frames = np.stack(outputs).astype(np.float32)
    loss_frames = np.stack(all_losses).astype(np.float32)
    output = {
        "grasp_seqs": output_frames,
        "optimization_loss_per_frame": loss_frames,
        "source_trajectory_indices": np.asarray(indices, dtype=np.int64),
        "obj_rotmat": np.asarray(source_data["obj_rotmat"])[indices],
        "obj_scale": np.asarray(source_data["obj_scale"])[indices],
        "retarget_method": METHOD,
        "vector_config": vector_path,
        "vector_config_sha256": vector_sha,
        "vector_scales": scales.astype(np.float32),
        "mapping_semantics": [item["semantic"] for item in pairs],
        "source_z_offset": float(args.source_z_offset),
        "maxeval": int(args.maxeval),
        "contact_weight": float(args.contact_weight),
        "contact_threshold": float(args.contact_threshold),
        "lift_delta": float(args.lift_delta),
        "grip_flexion_weight": float(args.grip_flexion_weight),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, output, allow_pickle=True)
    print(f"trajectories={len(output_frames)}")
    print(f"output_shape={output_frames.shape}")
    print(f"mean_vector_loss={loss_frames.mean():.6f}")
    print(f"output={args.output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trajectory-indices", type=int, nargs="*")
    parser.add_argument("--maxeval", type=int, default=50)
    parser.add_argument("--translation-bound", type=float, default=2.0)
    parser.add_argument("--source-z-offset", type=float, default=0.4)
    parser.add_argument("--vector-config", type=Path, default=DEFAULT_VECTOR_CONFIG)
    parser.add_argument("--contact-weight", type=float, default=0.0)
    parser.add_argument("--contact-threshold", type=float, default=0.02)
    parser.add_argument("--lift-delta", type=float, default=0.03)
    parser.add_argument("--object-clearance", type=float, default=0.005)
    parser.add_argument("--object-root", type=Path,
                        default=RETARGET_ROOT.parent / "reference" / "HandRetargetTask2026"
                                / "scripts" / "data" / "sorting" / "object_41")
    parser.add_argument("--object-name")
    parser.add_argument("--grip-flexion-weight", type=float, default=0.0)
    parser.add_argument("--contact-fallback", choices=("error", "nearest"), default="nearest")
    parser.add_argument("--warm-start-dir", type=Path)
    retarget_file(parser.parse_args())


if __name__ == "__main__":
    main()
