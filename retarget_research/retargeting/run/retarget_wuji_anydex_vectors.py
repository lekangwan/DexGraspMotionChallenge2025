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

from object_geometry import transformed_object_surface  # noqa: E402
from phase_contact import build_phase_contact_plan  # noqa: E402
from retarget_wuji_keypoints import (  # noqa: E402
    apply_anatomy_profile,
    build_shadow_model,
    build_wuji_model,
    clip_start_to_bounds,
    initial_values as make_initial_values,
    robust_compute_orth6d_from_eulerXYZ,
    shadow_keypoints,
)


METHOD = "anydex_style_segment_vectors_v1"
DEFAULT_VECTOR_CONFIG = RETARGET_ROOT / "configs" / "wuji_anydex_vectors_v1.json"
SHADOW_KEYPOINT_COUNT = 21
WUJI_KEYPOINT_COUNT = 26
SOURCE_TIP_INDICES = [4, 8, 12, 16, 20]
SOURCE_TIP_MAP = {"index": 4, "middle": 8, "ring": 12, "little": 16, "thumb": 20}
WUJI_TIP_INDICES = {"index": 10, "middle": 15, "ring": 20, "little": 25, "thumb": 5}


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
        if not 0 <= int(item["wuji_origin"]) < WUJI_KEYPOINT_COUNT:
            raise ValueError(f"Wuji索引越界: {item['semantic']}")
        if not 0 <= int(item["wuji_task"]) < WUJI_KEYPOINT_COUNT:
            raise ValueError(f"Wuji索引越界: {item['semantic']}")
        if not np.isfinite(float(item["weight"])) or float(item["weight"]) <= 0:
            raise ValueError(f"向量权重必须为有限正数: {item['semantic']}")
    for key in ("huber_delta_m", "direction_huber_delta", "neutral_joint_weight",
                "previous_joint_weight", "palm_position_weight"):
        value = float(config[key])
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{key}必须是有效的非负值")
    return config, str(resolved), hashlib.sha256(raw).hexdigest()


def wuji_points(model, joints, translation=None, euler=None):
    joints = joints.view(1, -1)
    translation = torch.zeros((1, 3), dtype=joints.dtype) if translation is None else translation.view(1, 3)
    euler = torch.zeros((1, 3), dtype=joints.dtype) if euler is None else euler.view(1, 3)
    rotation = robust_compute_orth6d_from_eulerXYZ(euler)
    q = torch.cat([translation, rotation, joints], dim=1).to(model.device)
    return model.get_penetraion_keypoints(q=q)[0]


def zero_pose_scales(shadow_zero, wuji_zero, pairs):
    scales = []
    for item in pairs:
        source_length = np.linalg.norm(
            shadow_zero[int(item["shadow_task"])] - shadow_zero[int(item["shadow_origin"])])
        target_length = np.linalg.norm(
            wuji_zero[int(item["wuji_task"])] - wuji_zero[int(item["wuji_origin"])])
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


def frame_contact_anchors(frame_points, contact_plan, frame_index):
    if contact_plan is None:
        return None
    frame_phase = contact_plan["frames"][frame_index]["phase"]
    mask = np.asarray(contact_plan["source_contact_mask"][frame_index], dtype=bool)
    if frame_phase != "close" or not mask.any():
        return None
    semantics = list(SOURCE_TIP_MAP.keys())
    anchor_points = np.stack([
        frame_points[SOURCE_TIP_MAP[semantics[i]]]
        for i in np.flatnonzero(mask)
    ]).astype(np.float32)
    anchor_tip_indices = np.asarray(
        [WUJI_TIP_INDICES[semantics[i]] for i in np.flatnonzero(mask)], dtype=np.int64)
    return {
        "points": torch.as_tensor(anchor_points, dtype=torch.float32),
        "wuji_tip_indices": anchor_tip_indices,
    }


class WujiAnydexObjective:
    def __init__(self, model, target_palm, target_vectors, target_directions,
                 pairs, config, previous_values=None, flexion_couplings=None, alphas=None,
                 grip_flexion_weight=0.0, grip_flexion_targets=None, grip_active=False,
                 contact_anchors=None, contact_weight=0.0):
        self.model = model
        self.target_palm = torch.as_tensor(target_palm, dtype=torch.float32)
        self.target_vectors = torch.as_tensor(target_vectors, dtype=torch.float32)
        self.target_directions = torch.as_tensor(target_directions, dtype=torch.float32)
        position_mask = np.asarray([item["kind"] == "position" for item in pairs], dtype=bool)
        direction_mask = ~position_mask
        self.pos_origin = np.asarray(
            [item["wuji_origin"] for item, ok in zip(pairs, position_mask) if ok], dtype=np.int64)
        self.pos_task = np.asarray(
            [item["wuji_task"] for item, ok in zip(pairs, position_mask) if ok], dtype=np.int64)
        self.pos_weights = torch.as_tensor(
            [item["weight"] for item, ok in zip(pairs, position_mask) if ok], dtype=torch.float32)
        self.dir_origin = np.asarray(
            [item["wuji_origin"] for item, ok in zip(pairs, direction_mask) if ok], dtype=np.int64)
        self.dir_task = np.asarray(
            [item["wuji_task"] for item, ok in zip(pairs, direction_mask) if ok], dtype=np.int64)
        self.dir_weights = torch.as_tensor(
            [item["weight"] for item, ok in zip(pairs, direction_mask) if ok], dtype=torch.float32)
        self.alphas = None if alphas is None else torch.as_tensor(alphas, dtype=torch.float32)
        self.alpha_blend = config.get("alpha_blend", {})
        if self.alphas is not None and self.alpha_blend:
            tip_open = float(self.alpha_blend.get("tip_open_fraction", 0.6))
            dir_open = float(self.alpha_blend.get("direction_open_fraction", 0.5))
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
        self.flexion_couplings = list(flexion_couplings or [])
        self.grip_flexion_weight = float(grip_flexion_weight)
        self.grip_flexion_targets = (
            None if grip_flexion_targets is None
            else torch.as_tensor(grip_flexion_targets, dtype=torch.float32))
        self.grip_active = bool(grip_active)
        self.contact_anchors = contact_anchors
        self.contact_weight = float(contact_weight)

    def __call__(self, values, gradient=None):
        value = torch.tensor(np.asarray(values, dtype=np.float32), requires_grad=True)
        joints = value[:20]
        translation = value[20:23]
        euler = value[23:26]
        points = wuji_points(self.model, joints, translation, euler)
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
        if self.previous is not None:
            loss = loss + self.previous_weight * torch.mean((joints - self.previous[:20]) ** 2)
        for coupling in self.flexion_couplings:
            proximal = joints[coupling["proximal_index"]]
            distal = joints[coupling["distal_index"]]
            expected = coupling["ratio"] * proximal + coupling["offset_rad"]
            loss = loss + coupling["weight"] * (distal - expected) ** 2
        if self.grip_active and self.grip_flexion_weight > 0 and self.grip_flexion_targets is not None:
            deficit = torch.clamp(self.grip_flexion_targets - joints, min=0.0)
            loss = loss + self.grip_flexion_weight * torch.mean(deficit * deficit)
        if self.contact_anchors is not None and self.contact_weight > 0:
            anchor_points = self.contact_anchors["points"]
            anchor_tip_indices = self.contact_anchors["wuji_tip_indices"]
            tip_positions = points[anchor_tip_indices]
            contact_loss = torch.mean(torch.sum((tip_positions - anchor_points) ** 2, dim=1))
            loss = loss + self.contact_weight * contact_loss * 1000.0
        if gradient is not None and len(gradient) > 0:
            loss.backward()
            gradient[:] = value.grad.detach().numpy().astype(np.float64)
        return float(loss.detach())


def retarget_vector_trajectory(source_frames, source_points, model, pairs, scales, config,
                               maxeval, translation_bound, lower, upper,
                               flexion_couplings=None, contact_plan=None,
                               grip_flexion_weight=0.0, contact_weight=0.0,
                               initial_values=None):
    grip_flexion_targets = np.asarray(config.get("grip_flexion_targets", []), dtype=np.float32)
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
                frame_points[SOURCE_TIP_INDICES] - frame_points[20], axis=1)
            alphas = np.clip((d2 - distances) / (d2 - d1), 0.0, 1.0)
        if previous is None and initial_values is not None:
            start = np.asarray(initial_values, dtype=np.float64).copy()
        elif previous is None:
            start = make_initial_values(source_frames[frame_index], 20)
        else:
            start = previous.copy()
        start = clip_start_to_bounds(start, lower, upper)
        grip_active = bool(
            contact_plan is not None
            and grip_flexion_weight > 0
            and contact_plan["frames"][frame_index]["phase"] in ("close", "lift")
        )
        objective = WujiAnydexObjective(
            model, frame_points[0], target_vectors, target_directions, pairs, config,
            previous_values=previous, flexion_couplings=flexion_couplings, alphas=alphas,
            grip_flexion_weight=grip_flexion_weight,
            grip_flexion_targets=grip_flexion_targets,
            grip_active=grip_active,
            contact_anchors=frame_contact_anchors(frame_points, contact_plan, frame_index),
            contact_weight=contact_weight,
        )
        optimizer = nlopt.opt(nlopt.LD_SLSQP, 26)
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
    shadow_model = build_shadow_model()
    wuji_model = build_wuji_model()
    joint_names = list(wuji_model.robot.get_joint_parameter_names())
    with torch.no_grad():
        shadow_zero = shadow_keypoints(np.zeros((1, 28), dtype=np.float32), shadow_model)[0]
        wuji_zero = wuji_points(wuji_model, torch.zeros(20, dtype=torch.float32)).cpu().numpy()
    scales = zero_pose_scales(shadow_zero, wuji_zero, pairs)
    segment_scaling = vector_config.get("segment_scaling", {})
    if segment_scaling:
        for index, item in enumerate(pairs):
            semantic = item["semantic"]
            finger = semantic.split("_")[0]
            factors = segment_scaling.get(finger)
            if not factors:
                continue
            factor = factors[-1] if semantic.endswith("_tip_reach") else factors[1]
            scales[index] *= float(factor)
    urdf_lower = wuji_model.revolute_joints_q_lower[0].detach().numpy()
    urdf_upper = wuji_model.revolute_joints_q_upper[0].detach().numpy()
    anatomy_path, anatomy_sha, couplings = None, None, None
    lower_joints, upper_joints = urdf_lower, urdf_upper
    if args.anatomy_config is not None:
        lower_joints, upper_joints, couplings = apply_anatomy_profile(
            joint_names, urdf_lower, urdf_upper,
            json.loads(args.anatomy_config.read_text(encoding="utf-8")))
        anatomy_path = str(args.anatomy_config.resolve())
        anatomy_sha = hashlib.sha256(args.anatomy_config.read_bytes()).hexdigest()
    lower = np.concatenate([lower_joints, np.full(3, -args.translation_bound), np.full(3, -np.pi)])
    upper = np.concatenate([upper_joints, np.full(3, args.translation_bound), np.full(3, np.pi)])
    outputs, all_losses = [], []
    object_name = args.object_name or args.source.stem
    warm_start_by_index = {}
    if args.warm_start_dir is not None:
        warm_data = np.load(
            args.warm_start_dir / f"{object_name}.npy",
            allow_pickle=True).item()
        warm_indices = np.asarray(warm_data["source_trajectory_indices"], dtype=np.int64)
        for warm_row, warm_index in enumerate(warm_indices):
            saved = np.asarray(warm_data["grasp_seqs"][warm_row, 0], dtype=np.float64)
            warm_start_by_index[int(warm_index)] = np.concatenate(
                [saved[6:26], saved[:6]])
    for trajectory_number, source_index in enumerate(indices):
        source_frames = np.asarray(
            source_data["grasp_seqs"][source_index], dtype=np.float32).copy()
        source_frames[:, 2] += args.source_z_offset
        with torch.no_grad():
            source_points = shadow_keypoints(source_frames, shadow_model)
        contact_plan = None
        if args.grip_flexion_weight > 0 or args.contact_weight > 0:
            object_vertices, object_normals = transformed_object_surface(
                args.object_root / object_name,
                np.asarray(source_data["obj_scale"])[source_index],
                np.asarray(source_data["obj_rotmat"])[source_index],
                args.object_clearance,
            )
            tip_points = {
                semantic: source_points[:, point_index, :]
                for semantic, point_index in SOURCE_TIP_MAP.items()
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
            source_frames, source_points, wuji_model, pairs, scales, vector_config,
            args.maxeval, args.translation_bound, lower, upper,
            flexion_couplings=couplings,
            contact_plan=contact_plan, grip_flexion_weight=args.grip_flexion_weight,
            contact_weight=args.contact_weight,
            initial_values=warm_start_by_index.get(source_index),
        )
        outputs.append(np.concatenate([internal[:, 20:26], internal[:, :20]], axis=1))
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
        "mapping_semantics": [
            pair["semantic"]
            for pair in json.loads(
                (RETARGET_ROOT / "configs" / "wuji_keypoint_map.json")
                .read_text(encoding="utf-8"))["pairs"]
        ],
        "wuji_joint_names": joint_names,
        "anatomy_config": anatomy_path,
        "anatomy_config_sha256": anatomy_sha,
        "source_z_offset": float(args.source_z_offset),
        "maxeval": int(args.maxeval),
        "grip_flexion_weight": float(args.grip_flexion_weight),
        "contact_weight": float(args.contact_weight),
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
    parser.add_argument("--anatomy-config", type=Path)
    parser.add_argument("--grip-flexion-weight", type=float, default=0.0)
    parser.add_argument("--contact-weight", type=float, default=0.0)
    parser.add_argument("--contact-threshold", type=float, default=0.02)
    parser.add_argument("--lift-delta", type=float, default=0.03)
    parser.add_argument("--object-clearance", type=float, default=0.005)
    parser.add_argument("--object-root", type=Path,
                        default=RETARGET_ROOT.parent / "reference" / "HandRetargetTask2026"
                                / "scripts" / "data" / "sorting" / "object_41")
    parser.add_argument("--object-name")
    parser.add_argument("--contact-fallback", choices=("error", "nearest"), default="nearest")
    parser.add_argument("--warm-start-dir", type=Path)
    retarget_file(parser.parse_args())


if __name__ == "__main__":
    main()
