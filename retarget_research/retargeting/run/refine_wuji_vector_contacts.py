#!/usr/bin/env python3
"""在Wuji功能向量轨迹上加入真实指腹—物体表面接触约束。

输入：Shadow源轨迹、纯功能向量Wuji候选、物体网格和物理校准指腹。
输出：标准`(N,70,26)`混合重定向轨迹及阶段、接触和loss审计信息。
内部逻辑：接近段保留向量解；闭合段固定手腕，在保留功能向量的
同时优化指腹距离、法向和穿透；抬升段锁定最终抓形。
作用：将“跨手形态的功能关系”与“对当前物体真正形成接触”组合，
避免回到错配内部指骨的15点绝对回归。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import nlopt
import numpy as np
import torch


RUN_DIR = Path(__file__).resolve().parent
RETARGET_ROOT = RUN_DIR.parent
PREPARE_DIR = RETARGET_ROOT / "prepare"
for path in (RUN_DIR, PREPARE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from object_geometry import transformed_object_surface  # noqa: E402
from phase_contact import (  # noqa: E402
    build_phase_contact_plan,
    load_pad_config,
    pad_contact_terms,
    world_pad_regions,
)
from refine_wuji_pad_contacts import (  # noqa: E402
    OBJECT_ROOT,
    SOURCE_TIP_INDICES,
    internal_to_saved,
    saved_to_internal,
    wuji_model_q,
)
from retarget_wuji_keypoints import (  # noqa: E402
    apply_anatomy_profile,
    build_shadow_model,
    build_wuji_model,
    load_anatomy_profile,
    shadow_keypoints,
)
from retarget_wuji_vectors import huber_norm, load_vector_config  # noqa: E402


METHOD = "dexpilot_style_functional_vectors_plus_surface_contact_v1"


def numpy_pad_regions(model, pad_config, saved_frame):
    """计算一帧候选姿态下五指真实指腹的可达世界几何。

    输入：Wuji模型、指腹配置和`[腕6,关节20]`一帧。
    输出：五指到指腹点/法向NumPy数组的字典。
    内部逻辑：先正向运动学更新link位姿，再变换物理校准的局部指腹点。
    作用：让接触区域由目标手真实可达性决定，而不是照搬Shadow位置。
    """
    internal = saved_to_internal(np.asarray(saved_frame)[None])[0]
    joints = torch.as_tensor(internal[:20], dtype=torch.float32)
    model.get_penetraion_keypoints(q=wuji_model_q(joints, internal[20:26], model.device))
    regions = world_pad_regions(model, pad_config)
    return {
        name: (
            points.detach().cpu().numpy(),
            normals.detach().cpu().numpy(),
        )
        for name, (points, normals) in regions.items()
    }


class WujiVectorContactObjective:
    """单帧功能向量保真与真实指腹接触的联合目标。"""

    def __init__(
        self,
        model,
        wrist,
        target_vectors,
        pairs,
        vector_config,
        pad_config,
        contact_targets,
        object_surface,
        reference_joints,
        flexion_couplings,
        args,
    ):
        """保存单帧固定目标，预转换索引和张量以供SLSQP反复求值。"""
        self.model = model
        self.wrist = np.asarray(wrist, dtype=np.float32)
        self.target_vectors = torch.as_tensor(target_vectors, dtype=torch.float32)
        self.origins = np.asarray([item["wuji_origin"] for item in pairs], dtype=np.int64)
        self.tasks = np.asarray([item["wuji_task"] for item in pairs], dtype=np.int64)
        self.vector_weights = torch.as_tensor([item["weight"] for item in pairs], dtype=torch.float32)
        self.huber_delta = float(vector_config["huber_delta_m"])
        self.pad_config = pad_config
        self.contact_targets = contact_targets
        self.object_surface = object_surface
        self.reference = torch.as_tensor(reference_joints, dtype=torch.float32)
        self.couplings = list(flexion_couplings)
        self.args = args
        self.last_components = {}

    def __call__(self, values, gradient=None):
        """输入20关节候选，返回联合loss，并在需要时回填自动微分梯度。"""
        joints = torch.tensor(np.asarray(values, dtype=np.float32), requires_grad=True)
        points = self.model.get_penetraion_keypoints(
            q=wuji_model_q(joints, self.wrist, self.model.device)
        )[0]
        vectors = points[self.tasks] - points[self.origins]
        vector = (
            torch.mean(
                self.vector_weights
                * huber_norm(vectors - self.target_vectors, self.huber_delta)
            )
            * 1000.0
            * float(self.args.vector_weight)
        )
        loss = vector
        components = {"functional_vector": vector}
        terms = pad_contact_terms(
            world_pad_regions(self.model, self.pad_config),
            self.contact_targets,
            self.object_surface,
            self.args.contact_offset,
            self.args.min_signed_distance,
        )
        for name, weight, scale in (
            ("contact", self.args.contact_weight, 1000.0),
            ("normal", self.args.normal_weight, 1.0),
            ("penetration", self.args.penetration_weight, 1000.0),
        ):
            if name in terms and weight > 0:
                value = float(weight) * scale * terms[name]
                loss = loss + value
                components[name] = value
        if self.args.joint_prior_weight > 0:
            prior = float(self.args.joint_prior_weight) * torch.mean(
                (joints - self.reference) ** 2
            )
            loss = loss + prior
            components["joint_prior"] = prior
        for coupling in self.couplings:
            expected = (
                coupling["ratio"] * joints[coupling["proximal_index"]]
                + coupling["offset_rad"]
            )
            value = coupling["weight"] * (
                joints[coupling["distal_index"]] - expected
            ) ** 2
            loss = loss + value
            components.setdefault("anatomy_coupling", value.new_zeros(()))
            components["anatomy_coupling"] = components["anatomy_coupling"] + value
        self.last_components = {
            name: float(value.detach().cpu()) for name, value in components.items()
        }
        if gradient is not None and len(gradient) > 0:
            loss.backward()
            gradient[:] = joints.grad.detach().cpu().numpy().astype(np.float64)
        return float(loss.detach().cpu())


def refine_trajectory(
    baseline_saved,
    source_points,
    pairs,
    scales,
    vector_config,
    plan,
    model,
    pad_config,
    lower,
    upper,
    couplings,
    args,
):
    """对一条纯向量轨迹做分阶段接触细化。

    输入：基线轨迹、Shadow关键点/向量定义、接触计划、边界和权重。
    输出：细化后轨迹、逐帧loss和分项。
    内部逻辑：只在close阶段优化，lift阶段复用最终抓形但保留原手腕。
    作用：限制计算量，并防止抬升过程手指再度松开。
    """
    baseline = saved_to_internal(baseline_saved)
    source_origins = np.asarray([item["shadow_origin"] for item in pairs])
    source_tasks = np.asarray([item["shadow_task"] for item in pairs])
    desired = (
        source_points[:, source_tasks] - source_points[:, source_origins]
    ) * scales[None, :, None]
    outputs, losses, components = [], [], []
    grasp_joints = None
    for frame_index, frame in enumerate(baseline):
        phase = plan[frame_index]
        objective = WujiVectorContactObjective(
            model,
            frame[20:26],
            desired[frame_index],
            pairs,
            vector_config,
            pad_config,
            phase["targets"],
            (phase["object_vertices"], phase["object_normals"]),
            frame[:20],
            couplings,
            args,
        )
        if phase["phase"] == "approach":
            joints = frame[:20].copy()
        elif phase["phase"] == "lift":
            if grasp_joints is None:
                grasp_joints = outputs[-1][:20].copy()
            joints = grasp_joints.copy()
        else:
            optimizer = nlopt.opt(nlopt.LD_SLSQP, 20)
            optimizer.set_min_objective(objective)
            optimizer.set_lower_bounds(lower.tolist())
            optimizer.set_upper_bounds(upper.tolist())
            optimizer.set_maxeval(int(args.maxeval))
            optimizer.set_xtol_rel(1e-6)
            optimizer.set_ftol_rel(1e-8)
            try:
                joints = optimizer.optimize(np.clip(frame[:20], lower, upper))
            except (nlopt.RoundoffLimited, RuntimeError):
                joints = frame[:20].copy()
        result = frame.copy()
        result[:20] = np.clip(np.asarray(joints, dtype=np.float32), lower, upper)
        outputs.append(result)
        losses.append(objective(result[:20]))
        components.append(objective.last_components.copy())
    return internal_to_saved(np.stack(outputs)), np.asarray(losses, dtype=np.float32), components


def refine_file(args):
    """读取一个源文件及向量基线，构造接触计划并保存混合候选。"""
    source_data = np.load(args.source, allow_pickle=True).item()
    baseline_data = np.load(args.initial_target, allow_pickle=True).item()
    if baseline_data.get("retarget_method") != "dexpilot_style_functional_vectors_v1":
        raise ValueError("初始候选必须是纯Wuji功能向量v1")
    indices = [int(value) for value in (args.trajectory_indices or baseline_data["source_trajectory_indices"])]
    baseline_indices = np.asarray(baseline_data["source_trajectory_indices"], dtype=np.int64)
    baseline_by_index = {
        int(index): frames for index, frames in zip(baseline_indices, baseline_data["grasp_seqs"])
    }
    missing = sorted(set(indices) - set(baseline_by_index))
    if missing:
        raise ValueError(f"功能向量基线缺少源索引: {missing}")
    vector_config, vector_path, vector_sha = load_vector_config(Path(baseline_data["vector_config"]))
    if vector_sha != baseline_data.get("vector_config_sha256"):
        raise ValueError("功能向量配置已改变，拒绝细化旧候选")
    pairs = vector_config["pairs"]
    scales = np.asarray(baseline_data["vector_scales"], dtype=np.float32)
    pad_config = load_pad_config(args.contact_pad_config, expected_hand="wuji")
    pad_sha = hashlib.sha256(args.contact_pad_config.read_bytes()).hexdigest()
    shadow_model, wuji_model = build_shadow_model(), build_wuji_model()
    joint_names = list(wuji_model.robot.get_joint_parameter_names())
    urdf_lower = wuji_model.revolute_joints_q_lower[0].detach().cpu().numpy()
    urdf_upper = wuji_model.revolute_joints_q_upper[0].detach().cpu().numpy()
    anatomy_path = baseline_data.get("anatomy_config")
    anatomy_file = None if anatomy_path is None else Path(str(anatomy_path))
    _, _, _, _, anatomy_sha = load_anatomy_profile(
        anatomy_file, joint_names, urdf_lower, urdf_upper
    )
    if anatomy_sha != baseline_data.get("anatomy_config_sha256"):
        raise ValueError("解剖配置与向量基线不一致")
    anatomy_profile = {} if anatomy_file is None else json.loads(anatomy_file.read_text(encoding="utf-8"))
    lower, upper, couplings = apply_anatomy_profile(
        joint_names, urdf_lower, urdf_upper, anatomy_profile
    )
    object_name = args.object_name or args.source.stem
    outputs, all_losses, all_components, all_phases = [], [], [], []
    for source_index in indices:
        source_frames = np.asarray(source_data["grasp_seqs"][source_index], dtype=np.float32).copy()
        source_frames[:, 2] += float(baseline_data.get("source_z_offset", 0.4))
        source_points = shadow_keypoints(source_frames, shadow_model)
        vertices, normals = transformed_object_surface(
            args.object_root / object_name,
            np.asarray(source_data["obj_scale"])[source_index],
            np.asarray(source_data["obj_rotmat"])[source_index],
            args.object_clearance,
        )
        source_tips = {
            name: source_points[:, point_index]
            for name, point_index in SOURCE_TIP_INDICES.items()
        }
        preliminary = build_phase_contact_plan(
            source_frames,
            source_tips,
            vertices,
            normals,
            contact_threshold=args.contact_threshold,
            min_contact_tips=args.min_contact_tips,
            lift_delta=args.lift_delta,
            region_neighbors=args.region_neighbors,
            contact_fallback="nearest",
        )
        reachable = numpy_pad_regions(
            wuji_model,
            pad_config,
            baseline_by_index[source_index][preliminary["grasp_frame"]],
        )
        plan = build_phase_contact_plan(
            source_frames,
            source_tips,
            vertices,
            normals,
            contact_threshold=args.contact_threshold,
            min_contact_tips=args.min_contact_tips,
            lift_delta=args.lift_delta,
            region_neighbors=args.region_neighbors,
            opposition_candidate_neighbors=args.opposition_candidate_neighbors,
            opposition_distance_scale=args.opposition_distance_scale,
            opposition_weight=args.opposition_weight,
            opposition_refine_frames=args.opposition_refine_frames,
            reachable_pads=reachable,
            reachable_pad_alignment_weight=args.pad_alignment_weight,
            reachable_min_opposing_fingers=args.min_opposing_fingers,
            friction_stability_weight=args.friction_stability_weight,
            friction_coefficient=args.friction_coefficient,
            friction_cone_edges=args.friction_cone_edges,
            max_reachable_distance=args.max_reachable_distance,
            contact_fallback="nearest",
        )
        refined, losses, components = refine_trajectory(
            baseline_by_index[source_index], source_points, pairs, scales,
            vector_config, plan["frames"], wuji_model, pad_config,
            lower, upper, couplings, args,
        )
        outputs.append(refined)
        all_losses.append(losses)
        all_components.append(components)
        all_phases.append({
            "close_start_frame": int(plan["close_start_frame"]),
            "lift_start_frame": int(plan["lift_start_frame"]),
            "grasp_frame": int(plan["grasp_frame"]),
            "contact_fallback_used": bool(plan["contact_fallback_used"]),
            "opposition_start_frame": plan["opposition_start_frame"],
            "opposition_diagnostics": plan["opposition_diagnostics"],
        })
    output = {
        "grasp_seqs": np.stack(outputs).astype(np.float32),
        "optimization_loss_per_frame": np.stack(all_losses),
        "optimization_loss_components_per_frame": all_components,
        "source_trajectory_indices": np.asarray(indices, dtype=np.int64),
        "obj_rotmat": np.asarray(source_data["obj_rotmat"])[indices],
        "obj_scale": np.asarray(source_data["obj_scale"])[indices],
        "retarget_method": METHOD,
        "initial_target": str(args.initial_target.resolve()),
        "vector_config": vector_path,
        "vector_config_sha256": vector_sha,
        "vector_scales": scales,
        "mapping_semantics": [item["semantic"] for item in pairs],
        "contact_pad_config": str(args.contact_pad_config.resolve()),
        "contact_pad_config_sha256": pad_sha,
        "wuji_joint_names": baseline_data["wuji_joint_names"],
        "anatomy_config": baseline_data.get("anatomy_config"),
        "anatomy_config_sha256": baseline_data.get("anatomy_config_sha256"),
        "source_z_offset": float(baseline_data.get("source_z_offset", 0.4)),
        "phase_metadata": all_phases,
        "maxeval": int(args.maxeval),
    }
    for name in (
        "object_clearance", "vector_weight", "contact_weight", "normal_weight",
        "penetration_weight", "joint_prior_weight", "contact_offset",
        "min_signed_distance", "contact_threshold", "lift_delta",
        "opposition_distance_scale", "opposition_weight", "pad_alignment_weight",
        "friction_stability_weight", "friction_coefficient", "max_reachable_distance",
    ):
        output[name] = float(getattr(args, name))
    for name in (
        "min_contact_tips", "region_neighbors", "opposition_candidate_neighbors",
        "opposition_refine_frames", "min_opposing_fingers", "friction_cone_edges",
    ):
        output[name] = int(getattr(args, name))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, output, allow_pickle=True)
    print(f"trajectories={len(outputs)}")
    print(f"output_shape={output['grasp_seqs'].shape}")
    print(f"mean_joint_refinement_loss={float(np.mean(output['optimization_loss_per_frame'])):.6f}")
    print(f"output={args.output}")


def main():
    """解析混合重定向参数、验证取值后执行文件级细化。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--initial-target", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trajectory-indices", type=int, nargs="*")
    parser.add_argument("--object-name")
    parser.add_argument("--object-root", type=Path)
    parser.add_argument("--contact-pad-config", type=Path, default=RETARGET_ROOT / "configs" / "wuji_contact_pads_v1.json")
    parser.add_argument("--object-clearance", type=float, default=0.005)
    parser.add_argument("--maxeval", type=int, default=40)
    parser.add_argument("--vector-weight", type=float, default=1.0)
    parser.add_argument("--contact-weight", type=float, default=5.0)
    parser.add_argument("--normal-weight", type=float, default=0.05)
    parser.add_argument("--penetration-weight", type=float, default=1.0)
    parser.add_argument("--joint-prior-weight", type=float, default=1.0)
    parser.add_argument("--contact-offset", type=float, default=-0.002)
    parser.add_argument("--min-signed-distance", type=float, default=-0.005)
    parser.add_argument("--contact-threshold", type=float, default=0.02)
    parser.add_argument("--min-contact-tips", type=int, default=2)
    parser.add_argument("--lift-delta", type=float, default=0.03)
    parser.add_argument("--region-neighbors", type=int, default=32)
    parser.add_argument("--opposition-candidate-neighbors", type=int, default=24)
    parser.add_argument("--opposition-distance-scale", type=float, default=0.03)
    parser.add_argument("--opposition-weight", type=float, default=3.0)
    parser.add_argument("--opposition-refine-frames", type=int, default=5)
    parser.add_argument("--pad-alignment-weight", type=float, default=1.0)
    parser.add_argument("--min-opposing-fingers", type=int, default=2)
    parser.add_argument("--friction-stability-weight", type=float, default=0.0)
    parser.add_argument("--friction-coefficient", type=float, default=1.0)
    parser.add_argument("--friction-cone-edges", type=int, default=4)
    parser.add_argument("--max-reachable-distance", type=float, default=0.03)
    args = parser.parse_args()
    # GraspM3标准布局为`external_data/dataset`+`external_data/meshdata`。
    # 从源文件推导可避免把旧参考仓库的空object_41路径写死。
    if args.object_root is None:
        args.object_root = args.source.resolve().parent.parent / "meshdata"
    nonnegative = (
        "vector_weight", "contact_weight", "normal_weight", "penetration_weight",
        "joint_prior_weight", "contact_threshold", "lift_delta",
        "opposition_distance_scale", "opposition_weight", "pad_alignment_weight",
        "friction_stability_weight", "friction_coefficient", "max_reachable_distance",
    )
    if any(getattr(args, name) < 0 for name in nonnegative):
        parser.error("损失权重、距离和摩擦系数不能为负")
    if args.min_signed_distance > args.contact_offset:
        parser.error("--min-signed-distance必须小于等于--contact-offset")
    if args.maxeval < 1 or args.region_neighbors < 1 or args.opposition_candidate_neighbors < 1:
        parser.error("迭代和候选数必须为正整数")
    refine_file(args)


if __name__ == "__main__":
    main()
