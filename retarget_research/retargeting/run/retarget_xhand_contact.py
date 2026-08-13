#!/usr/bin/env python3
"""在官方XHand语义基线上做单次分阶段真实指腹接触细化。

输入：Shadow源npy、官方18维基线、源轨迹索引、XHand指腹配置和物体表面。
输出：同形状18维单候选轨迹、逐帧loss分解和自动阶段元数据。
内部逻辑：接近段复制基线，闭合段SLSQP联合优化关键点/接触/法向/穿透/关节先验，抬升段冻结抓形。
作用：在不修改官方参考源码、不做多候选物理筛选的前提下加入物体感知重定向。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import nlopt
import numpy as np
import torch


RETARGET_ROOT = Path(__file__).resolve().parents[1]
EVALUATE_DIR = RETARGET_ROOT / "evaluate"
PREPARE_DIR = RETARGET_ROOT / "prepare"
for path in (EVALUATE_DIR, PREPARE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from evaluate_xhand_geometry import (  # noqa: E402
    build_models,
    shadow_to_model_q,
)
from object_geometry import transformed_object_surface  # noqa: E402
try:
    from .phase_contact import (  # noqa: E402
        build_phase_contact_plan,
        load_pad_config,
        pad_contact_terms,
        world_pad_regions,
    )
except ImportError:
    from phase_contact import (  # noqa: E402
        build_phase_contact_plan,
        load_pad_config,
        pad_contact_terms,
        world_pad_regions,
    )
from utils.rot6d import robust_compute_orth6d_from_eulerXYZ  # noqa: E402


REFERENCE_SCRIPTS = RETARGET_ROOT.parent / "reference" / "HandRetargetTask2026" / "scripts"
OBJECT_ROOT = REFERENCE_SCRIPTS / "data" / "sorting" / "object_41"
SOURCE_TIP_INDICES = {"index": 4, "middle": 8, "ring": 12, "little": 16, "thumb": 20}


def load_mapping():
    """读取15对Shadow—XHand语义关键点索引。

    输入：固定配置`configs/xhand_keypoint_map.json`。
    输出：源索引、目标索引和语义名称三个等长列表。
    内部逻辑：保持JSON顺序，不依赖模型内部遍历的隐含含义。
    作用：让接触细化的几何主目标与官方基线完全同口径。
    """
    config = json.loads(
        (RETARGET_ROOT / "configs" / "xhand_keypoint_map.json").read_text(
            encoding="utf-8"
        )
    )
    return (
        [item["shadow_index"] for item in config["pairs"]],
        [item["xhand_index"] for item in config["pairs"]],
        [item["semantic"] for item in config["pairs"]],
    )


def saved_to_internal(frames):
    """把保存顺序`[腕6,关节12]`改成优化顺序`[关节12,腕6]`。

    输入：`(T,18)`XHand基线轨迹。
    输出：同形状内部优化轨迹。
    内部逻辑：仅重排列，不改变任何数值。
    作用：兼容官方目标函数的变量布局，并保留物理入口的既有保存格式。
    """
    frames = np.asarray(frames, dtype=np.float32)
    if frames.ndim != 2 or frames.shape[1] != 18:
        raise ValueError(f"XHand轨迹应为(T,18)，实际为{frames.shape}")
    return np.concatenate([frames[:, 6:], frames[:, :6]], axis=1)


def internal_to_saved(frames):
    """把内部`[关节12,腕6]`恢复为标准保存顺序。

    输入：`(T,18)`内部优化轨迹。
    输出：`[腕平移3,腕欧拉3,手指关节12]`。
    内部逻辑：执行`saved_to_internal`的逆列重排。
    作用：让现有几何评估和Isaac重放无需任何修改。
    """
    frames = np.asarray(frames, dtype=np.float32)
    return np.concatenate([frames[:, 12:18], frames[:, :12]], axis=1)


def model_q(values, device):
    """把18维内部变量转换成XHand运动学模型输入。

    输入：`[关节12,平移3,欧拉3]`张量及设备。
    输出：`(1,21)`的`[平移3,旋转6D,关节12]`。
    内部逻辑：只对末3维欧拉角做连续旋转6D转换。
    作用：连接NLopt变量和可微XHand正向运动学。
    """
    joints = values[:12].view(1, 12)
    translation = values[12:15].view(1, 3)
    rotation = robust_compute_orth6d_from_eulerXYZ(values[15:18].view(1, 3))
    return torch.cat([translation, rotation, joints], dim=1).to(device)


class XHandContactObjective:
    """XHand单帧语义几何与真实指腹接触联合目标。"""

    def __init__(
        self,
        model,
        target_keypoints,
        target_indices,
        pad_config,
        contact_targets,
        object_surface,
        reference_values,
        contact_weight,
        normal_weight,
        penetration_weight,
        joint_prior_weight,
        contact_offset,
        min_signed_distance,
    ):
        """保存单帧不变目标和权重。

        输入：XHand模型、Shadow目标点、目标索引、指腹/物体几何、官方基线和各权重。
        输出：无返回值；构造可被NLopt反复调用的目标对象。
        内部逻辑：所有非优化数组只转换一次，最终调用时再建立可求导变量。
        作用：明确区分语义模仿主项和物理接触细化项。
        """
        self.model = model
        self.target = torch.as_tensor(target_keypoints, dtype=torch.float32)
        self.target_indices = np.asarray(target_indices, dtype=np.int64)
        self.pad_config = pad_config
        self.contact_targets = contact_targets
        self.object_surface = object_surface
        self.reference = torch.as_tensor(reference_values, dtype=torch.float32)
        self.contact_weight = float(contact_weight)
        self.normal_weight = float(normal_weight)
        self.penetration_weight = float(penetration_weight)
        self.joint_prior_weight = float(joint_prior_weight)
        self.contact_offset = float(contact_offset)
        self.min_signed_distance = float(min_signed_distance)
        self.last_components = {}

    def __call__(self, values, gradient=None):
        """计算联合loss并按需写回18维梯度。

        输入：18维内部候选和可选NLopt梯度缓冲区。
        输出：总标量loss；同时更新最近一次分项字典。
        内部逻辑：做正向运动学后调用共享指腹项，再加入与官方关节的二次先验。
        作用：只在闭合阶段把官方几何解小幅推向更可执行的接触解。
        """
        values_tensor = torch.tensor(
            np.asarray(values, dtype=np.float32),
            dtype=torch.float32,
            requires_grad=True,
        )
        keypoints = self.model.get_penetraion_keypoints(
            q=model_q(values_tensor, self.model.device)
        )[0]
        difference = keypoints[self.target_indices] - self.target
        geometry = torch.mean(torch.sum(difference * difference, dim=1)) * 1000.0
        loss = geometry
        components = {"geometry": geometry}
        pads = world_pad_regions(self.model, self.pad_config)
        terms = pad_contact_terms(
            pads,
            self.contact_targets,
            self.object_surface,
            self.contact_offset,
            self.min_signed_distance,
        )
        if "contact" in terms and self.contact_weight > 0:
            contact = self.contact_weight * terms["contact"] * 1000.0
            loss = loss + contact
            components["phase_contact"] = contact
        if "normal" in terms and self.normal_weight > 0:
            normal = self.normal_weight * terms["normal"]
            loss = loss + normal
            components["phase_normal"] = normal
        if "penetration" in terms and self.penetration_weight > 0:
            penetration = self.penetration_weight * terms["penetration"] * 1000.0
            loss = loss + penetration
            components["phase_penetration"] = penetration
        if self.joint_prior_weight > 0:
            prior = self.joint_prior_weight * torch.mean(
                (values_tensor[:12] - self.reference[:12]) ** 2
            )
            loss = loss + prior
            components["joint_prior"] = prior
        self.last_components = {
            name: float(value.detach().cpu().item())
            for name, value in components.items()
        }
        if gradient is not None and len(gradient) > 0:
            loss.backward()
            gradient[:] = values_tensor.grad.detach().cpu().numpy().astype(np.float64)
        return float(loss.detach().cpu().item())


def refine_trajectory(
    baseline_internal,
    source_keypoints,
    target_indices,
    phase_plan,
    model,
    pad_config,
    lower,
    upper,
    args,
):
    """细化一条XHand轨迹并硬保持抬升抓形。

    输入：内部基线、Shadow关键点、映射索引、阶段计划、模型/边界和参数。
    输出：细化轨迹、逐帧总loss和分项列表。
    内部逻辑：接近段复制；闭合段从同帧官方解启动SLSQP；抬升段用官方腕姿态+闭合末帧关节。
    作用：把额外计算集中到约8–10个闭合帧，控制1000轨迹扩展成本。
    """
    results, losses, components = [], [], []
    grasp_joints = None
    for frame_index, baseline in enumerate(baseline_internal):
        safe_baseline = np.clip(
            np.asarray(baseline, dtype=np.float64), lower + 1e-6, upper - 1e-6
        )
        phase = phase_plan[frame_index]
        objective = XHandContactObjective(
            model,
            source_keypoints[frame_index],
            target_indices,
            pad_config,
            phase["targets"],
            (phase["object_vertices"], phase["object_normals"]),
            safe_baseline,
            args.contact_weight,
            args.normal_weight,
            args.penetration_weight,
            args.joint_prior_weight,
            args.contact_offset,
            args.min_signed_distance,
        )
        if phase["phase"] == "approach":
            result = safe_baseline.copy()
        elif phase["phase"] == "lift":
            if grasp_joints is None:
                grasp_joints = results[-1][:12].copy()
            result = safe_baseline.copy()
            result[:12] = grasp_joints
        else:
            optimizer = nlopt.opt(nlopt.LD_SLSQP, 18)
            optimizer.set_min_objective(objective)
            optimizer.set_lower_bounds(lower.tolist())
            optimizer.set_upper_bounds(upper.tolist())
            optimizer.set_maxeval(args.maxeval)
            optimizer.set_xtol_rel(1e-6)
            optimizer.set_ftol_rel(1e-8)
            try:
                result = optimizer.optimize(safe_baseline)
            except (nlopt.RoundoffLimited, RuntimeError):
                result = safe_baseline.copy()
        result = np.clip(np.asarray(result, dtype=np.float32), lower, upper)
        results.append(result)
        losses.append(objective(result))
        components.append(objective.last_components.copy())
    return np.stack(results), np.asarray(losses, dtype=np.float32), components


def retarget_file(args):
    """读取源/官方基线，逐条细化并保存标准18维候选。

    输入：命令行文件、索引、指腹配置、物体目录和目标权重。
    输出：带方法元数据的XHand npy。
    内部逻辑：按源索引对齐基线行，构造源关键点/物体阶段计划，逐条调用闭合细化。
    作用：作为XHand物体感知单候选方法的可复现文件入口。
    """
    source_data = np.load(args.source, allow_pickle=True).item()
    baseline_data = np.load(args.initial_target, allow_pickle=True).item()
    indices = [int(index) for index in (args.trajectory_indices or [0])]
    baseline_indices = np.asarray(
        baseline_data["source_trajectory_indices"], dtype=np.int64
    )
    baseline_frames = np.asarray(baseline_data["grasp_seqs"], dtype=np.float32)
    baseline_by_index = {
        int(index): frames for index, frames in zip(baseline_indices, baseline_frames)
    }
    missing = sorted(set(indices) - set(baseline_by_index))
    if missing:
        raise ValueError(f"官方XHand基线缺少源索引: {missing}")
    pad_config = load_pad_config(args.contact_pad_config, expected_hand="xhand")
    source_indices, target_indices, semantics = load_mapping()
    shadow_model, xhand_model = build_models()
    joint_lower = xhand_model.revolute_joints_q_lower[0].detach().cpu().numpy()
    joint_upper = xhand_model.revolute_joints_q_upper[0].detach().cpu().numpy()
    lower = np.concatenate(
        [joint_lower, np.full(3, -args.translation_bound), np.full(3, -np.pi)]
    )
    upper = np.concatenate(
        [joint_upper, np.full(3, args.translation_bound), np.full(3, np.pi)]
    )
    outputs, all_losses, all_components, all_phases = [], [], [], []
    object_name = args.object_name or args.source.stem
    for source_index in indices:
        source_frames = np.asarray(
            source_data["grasp_seqs"][source_index], dtype=np.float32
        ).copy()
        source_frames[:, 2] += args.source_z_offset
        with torch.no_grad():
            shadow_model.set_parameters(shadow_to_model_q(source_frames))
            all_source_points = shadow_model.get_penetraion_keypoints().cpu().numpy()
        object_vertices, object_normals = transformed_object_surface(
            args.object_root / object_name,
            np.asarray(source_data["obj_scale"])[source_index],
            np.asarray(source_data["obj_rotmat"])[source_index],
            args.object_clearance,
        )
        tips = {
            semantic: all_source_points[:, point_index, :]
            for semantic, point_index in SOURCE_TIP_INDICES.items()
        }
        plan = build_phase_contact_plan(
            source_frames,
            tips,
            object_vertices,
            object_normals,
            args.contact_threshold,
            args.min_contact_tips,
            args.lift_delta,
            args.region_neighbors,
        )
        internal, losses, components = refine_trajectory(
            saved_to_internal(baseline_by_index[source_index]),
            all_source_points[:, source_indices, :],
            target_indices,
            plan["frames"],
            xhand_model,
            pad_config,
            lower,
            upper,
            args,
        )
        outputs.append(internal_to_saved(internal))
        all_losses.append(losses)
        all_components.append(components)
        all_phases.append(
            {
                "close_start_frame": int(plan["close_start_frame"]),
                "lift_start_frame": int(plan["lift_start_frame"]),
                "grasp_frame": int(plan["grasp_frame"]),
                "source_contact_tip_count": np.asarray(
                    plan["source_contact_tip_count"]
                ).tolist(),
            }
        )
    output = {
        "grasp_seqs": np.stack(outputs).astype(np.float32),
        "source_trajectory_indices": np.asarray(indices, dtype=np.int64),
        "obj_rotmat": np.asarray(source_data["obj_rotmat"])[indices],
        "obj_scale": np.asarray(source_data["obj_scale"])[indices],
        "mapping_semantics": semantics,
        "optimization_loss_per_frame": np.stack(all_losses),
        "optimization_loss_components_per_frame": all_components,
        "phase_metadata": all_phases,
        "method": "xhand_official_baseline_phase_contact_refinement_v1",
        "initial_target": str(args.initial_target.resolve()),
        "contact_pad_config": str(args.contact_pad_config.resolve()),
        "maxeval": int(args.maxeval),
        "contact_weight": float(args.contact_weight),
        "normal_weight": float(args.normal_weight),
        "penetration_weight": float(args.penetration_weight),
        "joint_prior_weight": float(args.joint_prior_weight),
        "contact_offset": float(args.contact_offset),
        "min_signed_distance": float(args.min_signed_distance),
        "contact_threshold": float(args.contact_threshold),
        "min_contact_tips": int(args.min_contact_tips),
        "lift_delta": float(args.lift_delta),
        "region_neighbors": int(args.region_neighbors),
        "object_clearance": float(args.object_clearance),
        "source_z_offset": float(args.source_z_offset),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, output, allow_pickle=True)
    print(f"trajectories={len(outputs)}")
    print(f"output_shape={output['grasp_seqs'].shape}")
    print(f"mean_loss={float(np.mean(output['optimization_loss_per_frame'])):.6f}")
    print(f"output={args.output}")


def main():
    """解析XHand闭合接触细化参数并执行文件入口。

    输入：源/官方基线/输出、轨迹索引、校准配置、阶段阈值和loss权重。
    输出：单候选npy与终端摘要。
    内部逻辑：检查符号距离、权重和邻域参数后调用`retarget_file`。
    作用：提供可由未来manifest批处理调用的稳定命令行接口。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--initial-target", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trajectory-indices", type=int, nargs="*")
    parser.add_argument("--object-name")
    parser.add_argument("--contact-pad-config", type=Path, required=True)
    parser.add_argument("--object-root", type=Path, default=OBJECT_ROOT)
    parser.add_argument("--source-z-offset", type=float, default=0.4)
    parser.add_argument("--object-clearance", type=float, default=0.005)
    parser.add_argument("--translation-bound", type=float, default=2.0)
    parser.add_argument("--maxeval", type=int, default=20)
    parser.add_argument("--contact-weight", type=float, default=3.0)
    parser.add_argument("--normal-weight", type=float, default=0.02)
    parser.add_argument("--penetration-weight", type=float, default=1.0)
    parser.add_argument("--joint-prior-weight", type=float, default=5.0)
    parser.add_argument("--contact-threshold", type=float, default=0.02)
    parser.add_argument("--min-contact-tips", type=int, default=2)
    parser.add_argument("--lift-delta", type=float, default=0.03)
    parser.add_argument("--region-neighbors", type=int, default=32)
    parser.add_argument("--contact-offset", type=float, default=-0.001)
    parser.add_argument("--min-signed-distance", type=float, default=-0.003)
    args = parser.parse_args()
    for name in (
        "contact_weight",
        "normal_weight",
        "penetration_weight",
        "joint_prior_weight",
        "contact_threshold",
        "lift_delta",
    ):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')}不能为负数")
    if args.min_signed_distance > args.contact_offset:
        parser.error("--min-signed-distance必须小于等于--contact-offset")
    if not 1 <= args.min_contact_tips <= 5:
        parser.error("--min-contact-tips必须在1到5之间")
    if args.region_neighbors < 1 or args.maxeval < 1:
        parser.error("--region-neighbors和--maxeval必须为正整数")
    retarget_file(args)


if __name__ == "__main__":
    main()
