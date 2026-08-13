#!/usr/bin/env python3
"""在Wuji v2基线上做分阶段真实指腹接触细化。

输入：Shadow源npy、Wuji v2基线、源索引、物理校准指腹和物体表面。
输出：单一26维细化轨迹、逐帧loss分解和阶段/对向区域元数据。
内部逻辑：接近段复制基线；闭合段固定基线手腕，仅优化20个手指关节；抬升段冻结抓形。
作用：避免旧方法用link中心冒充接触点，并保持一次优化即可扩展到大规模轨迹。
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
PREPARE_DIR = RETARGET_ROOT / "prepare"
for path in (Path(__file__).resolve().parent, PREPARE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from object_geometry import transformed_object_surface  # noqa: E402
from phase_contact import (  # noqa: E402
    build_phase_contact_plan,
    load_pad_config,
    pad_contact_terms,
    world_pad_regions,
)
from retarget_wuji_keypoints import (  # noqa: E402
    build_shadow_model,
    build_wuji_model,
    load_pairs,
    shadow_keypoints,
)
from utils.rot6d import robust_compute_orth6d_from_eulerXYZ  # noqa: E402


REFERENCE_SCRIPTS = RETARGET_ROOT.parent / "reference" / "HandRetargetTask2026" / "scripts"
OBJECT_ROOT = REFERENCE_SCRIPTS / "data" / "sorting" / "object_41"
SOURCE_TIP_INDICES = {
    "index": 4,
    "middle": 8,
    "ring": 12,
    "little": 16,
    "thumb": 20,
}


def saved_to_internal(frames):
    """把保存顺序的Wuji轨迹改成优化器内部顺序。

    输入：`(T,26)`的`[腕6,关节20]`轨迹。
    输出：`(T,26)`的`[关节20,腕6]`轨迹。
    内部逻辑：只重排列，不改变数值。
    作用：让关节变量连续放在前20维，便于固定手腕做局部细化。
    """
    frames = np.asarray(frames, dtype=np.float32)
    if frames.ndim != 2 or frames.shape[1] != 26:
        raise ValueError(f"Wuji轨迹应为(T,26)，实际为{frames.shape}")
    return np.concatenate([frames[:, 6:], frames[:, :6]], axis=1)


def internal_to_saved(frames):
    """把内部Wuji变量恢复为物理重放使用的保存顺序。

    输入：`(T,26)`的`[关节20,腕6]`轨迹。
    输出：`[腕6,关节20]`轨迹。
    内部逻辑：执行`saved_to_internal`的逆列重排。
    作用：复用现有几何评估和Isaac Gym重放入口。
    """
    frames = np.asarray(frames, dtype=np.float32)
    return np.concatenate([frames[:, 20:26], frames[:, :20]], axis=1)


def wuji_model_q(joints, wrist, device):
    """将20维关节和固定6维手腕转换为Wuji运动学输入。

    输入：可求导关节张量、`[平移3,欧拉3]`腕姿态和模型设备。
    输出：`(1,29)`的`[平移3,旋转6D,关节20]`。
    内部逻辑：仅把腕欧拉角转成旋转6D，关节梯度完整保留。
    作用：让优化器只改变手指，不重演Linker闭合后手腕跳变问题。
    """
    wrist = torch.as_tensor(wrist, dtype=torch.float32, device=device)
    rotation = robust_compute_orth6d_from_eulerXYZ(wrist[3:6].view(1, 3))
    return torch.cat([wrist[:3].view(1, 3), rotation, joints.view(1, 20)], dim=1)


class WujiPadContactObjective:
    """Wuji单帧语义几何和真实指腹接触联合目标。"""

    def __init__(
        self,
        model,
        wrist,
        source_targets,
        target_indices,
        pad_config,
        contact_targets,
        object_surface,
        reference_joints,
        args,
    ):
        """保存单帧固定目标和权重。

        输入：模型、固定腕、15点监督、指腹/物体表面、基线关节和参数。
        输出：无返回；构造可被NLopt多次调用的目标对象。
        内部逻辑：把不求导的数据预先转成张量，记录最近一次loss分项。
        作用：明确区分“复制Shadow姿态”和“形成真实物理接触”两类目标。
        """
        self.model = model
        self.wrist = np.asarray(wrist, dtype=np.float32)
        self.source_targets = torch.as_tensor(source_targets, dtype=torch.float32)
        self.target_indices = np.asarray(target_indices, dtype=np.int64)
        self.pad_config = pad_config
        self.contact_targets = contact_targets
        self.object_surface = object_surface
        self.reference = torch.as_tensor(reference_joints, dtype=torch.float32)
        self.args = args
        self.last_components = {}

    def __call__(self, values, gradient=None):
        """计算20维关节候选的联合损失和梯度。

        输入：20维关节数组和可选NLopt梯度缓冲区。
        输出：几何、接触、法向、穿透和关节先验之和。
        内部逻辑：正向运动学后复用共享真实指腹项，并按权重累加各分量。
        作用：把Wuji v2几何解小幅推向物理可抓取解，而不移动手腕。
        """
        joints = torch.tensor(
            np.asarray(values, dtype=np.float32),
            dtype=torch.float32,
            requires_grad=True,
        )
        keypoints = self.model.get_penetraion_keypoints(
            q=wuji_model_q(joints, self.wrist, self.model.device)
        )[0]
        difference = keypoints[self.target_indices] - self.source_targets
        geometry = torch.mean(torch.sum(difference * difference, dim=1)) * 1000.0
        loss = geometry
        components = {"geometry": geometry}
        terms = pad_contact_terms(
            world_pad_regions(self.model, self.pad_config),
            self.contact_targets,
            self.object_surface,
            self.args.contact_offset,
            self.args.min_signed_distance,
        )
        weighted = {
            "contact": (self.args.contact_weight, 1000.0, "phase_contact"),
            "normal": (self.args.normal_weight, 1.0, "phase_normal"),
            "penetration": (
                self.args.penetration_weight,
                1000.0,
                "phase_penetration",
            ),
        }
        for term_name, (weight, scale, output_name) in weighted.items():
            if term_name in terms and weight > 0:
                value = float(weight) * float(scale) * terms[term_name]
                loss = loss + value
                components[output_name] = value
        if self.args.joint_prior_weight > 0:
            prior = self.args.joint_prior_weight * torch.mean(
                (joints - self.reference) ** 2
            )
            loss = loss + prior
            components["joint_prior"] = prior
        self.last_components = {
            name: float(value.detach().cpu().item())
            for name, value in components.items()
        }
        if gradient is not None and len(gradient) > 0:
            loss.backward()
            gradient[:] = joints.grad.detach().cpu().numpy().astype(np.float64)
        return float(loss.detach().cpu().item())


def refine_trajectory(
    baseline_internal,
    source_targets,
    target_indices,
    phase_plan,
    model,
    pad_config,
    lower,
    upper,
    args,
):
    """细化闭合阶段并在抬升期保持最终抓形。

    输入：基线、15点监督、阶段计划、模型/指腹、关节边界和参数。
    输出：内部26维轨迹、逐帧总loss和分项列表。
    内部逻辑：接近复制；闭合只优化20关节；抬升使用基线手腕和闭合末帧关节。
    作用：把额外计算限制在约9帧，并避免逐帧接触目标造成抓形松动。
    """
    results, losses, components = [], [], []
    grasp_joints = None
    for frame_index, baseline in enumerate(baseline_internal):
        baseline = np.asarray(baseline, dtype=np.float32)
        phase = phase_plan[frame_index]
        objective = WujiPadContactObjective(
            model,
            baseline[20:26],
            source_targets[frame_index],
            target_indices,
            pad_config,
            phase["targets"],
            (phase["object_vertices"], phase["object_normals"]),
            baseline[:20],
            args,
        )
        if phase["phase"] == "approach":
            joints = baseline[:20].copy()
        elif phase["phase"] == "lift":
            if grasp_joints is None:
                grasp_joints = results[-1][:20].copy()
            joints = grasp_joints.copy()
        else:
            optimizer = nlopt.opt(nlopt.LD_SLSQP, 20)
            optimizer.set_min_objective(objective)
            optimizer.set_lower_bounds(lower.tolist())
            optimizer.set_upper_bounds(upper.tolist())
            optimizer.set_maxeval(args.maxeval)
            optimizer.set_xtol_rel(1e-6)
            optimizer.set_ftol_rel(1e-8)
            try:
                joints = optimizer.optimize(
                    np.clip(baseline[:20], lower + 1e-6, upper - 1e-6)
                )
            except (nlopt.RoundoffLimited, RuntimeError):
                joints = baseline[:20].copy()
        result = baseline.copy()
        result[:20] = np.clip(np.asarray(joints, dtype=np.float32), lower, upper)
        results.append(result)
        losses.append(objective(result[:20]))
        components.append(objective.last_components.copy())
    return np.stack(results), np.asarray(losses, dtype=np.float32), components


def refine_file(args):
    """读取源和Wuji v2基线，细化指定轨迹并保存。

    输入：命令行源/基线/索引、物体、指腹、阶段和loss参数。
    输出：带完整追溯元数据的26维单候选npy。
    内部逻辑：按源索引对齐基线，逐条构造阶段计划并调用关节细化。
    作用：提供可独立物理评估、未来可接入manifest的文件级入口。
    """
    source_data = np.load(args.source, allow_pickle=True).item()
    baseline_data = np.load(args.initial_target, allow_pickle=True).item()
    indices = [int(index) for index in (args.trajectory_indices or [0])]
    baseline_indices = np.asarray(
        baseline_data["source_trajectory_indices"], dtype=np.int64
    )
    baseline_by_index = {
        int(index): frames
        for index, frames in zip(baseline_indices, baseline_data["grasp_seqs"])
    }
    missing = sorted(set(indices) - set(baseline_by_index))
    if missing:
        raise ValueError(f"Wuji v2基线缺少源索引: {missing}")
    mapping_config = Path(baseline_data["mapping_config"])
    pairs = load_pairs(mapping_config)
    source_indices = [pair["shadow_index"] for pair in pairs]
    target_indices = [pair["wuji_index"] for pair in pairs]
    semantics = [pair["semantic"] for pair in pairs]
    pad_config = load_pad_config(args.contact_pad_config, expected_hand="wuji")
    shadow_model = build_shadow_model()
    wuji_model = build_wuji_model()
    lower = wuji_model.revolute_joints_q_lower[0].detach().cpu().numpy()
    upper = wuji_model.revolute_joints_q_upper[0].detach().cpu().numpy()
    object_name = args.object_name or args.source.stem
    outputs, all_losses, all_components, all_phases = [], [], [], []
    for source_index in indices:
        source_frames = np.asarray(
            source_data["grasp_seqs"][source_index], dtype=np.float32
        ).copy()
        source_frames[:, 2] += args.source_z_offset
        all_source_points = shadow_keypoints(source_frames, shadow_model)
        vertices, normals = transformed_object_surface(
            args.object_root / object_name,
            np.asarray(source_data["obj_scale"])[source_index],
            np.asarray(source_data["obj_rotmat"])[source_index],
            args.object_clearance,
        )
        source_tips = {
            semantic: all_source_points[:, index, :]
            for semantic, index in SOURCE_TIP_INDICES.items()
        }
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
        )
        internal, losses, components = refine_trajectory(
            saved_to_internal(baseline_by_index[source_index]),
            all_source_points[:, source_indices, :],
            target_indices,
            plan["frames"],
            wuji_model,
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
                "opposition_start_frame": plan["opposition_start_frame"],
                "opposition_diagnostics": plan["opposition_diagnostics"],
            }
        )
    output = {
        "grasp_seqs": np.stack(outputs).astype(np.float32),
        "source_trajectory_indices": np.asarray(indices, dtype=np.int64),
        "obj_rotmat": np.asarray(source_data["obj_rotmat"])[indices],
        "obj_scale": np.asarray(source_data["obj_scale"])[indices],
        "mapping_semantics": semantics,
        "mapping_config": str(mapping_config.resolve()),
        "wuji_joint_names": baseline_data["wuji_joint_names"],
        "optimization_loss_per_frame": np.stack(all_losses),
        "optimization_loss_components_per_frame": all_components,
        "phase_metadata": all_phases,
        "method": "wuji_v2_phase_pad_contact_refinement_v1",
        "initial_target": str(args.initial_target.resolve()),
        "contact_pad_config": str(args.contact_pad_config.resolve()),
    }
    for name in (
        "source_z_offset", "object_clearance", "contact_weight", "normal_weight",
        "penetration_weight", "joint_prior_weight", "contact_offset",
        "min_signed_distance", "contact_threshold", "lift_delta",
        "opposition_distance_scale", "opposition_weight",
    ):
        output[name] = float(getattr(args, name))
    for name in (
        "maxeval", "min_contact_tips", "region_neighbors",
        "opposition_candidate_neighbors", "opposition_refine_frames",
    ):
        output[name] = int(getattr(args, name))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, output, allow_pickle=True)
    print(f"trajectories={len(outputs)}")
    print(f"output_shape={output['grasp_seqs'].shape}")
    print(f"mean_loss={float(np.mean(output['optimization_loss_per_frame'])):.6f}")
    print(f"output={args.output}")


def main():
    """解析参数、验证范围并执行Wuji真实指腹细化。

    输入：源/基线/输出及接触、阶段、对向区域选择超参数。
    输出：细化npy和终端摘要。
    内部逻辑：拒绝负权重、无效邻域和颠倒的穿透距离后调用`refine_file`。
    作用：形成明确可复现的单轨迹/单文件命令行入口。
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
    parser.add_argument("--maxeval", type=int, default=20)
    parser.add_argument("--contact-weight", type=float, default=5.0)
    parser.add_argument("--normal-weight", type=float, default=0.05)
    parser.add_argument("--penetration-weight", type=float, default=1.0)
    parser.add_argument("--joint-prior-weight", type=float, default=2.0)
    parser.add_argument("--contact-threshold", type=float, default=0.02)
    parser.add_argument("--min-contact-tips", type=int, default=2)
    parser.add_argument("--lift-delta", type=float, default=0.03)
    parser.add_argument("--region-neighbors", type=int, default=32)
    parser.add_argument("--contact-offset", type=float, default=-0.003)
    parser.add_argument("--min-signed-distance", type=float, default=-0.006)
    parser.add_argument("--opposition-candidate-neighbors", type=int, default=0)
    parser.add_argument("--opposition-distance-scale", type=float, default=0.03)
    parser.add_argument("--opposition-weight", type=float, default=3.0)
    parser.add_argument("--opposition-refine-frames", type=int, default=4)
    args = parser.parse_args()
    for name in (
        "contact_weight", "normal_weight", "penetration_weight",
        "joint_prior_weight", "contact_threshold", "lift_delta",
        "opposition_distance_scale", "opposition_weight",
    ):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')}不能为负数")
    if args.min_signed_distance > args.contact_offset:
        parser.error("--min-signed-distance必须小于等于--contact-offset")
    if args.maxeval < 1 or args.region_neighbors < 1:
        parser.error("--maxeval和--region-neighbors必须为正整数")
    if args.opposition_candidate_neighbors < 0 or args.opposition_refine_frames < 1:
        parser.error("对向候选数不能为负，细化帧数必须为正")
    refine_file(args)


if __name__ == "__main__":
    main()
