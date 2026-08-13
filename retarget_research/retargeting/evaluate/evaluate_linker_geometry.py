#!/usr/bin/env python3
"""独立比较Shadow源轨迹与Linker O6候选轨迹的语义关键点。

输入：Shadow原始`.npy`、Linker候选`.npy`、轨迹索引和源轨迹Z偏移。
输出：整体、逐语义、逐帧几何误差以及目标轨迹平滑性JSON。
内部逻辑：分别做正向运动学，再使用校准配置中的真实link局部点计算距离。
作用：不依赖优化器保存的loss，独立检查候选轨迹是否真的复现源手几何。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
import transforms3d


RETARGET_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = RETARGET_ROOT.parent
REFERENCE_SCRIPTS = PROJECT_ROOT / "reference" / "HandRetargetTask2026" / "scripts"
THIRD_PARTY_PK = (
    PROJECT_ROOT
    / "reference"
    / "HandRetargetTask2026"
    / "third_party"
    / "pytorch_kinematics"
)
for path in (REFERENCE_SCRIPTS, THIRD_PARTY_PK):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from utils.hand_model import HandModel as ShadowHandModel  # noqa: E402
from utils.HandModel_linkerhand import HandModel_Linkerhand  # noqa: E402
from utils.HandModel_xhand import HandModel_xhand  # noqa: E402
from utils.rot6d import robust_compute_orth6d_from_eulerXYZ  # noqa: E402


def build_models(joint_mode="coupled6"):
    """创建评估所需的Shadow和对应关节模式Linker模型。

    输入：无显式参数；从只读参考仓库读取机器人资产。
    输出：一个Shadow模型和一个6轴耦合或11轴解耦Linker模型。
    逻辑：使用与生成轨迹时相同的URDF/MJCF，由候选元数据选择模型类，
    但不复用优化目标对象。
    作用：让评估根据保存轨迹重新计算几何，形成独立验证链路。
    """
    assets = REFERENCE_SCRIPTS / "assets"
    shadow_base = assets / "mjcf_free"
    shadow = ShadowHandModel(
        mjcf_path=str(shadow_base / "shadow_hand_vis_new.xml"),
        mesh_path=str(shadow_base / "meshes"),
        contact_points_path=str(shadow_base / "contact_points.json"),
        penetration_points_path=str(shadow_base / "penetration_points.json"),
        n_surface_points=0,
        device="cpu",
        use_joint21=True,
    )
    linker_asset = assets / "linkerhand" / "o6" / "right"
    if joint_mode not in {"coupled6", "independent11"}:
        raise ValueError(f"未知Linker关节模式: {joint_mode}")
    model_class = (
        HandModel_Linkerhand if joint_mode == "coupled6" else HandModel_xhand
    )
    linker = model_class(
        robot_name="linkerhand",
        urdf_filename="linkerhand_o6_right.urdf",
        mesh_path="",
        batch_size=1,
        device=torch.device("cpu"),
        mesh_nsp=128,
        hand_scale=1.0,
        asset_dir=str(linker_asset),
        allow_missing_contacts=True,
    )
    return shadow, linker


def load_pairs(semantics):
    """按候选轨迹记录的语义名称读取校准点对。

    输入：候选文件中保存的语义名称列表。
    输出：顺序与名称列表完全一致的配置pair列表。
    逻辑：以semantic为键查找配置，并拒绝未知或重复缺失的名称。
    作用：兼容10点基线、11点拇指消融和15点密集消融，防止口径混淆。
    """
    config = json.loads(
        (RETARGET_ROOT / "configs" / "linker_o6_keypoint_map.json").read_text(
            encoding="utf-8"
        )
    )
    by_name = {pair["semantic"]: pair for pair in config["pairs"]}
    unknown = [name for name in semantics if name not in by_name]
    if unknown:
        raise ValueError(f"候选轨迹包含配置中不存在的语义点: {unknown}")
    return [by_name[name] for name in semantics]


def shadow_to_model_q(frames):
    """把28维Shadow保存轨迹转换为31维运动学输入。

    输入：形状`(T,28)`，顺序为平移3、欧拉角3、关节22。
    输出：形状`(T,31)`，欧拉角被连续旋转6D替代。
    逻辑：保留平移和关节，只转换模型所需的旋转表达。
    作用：批量恢复每帧Shadow关键点世界坐标。
    """
    frames = torch.as_tensor(frames, dtype=torch.float32)
    result = torch.zeros((len(frames), 31), dtype=torch.float32)
    result[:, :3] = frames[:, :3]
    result[:, 3:9] = robust_compute_orth6d_from_eulerXYZ(frames[:, 3:6])
    result[:, 9:] = frames[:, 6:]
    return result


def linker_to_model_q(frames):
    """把Linker保存轨迹转换为运动学输入。

    输入：`(T,12)`O6或`(T,17)`解耦轨迹，均为手腕6在前。
    输出：`(T,15)`或`(T,20)`，把欧拉角换成旋转6D后保留手指角。
    逻辑：6轴模式随后由模型展开mimic；11轴模式直接完成正向运动学。
    作用：以候选实际控制维度重建整只手的姿态。
    """
    frames = torch.as_tensor(frames, dtype=torch.float32)
    return torch.cat(
        [
            frames[:, :3],
            robust_compute_orth6d_from_eulerXYZ(frames[:, 3:6]),
            frames[:, 6:],
        ],
        dim=1,
    )


def compute_linker_points(model, model_q, pairs):
    """逐帧计算校准后的Linker物理点世界坐标。

    输入：Linker模型、`(T,15)`姿态和M个语义点对。
    输出：形状`(T,M,3)`的NumPy坐标数组。
    逻辑：逐帧更新运动学，将每个link局部点变换到手坐标再变换到世界坐标。
    作用：绕过参考代码中尚未校准的全零关键点文件。
    """
    trajectories = []
    with torch.no_grad():
        for frame_q in model_q:
            model.update_kinematics(frame_q.view(1, -1))
            frame_points = []
            for pair in pairs:
                local = torch.tensor(
                    pair["linker_local_xyz"], dtype=torch.float32
                ).view(1, 1, 3)
                hand = model.current_status[pair["linker_link"]].transform_points(
                    local
                )[0, 0]
                world = hand @ model.global_rotation[0].T + model.global_translation[0]
                frame_points.append(world * float(model.scale))
            trajectories.append(torch.stack(frame_points))
    return torch.stack(trajectories).cpu().numpy()


def rotation_step_angles(euler_xyz):
    """计算相邻手腕姿态之间的真实旋转角。

    输入：形状`(T,3)`的静态XYZ欧拉角。
    输出：形状`(T-1,)`的相邻旋转角，单位为弧度。
    逻辑：先转旋转矩阵，再由相对矩阵的trace求SO(3)测地角。
    作用：避免直接相减欧拉角在正负π边界产生虚假的大跳变。
    """
    matrices = np.stack(
        [
            transforms3d.euler.euler2mat(*frame, axes="sxyz")
            for frame in np.asarray(euler_xyz)
        ]
    )
    relative = np.einsum("tji,tjk->tik", matrices[:-1], matrices[1:])
    cosine = np.clip((np.trace(relative, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    return np.arccos(cosine)


def coupling_residuals(independent_joints):
    """量化11轴结果偏离原O6固定联动关系的程度。

    输入：形状`(T,11)`，顺序与Linker URDF的全部手指关节一致。
    输出：`(T,5)`残差，依次为拇指IP和四指DIP减去原mimic预测值。
    逻辑：原O6关系为`IP=1.86*pitch`、`DIP=0.89*MCP`。
    作用：判断成功率提升是否真正使用了新增自由度，以及需要多大机械解耦。
    """
    joints = np.asarray(independent_joints, dtype=np.float32)
    if joints.ndim != 2 or joints.shape[1] != 11:
        raise ValueError(f"独立Linker关节应为(T,11)，实际为{joints.shape}")
    return np.stack(
        [
            joints[:, 2] - 1.86 * joints[:, 1],
            joints[:, 4] - 0.89 * joints[:, 3],
            joints[:, 6] - 0.89 * joints[:, 5],
            joints[:, 8] - 0.89 * joints[:, 7],
            joints[:, 10] - 0.89 * joints[:, 9],
        ],
        axis=1,
    )


def main():
    """运行单条Linker候选轨迹的独立几何评估。

    输入：命令行源/目标文件、双方索引、Z偏移和报告输出路径。
    输出：写入JSON报告，并在终端打印均值、最大值和最差点。
    逻辑：校验数据、恢复双方物理点，统计距离与相邻帧动作变化。
    作用：作为evaluate分区中Linker数学重定向质量的标准入口。
    """
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
    source_frames = np.asarray(
        source_data["grasp_seqs"][args.source_index], dtype=np.float32
    ).copy()
    target_frames = np.asarray(
        target_data["grasp_seqs"][args.target_index], dtype=np.float32
    )
    if source_frames.shape[0] != target_frames.shape[0]:
        raise ValueError(
            f"源和目标帧数不同: {source_frames.shape[0]} vs {target_frames.shape[0]}"
        )
    if target_frames.shape[1] not in {12, 17}:
        raise ValueError(
            f"Linker轨迹应为12维O6或17维解耦模式，实际为{target_frames.shape[1]}"
        )
    inferred_mode = "coupled6" if target_frames.shape[1] == 12 else "independent11"
    joint_mode = str(target_data.get("joint_mode", inferred_mode))
    if joint_mode != inferred_mode:
        raise ValueError(
            f"候选元数据joint_mode={joint_mode}与维度{target_frames.shape[1]}不一致"
        )
    source_z_offset = (
        float(target_data.get("source_z_offset", 0.4))
        if args.source_z_offset is None
        else args.source_z_offset
    )
    source_frames[:, 2] += source_z_offset
    semantics = list(target_data.get("mapping_semantics", []))
    if not semantics:
        raise ValueError("候选轨迹没有mapping_semantics，无法确定评估口径")
    pairs = load_pairs(semantics)

    shadow, linker = build_models(joint_mode)
    with torch.no_grad():
        shadow.set_parameters(shadow_to_model_q(source_frames))
        all_shadow_points = shadow.get_penetraion_keypoints().cpu().numpy()
    source_indices = [pair["shadow_index"] for pair in pairs]
    source_points = all_shadow_points[:, source_indices, :]
    target_points = compute_linker_points(
        linker, linker_to_model_q(target_frames), pairs
    )
    distances = np.linalg.norm(target_points - source_points, axis=-1)

    joint_delta = np.diff(target_frames[:, 6:], axis=0)
    wrist_delta = np.diff(target_frames[:, :3], axis=0)
    wrist_rotation_delta = rotation_step_angles(target_frames[:, 3:6])
    worst_frame, worst_pair = np.unravel_index(distances.argmax(), distances.shape)
    report = {
        "source": str(args.source.resolve()),
        "target": str(args.target.resolve()),
        "source_trajectory_index": args.source_index,
        "target_trajectory_index": args.target_index,
        "frames": int(len(source_frames)),
        "pair_count": len(pairs),
        "joint_mode": joint_mode,
        "finger_joint_count": int(target_frames.shape[1] - 6),
        "source_z_offset_m": source_z_offset,
        "keypoint_mean_distance_m": float(distances.mean()),
        "keypoint_max_distance_m": float(distances.max()),
        "worst_frame": int(worst_frame),
        "worst_semantic": semantics[int(worst_pair)],
        "mean_joint_step_l2_rad": float(np.linalg.norm(joint_delta, axis=1).mean()),
        "max_joint_step_l2_rad": float(np.linalg.norm(joint_delta, axis=1).max()),
        "mean_wrist_translation_step_m": float(
            np.linalg.norm(wrist_delta, axis=1).mean()
        ),
        "max_wrist_translation_step_m": float(
            np.linalg.norm(wrist_delta, axis=1).max()
        ),
        "mean_wrist_rotation_step_rad": float(wrist_rotation_delta.mean()),
        "max_wrist_rotation_step_rad": float(wrist_rotation_delta.max()),
        "per_semantic": {
            name: {
                "mean_distance_m": float(distances[:, index].mean()),
                "max_distance_m": float(distances[:, index].max()),
            }
            for index, name in enumerate(semantics)
        },
        "per_frame_mean_distance_m": distances.mean(axis=1).tolist(),
    }
    if joint_mode == "independent11":
        residuals = coupling_residuals(target_frames[:, 6:])
        report["coupling_residual_mean_abs_rad"] = float(
            np.mean(np.abs(residuals))
        )
        report["coupling_residual_max_abs_rad"] = float(
            np.max(np.abs(residuals))
        )
        report["coupling_residual_per_joint_mean_abs_rad"] = {
            name: float(np.mean(np.abs(residuals[:, index])))
            for index, name in enumerate(
                ["thumb_ip", "index_dip", "middle_dip", "ring_dip", "pinky_dip"]
            )
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"frames={report['frames']}")
    print(f"pair_count={report['pair_count']}")
    print(f"keypoint_mean_distance_m={report['keypoint_mean_distance_m']:.6f}")
    print(f"keypoint_max_distance_m={report['keypoint_max_distance_m']:.6f}")
    print(f"worst={report['worst_semantic']}@frame{report['worst_frame']}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
