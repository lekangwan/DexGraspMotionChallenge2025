#!/usr/bin/env python3
"""比较一条Shadow源轨迹与XHand候选轨迹的15对语义关键点。

输入：Shadow原始npy、XHand候选npy、轨迹索引和源轨迹Z偏移。
输出：70帧整体/逐语义关键点误差及相邻帧变化JSON。
内部逻辑：双方分别做正向运动学，再按语义配置计算世界坐标距离。
作用：独立评价数学重定向质量，不使用也不替代Isaac Gym物理成功率。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch


RETARGET_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = RETARGET_ROOT.parent
REFERENCE_SCRIPTS = (
    PROJECT_ROOT / "reference" / "HandRetargetTask2026" / "scripts"
)
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
from utils.HandModel_xhand import HandModel_xhand as XHandModel  # noqa: E402
from utils.rot6d import robust_compute_orth6d_from_eulerXYZ  # noqa: E402


def build_models():
    """创建评测所需的Shadow和XHand CPU运动学模型。

    输入：无显式参数；从固定参考资产目录读取模型文件。
    输出：Shadow模型和XHand模型。
    逻辑：使用与参考优化器一致的模型类、资产和`use_joint21`设置。
    作用：保证独立评测与被评测的参考基线使用相同几何定义。
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
    xhand_asset = assets / "xhand_right" / "urdf"
    xhand = XHandModel(
        robot_name="xhand",
        urdf_filename="xhand_right.urdf",
        mesh_path="",
        batch_size=1,
        device=torch.device("cpu"),
        mesh_nsp=128,
        hand_scale=1.0,
        asset_dir=str(xhand_asset),
        allow_missing_contacts=True,
    )
    return shadow, xhand


def shadow_to_model_q(frames):
    """把Shadow保存格式转换成运动学模型格式。

    输入：形状`(T,28)`，内容为平移3、欧拉角3和关节22。
    输出：形状`(T,31)`，把欧拉角转换成连续的旋转6D表示。
    逻辑：保留平移和关节，调用参考旋转函数生成6维旋转。
    作用：让Shadow轨迹可以批量进入正向运动学模型。
    """
    frames = torch.as_tensor(frames, dtype=torch.float32)
    result = torch.zeros((len(frames), 31), dtype=torch.float32)
    result[:, :3] = frames[:, :3]
    result[:, 3:9] = robust_compute_orth6d_from_eulerXYZ(frames[:, 3:6])
    result[:, 9:] = frames[:, 6:]
    return result


def xhand_to_model_q(frames):
    """把XHand保存格式转换成运动学模型格式。

    输入：形状`(T,18)`，内容为平移3、欧拉角3和XHand关节12。
    输出：形状`(T,21)`，内容为平移3、旋转6D和关节12。
    逻辑：转换欧拉角后按XHand模型要求重新拼接。
    作用：让候选轨迹可以批量计算XHand世界坐标关键点。
    """
    frames = torch.as_tensor(frames, dtype=torch.float32)
    translation = frames[:, :3]
    rotation = robust_compute_orth6d_from_eulerXYZ(frames[:, 3:6])
    joints = frames[:, 6:]
    return torch.cat([translation, rotation, joints], dim=1)


def main():
    """读取一对源/目标轨迹并写出几何评测报告。

    输入：命令行源文件、目标文件、双方轨迹索引、Z偏移和输出路径。
    输出：无Python返回值；写JSON并在终端打印核心指标。
    逻辑：校验维度，做双方正向运动学，统计15对点误差和帧间变化。
    作用：作为evaluate阶段的XHand单轨迹几何评测入口。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--source-index", type=int, default=0)
    parser.add_argument("--target-index", type=int, default=0)
    parser.add_argument("--source-z-offset", type=float, default=0.4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source_data = np.load(args.source, allow_pickle=True).item()
    target_data = np.load(args.target, allow_pickle=True).item()
    source_frames = np.asarray(
        source_data["grasp_seqs"][args.source_index], dtype=np.float32
    ).copy()
    source_frames[:, 2] += args.source_z_offset
    target_frames = np.asarray(
        target_data["grasp_seqs"][args.target_index], dtype=np.float32
    )
    if source_frames.shape[0] != target_frames.shape[0]:
        raise ValueError(
            f"源和目标帧数不同: {source_frames.shape[0]} vs {target_frames.shape[0]}"
        )
    if target_frames.shape[1] != 18:
        raise ValueError(f"XHand轨迹应为18维，实际为{target_frames.shape[1]}")

    mapping = json.loads(
        (RETARGET_ROOT / "configs" / "xhand_keypoint_map.json").read_text()
    )
    shadow, xhand = build_models()
    with torch.no_grad():
        shadow.set_parameters(shadow_to_model_q(source_frames))
        source_points = shadow.get_penetraion_keypoints().cpu().numpy()
        target_points = (
            xhand.get_penetraion_keypoints(q=xhand_to_model_q(target_frames))
            .cpu()
            .numpy()
        )

    source_indices = [pair["shadow_index"] for pair in mapping["pairs"]]
    target_indices = [pair["xhand_index"] for pair in mapping["pairs"]]
    semantics = [pair["semantic"] for pair in mapping["pairs"]]
    difference = (
        target_points[:, target_indices, :] - source_points[:, source_indices, :]
    )
    distances = np.linalg.norm(difference, axis=-1)

    joint_delta = np.diff(target_frames[:, 6:], axis=0)
    wrist_delta = np.diff(target_frames[:, :3], axis=0)
    report = {
        "source": str(args.source.resolve()),
        "target": str(args.target.resolve()),
        "frames": int(len(source_frames)),
        "pair_count": len(semantics),
        "source_z_offset_m": args.source_z_offset,
        "keypoint_mean_distance_m": float(distances.mean()),
        "keypoint_max_distance_m": float(distances.max()),
        "worst_frame": int(np.unravel_index(distances.argmax(), distances.shape)[0]),
        "worst_semantic": semantics[
            int(np.unravel_index(distances.argmax(), distances.shape)[1])
        ],
        "mean_joint_step_l2_rad": float(np.linalg.norm(joint_delta, axis=1).mean()),
        "max_joint_step_l2_rad": float(np.linalg.norm(joint_delta, axis=1).max()),
        "mean_wrist_translation_step_m": float(
            np.linalg.norm(wrist_delta, axis=1).mean()
        ),
        "per_semantic": {
            name: {
                "mean_distance_m": float(distances[:, index].mean()),
                "max_distance_m": float(distances[:, index].max()),
            }
            for index, name in enumerate(semantics)
        },
        "per_frame_mean_distance_m": distances.mean(axis=1).tolist(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"frames={report['frames']}")
    print(f"keypoint_mean_distance_m={report['keypoint_mean_distance_m']:.6f}")
    print(f"keypoint_max_distance_m={report['keypoint_max_distance_m']:.6f}")
    print(f"worst={report['worst_semantic']}@frame{report['worst_frame']}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
