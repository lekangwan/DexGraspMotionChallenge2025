#!/usr/bin/env python3
"""独立比较Shadow源轨迹与Wuji 20自由度候选的15对语义关键点。

输入：Shadow原始`.npy`、Wuji候选`.npy`及双方轨迹索引。
输出：整体/逐语义几何误差和关节、手腕相邻帧变化JSON。
内部逻辑：从保存动作重新做双方正向运动学，不读取优化器保存的loss。
作用：验证26维候选确实复现了源几何，并量化高自由度轨迹的连续性。
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

from utils.HandModel_xhand import HandModel_xhand  # noqa: E402
from utils.hand_model import HandModel as ShadowHandModel  # noqa: E402
from utils.rot6d import robust_compute_orth6d_from_eulerXYZ  # noqa: E402
from wuji_candidate_utils import trajectory_mapping_metadata  # noqa: E402


def build_models():
    """创建独立评估所需的Shadow和Wuji CPU模型。

    输入：只读参考资产中的Shadow MJCF与Wuji右手URDF/关键点。
    输出：21点Shadow模型和26点、20关节Wuji模型。
    内部逻辑：按各自原生解析器加载，不创建或调用优化目标。
    作用：让几何报告由保存动作重新计算，而非相信生成阶段的loss。
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
    wuji_asset = assets / "wujihand_urdf" / "urdf"
    wuji = HandModel_xhand(
        robot_name="wuji_right",
        urdf_filename="right.urdf",
        mesh_path="../",
        batch_size=1,
        device=torch.device("cpu"),
        mesh_nsp=128,
        hand_scale=1.0,
        asset_dir=str(wuji_asset),
        allow_missing_contacts=True,
    )
    return shadow, wuji


def load_pairs(semantics, mapping_config):
    """按候选文件声明的顺序读取Wuji语义映射。

    输入：候选npy中的语义名称列表及生成时使用的映射JSON。
    输出：同顺序的配置pair列表。
    内部逻辑：用semantic建立查找表，遇到未知名称立即报错。
    作用：防止评估阶段使用与生成阶段不同的数组索引。
    """
    config = json.loads(
        Path(mapping_config).read_text()
    )
    by_name = {pair["semantic"]: pair for pair in config["pairs"]}
    unknown = [name for name in semantics if name not in by_name]
    if unknown:
        raise ValueError(f"未知Wuji语义点: {unknown}")
    return [by_name[name] for name in semantics]


def shadow_to_model_q(frames):
    """把Shadow保存的28维帧转换为31维模型参数。

    输入：`(T,28)`平移、欧拉角和22关节。
    输出：`(T,31)`平移、旋转6D和22关节。
    内部逻辑：仅转换手腕旋转表示，其余数值保持不变。
    作用：批量恢复Shadow的21个世界关键点。
    """
    frames = torch.as_tensor(frames, dtype=torch.float32)
    q = torch.zeros((len(frames), 31), dtype=torch.float32)
    q[:, :3] = frames[:, :3]
    q[:, 3:9] = robust_compute_orth6d_from_eulerXYZ(frames[:, 3:6])
    q[:, 9:] = frames[:, 6:]
    return q


def wuji_to_model_q(frames):
    """把Wuji保存的26维帧转换为29维模型参数。

    输入：`(T,26)`，顺序为平移3、欧拉角3、关节20。
    输出：`(T,29)`，顺序为平移3、旋转6D、关节20。
    内部逻辑：将欧拉角改写为模型使用的连续旋转6D。
    作用：批量恢复Wuji 26个世界关键点。
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


def rotation_step_angles(euler_xyz):
    """计算相邻Wuji手腕之间的SO(3)真实旋转角。

    输入：`(T,3)`静态XYZ欧拉角。
    输出：`(T-1,)`弧度制相邻旋转角。
    内部逻辑：转换旋转矩阵后，由相对矩阵trace求测地角。
    作用：避免欧拉角在正负π换支时产生虚假跳变。
    """
    matrices = np.stack(
        [
            transforms3d.euler.euler2mat(*frame, axes="sxyz")
            for frame in np.asarray(euler_xyz)
        ]
    )
    relative = np.einsum("tji,tjk->tik", matrices[:-1], matrices[1:])
    cosine = np.clip(
        (np.trace(relative, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0
    )
    return np.arccos(cosine)


def main():
    """运行单条Wuji候选的独立几何与连续性评估。

    输入：命令行源/目标文件、双方索引、可选源Z偏移和输出JSON。
    输出：完整报告及终端均值、最大值、最差语义摘要。
    内部逻辑：校验26维和帧数，恢复双方15对点，再统计误差与动作变化。
    作用：作为Wuji通过运动学门G2前的标准证据入口。
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
        raise ValueError("Shadow和Wuji轨迹帧数不同")
    if target_frames.shape[1] != 26:
        raise ValueError(f"Wuji轨迹应为26维，实际为{target_frames.shape[1]}")
    source_z_offset = (
        float(target_data.get("source_z_offset", 0.4))
        if args.source_z_offset is None
        else args.source_z_offset
    )
    source_frames[:, 2] += source_z_offset
    mapping_config, semantics = trajectory_mapping_metadata(
        target_data,
        args.target_index,
        RETARGET_ROOT / "configs" / "wuji_keypoint_map.json",
    )
    if not semantics:
        raise ValueError("Wuji候选没有mapping_semantics")
    pairs = load_pairs(semantics, mapping_config)
    shadow, wuji = build_models()
    with torch.no_grad():
        shadow.set_parameters(shadow_to_model_q(source_frames))
        source_all = shadow.get_penetraion_keypoints()
        target_all = wuji.get_penetraion_keypoints(q=wuji_to_model_q(target_frames))
    source_indices = [pair["shadow_index"] for pair in pairs]
    target_indices = [pair["wuji_index"] for pair in pairs]
    source_points = source_all[:, source_indices].numpy()
    target_points = target_all[:, target_indices].numpy()
    distances = np.linalg.norm(target_points - source_points, axis=-1)
    joint_delta = np.diff(target_frames[:, 6:], axis=0)
    wrist_delta = np.diff(target_frames[:, :3], axis=0)
    rotation_delta = rotation_step_angles(target_frames[:, 3:6])
    worst_frame, worst_pair = np.unravel_index(distances.argmax(), distances.shape)
    report = {
        "source": str(args.source.resolve()),
        "target": str(args.target.resolve()),
        "source_trajectory_index": args.source_index,
        "target_trajectory_index": args.target_index,
        "frames": int(len(source_frames)),
        "pair_count": len(pairs),
        "source_z_offset_m": source_z_offset,
        "mapping_config": str(mapping_config.resolve()),
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
        "mean_wrist_rotation_step_rad": float(rotation_delta.mean()),
        "max_wrist_rotation_step_rad": float(rotation_delta.max()),
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
    print(f"pair_count={report['pair_count']}")
    print(f"keypoint_mean_distance_m={report['keypoint_mean_distance_m']:.6f}")
    print(f"keypoint_max_distance_m={report['keypoint_max_distance_m']:.6f}")
    print(f"worst={report['worst_semantic']}@frame{report['worst_frame']}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
