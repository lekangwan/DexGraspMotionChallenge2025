#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import transforms3d


METHOD = "linker_joint_normalized_v1"
R_ALIGN = np.array(
    [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32
)
WRIST_OFFSET = np.array([0.003, 0.002, -0.01], dtype=np.float32)

# GraspM3 的22个关节顺序：FF(4)、MF(4)、RF(4)、LF(5)、TH(5)。
SOURCE_LOWER = np.array(
    [-0.349, 0.0, 0.0, 0.0] * 3
    + [0.0, -0.349, 0.0, 0.0, 0.0]
    + [-1.047, 0.0, -0.209, -0.524, -1.571],
    dtype=np.float32,
)
SOURCE_UPPER = np.array(
    [0.349, 1.571, 1.571, 1.571] * 3
    + [0.785, 0.349, 1.571, 1.571, 1.571]
    + [1.047, 1.222, 0.209, 0.524, 0.0],
    dtype=np.float32,
)
TARGET_UPPER = np.array([1.36, 0.58, 1.6, 1.6, 1.6, 1.6], dtype=np.float32)
MAPPING_SEMANTICS = [
    "palm",
    "index_proximal_end", "index_tip",
    "middle_proximal_end", "middle_tip",
    "ring_proximal_end", "ring_tip",
    "little_proximal_end", "little_tip",
    "thumb_tip",
]


def normalize(values):
    """把Shadow关节按其物理范围映射到[0,1]。"""
    ratio = (values - SOURCE_LOWER) / (SOURCE_UPPER - SOURCE_LOWER)
    return np.clip(ratio, 0.0, 1.0)


def map_joints(shadow_joints, flex_mode="mean"):
    """将22维Shadow关节压缩为Linker的6个主动关节。"""
    normalized = normalize(np.asarray(shadow_joints, dtype=np.float32))
    # 普通手指忽略侧摆，只保留三个屈曲关节的平均闭合程度。
    aggregate = np.max if flex_mode == "max" else np.mean
    index = aggregate(normalized[1:4])
    middle = aggregate(normalized[5:8])
    ring = aggregate(normalized[9:12])
    little = aggregate(normalized[14:17])
    thumb_yaw = normalized[17]
    thumb_pitch = aggregate(normalized[[18, 21]]) if flex_mode == "max" else normalized[18]
    return TARGET_UPPER * np.array(
        [thumb_yaw, thumb_pitch, index, middle, ring, little], dtype=np.float32
    )


def retarget_trajectory(source_frames, source_z_offset=0.4, flex_mode="mean"):
    """输入(T,28) Shadow轨迹，输出(T,12) Linker轨迹。"""
    source_frames = np.asarray(source_frames, dtype=np.float32).copy()
    source_frames[:, 2] += float(source_z_offset)
    output = np.empty((len(source_frames), 12), dtype=np.float32)
    for frame_index, frame in enumerate(source_frames):
        rotation = transforms3d.euler.euler2mat(*frame[3:6], axes="sxyz")
        euler = transforms3d.euler.mat2euler(R_ALIGN @ rotation, axes="sxyz")
        output[frame_index, :3] = frame[:3] + WRIST_OFFSET
        output[frame_index, 3:6] = euler
        output[frame_index, 6:] = map_joints(frame[6:], flex_mode)
    return output


def retarget_file(source, output, trajectory_indices, source_z_offset=0.4, flex_mode="mean"):
    """读取源npy指定轨迹并保存标准Linker候选文件。"""
    data = np.load(source, allow_pickle=True).item()
    indices = np.asarray(trajectory_indices, dtype=np.int64)
    source_frames = np.asarray(data["grasp_seqs"])[indices]
    trajectories = np.stack(
        [retarget_trajectory(frames, source_z_offset, flex_mode) for frames in source_frames]
    )
    result = {
        "grasp_seqs": trajectories,
        "source_trajectory_indices": indices,
        "obj_rotmat": np.asarray(data["obj_rotmat"])[indices],
        "obj_scale": np.asarray(data["obj_scale"])[indices],
        "retarget_method": METHOD,
        "flex_mode": flex_mode,
        "joint_mode": "coupled6",
        "finger_joint_count": 6,
        "target_dimension": 12,
        "source_z_offset": float(source_z_offset),
        "wrist_alignment": R_ALIGN,
        "wrist_offset": WRIST_OFFSET,
        "source_joint_lower": SOURCE_LOWER,
        "source_joint_upper": SOURCE_UPPER,
        "target_joint_upper": TARGET_UPPER,
        "mapping_semantics": MAPPING_SEMANTICS,
        "optimization_loss_per_frame": np.zeros(trajectories.shape[:2], dtype=np.float32),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, result, allow_pickle=True)
    print(f"output_shape={trajectories.shape}")
    print(f"output={output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trajectory-indices", type=int, nargs="+", required=True)
    parser.add_argument("--source-z-offset", type=float, default=0.4)
    parser.add_argument("--flex-mode", choices=("mean", "max"), default="mean")
    args = parser.parse_args()
    retarget_file(args.source, args.output, args.trajectory_indices, args.source_z_offset, args.flex_mode)


if __name__ == "__main__":
    main()
