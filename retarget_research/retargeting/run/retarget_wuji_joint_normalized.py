#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import transforms3d

RETARGET_ROOT = Path(__file__).resolve().parents[1]


METHOD = "wuji_joint_normalized_v1"
R_ALIGN = np.array(
    [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32
)

# GraspM3 22关节顺序：FF(4)、MF(4)、RF(4)、LF(5)、TH(5)。
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
# Wuji 20关节：finger1拇指、finger2-5普通指，均为4关节。
WUJI_FINGER_FLEX_UPPER = np.array([1.636, 1.627, 1.627], dtype=np.float32)
WUJI_FINGER_ABD_RANGE = np.array([-0.495, 0.495], dtype=np.float32)
WUJI_THUMB_UPPER = np.array([1.651, 0.934, 1.627, 1.627], dtype=np.float32)
WUJI_THUMB_LOWER = np.array([-0.045, -0.166, -0.493, -0.493], dtype=np.float32)
MAPPING_SEMANTICS = None


def load_semantics():
    config = json.loads(
        (RETARGET_ROOT / "configs" / "wuji_keypoint_map.json").read_text(encoding="utf-8"))
    return [pair["semantic"] for pair in config["pairs"]]


def normalize(values, lower, upper):
    ratio = (np.asarray(values, dtype=np.float32) - lower) / (upper - lower)
    return np.clip(ratio, 0.0, 1.0)


def map_finger(shadow_finger, wuji_abd_range, wuji_flex_upper, little=False):
    if little:
        abd = max(normalize(shadow_finger[0], 0.0, 0.785),
                  normalize(shadow_finger[1], -0.349, 0.349))
        flex = shadow_finger[2:5]
    else:
        abd = normalize(shadow_finger[0], -0.349, 0.349)
        flex = shadow_finger[1:4]
    flexion = max(normalize(value, 0.0, 1.571) for value in flex)
    joints = np.empty(4, dtype=np.float32)
    joints[0] = wuji_flex_upper[0] * flexion
    joints[1] = wuji_abd_range[0] + 2.0 * abd * wuji_abd_range[1]
    joints[2] = wuji_flex_upper[1] * flexion
    joints[3] = wuji_flex_upper[2] * flexion
    return joints


def map_thumb(shadow_thumb):
    flexion = max(normalize(shadow_thumb[1], 0.0, 1.222),
                  normalize(-shadow_thumb[4], 0.0, 1.571))
    joints = np.empty(4, dtype=np.float32)
    joints[0] = WUJI_THUMB_UPPER[0] * flexion
    joints[1] = WUJI_THUMB_LOWER[1] + (
        WUJI_THUMB_UPPER[1] - WUJI_THUMB_LOWER[1]
    ) * normalize(shadow_thumb[2], -0.209, 0.209)
    joints[2] = WUJI_THUMB_UPPER[2] * flexion
    joints[3] = WUJI_THUMB_UPPER[3] * flexion
    return joints


def map_joints(shadow_joints):
    shadow_joints = np.asarray(shadow_joints, dtype=np.float32)
    joints = np.empty(20, dtype=np.float32)
    joints[0:4] = map_thumb(shadow_joints[17:22])
    joints[4:8] = map_finger(shadow_joints[0:4], WUJI_FINGER_ABD_RANGE, WUJI_FINGER_FLEX_UPPER)
    joints[8:12] = map_finger(shadow_joints[4:8], WUJI_FINGER_ABD_RANGE, WUJI_FINGER_FLEX_UPPER)
    joints[12:16] = map_finger(shadow_joints[8:12], WUJI_FINGER_ABD_RANGE, WUJI_FINGER_FLEX_UPPER)
    joints[16:20] = map_finger(shadow_joints[12:17], WUJI_FINGER_ABD_RANGE,
                               WUJI_FINGER_FLEX_UPPER, little=True)
    return joints


def retarget_trajectory(source_frames, source_z_offset=0.4):
    source_frames = np.asarray(source_frames, dtype=np.float32).copy()
    source_frames[:, 2] += float(source_z_offset)
    output = np.empty((len(source_frames), 26), dtype=np.float32)
    for frame_index, frame in enumerate(source_frames):
        rotation = transforms3d.euler.euler2mat(*frame[3:6], axes="sxyz")
        euler = transforms3d.euler.mat2euler(R_ALIGN @ rotation, axes="sxyz")
        output[frame_index, :3] = frame[:3]
        output[frame_index, 3:6] = np.asarray(euler, dtype=np.float32)
        output[frame_index, 6:] = map_joints(frame[6:])
    return output


def retarget_file(args):
    data = np.load(args.source, allow_pickle=True).item()
    indices = np.asarray(args.trajectory_indices or [0], dtype=np.int64)
    outputs, losses = [], []
    for source_index in indices:
        frames = retarget_trajectory(
            data["grasp_seqs"][source_index], args.source_z_offset)
        outputs.append(frames)
        losses.append(np.zeros(len(frames), dtype=np.float32))
    output_frames = np.stack(outputs).astype(np.float32)
    output = {
        "grasp_seqs": output_frames,
        "optimization_loss_per_frame": np.stack(losses).astype(np.float32),
        "source_trajectory_indices": indices,
        "obj_rotmat": np.asarray(data["obj_rotmat"])[indices],
        "obj_scale": np.asarray(data["obj_scale"])[indices],
        "retarget_method": METHOD,
        "wuji_joint_names": ["finger1_joint1", "finger1_joint2", "finger1_joint3",
                             "finger1_joint4", "finger2_joint1", "finger2_joint2",
                             "finger2_joint3", "finger2_joint4", "finger3_joint1",
                             "finger3_joint2", "finger3_joint3", "finger3_joint4",
                             "finger4_joint1", "finger4_joint2", "finger4_joint3",
                             "finger4_joint4", "finger5_joint1", "finger5_joint2",
                             "finger5_joint3", "finger5_joint4"],
        "mapping_semantics": load_semantics(),
        "source_z_offset": float(args.source_z_offset),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, output, allow_pickle=True)
    print(f"trajectories={len(output_frames)}")
    print(f"output_shape={output_frames.shape}")
    print(f"output={args.output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trajectory-indices", type=int, nargs="*")
    parser.add_argument("--source-z-offset", type=float, default=0.4)
    retarget_file(parser.parse_args())


if __name__ == "__main__":
    main()
