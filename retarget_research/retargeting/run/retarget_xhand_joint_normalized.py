#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import transforms3d

RETARGET_ROOT = Path(__file__).resolve().parents[1]


METHOD = "xhand_joint_normalized_v1"
R_ALIGN = np.array(
    [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32
)

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


def normalize(values, lower, upper):
    ratio = (np.asarray(values, dtype=np.float32) - lower) / (upper - lower)
    return np.clip(ratio, 0.0, 1.0)


def map_finger(shadow_finger, with_abd, little=False):
    if little:
        abd = max(normalize(shadow_finger[0], 0.0, 0.785),
                  normalize(shadow_finger[1], -0.349, 0.349))
        flex = shadow_finger[2:5]
    else:
        abd = normalize(shadow_finger[0], -0.349, 0.349)
        flex = shadow_finger[1:4]
    flexion = max(normalize(value, 0.0, 1.571) for value in flex)
    if with_abd:
        joints = np.array([1.919 * flexion,
                           (0.348 / 2.0) * (2.0 * abd - 1.0),
                           1.919 * flexion], dtype=np.float32)
    else:
        joints = np.array([1.919 * flexion, 1.919 * flexion], dtype=np.float32)
    return joints


def map_thumb(shadow_thumb):
    flexion = max(normalize(shadow_thumb[1], 0.0, 1.222),
                  normalize(-shadow_thumb[4], 0.0, 1.571))
    joints = np.array([
        1.832 * flexion,
        -0.698 + 2.268 * normalize(shadow_thumb[0], -1.047, 1.047),
        1.570 * normalize(shadow_thumb[2], -0.209, 0.209),
    ], dtype=np.float32)
    return joints


def map_joints(shadow_joints):
    shadow_joints = np.asarray(shadow_joints, dtype=np.float32)
    joints = np.empty(12, dtype=np.float32)
    joints[0:3] = map_thumb(shadow_joints[17:22])
    joints[3:6] = map_finger(shadow_joints[0:4], with_abd=True)
    joints[6:8] = map_finger(shadow_joints[4:8], with_abd=False)
    joints[8:10] = map_finger(shadow_joints[8:12], with_abd=False)
    joints[10:12] = map_finger(shadow_joints[12:17], with_abd=False, little=True)
    return joints


def retarget_file(args):
    data = np.load(args.source, allow_pickle=True).item()
    indices = np.asarray(args.trajectory_indices or [0], dtype=np.int64)
    outputs = []
    for source_index in indices:
        source_frames = np.asarray(
            data["grasp_seqs"][source_index], dtype=np.float32).copy()
        source_frames[:, 2] += args.source_z_offset
        output = np.empty((len(source_frames), 18), dtype=np.float32)
        for frame_index, frame in enumerate(source_frames):
            rotation = transforms3d.euler.euler2mat(*frame[3:6], axes="sxyz")
            euler = transforms3d.euler.mat2euler(R_ALIGN @ rotation, axes="sxyz")
            output[frame_index, :3] = frame[:3]
            output[frame_index, 3:6] = np.asarray(euler, dtype=np.float32)
            output[frame_index, 6:] = map_joints(frame[6:])
        outputs.append(output)
    output_frames = np.stack(outputs).astype(np.float32)
    output = {
        "grasp_seqs": output_frames,
        "optimization_loss_per_frame": np.zeros(
            (len(indices), len(outputs[0])), dtype=np.float32),
        "source_trajectory_indices": indices,
        "obj_rotmat": np.asarray(data["obj_rotmat"])[indices],
        "obj_scale": np.asarray(data["obj_scale"])[indices],
        "retarget_method": METHOD,
        "mapping_semantics": [
            pair["semantic"]
            for pair in json.loads(
                (RETARGET_ROOT / "configs" / "xhand_keypoint_map.json")
                .read_text(encoding="utf-8"))["pairs"]
        ],
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
