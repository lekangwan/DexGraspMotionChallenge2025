#!/usr/bin/env python3
"""在Isaac Gym场景中逐帧回放正式评测保存的状态并录制视频。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from isaacgym import gymapi
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVALUATE_DIR = PROJECT_ROOT / "retarget_research/retargeting/evaluate"
sys.path.insert(0, str(EVALUATE_DIR))

from isaac_replay_common import (  # noqa: E402
    IsaacCameraRecorder,
    create_cpu_sim,
    load_object_asset,
    object_start_pose,
    set_dof_state_and_target,
)
from replay_linker_isaac import load_linker_asset  # noqa: E402
from replay_wuji_isaac import load_wuji_asset  # noqa: E402
from replay_xhand_isaac import load_xhand_asset  # noqa: E402


def load_hand(gym, sim, hand, target_data):
    if hand == "linker":
        return load_linker_asset(gym, sim)
    if hand == "xhand":
        return load_xhand_asset(gym, sim)
    optimizer_names = list(target_data.get("wuji_joint_names", []))
    if len(optimizer_names) != 20:
        raise ValueError("Wuji候选文件缺少20个wuji_joint_names")
    return load_wuji_asset(gym, sim, optimizer_names)


def set_object_pose(gym, env, actor, position, quaternion):
    states = gym.get_actor_rigid_body_states(env, actor, gymapi.STATE_ALL)
    states["pose"]["p"][0] = tuple(np.asarray(position, dtype=np.float32))
    states["pose"]["r"][0] = tuple(np.asarray(quaternion, dtype=np.float32))
    states["vel"]["linear"][:] = 0.0
    states["vel"]["angular"][:] = 0.0
    gym.set_actor_rigid_body_states(env, actor, states, gymapi.STATE_ALL)


def render(args):
    if args.state.suffix == ".npz":
        with np.load(args.state, allow_pickle=False) as archive:
            report = json.loads(str(archive["metadata_json"].item()))
            dof_positions = np.asarray(archive["hand_dof_position"], dtype=np.float32)
            object_positions = np.asarray(archive["object_position"], dtype=np.float32)
            object_quaternions = np.asarray(
                archive["object_quaternion_xyzw"], dtype=np.float32
            )
        report["success"] = None
    else:
        report = json.loads(args.state.read_text(encoding="utf-8"))
        required = {
            "actual_hand_dof_positions", "object_positions_m",
            "object_quaternions_xyzw",
        }
        missing = required - set(report)
        if missing:
            raise ValueError(f"状态报告缺少字段: {sorted(missing)}")
        dof_positions = np.asarray(
            report["actual_hand_dof_positions"], dtype=np.float32
        )
        object_positions = np.asarray(report["object_positions_m"], dtype=np.float32)
        object_quaternions = np.asarray(
            report["object_quaternions_xyzw"], dtype=np.float32
        )
    required = {"hand", "source", "target", "source_trajectory_index", "physics_dof_names"}
    missing = required - set(report)
    if missing:
        raise ValueError(f"状态元数据缺少字段: {sorted(missing)}")

    source_data = np.load(report["source"], allow_pickle=True).item()
    target_data = np.load(report["target"], allow_pickle=True).item()
    source_index = int(report["source_trajectory_index"])
    scale = float(np.asarray(source_data["obj_scale"])[source_index])
    rotation = np.asarray(source_data["obj_rotmat"])[source_index]
    object_dir = args.object_dir or Path(report["object_dir"])
    frame_count = min(len(dof_positions), len(object_positions), len(object_quaternions))

    gym, sim = create_cpu_sim(args.dt, args.substeps, enable_graphics=True)
    recorder = None
    try:
        hand_asset, dof_properties, dof_names = load_hand(
            gym, sim, report["hand"], target_data
        )
        if list(report["physics_dof_names"]) != list(dof_names):
            raise ValueError("保存状态的DOF顺序与当前URDF不一致")
        object_asset = load_object_asset(gym, sim, object_dir)
        env = gym.create_env(
            sim, gymapi.Vec3(-1.0, -1.0, -0.2), gymapi.Vec3(1.0, 1.0, 1.0), 1
        )
        hand_actor = gym.create_actor(
            env, hand_asset, gymapi.Transform(), report["hand"], 0, 1
        )
        gym.set_actor_dof_properties(env, hand_actor, dof_properties)
        object_pose, _ = object_start_pose(object_dir, scale, rotation, args.clearance)
        object_actor = gym.create_actor(env, object_asset, object_pose, "object", 0, 0)
        gym.set_actor_scale(env, object_actor, scale)
        recorder = IsaacCameraRecorder(
            gym, sim, env, args.output, args.width, args.height,
            args.fps, args.capture_every,
        )
        for step in range(frame_count):
            set_dof_state_and_target(gym, env, hand_actor, dof_positions[step])
            set_object_pose(
                gym, env, object_actor, object_positions[step], object_quaternions[step]
            )
            recorder.capture(step)
        video = recorder.close()
        recorder = None
    finally:
        if recorder is not None:
            recorder.close()
        gym.destroy_sim(sim)

    metadata = {
        "renderer": "isaac_state_replay",
        "state": str(args.state.resolve()),
        "video": video["video"],
        "video_frame_count": video["video_frame_count"],
        "source_state_count": frame_count,
        "success": report.get("success"),
        "object_name": report.get("object_name"),
        "source_trajectory_index": source_index,
    }
    sidecar = args.output.with_suffix(".json")
    sidecar.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--object-dir", type=Path)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--capture-every", type=int, default=3)
    parser.add_argument("--dt", type=float, default=1.0 / 60.0)
    parser.add_argument("--substeps", type=int, default=2)
    parser.add_argument("--clearance", type=float, default=0.002)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    render(args)


if __name__ == "__main__":
    main()
