#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--online-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    online_dir = args.online_dir.expanduser().resolve()
    data_dir = args.data_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with np.load(data_dir / "train.npz", allow_pickle=False) as archive:
        offline = {name: archive[name] for name in archive.files}
    n_offline = len(offline["observations"])
    mappings = json.loads((data_dir / "mappings.json").read_text(encoding="utf-8"))
    category_to_id = mappings["category_to_id"]

    rows_obs, rows_actions, rows_prev, rows_cat, rows_traj, rows_src, rows_frame = \
        [], [], [], [], [], [], []
    next_trajectory_id = int(offline["trajectory_id"].max()) + 1
    online_count = 0
    for path in sorted(online_dir.rglob("*.npz")):
        with np.load(path, allow_pickle=False) as archive:
            observations = archive["observations"].astype(np.float32)
            teacher_actions = archive["teacher_actions"].astype(np.float32)
            executed_actions = archive["executed_actions"].astype(np.float32)
            metadata = json.loads(str(archive["metadata_json"].item()))
        category_id = int(category_to_id[metadata["category"]])
        previous = np.empty_like(executed_actions)
        previous[0] = 2.0 * teacher_actions[0] - teacher_actions[1]
        previous[1:] = executed_actions[:-1]
        for step in range(len(observations)):
            rows_obs.append(observations[step])
            rows_actions.append(teacher_actions[step])
            rows_prev.append(previous[step])
            rows_cat.append(category_id)
            rows_traj.append(next_trajectory_id)
            rows_src.append(int(metadata["source_trajectory_index"]))
            rows_frame.append(step)
        next_trajectory_id += 1
        online_count += len(observations)
        print(f"merged {path.name} ({len(observations)} steps)")

    offline_prev = np.empty_like(offline["actions"])
    for trajectory_id in np.unique(offline["trajectory_id"]):
        indices = np.flatnonzero(offline["trajectory_id"] == trajectory_id)
        raw = offline["actions"][indices]
        offline_prev[indices[0]] = 2.0 * raw[0] - raw[1]
        offline_prev[indices[1:]] = raw[:-1]
    merged = {
        "observations": np.concatenate(
            [offline["observations"], np.stack(rows_obs)]).astype(np.float32),
        "actions": np.concatenate(
            [offline["actions"], np.stack(rows_actions)]).astype(np.float32),
        "previous_commands": np.concatenate(
            [offline_prev, np.stack(rows_prev)]).astype(np.float32),
        "trajectory_id": np.concatenate(
            [offline["trajectory_id"], np.asarray(rows_traj)]).astype(np.int64),
        "category_id": np.concatenate(
            [offline["category_id"], np.asarray(rows_cat)]).astype(np.int64),
        "object_id": np.concatenate(
            [offline["object_id"],
             np.full(online_count, -1, dtype=np.int64)]).astype(np.int64),
        "source_trajectory_index": np.concatenate(
            [offline["source_trajectory_index"],
             np.asarray(rows_src)]).astype(np.int64),
        "source_frame_index": np.concatenate(
            [offline["source_frame_index"],
             np.asarray(rows_frame)]).astype(np.int64),
        "is_hold": np.concatenate(
            [offline["is_hold"],
             np.zeros(online_count, dtype=bool)]).astype(bool),
        "expert_replay_success": np.concatenate(
            [offline["expert_replay_success"],
             np.full(online_count, False, dtype=bool)]).astype(bool),
    }
    np.savez(output_dir / "train.npz", **merged)
    print(f"MERGED offline={n_offline} online={online_count} total={n_offline + online_count}")
    print(f"output={output_dir / 'train.npz'}")


if __name__ == "__main__":
    main()
