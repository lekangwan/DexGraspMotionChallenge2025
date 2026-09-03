#!/usr/bin/env python3
"""把逐轨迹DAgger查询结果合并为可训练的Online-R1数据集。

输入：在线NPZ根目录、类别映射和唯一输出路径。
输出：含原始观测、教师动作、学生实际动作、轨迹/类别ID的单个NPZ及JSON摘要。
内部逻辑：递归读取全部轨迹，验证同一手/学生/教师checkpoint和对齐协议；监督
动作取固定类别教师输出，历史动作保留真正由学生执行的命令，二者绝不混淆。
作用：让后续训练按25%在线、75%离线聚合；这一步只合并数据，不再次查询物理结果。
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np


def load_online_trajectory(path, expected_observation_dim, expected_action_dim):
    """读取并验证一条在线轨迹，返回数组和元数据。"""
    with np.load(path, allow_pickle=False) as archive:
        required = {"observations", "teacher_actions", "executed_actions", "metadata_json"}
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"{path}缺少字段: {sorted(missing)}")
        observations = archive["observations"].astype(np.float32)
        teacher_actions = archive["teacher_actions"].astype(np.float32)
        executed_actions = archive["executed_actions"].astype(np.float32)
        metadata = json.loads(str(archive["metadata_json"].item()))
    length = len(observations)
    if observations.shape != (length, expected_observation_dim):
        raise ValueError(f"{path}观测尺寸错误: {observations.shape}")
    if teacher_actions.shape != (length, expected_action_dim):
        raise ValueError(f"{path}教师动作尺寸错误: {teacher_actions.shape}")
    if executed_actions.shape != teacher_actions.shape:
        raise ValueError(f"{path}学生动作尺寸错误")
    if not all(np.isfinite(value).all() for value in (observations, teacher_actions, executed_actions)):
        raise ValueError(f"{path}含NaN或Inf")
    if metadata.get("alignment") not in {
        "student_pre_action_observation_to_category_teacher_action_v1",
        "student_pre_action_observation_to_state_aligned_expert_action_v2",
    }:
        raise ValueError(f"{path}对齐协议错误")
    return observations, teacher_actions, executed_actions, metadata


def aggregate(online_dir, data_dir, output):
    """汇总一个手的一轮在线查询。

    输入：在线轨迹目录、该手正式数据目录和输出路径。
    输出：数据摘要字典，并写NPZ/JSON。
    内部逻辑：从正式数据读取维度与类别ID；为每个文件分配新trajectory_id，
    将教师动作写为训练`actions`，另存学生`executed_actions`供Temporal历史使用。
    作用：统一单帧Online-R1、Temporal3和Diffusion对在线状态的读取口径。
    """
    online_dir = Path(online_dir).expanduser().resolve()
    data_dir = Path(data_dir).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    mappings = json.loads((data_dir / "mappings.json").read_text(encoding="utf-8"))
    with np.load(data_dir / "train.npz", allow_pickle=False) as archive:
        observation_dim = int(archive["observations"].shape[1])
        action_dim = int(archive["actions"].shape[1])
    paths = sorted(online_dir.glob("**/*.npz"))
    if not paths:
        raise ValueError(f"没有在线轨迹: {online_dir}")
    chunks = []
    reference = None
    category_counts = Counter()
    for trajectory_id, path in enumerate(paths):
        observations, teacher_actions, executed_actions, metadata = load_online_trajectory(
            path, observation_dim, action_dim
        )
        identity = (
            metadata["hand"],
            metadata["student_checkpoint"],
            metadata["teacher_checkpoint"],
        )
        if reference is None:
            reference = identity
        elif identity != reference:
            raise ValueError(f"{path}来自不同手或不同师生checkpoint")
        category = metadata["category"]
        if category not in mappings["category_to_id"]:
            raise ValueError(f"未知类别: {category}")
        category_counts[category] += 1
        length = len(observations)
        chunks.append(
            {
                "observations": observations,
                "actions": teacher_actions,
                "executed_actions": executed_actions,
                "trajectory_id": np.full(length, trajectory_id, dtype=np.int64),
                "category_id": np.full(
                    length, mappings["category_to_id"][category], dtype=np.int64
                ),
                "is_hold": np.arange(length, dtype=np.int64) >= 210,
            }
        )
    merged = {
        key: np.concatenate([chunk[key] for chunk in chunks], axis=0)
        for key in chunks[0]
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **merged)
    summary = {
        "schema_version": 1,
        "hand": reference[0],
        "student_checkpoint": reference[1],
        "teacher_checkpoint": reference[2],
        "trajectory_count": len(chunks),
        "step_count": len(merged["actions"]),
        "observation_dimension": observation_dim,
        "action_dimension": action_dim,
        "category_trajectory_counts": dict(sorted(category_counts.items())),
        "target_rule": "fixed_category_teacher_on_student_visited_pre_action_state",
        "history_rule": "executed_student_action_not_teacher_target",
        "output": str(output),
    }
    output.with_suffix(".json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main():
    """解析CLI、合并并打印Online-R1完成标志。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--online-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = aggregate(args.online_dir, args.data_dir, args.output)
    print(
        f"trajectories={summary['trajectory_count']} steps={summary['step_count']} "
        f"categories={len(summary['category_trajectory_counts'])}"
    )
    print(f"ONLINE_R1_DATA={summary['output']}")


if __name__ == "__main__":
    main()
