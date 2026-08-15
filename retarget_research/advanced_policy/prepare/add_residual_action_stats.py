#!/usr/bin/env python3
"""用成功train轨迹补充动作增量的均值和标准差。

输入：策略数据目录中的`train.npz`与`normalization.npz`。
输出：原归一化文件新增`action_delta_mean/std`，其他字段逐元素保持不变。
内部逻辑：轨迹首步的上一命令由20 Hz首帧三步线性插值反解为`2*a0-a1`；
其余步骤只比较同一trajectory内相邻动作，绝不跨文件边界或读取valid/test。
作用：让PhaseResidual策略在训练和闭环中使用完全相同的专家增量尺度。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def compute_delta_stats(actions, trajectory_ids):
    """输入train动作及轨迹ID，输出全部合法delta、逐维均值和安全标准差。"""
    actions = np.asarray(actions, dtype=np.float32)
    trajectory_ids = np.asarray(trajectory_ids, dtype=np.int64)
    if actions.ndim != 2 or len(actions) != len(trajectory_ids):
        raise ValueError("动作和trajectory_id形状不一致")
    deltas = np.empty_like(actions)
    for trajectory_id in np.unique(trajectory_ids):
        indices = np.flatnonzero(trajectory_ids == trajectory_id)
        if len(indices) < 2 or not np.array_equal(
            indices, np.arange(indices[0], indices[-1] + 1)
        ):
            raise ValueError(f"trajectory {trajectory_id}不连续或过短")
        open_command = 2.0 * actions[indices[0]] - actions[indices[1]]
        deltas[indices[0]] = actions[indices[0]] - open_command
        deltas[indices[1:]] = actions[indices[1:]] - actions[indices[:-1]]
    mean = deltas.mean(axis=0).astype(np.float32)
    std = np.maximum(deltas.std(axis=0), 1e-6).astype(np.float32)
    return deltas, mean, std


def add_stats(data_dir):
    """读取一个手的数据、验证或原子更新增量统计并返回摘要。

    输入：包含train/normalization的目录。
    输出：动作维度、delta L2均值/99.5%分位及是否已存在的摘要。
    内部逻辑：若已有字段则要求与重算值一致；否则保留全部旧数组写临时文件再替换。
    作用：脚本可安全重复运行，并防止后续数据改变后继续沿用旧增量尺度。
    """
    data_dir = Path(data_dir).expanduser().resolve()
    with np.load(data_dir / "train.npz", allow_pickle=False) as archive:
        actions = archive["actions"].astype(np.float32)
        trajectory_ids = archive["trajectory_id"].astype(np.int64)
    deltas, mean, std = compute_delta_stats(actions, trajectory_ids)
    normalization_path = data_dir / "normalization.npz"
    with np.load(normalization_path, allow_pickle=False) as archive:
        values = {name: archive[name].copy() for name in archive.files}
    existed = "action_delta_mean" in values or "action_delta_std" in values
    if existed:
        if not {"action_delta_mean", "action_delta_std"} <= set(values):
            raise ValueError("增量统计只存在一半字段")
        np.testing.assert_allclose(values["action_delta_mean"], mean, rtol=0, atol=1e-7)
        np.testing.assert_allclose(values["action_delta_std"], std, rtol=0, atol=1e-7)
    else:
        values["action_delta_mean"] = mean
        values["action_delta_std"] = std
        temporary = normalization_path.with_suffix(".npz.tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **values)
        temporary.replace(normalization_path)
    norms = np.linalg.norm(deltas, axis=1)
    return {
        "action_dimension": int(actions.shape[1]),
        "step_count": int(len(actions)),
        "mean_delta_l2": float(norms.mean()),
        "q995_delta_l2": float(np.quantile(norms, 0.995)),
        "already_present": bool(existed),
        "normalization": str(normalization_path),
    }


def main():
    """解析数据目录、补充统计并打印完成标志。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = add_stats(args.data_dir)
    print(
        f"steps={summary['step_count']} mean_delta_l2={summary['mean_delta_l2']:.6f} "
        f"q995_delta_l2={summary['q995_delta_l2']:.6f}"
    )
    print(f"RESIDUAL_ACTION_STATS={summary['normalization']}")


if __name__ == "__main__":
    main()
