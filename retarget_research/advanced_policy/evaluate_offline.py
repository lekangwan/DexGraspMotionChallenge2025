#!/usr/bin/env python3
"""在冻结test NPZ上测量训练策略的动作预测误差。

输入：checkpoint、对应手的数据目录、设备和可选最大轨迹/步骤数。
输出：整体及逐类别MAE/RMSE JSON，可选预测数组。
内部逻辑：按trajectory_id顺序重置统一PolicyRunner，逐步预测并与未归一化专家动作配对。
作用：作为训练是否学会专家动作的快速诊断；它不是物理抓取成功率，不能替代闭环评测。
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

try:
    from .runtime import PolicyRunner
except ImportError:
    from runtime import PolicyRunner


def error_metrics(predictions, targets):
    """计算一组未标准化动作的元素级MAE、RMSE和逐维MAE。"""
    predictions = np.asarray(predictions, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)
    difference = predictions - targets
    return {
        "sample_count": len(predictions),
        "mae": float(np.mean(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(np.square(difference)))),
        "per_action_dimension_mae": np.mean(np.abs(difference), axis=0).tolist(),
    }


def evaluate(args):
    """按完整轨迹运行离线推理并汇总误差。

    输入：解析后的checkpoint、数据、设备和限制参数。
    输出：报告字典及预测/目标/标签数组。
    内部逻辑：轨迹边界处reset，类别ID反查类别名；限制始终以完整轨迹为单位。
    作用：同时验证checkpoint可加载、历史逻辑可运行和数据类别映射一致。
    """
    with np.load(args.data_dir / "test.npz", allow_pickle=False) as archive:
        data = {name: archive[name].copy() for name in archive.files}
    mappings = json.loads((args.data_dir / "mappings.json").read_text(encoding="utf-8"))
    id_to_category = {int(value): key for key, value in mappings["category_to_id"].items()}
    runner = PolicyRunner(
        args.checkpoint,
        args.data_dir,
        args.device,
        args.diffusion_execute_steps,
        args.normalized_action_clip,
    )
    trajectory_ids = list(dict.fromkeys(data["trajectory_id"].astype(int).tolist()))
    if args.max_trajectories > 0:
        trajectory_ids = trajectory_ids[: args.max_trajectories]
    predictions, targets, categories, trajectory_labels = [], [], [], []
    for trajectory_id in trajectory_ids:
        indices = np.flatnonzero(data["trajectory_id"] == trajectory_id)
        if args.max_steps_per_trajectory > 0:
            indices = indices[: args.max_steps_per_trajectory]
        category_id = int(data["category_id"][indices[0]])
        category_name = id_to_category[category_id]
        runner.reset(category_name, data["observations"][indices[0]])
        for index in indices:
            predictions.append(runner.act(data["observations"][index]))
            targets.append(data["actions"][index])
            categories.append(category_id)
            trajectory_labels.append(trajectory_id)
    predictions = np.asarray(predictions, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)
    categories = np.asarray(categories, dtype=np.int64)
    per_category = {}
    for category_id in sorted(set(categories.tolist())):
        mask = categories == category_id
        per_category[id_to_category[category_id]] = error_metrics(predictions[mask], targets[mask])
    report = {
        "status": "complete",
        "metric_boundary": "offline expert action error; not closed-loop grasp success",
        "checkpoint": str(args.checkpoint.resolve()),
        "data_dir": str(args.data_dir.resolve()),
        "model_type": runner.model_type,
        "trajectory_count": len(trajectory_ids),
        "overall": error_metrics(predictions, targets),
        "per_category": per_category,
    }
    arrays = {
        "predictions": predictions,
        "targets": targets,
        "category_id": categories,
        "trajectory_id": np.asarray(trajectory_labels, dtype=np.int64),
    }
    return report, arrays


def main():
    """解析参数、执行离线评估并保存JSON/可选NPZ。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prediction-output", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-trajectories", type=int, default=0)
    parser.add_argument("--max-steps-per-trajectory", type=int, default=0)
    parser.add_argument("--diffusion-execute-steps", type=int, default=2)
    parser.add_argument("--normalized-action-clip", type=float, default=5.0)
    args = parser.parse_args()
    report, arrays = evaluate(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.prediction_output is not None:
        args.prediction_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.prediction_output, **arrays)
    print(f"trajectories={report['trajectory_count']}")
    print(f"mae={report['overall']['mae']:.6f}")
    print(f"rmse={report['overall']['rmse']:.6f}")
    print(f"OFFLINE_EVALUATION={args.output.resolve()}")


if __name__ == "__main__":
    main()
