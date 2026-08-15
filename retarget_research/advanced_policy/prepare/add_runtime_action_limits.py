#!/usr/bin/env python3
"""为现有正式策略数据补充仅由train计算的动作限速统计。

输入：策略数据根目录、一个或多个手名及分位数。
输出：原子更新各手`normalization.npz`和`dataset_summary.json`。
内部逻辑：保留已有归一化字段，只读取train同轨迹相邻动作计算逐维/L2高分位。
作用：无需重新解析3000条物理trace，即可让既有checkpoint使用新的闭环安全层。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from .prepare_policy_dataset import compute_action_delta_limits
except ImportError:
    from prepare_policy_dataset import compute_action_delta_limits


def update_hand_data(data_dir: Path, quantile: float) -> dict:
    """计算并原子写入一只手的运行时动作限速统计。

    输入：包含train/normalization/summary的数据目录和分位数。
    输出：便于终端核对的精简统计字典。
    内部逻辑：从train提取动作与轨迹边界，保留normalization旧字段后增加三个新字段。
    作用：兼容已训练checkpoint，避免为执行层限速重新生成策略数据或重新训练。
    """
    train_path = data_dir / "train.npz"
    normalization_path = data_dir / "normalization.npz"
    summary_path = data_dir / "dataset_summary.json"
    for path in (train_path, normalization_path, summary_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    with np.load(train_path, allow_pickle=False) as archive:
        per_dimension, vector_norm = compute_action_delta_limits(
            archive["actions"], archive["trajectory_id"], quantile
        )
    with np.load(normalization_path, allow_pickle=False) as archive:
        normalization = {name: archive[name].copy() for name in archive.files}
    normalization.update(
        {
            "action_delta_limit": per_dimension,
            "action_delta_norm_limit": np.asarray(vector_norm, dtype=np.float32),
            "action_delta_quantile": np.asarray(quantile, dtype=np.float32),
        }
    )
    temporary_npz = normalization_path.with_name("normalization.tmp.npz")
    np.savez_compressed(temporary_npz, **normalization)
    temporary_npz.replace(normalization_path)

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["runtime_action_rate_limit"] = {
        "source": "train_same_trajectory_adjacent_action_delta",
        "quantile": float(quantile),
        "per_dimension_limit": per_dimension.tolist(),
        "vector_l2_limit": float(vector_norm),
    }
    temporary_json = summary_path.with_suffix(".json.tmp")
    temporary_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary_json.replace(summary_path)
    return {
        "hand": summary["hand"],
        "action_dimension": len(per_dimension),
        "quantile": float(quantile),
        "vector_l2_limit": float(vector_norm),
    }


def main() -> None:
    """解析数据根目录并依次升级指定手的数据。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--hand", action="append", choices=["linker", "xhand", "wuji"], required=True
    )
    parser.add_argument("--quantile", type=float, default=0.995)
    args = parser.parse_args()
    for hand in args.hand:
        result = update_hand_data(args.data_root / hand, args.quantile)
        print(json.dumps(result, ensure_ascii=False))
    print("RUNTIME_ACTION_LIMITS=READY")


if __name__ == "__main__":
    main()
