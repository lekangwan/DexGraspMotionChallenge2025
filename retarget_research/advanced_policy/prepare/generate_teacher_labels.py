#!/usr/bin/env python3
"""用固定共享类别教师为离线数据生成蒸馏动作标签。

输入：类别教师checkpoint、标准策略数据目录、split和输出NPZ。
输出：与原split逐步骤严格对齐的标准化`teacher_actions`及轨迹ID。
内部逻辑：按原始顺序遍历、不shuffle；教师根据category_id自动路由轻量类别头；
输出前核对样本数并保存checkpoint绝对路径，防止学生误读另一批标签。
作用：把“类别教师 -> Task-ID统一学生”从口头设计变成可审计的离线蒸馏数据。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from ..dataset import TargetHandPolicyDataset
    from ..train import build_model, load_checkpoint_file
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from dataset import TargetHandPolicyDataset
    from train import build_model, load_checkpoint_file


def generate_labels(checkpoint_path, data_dir, split, output, device, batch_size):
    """执行一个split的教师标签推理。

    输入：checkpoint、数据目录、split名、输出路径、设备和batch大小。
    输出：摘要字典，并写入压缩NPZ。
    内部逻辑：严格要求checkpoint类型为`category_teacher`；直接使用数据集已经
    标准化的观测，标签也保持标准化，避免一次反归一化/再归一化的数值误差。
    作用：提供统一学生的监督target，同时保留与演示动作相同的尺度。
    """
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    data_dir = Path(data_dir).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    payload = load_checkpoint_file(checkpoint_path, device)
    config, dimensions = payload["config"], payload["dimensions"]
    if config.get("model_type") != "category_teacher":
        raise ValueError("教师checkpoint的model_type不是category_teacher")
    dataset = TargetHandPolicyDataset(
        data_dir / f"{split}.npz",
        data_dir / "normalization.npz",
        mode="bc",
    )
    if dataset.observations.shape[1] != dimensions["observation_dim"]:
        raise ValueError("教师观测维度与数据不一致")
    if dataset.actions.shape[1] != dimensions["action_dim"]:
        raise ValueError("教师动作维度与数据不一致")
    model = build_model(
        config,
        dimensions["observation_dim"],
        dimensions["action_dim"],
        dimensions["category_count"],
    ).to(device)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=False, num_workers=0)
    chunks = []
    with torch.no_grad():
        for batch in loader:
            prediction = model(
                batch["observations"].to(device),
                batch["category_id"].to(device),
            )
            chunks.append(prediction.cpu().numpy().astype(np.float32))
    labels = np.concatenate(chunks, axis=0)
    if labels.shape != dataset.actions.shape or not np.isfinite(labels).all():
        raise ValueError(f"教师标签非法: {labels.shape}")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        teacher_actions=labels,
        trajectory_id=dataset.data["trajectory_id"].astype(np.int64),
        category_id=dataset.data["category_id"].astype(np.int64),
        teacher_checkpoint=np.asarray(str(checkpoint_path)),
        split=np.asarray(split),
    )
    return {
        "split": split,
        "sample_count": len(labels),
        "action_dimension": labels.shape[1],
        "teacher_checkpoint": str(checkpoint_path),
        "output": str(output),
    }


def main():
    """解析CLI、生成单个split标签并打印完成标志。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "valid"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("请求CUDA生成标签，但当前PyTorch不可见GPU")
    summary = generate_labels(
        args.checkpoint,
        args.data_dir,
        args.split,
        args.output,
        torch.device(args.device),
        args.batch_size,
    )
    print(json.dumps(summary, ensure_ascii=False))
    print(f"TEACHER_LABELS={summary['output']}")


if __name__ == "__main__":
    main()
