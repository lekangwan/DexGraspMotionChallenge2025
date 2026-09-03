#!/usr/bin/env python3
"""把多个同构BC checkpoint做参数平均，生成推理成本不变的BC Soup。

输入：两个或更多`--ingredient` checkpoint、可选对应权重和唯一输出路径。
输出：带`model_soup`来源记录的新checkpoint，以及同名JSON审计文件。
内部逻辑：严格核对数据维度、参数键/形状/类型；浮点参数以float64累加后还原，
非浮点状态必须完全相同；删除已经不再对应新参数的优化器和随机数状态。
作用：复现旧Shadow流程的BC Soup阶段，为后续类别教师提供更稳的共同初始化。
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch


def load_local_checkpoint(path):
    """读取本项目可信checkpoint并兼容不同PyTorch的`weights_only`参数。"""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def normalize_weights(values, count):
    """输入可选原始权重和模型数，输出和为1的非负权重。"""
    weights = [1.0] * count if not values else [float(value) for value in values]
    if len(weights) != count:
        raise ValueError("每个ingredient必须恰好对应一个weight")
    if any(value < 0.0 for value in weights) or sum(weights) <= 0.0:
        raise ValueError("Soup权重必须非负且总和大于0")
    total = sum(weights)
    return [value / total for value in weights]


def average_states(states, weights):
    """逐参数平均兼容state_dict。

    输入：至少两个state_dict和归一化权重。
    输出：新的平均state_dict。
    内部逻辑：浮点张量用float64避免累加精度损失；整数/布尔buffer拒绝不一致。
    作用：确保参数Soup不是静默拼接了不同网络或不同类别映射。
    """
    keys = list(states[0])
    if any(list(state) != keys for state in states[1:]):
        raise ValueError("ingredient的模型参数键或顺序不同")
    averaged = {}
    for key in keys:
        tensors = [state[key] for state in states]
        reference = tensors[0]
        if any(
            tensor.shape != reference.shape or tensor.dtype != reference.dtype
            for tensor in tensors[1:]
        ):
            raise ValueError(f"参数形状或类型不兼容: {key}")
        if reference.is_floating_point():
            value = torch.zeros_like(reference, dtype=torch.float64)
            for weight, tensor in zip(weights, tensors):
                value.add_(tensor.to(torch.float64), alpha=weight)
            averaged[key] = value.to(reference.dtype)
        elif any(not torch.equal(reference, tensor) for tensor in tensors[1:]):
            raise ValueError(f"非浮点buffer不一致: {key}")
        else:
            averaged[key] = reference.clone()
    return averaged


def make_soup(paths, weights, output):
    """执行完整Soup构建并返回审计元数据。

    输入：checkpoint路径、归一化前权重和不存在的输出路径。
    输出：写入checkpoint/JSON并返回metadata字典。
    内部逻辑：除state形状外还要求`dimensions`完全相同、所有模型均为单帧BC；
    输出沿用第一份结构配置，但显式记录所有原料，且不伪装成可续训checkpoint。
    作用：为后续warm start提供可追溯、可直接推理的平均模型。
    """
    paths = [Path(path).expanduser().resolve() for path in paths]
    output = Path(output).expanduser().resolve()
    if len(paths) < 2:
        raise ValueError("BC Soup至少需要两个独立checkpoint")
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    if output.exists():
        raise FileExistsError(output)
    normalized_weights = normalize_weights(weights, len(paths))
    checkpoints = [load_local_checkpoint(path) for path in paths]
    model_types = [item.get("config", {}).get("model_type") for item in checkpoints]
    if any(value != model_types[0] for value in model_types[1:]):
        raise ValueError("Soup原料的model_type不一致")
    dimensions = checkpoints[0].get("dimensions")
    if any(item.get("dimensions") != dimensions for item in checkpoints[1:]):
        raise ValueError("ingredient的数据维度或类别数不同")
    states = [item.get("model_state") for item in checkpoints]
    if any(state is None for state in states):
        raise ValueError("ingredient缺少model_state")
    result = copy.deepcopy(checkpoints[0])
    result["model_state"] = average_states(states, normalized_weights)
    metadata = {
        "method": "weighted_parameter_average",
        "ingredients": [str(path) for path in paths],
        "normalized_weights": normalized_weights,
        "dimensions": dimensions,
        "model_type": model_types[0],
        "same_inference_architecture": True,
        "optimizer_state_removed": True,
    }
    result["model_soup"] = metadata
    for key in (
        "optimizer_state",
        "scheduler_state",
        "data_loader_generator_state",
        "python_random_state",
        "numpy_random_state",
        "torch_random_state",
        "cuda_random_state",
    ):
        result.pop(key, None)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, output)
    output.with_suffix(".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return metadata


def main():
    """解析CLI、生成Soup并打印机器可识别完成标志。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--ingredient", action="append", type=Path, required=True)
    parser.add_argument("--weight", action="append", type=float, default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metadata = make_soup(args.ingredient, args.weight, args.output)
    print(f"ingredients={len(metadata['ingredients'])}")
    print(f"normalized_weights={metadata['normalized_weights']}")
    print(f"MODEL_SOUP={args.output.expanduser().resolve()}")


if __name__ == "__main__":
    main()
