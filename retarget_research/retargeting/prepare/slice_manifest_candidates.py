#!/usr/bin/env python3
"""从已有完整候选目录切出与小manifest严格对齐的轻量候选目录。

输入：A或B manifest、已有候选目录、输出目录和可选动作维度。
输出：每个物体仅含manifest所选轨迹的npy，以及切片审计摘要JSON。
内部逻辑：按`source_trajectory_indices`查行，并同步切片所有轨迹级数组/列表字段。
作用：直接复用昂贵的1000条重定向结果，让小样本物理评测无需重新生成轨迹。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def selected_positions(source_indices: np.ndarray, requested_indices: list[int]) -> list[int]:
    """把源轨迹编号转换成候选文件中的行位置。

    输入：候选保存的源编号数组，以及manifest请求的编号列表。
    输出：与请求顺序一致的整数行位置列表。
    内部逻辑：建立唯一编号到行号的映射，并在缺失或重复时立即报错。
    作用：避免把“第几行”和“原始数据第几条轨迹”混为一谈。
    """
    values = [int(value) for value in np.asarray(source_indices).tolist()]
    if len(values) != len(set(values)):
        raise ValueError(f"候选source_trajectory_indices存在重复值: {values}")
    by_index = {value: position for position, value in enumerate(values)}
    missing = [value for value in requested_indices if value not in by_index]
    if missing:
        raise ValueError(f"完整候选缺少manifest轨迹索引: {missing}")
    return [by_index[value] for value in requested_indices]


def slice_trajectory_field(value: Any, positions: list[int], source_count: int) -> Any:
    """在确认第一维属于轨迹维时同步切片一个npy字段。

    输入：任意字段值、目标行号和原候选轨迹数。
    输出：轨迹级字段的子集；全局标量/配置字段保持原值。
    内部逻辑：只切第一维恰等于`source_count`的ndarray、list或tuple。
    作用：让动作、尺度、旋转、loss和阶段元数据始终保持行对齐。
    """
    if isinstance(value, np.ndarray) and value.ndim >= 1 and len(value) == source_count:
        return value[np.asarray(positions, dtype=np.int64)]
    if isinstance(value, list) and len(value) == source_count:
        return [value[position] for position in positions]
    if isinstance(value, tuple) and len(value) == source_count:
        return tuple(value[position] for position in positions)
    return value


def slice_candidate(data: dict, requested_indices: list[int], dimension: int | None) -> dict:
    """切出一个物体的若干候选轨迹并执行形状/数值校验。

    输入：完整候选字典、manifest源索引和可选期望动作维度。
    输出：字段对齐的新候选字典。
    内部逻辑：先定位行，再统一切片轨迹级字段，最后覆盖源索引并检查70帧有限值。
    作用：提供可单元测试的核心切片逻辑，与文件系统遍历解耦。
    """
    if "source_trajectory_indices" not in data or "grasp_seqs" not in data:
        raise ValueError("候选缺少source_trajectory_indices或grasp_seqs")
    source_indices = np.asarray(data["source_trajectory_indices"], dtype=np.int64)
    frames = np.asarray(data["grasp_seqs"])
    if frames.ndim != 3 or len(frames) != len(source_indices):
        raise ValueError(
            f"候选动作与索引第一维不一致: {frames.shape} vs {source_indices.shape}"
        )
    positions = selected_positions(source_indices, requested_indices)
    output = {
        key: slice_trajectory_field(value, positions, len(source_indices))
        for key, value in data.items()
    }
    output["source_trajectory_indices"] = np.asarray(
        requested_indices, dtype=np.int64
    )
    selected_frames = np.asarray(output["grasp_seqs"])
    expected_tail = (70, dimension) if dimension is not None else (70, frames.shape[2])
    if selected_frames.shape != (len(requested_indices), *expected_tail):
        raise ValueError(
            f"切片后动作形状错误: {selected_frames.shape}，期望"
            f"{(len(requested_indices), *expected_tail)}"
        )
    if not np.isfinite(selected_frames).all():
        raise ValueError("切片候选动作含NaN或Inf")
    return output


def slice_manifest(
    manifest: dict,
    source_dir: Path,
    output_dir: Path,
    dimension: int | None,
) -> dict:
    """按manifest逐物体切片整个候选目录。

    输入：已解析manifest、源/输出目录和可选动作维度。
    输出：含物体数、轨迹数和逐文件来源的审计摘要。
    内部逻辑：每个条目加载同名npy、调用纯切片函数并保存到新目录。
    作用：为Linker成对比较和XHand轻量后处理准备严格对齐输入。
    """
    records = []
    trajectory_count = 0
    output_dir.mkdir(parents=True, exist_ok=True)
    for entry in manifest.get("entries", []):
        object_name = str(entry["object_name"])
        requested = [int(value) for value in entry["trajectory_indices"]]
        source_path = source_dir / f"{object_name}.npy"
        if not source_path.is_file():
            raise FileNotFoundError(f"完整候选不存在: {source_path}")
        data = np.load(source_path, allow_pickle=True).item()
        output = slice_candidate(data, requested, dimension)
        output_path = output_dir / source_path.name
        np.save(output_path, output, allow_pickle=True)
        records.append(
            {
                "object_name": object_name,
                "source": str(source_path.resolve()),
                "output": str(output_path.resolve()),
                "source_trajectory_indices": requested,
            }
        )
        trajectory_count += len(requested)
    expected = int(manifest.get("trajectory_count", trajectory_count))
    if trajectory_count != expected:
        raise ValueError(f"切片轨迹数{trajectory_count}与manifest声明{expected}不符")
    return {
        "manifest_purpose": manifest.get("purpose"),
        "source_candidate_dir": str(source_dir.resolve()),
        "output_candidate_dir": str(output_dir.resolve()),
        "object_count": len(records),
        "trajectory_count": trajectory_count,
        "target_dimension": dimension,
        "records": records,
    }


def main() -> None:
    """解析目录参数，执行批量切片并写出审计摘要。

    输入：`--manifest/--source-dir/--output-dir/--dimension`。
    输出：对齐候选npy与`candidate_slice_summary.json`。
    内部逻辑：读取JSON后调用`slice_manifest`，不触发运动学优化或物理仿真。
    作用：把约10小时正式重定向结果转换成秒级可复用的小样本输入。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dimension", type=int)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    summary = slice_manifest(
        manifest, args.source_dir, args.output_dir, args.dimension
    )
    summary_path = args.output_dir / "candidate_slice_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"objects={summary['object_count']} trajectories={summary['trajectory_count']}"
    )
    print(f"summary={summary_path.resolve()}")


if __name__ == "__main__":
    main()
