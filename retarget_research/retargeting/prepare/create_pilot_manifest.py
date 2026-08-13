#!/usr/bin/env python3
"""从现有Shadow轨迹目录确定性抽取重定向开发manifest。

输入：包含`.npy`物体文件的目录、随机种子、物体数、每物体轨迹数和排除项。
输出：记录源文件哈希与轨迹索引的JSON manifest。
内部逻辑：先校验每个文件的`grasp_seqs`维度，再用单一NumPy随机生成器无放回抽样。
作用：在运行实验前冻结样本，防止根据物理结果选择性保留容易成功的轨迹。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def file_sha256(path):
    """计算源轨迹文件的SHA-256摘要。

    输入：一个本地文件路径。
    输出：64字符十六进制哈希字符串。
    逻辑：按1 MiB分块读取并累计摘要，避免一次把整个文件读入内存。
    作用：以后即使文件名相同，也能发现数据内容已发生变化。
    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def inspect_source_file(path):
    """读取并校验一个Shadow轨迹文件的最小元数据。

    输入：预期包含`grasp_seqs/obj_rotmat/obj_scale`的`.npy`路径。
    输出：物体名、轨迹数、帧数、动作维度、路径和文件哈希字典。
    逻辑：只允许`(N,T,28)`且三个必需字段长度一致的数据进入候选池。
    作用：让抽样失败尽早暴露，而不是在昂贵优化运行中途才报维度错误。
    """
    data = np.load(path, allow_pickle=True).item()
    required = {"grasp_seqs", "obj_rotmat", "obj_scale"}
    missing = sorted(required - set(data))
    if missing:
        raise ValueError(f"{path.name} 缺少字段: {missing}")
    sequences = np.asarray(data["grasp_seqs"])
    if sequences.ndim != 3 or sequences.shape[2] != 28:
        raise ValueError(f"{path.name} grasp_seqs应为(N,T,28)，实际{sequences.shape}")
    if len(data["obj_rotmat"]) != len(sequences) or len(data["obj_scale"]) != len(
        sequences
    ):
        raise ValueError(f"{path.name} 三个轨迹级字段长度不一致")
    return {
        "object_name": path.stem,
        "source_path": str(path.resolve()),
        "source_sha256": file_sha256(path),
        "available_trajectory_count": int(sequences.shape[0]),
        "frame_count": int(sequences.shape[1]),
        "action_dimension": int(sequences.shape[2]),
    }


def sample_manifest(candidates, object_count, trajectories_per_object, seed):
    """从已校验候选物体中无放回抽取物体和轨迹索引。

    输入：候选元数据、物体数、每物体轨迹数和随机种子。
    输出：按物体名排序、带`trajectory_indices`的条目列表。
    逻辑：同一随机生成器先抽物体，再为每个物体独立抽轨迹并排序。
    作用：用可复现随机选择代替手工挑选成功样本。
    """
    eligible = [
        item
        for item in candidates
        if item["available_trajectory_count"] >= trajectories_per_object
    ]
    if len(eligible) < object_count:
        raise ValueError(
            f"合格物体仅{len(eligible)}个，无法抽取{object_count}个"
        )
    rng = np.random.default_rng(seed)
    selected_positions = rng.choice(len(eligible), size=object_count, replace=False)
    selected = []
    for position in selected_positions:
        item = dict(eligible[int(position)])
        count = item["available_trajectory_count"]
        indices = rng.choice(count, size=trajectories_per_object, replace=False)
        item["trajectory_indices"] = sorted(int(index) for index in indices)
        selected.append(item)
    return sorted(selected, key=lambda item: item["object_name"])


def collect_excluded_objects(explicit_names, manifest_paths):
    """合并命令行排除项和历史manifest中已经暴露的物体。

    输入：显式物体名列表，以及零个或多个历史manifest路径。
    输出：去重后的物体名集合和已解析历史manifest的绝对路径列表。
    内部逻辑：同时读取历史清单的`entries`与旧版`excluded_tuning_objects`；
    前者排除真正评测过的物体，后者继续排除更早用于单例调参的物体。
    作用：建立独立验证集时自动防止物体泄漏，避免人工复制名称漏项。
    """
    excluded = {str(name) for name in explicit_names}
    resolved_manifests = []
    for path in manifest_paths:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"排除manifest不存在: {path}")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        entries = manifest.get("entries")
        if not isinstance(entries, list):
            raise ValueError(f"排除manifest缺少entries列表: {path}")
        for entry in entries:
            if "object_name" not in entry:
                raise ValueError(f"排除manifest条目缺少object_name: {path}")
            excluded.add(str(entry["object_name"]))
        excluded.update(
            str(name) for name in manifest.get("excluded_tuning_objects", [])
        )
        excluded.update(str(name) for name in manifest.get("excluded_objects", []))
        resolved_manifests.append(str(path.resolve()))
    return excluded, resolved_manifests


def frozen_method_record(config_path):
    """为验证清单记录冻结方法配置及其内容哈希。

    输入：可选JSON配置路径。
    输出：未提供时为`None`，否则返回绝对路径、SHA-256和方法名。
    内部逻辑：先解析JSON确认配置有效，再对原文件字节计算摘要。
    作用：证明验证数据冻结时使用的是哪一版方法，防止事后静默改参数。
    """
    if config_path is None:
        return None
    config_path = Path(config_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"冻结方法配置不存在: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if "method" not in config:
        raise ValueError(f"冻结方法配置缺少method字段: {config_path}")
    return {
        "path": str(config_path.resolve()),
        "sha256": file_sha256(config_path),
        "method": str(config["method"]),
    }


def main():
    """解析参数、冻结随机样本并写出manifest。

    输入：源目录、输出路径、抽样规模、种子和可重复的排除物体名。
    输出：JSON manifest与终端条目摘要。
    逻辑：按文件名扫描和校验，排除已用于调参的物体，再确定性抽样。
    作用：为下一轮未见小样本验证建立不可随结果更改的输入清单。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--object-count", type=int, default=5)
    parser.add_argument("--trajectories-per-object", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--exclude-object", action="append", default=[])
    parser.add_argument("--exclude-manifest", action="append", type=Path, default=[])
    parser.add_argument("--frozen-method-config", type=Path)
    parser.add_argument(
        "--purpose", default="development_pilot_not_official_submission"
    )
    args = parser.parse_args()

    excluded, excluded_manifests = collect_excluded_objects(
        args.exclude_object, args.exclude_manifest
    )
    source_files = sorted(args.source_dir.glob("*.npy"))
    candidates = [
        inspect_source_file(path)
        for path in source_files
        if path.stem not in excluded
    ]
    entries = sample_manifest(
        candidates,
        args.object_count,
        args.trajectories_per_object,
        args.seed,
    )
    manifest = {
        "purpose": args.purpose,
        "source_directory": str(args.source_dir.resolve()),
        "seed": args.seed,
        "object_count": len(entries),
        "trajectories_per_object": args.trajectories_per_object,
        "trajectory_count": len(entries) * args.trajectories_per_object,
        "excluded_tuning_objects": sorted(excluded),
        "excluded_objects": sorted(excluded),
        "excluded_manifests": excluded_manifests,
        "frozen_method": frozen_method_record(args.frozen_method_config),
        "entries": entries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"object_count={manifest['object_count']}")
    print(f"trajectory_count={manifest['trajectory_count']}")
    for item in entries:
        print(f"{item['object_name']}: {item['trajectory_indices']}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
