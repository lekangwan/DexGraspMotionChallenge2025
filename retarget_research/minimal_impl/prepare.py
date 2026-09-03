"""从校准集目标轨迹生成某只手的Rank-5协同基底。

输入：校准manifest和已完成Global CEM的目标NPY目录。
输出：一个``(5,F)`` NumPy基底文件。
内部逻辑：只收集manifest指定轨迹，提取闭合/抬升指关节变化并做SVD。
发挥作用：在进入测试集前冻结协同子空间，防止测试泄漏。
"""

import argparse
import json
from pathlib import Path

import numpy as np

from .cem import build_synergy_basis
from .data import load_npy


def trajectories_from_manifest(manifest_path, target_dir):
    """输入manifest和目标目录，输出指定轨迹数组；按source index对齐行，严格限定拟合数据。"""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    trajectories = []
    for entry in manifest["entries"]:
        target = load_npy(Path(target_dir) / f"{entry['object_name']}.npy")
        row_by_source = {int(index): row for row, index in enumerate(target["source_trajectory_indices"])}
        for source_index in entry["trajectory_indices"]:
            trajectories.append(target["grasp_seqs"][row_by_source[int(source_index)]])
    return np.asarray(trajectories, dtype=np.float32)


def main():
    """输入CLI参数，输出冻结基底NPY；提供Rank-5阶段的唯一训练前入口。"""
    parser = argparse.ArgumentParser(description="用校准轨迹拟合Rank-5协同基底")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--target-dir", required=True)
    parser.add_argument("--rank", type=int, default=5)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    trajectories = trajectories_from_manifest(args.manifest, args.target_dir)
    basis = build_synergy_basis(trajectories, args.rank)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, basis)
    print(f"trajectories={len(trajectories)} basis_shape={basis.shape} output={output}")


if __name__ == "__main__":
    main()
