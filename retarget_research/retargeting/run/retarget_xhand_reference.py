#!/usr/bin/env python3
"""用参考XHand优化器处理明确指定的Shadow轨迹索引。

输入：单个Shadow `.npy`、轨迹索引、输出路径和参考优化参数。
输出：`(N,70,18)` XHand候选及源索引、方法配置等可审计元数据。
内部逻辑：把指定轨迹写入临时子集，直接调用只读参考模块的`opt_run`，再规范化输出。
作用：弥补官方脚本只能扫描整目录且不保存原索引的问题，使其可接入冻结manifest。
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import tempfile

import numpy as np


RUN_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = RUN_DIR.parents[1]
REFERENCE_SCRIPTS = (
    PROJECT_ROOT / "reference" / "HandRetargetTask2026" / "scripts"
)
THIRD_PARTY_PK = (
    PROJECT_ROOT
    / "reference"
    / "HandRetargetTask2026"
    / "third_party"
    / "pytorch_kinematics"
)


def select_source_trajectories(source_data, indices):
    """从源字典抽取指定轨迹及其逐轨迹物体字段。

    输入：含`grasp_seqs/obj_rotmat/obj_scale`的源字典和索引列表。
    输出：只含选中轨迹、顺序与索引一致的新字典。
    内部逻辑：明确选择三个参考优化器必需字段，不复制无关大对象。
    作用：让官方优化器只处理冻结样本，并保持物体姿态和缩放一一对应。
    """
    indices = np.asarray(indices, dtype=np.int64)
    return {
        "grasp_seqs": np.asarray(source_data["grasp_seqs"])[indices].copy(),
        "obj_rotmat": np.asarray(source_data["obj_rotmat"])[indices].copy(),
        "obj_scale": np.asarray(source_data["obj_scale"])[indices].copy(),
    }


def load_reference_module():
    """在参考脚本要求的工作目录中导入XHand优化模块。

    输入：本项目固定的只读参考仓库路径。
    输出：`retarget_shadow2xhand_hm_new` Python模块。
    内部逻辑：加入scripts和第三方运动学路径，并切换cwd以解析其相对assets路径。
    作用：复用官方核心算法而不修改官方源码，也不依赖用户启动命令所在目录。
    """
    for path in (REFERENCE_SCRIPTS, THIRD_PARTY_PK):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    os.chdir(REFERENCE_SCRIPTS)
    import retarget_shadow2xhand_hm_new as reference

    return reference


def retarget_file(args):
    """抽取输入、调用参考优化器并保存规范XHand候选。

    输入：source/output、轨迹索引及参考优化器全部相关参数。
    输出：目标npy；Python层无返回值。
    内部逻辑：临时目录隔离输入输出，参考结果通过形状检查后补充源索引和方法元数据。
    作用：把参考单文件算法变成安全、可复现的本项目run入口。
    """
    source = args.source.resolve()
    output = args.output.resolve()
    source_data = np.load(source, allow_pickle=True).item()
    indices = args.trajectory_indices or [0]
    subset = select_source_trajectories(source_data, indices)
    object_name = args.object_name or source.stem
    reference = load_reference_module()
    with tempfile.TemporaryDirectory(prefix="xhand_retarget_") as directory:
        temporary = Path(directory)
        input_dir = temporary / "input"
        output_dir = temporary / "output"
        input_dir.mkdir()
        output_dir.mkdir()
        np.save(input_dir / f"{object_name}.npy", subset, allow_pickle=True)
        reference_args = argparse.Namespace(
            obj_name=object_name,
            seq_dir=str(input_dir),
            obj_dir=str(args.object_dir.resolve()),
            save_path=str(output_dir),
            code_name_file="",
            mesh_root="",
            sample_frame_num=args.sample_frame_num,
            select_gap=1,
            iter_num=args.iter_num,
            trans_lr=args.trans_lr,
            ang_lr=args.ang_lr,
            trans_bound=args.trans_bound,
            enlarge_scale=args.enlarge_scale,
            html_save=False,
            device=args.device,
        )
        reference.opt_run(reference_args, object_name)
        reference_path = output_dir / f"{object_name}.npy"
        if not reference_path.is_file():
            raise RuntimeError(f"参考XHand优化器未产生输出: {reference_path}")
        result = np.load(reference_path, allow_pickle=True).item()
    frames = np.asarray(result["grasp_seqs"])
    expected_shape = (len(indices), 70, 18)
    if frames.shape != expected_shape or not np.isfinite(frames).all():
        raise ValueError(f"XHand参考候选无效: {frames.shape} vs {expected_shape}")
    result.update(
        {
            "source_path": str(source),
            "source_trajectory_indices": np.asarray(indices, dtype=np.int64),
            "reference_script": str(
                (REFERENCE_SCRIPTS / "retarget_shadow2xhand_hm_new.py").resolve()
            ),
            "iter_num": int(args.iter_num),
            "sample_frame_num": int(args.sample_frame_num),
            "trans_lr": float(args.trans_lr),
            "ang_lr": float(args.ang_lr),
            "trans_bound": float(args.trans_bound),
            "enlarge_scale": float(args.enlarge_scale),
            "source_z_offset": 0.4,
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, result, allow_pickle=True)
    print(f"trajectories={frames.shape[0]}")
    print(f"frames={frames.shape[1]}")
    print(f"output_shape={frames.shape}")
    print(f"output={output}")


def main():
    """解析参数并运行指定索引的参考XHand重定向。

    输入：命令行源/输出/索引及参考迭代、插值和边界参数。
    输出：规范候选npy和终端形状摘要。
    内部逻辑：仅组装显式参数并交给`retarget_file`。
    作用：作为XHand冻结manifest批处理调用的单文件入口。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trajectory-indices", type=int, nargs="*")
    parser.add_argument("--object-name", type=str)
    parser.add_argument("--iter-num", type=int, default=100)
    parser.add_argument("--sample-frame-num", type=int, default=5)
    parser.add_argument("--trans-lr", type=float, default=5e-3)
    parser.add_argument("--ang-lr", type=float, default=1e-2)
    parser.add_argument("--trans-bound", type=float, default=2.0)
    parser.add_argument("--enlarge-scale", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--object-dir",
        type=Path,
        default=REFERENCE_SCRIPTS / "data" / "sorting" / "object_41",
    )
    retarget_file(parser.parse_args())


if __name__ == "__main__":
    main()
