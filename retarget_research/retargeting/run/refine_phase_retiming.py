#!/usr/bin/env python3
"""为已有目标手轨迹加入闭合后、抬升前的稳定接触时间。

输入：小manifest、已有候选目录、动作维度、稳定帧数和输出目录。
输出：仍为70帧的单候选轨迹，以及每条轨迹的阶段重定时审计信息。
内部逻辑：压缩无接触接近段，提前执行原闭合段，在原lift时刻前重复末闭合姿态；
原lift起始帧及其后全部动作逐元素不变。
作用：让响应较慢或有关节耦合的目标手在抬升前获得额外物理稳定时间，而不延长episode。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


RUN_DIR = Path(__file__).resolve().parent
PREPARE_DIR = RUN_DIR.parent / "prepare"
if str(PREPARE_DIR) not in sys.path:
    sys.path.insert(0, str(PREPARE_DIR))

from slice_manifest_candidates import slice_candidate  # noqa: E402


def resample_segment(segment: np.ndarray, target_count: int) -> np.ndarray:
    """把一段多维动作线性重采样为指定帧数。

    输入：`(T,D)`动作段和正整数目标帧数。
    输出：`(target_count,D)`数组，保留原段首尾值。
    内部逻辑：对归一化时间轴逐维做一维线性插值；单帧目标取原段末帧。
    作用：只压缩尚未接触物体的接近段，避免直接删帧产生额外命令跳变。
    """
    segment = np.asarray(segment, dtype=np.float32)
    if segment.ndim != 2 or len(segment) < 1:
        raise ValueError(f"待重采样动作应为非空(T,D)，实际{segment.shape}")
    if target_count < 1 or target_count > len(segment):
        raise ValueError(
            f"目标帧数应在[1,{len(segment)}]，实际{target_count}"
        )
    if target_count == len(segment):
        return segment.copy()
    if target_count == 1:
        return segment[-1:].copy()
    source_time = np.linspace(0.0, 1.0, len(segment), dtype=np.float64)
    target_time = np.linspace(0.0, 1.0, target_count, dtype=np.float64)
    output = np.stack(
        [np.interp(target_time, source_time, segment[:, column]) for column in range(segment.shape[1])],
        axis=1,
    )
    return output.astype(np.float32)


def add_pre_lift_settle(
    frames: np.ndarray,
    close_start_frame: int,
    lift_start_frame: int,
    settle_frames: int,
) -> tuple[np.ndarray, dict]:
    """在不改变总帧数和lift动作的前提下加入预抬升稳定段。

    输入：`(70,D)`候选、闭合/抬升首帧和要增加的稳定帧数。
    输出：重定时轨迹及阶段、最大步长和逐元素保持检查等审计字典。
    内部逻辑：接近段由`close`帧压缩为`close-settle`帧；原闭合段整体前移；
    空出的帧重复lift前末闭合姿态，随后从原`lift_start_frame`继续完整原轨迹。
    作用：把原本花在远离物体处的控制时间转移到已经闭合、但尚未抬升的阶段。
    """
    frames = np.asarray(frames, dtype=np.float32)
    if frames.ndim != 2 or len(frames) != 70:
        raise ValueError(f"候选轨迹应为(70,D)，实际{frames.shape}")
    close = int(close_start_frame)
    lift = int(lift_start_frame)
    settle = int(settle_frames)
    if not 0 < close <= lift < len(frames):
        raise ValueError(f"阶段应满足0<close<=lift<70，实际{close},{lift}")
    if not 1 <= settle < close:
        raise ValueError(f"稳定帧数应满足1<=settle<close，实际{settle}与close={close}")

    compressed_approach = resample_segment(frames[:close], close - settle)
    close_segment = frames[close:lift].copy()
    # 正常情况下重复lift前一帧，即闭合完成但尚未抬升的姿态。
    # 极少数close==lift的回退轨迹没有显式闭合段，只能提前保持首个完整闭合帧。
    settle_pose_index = lift - 1 if close < lift else lift
    settle_segment = np.repeat(
        frames[settle_pose_index : settle_pose_index + 1], settle, axis=0
    )
    lift_segment = frames[lift:].copy()
    output = np.concatenate(
        [compressed_approach, close_segment, settle_segment, lift_segment], axis=0
    ).astype(np.float32)
    if output.shape != frames.shape or not np.isfinite(output).all():
        raise ValueError(f"阶段重定时结果无效: {output.shape}")
    if not np.array_equal(output[lift:], frames[lift:]):
        raise AssertionError("阶段重定时意外改变了lift及之后的动作")
    old_steps = np.linalg.norm(np.diff(frames, axis=0), axis=1)
    new_steps = np.linalg.norm(np.diff(output, axis=0), axis=1)
    audit = {
        "close_start_frame": close,
        "retimed_close_start_frame": close - settle,
        "lift_start_frame": lift,
        "settle_start_frame": lift - settle,
        "settle_frames": settle,
        "settle_pose_source_frame": settle_pose_index,
        "lift_segment_unchanged": True,
        "original_max_action_step_l2": float(np.max(old_steps)),
        "retimed_max_action_step_l2": float(np.max(new_steps)),
    }
    return output, audit


def phase_metadata(data: dict, trajectory_count: int) -> list[dict]:
    """从不同手的候选协议中读取统一阶段元数据。

    输入：候选字典和预期轨迹数。
    输出：逐轨迹阶段字典列表。
    内部逻辑：优先读取Linker的`squeeze_phase_metadata`，其次读取
    通用中心模块推断的`shared_grasp_center_phase_metadata`，最后读`phase_metadata`。
    作用：让同一个阶段重定时模块后续可用于Linker、XHand和Wuji候选。
    """
    metadata = data.get(
        "squeeze_phase_metadata",
        data.get("shared_grasp_center_phase_metadata", data.get("phase_metadata")),
    )
    if not isinstance(metadata, list) or len(metadata) != trajectory_count:
        raise ValueError("候选缺少与轨迹数一致的阶段元数据")
    return metadata


def retime_manifest(
    manifest: dict,
    input_dir: Path,
    output_dir: Path,
    dimension: int,
    settle_frames: int,
) -> dict:
    """对manifest中所有轨迹应用同一稳定帧数并保存候选。

    输入：manifest、完整候选目录、输出目录、动作维度和全局稳定帧数。
    输出：批次摘要，同时写出与小manifest严格对齐的每物体npy。
    内部逻辑：按源索引切出既有候选，读取各自阶段，再调用纯重定时函数。
    作用：一次生成可公平物理评测的单方法候选，不按物体或成败切换参数。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for entry in manifest.get("entries", []):
        object_name = str(entry["object_name"])
        input_path = input_dir / f"{object_name}.npy"
        if not input_path.is_file():
            raise FileNotFoundError(input_path)
        full = np.load(input_path, allow_pickle=True).item()
        indices = [int(value) for value in entry["trajectory_indices"]]
        data = slice_candidate(full, indices, dimension)
        metadata = phase_metadata(data, len(indices))
        outputs, audits = [], []
        for position, source_index in enumerate(indices):
            phase = metadata[position]
            output, audit = add_pre_lift_settle(
                data["grasp_seqs"][position],
                int(phase["close_start_frame"]),
                int(phase["lift_start_frame"]),
                settle_frames,
            )
            audit["source_trajectory_index"] = source_index
            outputs.append(output)
            audits.append(audit)
            records.append({"object_name": object_name, **audit})
        result = dict(data)
        result.update(
            {
                "grasp_seqs": np.stack(outputs).astype(np.float32),
                "method": "phase_retiming_pre_lift_settle_v1",
                "phase_retiming_input": str(input_path.resolve()),
                "pre_lift_settle_frames": int(settle_frames),
                "phase_retiming_audit": audits,
            }
        )
        np.save(output_dir / input_path.name, result, allow_pickle=True)
    expected = int(manifest.get("trajectory_count", len(records)))
    if len(records) != expected:
        raise ValueError(f"输出轨迹数{len(records)}与manifest声明{expected}不符")
    return {
        "method": "phase_retiming_pre_lift_settle_v1",
        "manifest_purpose": manifest.get("purpose"),
        "input_dir": str(input_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "target_dimension": int(dimension),
        "pre_lift_settle_frames": int(settle_frames),
        "trajectory_count": len(records),
        "mean_original_max_action_step_l2": float(
            np.mean([item["original_max_action_step_l2"] for item in records])
        ),
        "mean_retimed_max_action_step_l2": float(
            np.mean([item["retimed_max_action_step_l2"] for item in records])
        ),
        "records": records,
    }


def main() -> None:
    """解析参数，生成统一阶段重定时候选和审计摘要。

    输入：`--manifest/--input-dir/--output-dir/--dimension/--settle-frames`。
    输出：候选npy与`phase_retiming_summary.json`。
    内部逻辑：检查稳定帧数和维度后调用`retime_manifest`，不启动物理仿真。
    作用：为2/4/6帧少量消融提供秒级轨迹生成入口。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dimension", type=int, required=True)
    parser.add_argument("--settle-frames", type=int, required=True)
    args = parser.parse_args()
    if args.dimension < 1 or args.settle_frames < 1:
        raise ValueError("动作维度和稳定帧数必须为正整数")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    summary = retime_manifest(
        manifest,
        args.input_dir,
        args.output_dir,
        args.dimension,
        args.settle_frames,
    )
    summary_path = args.output_dir / "phase_retiming_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"trajectories={summary['trajectory_count']}")
    print(
        "mean_max_step="
        f"{summary['mean_original_max_action_step_l2']:.6f}->"
        f"{summary['mean_retimed_max_action_step_l2']:.6f}"
    )
    print(f"summary={summary_path.resolve()}")


if __name__ == "__main__":
    main()
