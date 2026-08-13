#!/usr/bin/env python3
"""给Linker O6基线叠加统一的渐进夹紧控制。

输入：冻结manifest、12维O6基线目录、输出目录和全局夹紧/抬升整形参数。
输出：每物体同索引的`(N,70,12)`单候选轨迹及批处理摘要JSON。
内部逻辑：从Shadow指尖接触自动推断闭合/抬升帧；接近阶段不变，闭合阶段
线性增加6维主动关节命令；可在抬升后渐进二次夹紧、设置四指闭合下限，
并统一缩放或截断手腕抬升平移。
作用：不增加自由度、不查询单条物理结果，也不冻结动态抓形，研究低自由度手的夹紧与抬升控制。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

try:
    from .phase_contact import infer_motion_phases
    from .retarget_linker_keypoints import (
        SOURCE_TIP_INDICES,
        build_shadow_model,
        shadow_keypoints,
    )
except ImportError:
    from phase_contact import infer_motion_phases
    from retarget_linker_keypoints import (
        SOURCE_TIP_INDICES,
        build_shadow_model,
        shadow_keypoints,
    )

RETARGET_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = RETARGET_ROOT.parent
PREPARE_DIR = RETARGET_ROOT / "prepare"
import sys

if str(PREPARE_DIR) not in sys.path:
    sys.path.insert(0, str(PREPARE_DIR))

from object_geometry import transformed_object_vertices  # noqa: E402


JOINT_LOWER = np.zeros(6, dtype=np.float32)
JOINT_UPPER = np.asarray([1.36, 0.58, 1.60, 1.60, 1.60, 1.60], dtype=np.float32)
DEFAULT_OBJECT_ROOT = (
    PROJECT_ROOT
    / "reference"
    / "HandRetargetTask2026"
    / "scripts"
    / "data"
    / "sorting"
    / "object_41"
)


def squeeze_progress(frame_count, close_start, lift_start):
    """生成从闭合开始线性增大、抬升后保持为1的夹紧进度。

    输入：总帧数、闭合起始帧和抬升起始帧。
    输出：长度T的`[0,1]`数组。
    内部逻辑：闭合前为0；`close_start..lift_start`线性插值；之后为1。
    若专家首次接触与抬升发生在同一帧，则该帧直接从0切换到1。
    作用：避免在接近物体时提前闭手，同时消除抬升首帧命令跳变。
    """
    if not 0 <= close_start <= lift_start < frame_count:
        raise ValueError(
            f"阶段索引必须满足0<=close<=lift<T，实际{close_start},{lift_start},{frame_count}"
        )
    progress = np.zeros(frame_count, dtype=np.float32)
    if close_start == lift_start:
        progress[lift_start:] = 1.0
        return progress
    progress[close_start : lift_start + 1] = np.linspace(
        0.0, 1.0, lift_start - close_start + 1, dtype=np.float32
    )
    progress[lift_start + 1 :] = 1.0
    return progress


def lift_squeeze_progress(frame_count, lift_start, tighten_frames):
    """生成抬升开始后二次夹紧的渐进系数。

    输入：总帧数、抬升起始帧和达到完整二次夹紧所需帧数。
    输出：长度T的`[0,1]`数组；抬升前为0，随后线性增至1并保持。
    内部逻辑：`tighten_frames=0`表示关闭；正数N在第N个抬升后帧达到1。
    作用：针对物体已抬起但迅速滑落的轨迹逐渐补偿夹持力，避免瞬时加力冲开物体。
    """
    if not 0 <= lift_start < frame_count:
        raise ValueError(f"lift_start越界: {lift_start} vs {frame_count}")
    progress = np.zeros(frame_count, dtype=np.float32)
    if tighten_frames <= 0:
        return progress
    end = min(frame_count - 1, lift_start + int(tighten_frames))
    progress[lift_start : end + 1] = np.linspace(
        0.0, 1.0, end - lift_start + 1, dtype=np.float32
    )
    progress[end + 1 :] = 1.0
    return progress


def shape_lift_translation(frames, lift_start, scale, distance_cap=0.0):
    """按统一比例减慢抬升，并可在指定距离后停止手腕平移。

    输入：`(T,12)`轨迹、抬升起始帧、`(0,1]`位移比例和可选距离上限（米）。
    输出：同形状的新轨迹；抬升前及6维手指命令不变。
    内部逻辑：以抬升起始帧XYZ为锚点，先缩放相对平移；若长度超过上限，
    沿原方向截到上限并在后续保持。手腕姿态和手指命令均不变。
    作用：既能验证减速，也能形成“抬升—停稳”动作，避免手腕持续上升离开滑动物体。
    """
    result = np.asarray(frames, dtype=np.float32).copy()
    if result.ndim != 2 or result.shape[1] != 12:
        raise ValueError(f"Linker O6轨迹应为(T,12)，实际为{result.shape}")
    if not 0 < scale <= 1:
        raise ValueError(f"抬升平移比例必须在(0,1]，实际为{scale}")
    if distance_cap < 0:
        raise ValueError(f"抬升距离上限必须大于等于0，实际为{distance_cap}")
    if not 0 <= lift_start < len(result):
        raise ValueError(f"lift_start越界: {lift_start} vs {len(result)}")
    anchor = result[lift_start, :3].copy()
    offsets = scale * (result[lift_start:, :3] - anchor)
    if distance_cap > 0:
        lengths = np.linalg.norm(offsets, axis=1)
        factors = np.minimum(1.0, distance_cap / np.maximum(lengths, 1e-12))
        offsets *= factors[:, None]
    result[lift_start:, :3] = anchor + offsets
    return result


def apply_squeeze(
    frames,
    close_start,
    lift_start,
    thumb_yaw_delta=0.0,
    thumb_pitch_delta=0.0,
    finger_delta=0.0,
    lift_thumb_yaw_delta=0.0,
    lift_thumb_pitch_delta=0.0,
    lift_finger_delta=0.0,
    lift_tighten_frames=0,
    lift_translation_scale=1.0,
    lift_finger_floor=0.0,
    lift_translation_cap=0.0,
    joint_residuals=None,
):
    """在12维O6轨迹的6个主动角上叠加渐进夹紧量。

    输入：`(T,12)`基线、阶段帧、闭合期增量，以及可选抬升期二次增量/时长/平移比例。
    输出：同形状新轨迹及每关节被限位截断的帧数。
    内部逻辑：关节增量为进度乘`[thumb_yaw, thumb_pitch, finger×4]`；若设置
    四指闭合下限，则只给抬升开始时低于该值的手指补足差值，最后按真实限位裁剪。
    作用：保留基线逐帧包覆变化，只增加一个全局、可部署的闭合残差控制器。
    """
    result = shape_lift_translation(
        frames, lift_start, lift_translation_scale, lift_translation_cap
    )
    if result.ndim != 2 or result.shape[1] != 12:
        raise ValueError(f"Linker O6轨迹应为(T,12)，实际为{result.shape}")
    close_progress = squeeze_progress(len(result), close_start, lift_start)
    close_delta = np.asarray(
        [thumb_yaw_delta, thumb_pitch_delta] + [finger_delta] * 4,
        dtype=np.float32,
    )
    if joint_residuals is not None:
        joint_residuals = np.asarray(joint_residuals, dtype=np.float32)
        if joint_residuals.shape != (6,):
            raise ValueError("joint_residuals必须恰好为6维")
        close_delta = close_delta + joint_residuals
    lift_progress = lift_squeeze_progress(
        len(result), lift_start, lift_tighten_frames
    )
    lift_delta = np.asarray(
        [lift_thumb_yaw_delta, lift_thumb_pitch_delta] + [lift_finger_delta] * 4,
        dtype=np.float32,
    )
    raw = (
        result[:, 6:]
        + close_progress[:, None] * close_delta[None, :]
        + lift_progress[:, None] * lift_delta[None, :]
    )
    if lift_finger_floor > 0:
        floor_delta = np.maximum(
            float(lift_finger_floor) - raw[lift_start, 2:], 0.0
        )
        raw[:, 2:] += lift_progress[:, None] * floor_delta[None, :]
    clipped = np.clip(raw, JOINT_LOWER, JOINT_UPPER)
    clipped_counts = np.sum(np.abs(raw - clipped) > 1e-7, axis=0).astype(int)
    result[:, 6:] = clipped
    return result, clipped_counts


def infer_trajectory_phases(
    source_frames,
    shadow_model,
    object_vertices,
    contact_threshold,
    min_contact_tips,
    lift_delta,
):
    """从单条Shadow轨迹和物体表面推断夹紧开始/抬升帧。

    输入：加过Z偏移的28维源轨迹、Shadow模型、物体顶点和阶段阈值。
    输出：`infer_motion_phases`阶段字典。
    内部逻辑：正向运动学取五指指尖，再以至少若干指进入表面阈值判闭合。
    作用：让统一夹紧方法适用于不同轨迹，而不固定写死第28/37帧。
    """
    points = shadow_keypoints(source_frames, shadow_model)
    tips = {
        semantic: points[:, point_index, :]
        for semantic, point_index in SOURCE_TIP_INDICES.items()
    }
    return infer_motion_phases(
        source_frames,
        tips,
        object_vertices,
        contact_threshold,
        min_contact_tips,
        lift_delta,
    )


def refine_entry(entry, args, shadow_model):
    """处理manifest中一个物体的全部冻结轨迹。

    输入：manifest条目、命令行参数和共享Shadow模型。
    输出：包含物体名、输出路径、阶段和限位统计的摘要字典。
    内部逻辑：核对源索引与12维基线，逐轨迹恢复物体表面、推断阶段并夹紧。
    作用：保证输出顺序与冻结manifest严格一致，供统一物理评估直接读取。
    """
    started = time.perf_counter()
    source_path = Path(entry["source_path"])
    baseline_path = args.baseline_dir / f"{entry['object_name']}.npy"
    if not source_path.is_file() or not baseline_path.is_file():
        raise FileNotFoundError(f"缺少源或O6基线: {source_path}, {baseline_path}")
    source = np.load(source_path, allow_pickle=True).item()
    baseline = np.load(baseline_path, allow_pickle=True).item()
    indices = np.asarray(entry["trajectory_indices"], dtype=np.int64)
    actual_indices = np.asarray(baseline["source_trajectory_indices"], dtype=np.int64)
    if not np.array_equal(indices, actual_indices):
        raise ValueError(f"{entry['object_name']}基线索引与manifest不一致")
    frames = np.asarray(baseline["grasp_seqs"], dtype=np.float32)
    if frames.shape != (len(indices), 70, 12):
        raise ValueError(f"{entry['object_name']} O6基线形状错误: {frames.shape}")
    object_dir = Path(
        entry.get("object_asset_path", args.object_root / entry["object_name"])
    )
    outputs, phase_metadata, clipped_counts = [], [], []
    for local_index, source_index in enumerate(indices):
        source_frames = np.asarray(
            source["grasp_seqs"][source_index], dtype=np.float32
        ).copy()
        source_frames[:, 2] += args.source_z_offset
        vertices = transformed_object_vertices(
            object_dir,
            np.asarray(source["obj_scale"])[source_index],
            np.asarray(source["obj_rotmat"])[source_index],
            args.object_clearance,
        )
        phases = infer_trajectory_phases(
            source_frames,
            shadow_model,
            vertices,
            args.contact_threshold,
            args.min_contact_tips,
            args.lift_delta,
        )
        refined, clipped = apply_squeeze(
            frames[local_index],
            phases["close_start_frame"],
            phases["lift_start_frame"],
            args.thumb_yaw_delta,
            args.thumb_pitch_delta,
            args.finger_delta,
            args.lift_thumb_yaw_delta,
            args.lift_thumb_pitch_delta,
            args.lift_finger_delta,
            args.lift_tighten_frames,
            args.lift_translation_scale,
            args.lift_finger_floor,
            args.lift_translation_cap,
            args.joint_residuals,
        )
        outputs.append(refined)
        clipped_counts.append(clipped)
        phase_metadata.append(
            {
                "source_trajectory_index": int(source_index),
                "close_start_frame": int(phases["close_start_frame"]),
                "lift_start_frame": int(phases["lift_start_frame"]),
                "grasp_frame": int(phases["grasp_frame"]),
                "joint_clipped_frame_counts": clipped.tolist(),
            }
        )
    result = dict(baseline)
    result.update(
        {
            "grasp_seqs": np.stack(outputs).astype(np.float32),
            "method": args.method_name,
            "initial_target": str(baseline_path.resolve()),
            "thumb_yaw_delta": float(args.thumb_yaw_delta),
            "thumb_pitch_delta": float(args.thumb_pitch_delta),
            "finger_delta": float(args.finger_delta),
            "lift_thumb_yaw_delta": float(args.lift_thumb_yaw_delta),
            "lift_thumb_pitch_delta": float(args.lift_thumb_pitch_delta),
            "lift_finger_delta": float(args.lift_finger_delta),
            "lift_tighten_frames": int(args.lift_tighten_frames),
            "lift_translation_scale": float(args.lift_translation_scale),
            "lift_finger_floor": float(args.lift_finger_floor),
            "lift_translation_cap": float(args.lift_translation_cap),
            "joint_residuals": [float(value) for value in args.joint_residuals],
            "contact_threshold": float(args.contact_threshold),
            "min_contact_tips": int(args.min_contact_tips),
            "lift_delta": float(args.lift_delta),
            "squeeze_phase_metadata": phase_metadata,
        }
    )
    output_path = args.output_dir / f"{entry['object_name']}.npy"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, result, allow_pickle=True)
    return {
        "object_name": entry["object_name"],
        "trajectory_count": len(indices),
        "output": str(output_path.resolve()),
        "phase_metadata": phase_metadata,
        "total_clipped_frames_per_joint": np.sum(clipped_counts, axis=0).tolist(),
        "elapsed_seconds": time.perf_counter() - started,
    }


def main():
    """解析参数并对整个冻结manifest生成单一夹紧候选。

    输入：manifest/基线/输出目录、统一夹紧量和阶段阈值。
    输出：每物体npy与`manifest_run_summary.json`。
    内部逻辑：共享一个Shadow模型，顺序处理物体并记录每条轨迹的阶段/限位。
    作用：作为低成本、无物理查询的Linker O6夹紧实验标准入口。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--method-name",
        default="linker_o6_dynamic_squeeze_v1",
        help="写入候选和汇总的可审计方法名；只影响元数据，不改变轨迹数值",
    )
    parser.add_argument("--object-root", type=Path, default=DEFAULT_OBJECT_ROOT)
    parser.add_argument("--source-z-offset", type=float, default=0.4)
    parser.add_argument("--object-clearance", type=float, default=0.005)
    parser.add_argument("--thumb-yaw-delta", type=float, default=0.0)
    parser.add_argument("--thumb-pitch-delta", type=float, default=0.0)
    parser.add_argument("--finger-delta", type=float, default=0.0)
    parser.add_argument("--lift-thumb-yaw-delta", type=float, default=0.0)
    parser.add_argument("--lift-thumb-pitch-delta", type=float, default=0.0)
    parser.add_argument("--lift-finger-delta", type=float, default=0.0)
    parser.add_argument("--lift-tighten-frames", type=int, default=0)
    parser.add_argument("--lift-translation-scale", type=float, default=1.0)
    parser.add_argument(
        "--lift-translation-cap",
        type=float,
        default=0.0,
        help="抬升后手腕相对平移的最大距离（米）；0表示不限制",
    )
    parser.add_argument(
        "--lift-finger-floor",
        type=float,
        default=0.0,
        help="抬升期四指主动角的统一最低目标；0表示关闭",
    )
    parser.add_argument(
        "--joint-residuals",
        type=float,
        nargs=6,
        default=[0.0] * 6,
        metavar=("TY", "TP", "INDEX", "MIDDLE", "RING", "LITTLE"),
        help="额外叠加到闭合阶段的有符号6维主动关节残差",
    )
    parser.add_argument("--contact-threshold", type=float, default=0.02)
    parser.add_argument("--min-contact-tips", type=int, default=2)
    parser.add_argument("--lift-delta", type=float, default=0.03)
    args = parser.parse_args()
    deltas = [
        args.thumb_yaw_delta,
        args.thumb_pitch_delta,
        args.finger_delta,
        args.lift_thumb_yaw_delta,
        args.lift_thumb_pitch_delta,
        args.lift_finger_delta,
    ]
    if min(deltas) < 0 or args.lift_tighten_frames < 0 or args.lift_finger_floor < 0:
        parser.error("夹紧增量和抬升加力帧数必须大于等于0")
    if args.lift_finger_floor > 0 and args.lift_tighten_frames == 0:
        parser.error("设置--lift-finger-floor时必须同时设置--lift-tighten-frames")
    if not 0 < args.lift_translation_scale <= 1:
        parser.error("--lift-translation-scale必须在(0,1]范围内")
    if args.lift_translation_cap < 0:
        parser.error("--lift-translation-cap必须大于等于0")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    shadow_model = build_shadow_model()
    started = time.perf_counter()
    results = []
    for entry in manifest["entries"]:
        item = refine_entry(entry, args, shadow_model)
        results.append(item)
        print(
            f"{item['object_name']}: trajectories={item['trajectory_count']} "
            f"time={item['elapsed_seconds']:.2f}s",
            flush=True,
        )
    summary = {
        "method": args.method_name,
        "manifest": str(args.manifest.resolve()),
        "baseline_dir": str(args.baseline_dir.resolve()),
        "object_count": len(results),
        "trajectory_count": sum(item["trajectory_count"] for item in results),
        "thumb_yaw_delta": args.thumb_yaw_delta,
        "thumb_pitch_delta": args.thumb_pitch_delta,
        "finger_delta": args.finger_delta,
        "lift_thumb_yaw_delta": args.lift_thumb_yaw_delta,
        "lift_thumb_pitch_delta": args.lift_thumb_pitch_delta,
        "lift_finger_delta": args.lift_finger_delta,
        "lift_tighten_frames": args.lift_tighten_frames,
        "lift_translation_scale": args.lift_translation_scale,
        "lift_finger_floor": args.lift_finger_floor,
        "lift_translation_cap": args.lift_translation_cap,
        "joint_residuals": args.joint_residuals,
        "contact_threshold": args.contact_threshold,
        "min_contact_tips": args.min_contact_tips,
        "lift_delta": args.lift_delta,
        "wall_time_seconds": time.perf_counter() - started,
        "results": results,
    }
    summary_path = args.output_dir / "manifest_run_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"output={summary_path}")


if __name__ == "__main__":
    main()
