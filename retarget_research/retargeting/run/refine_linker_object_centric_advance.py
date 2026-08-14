#!/usr/bin/env python3
"""按目标手抓取中心与物体中心的偏差，渐进修正Linker手腕平移。

输入：冻结manifest、已有Linker 12维候选目录、最大修正距离和输出目录。
输出：仍为70帧的单候选轨迹，以及逐轨迹的方向、距离和不变量审计信息。
内部逻辑：在抬升首帧计算五个目标指尖的平均位置，以物体世界包围盒中心为
目标；把不超过全局上限的三维平移从闭合开始渐进加入，抬升后保持恒定。
作用：补偿不同手型造成的整体掌物错位，同时保留原手指抓形、手腕姿态和完整
抬升相对运动；该规则只看几何，不查询单条轨迹的物理成败。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


RUN_DIR = Path(__file__).resolve().parent
RETARGET_ROOT = RUN_DIR.parent
PREPARE_DIR = RETARGET_ROOT / "prepare"
EVALUATE_DIR = RETARGET_ROOT / "evaluate"
for path in (RUN_DIR, PREPARE_DIR, EVALUATE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from evaluate_linker_geometry import (  # noqa: E402
    build_models,
    compute_linker_points,
    linker_to_model_q,
    load_pairs,
)
from object_geometry import transformed_object_vertices  # noqa: E402
from refine_linker_squeeze import squeeze_progress  # noqa: E402
from slice_manifest_candidates import slice_candidate  # noqa: E402


TIP_SEMANTICS = (
    "index_tip",
    "middle_tip",
    "ring_tip",
    "little_tip",
    "thumb_tip",
)


def bounded_center_correction(
    grasp_center: np.ndarray,
    object_center: np.ndarray,
    max_advance_m: float,
) -> tuple[np.ndarray, dict]:
    """计算指向物体中心、但长度受全局上限约束的三维平移。

    输入：三维抓取中心、三维物体中心和正的最大修正距离（米）。
    输出：三维修正向量与原始距离、实际长度、单位方向等审计字典。
    内部逻辑：距离小于上限时完整对齐，否则只沿中心连线前进上限长度；
    两中心数值重合时返回零向量，避免除零。
    作用：让同一全局超参数自动适配不同大小物体，又不按成败挑选修正量。
    """
    grasp = np.asarray(grasp_center, dtype=np.float32)
    object_ = np.asarray(object_center, dtype=np.float32)
    if grasp.shape != (3,) or object_.shape != (3,):
        raise ValueError(f"中心坐标必须为三维，实际{grasp.shape}与{object_.shape}")
    if not np.isfinite(grasp).all() or not np.isfinite(object_).all():
        raise ValueError("中心坐标包含非有限值")
    if max_advance_m <= 0:
        raise ValueError(f"最大修正距离必须为正数，实际{max_advance_m}")

    residual = object_ - grasp
    original_distance = float(np.linalg.norm(residual))
    if original_distance <= 1e-9:
        direction = np.zeros(3, dtype=np.float32)
        correction = direction.copy()
    else:
        direction = residual / original_distance
        correction = direction * min(float(max_advance_m), original_distance)
    actual_advance = float(np.linalg.norm(correction))
    return correction.astype(np.float32), {
        "grasp_center_xyz_m": grasp.tolist(),
        "object_bbox_center_xyz_m": object_.tolist(),
        "center_distance_before_m": original_distance,
        "advance_direction_xyz": direction.tolist(),
        "requested_max_advance_m": float(max_advance_m),
        "actual_advance_m": actual_advance,
        "center_distance_after_m": max(original_distance - actual_advance, 0.0),
    }


def apply_object_centric_advance(
    frames: np.ndarray,
    close_start_frame: int,
    lift_start_frame: int,
    grasp_center: np.ndarray,
    object_center: np.ndarray,
    max_advance_m: float,
) -> tuple[np.ndarray, dict]:
    """把物体中心指向修正平滑叠加到一条Linker轨迹。

    输入：`(70,12)`动作、闭合/抬升首帧、两个中心和全局距离上限。
    输出：同形状新轨迹及修正幅度、动作保持检查和步长统计。
    内部逻辑：闭合前修正为0，闭合至抬升线性增至完整值，抬升后保持；
    只改前三维平移，不改欧拉角与六个主动关节。
    作用：避免瞬移，同时保证抬升阶段每两个相邻帧的相对位移与原轨迹相同。
    """
    original = np.asarray(frames, dtype=np.float32)
    if original.shape != (70, 12):
        raise ValueError(f"Linker O6候选应为(70,12)，实际{original.shape}")
    close = int(close_start_frame)
    lift = int(lift_start_frame)
    progress = squeeze_progress(len(original), close, lift)
    correction, audit = bounded_center_correction(
        grasp_center, object_center, max_advance_m
    )
    output = original.copy()
    output[:, :3] += progress[:, None] * correction[None, :]
    if not np.isfinite(output).all():
        raise ValueError("物体中心指向修正产生了非有限动作")
    if not np.array_equal(output[:, 3:], original[:, 3:]):
        raise AssertionError("修正意外改变了手腕旋转或手指关节")
    lift_relative_unchanged = np.allclose(
        np.diff(output[lift:, :3], axis=0),
        np.diff(original[lift:, :3], axis=0),
        rtol=0.0,
        atol=1e-7,
    )
    if not lift_relative_unchanged:
        raise AssertionError("修正意外改变了抬升阶段的相对平移")
    old_steps = np.linalg.norm(np.diff(original[:, :3], axis=0), axis=1)
    new_steps = np.linalg.norm(np.diff(output[:, :3], axis=0), axis=1)
    audit.update(
        {
            "close_start_frame": close,
            "lift_start_frame": lift,
            "rotation_and_joint_commands_unchanged": True,
            "lift_relative_translation_unchanged": True,
            "original_max_translation_step_m": float(np.max(old_steps)),
            "corrected_max_translation_step_m": float(np.max(new_steps)),
        }
    )
    return output.astype(np.float32), audit


def target_grasp_center(
    linker_model,
    frames: np.ndarray,
    lift_start_frame: int,
    mapping_semantics: list[str],
) -> np.ndarray:
    """由正向运动学计算抬升首帧的五指抓取中心。

    输入：共享Linker模型、单条12维轨迹、抬升帧和候选语义点顺序。
    输出：五个目标指尖世界坐标的平均值，形状为`(3,)`。
    内部逻辑：复用独立几何评估的校准link局部点，定位五个`*_tip`后求均值。
    作用：用真正包围物体的指尖中心取代位于手后方的腕部根节点。
    """
    missing = [name for name in TIP_SEMANTICS if name not in mapping_semantics]
    if missing:
        raise ValueError(f"候选关键点缺少五指指尖: {missing}")
    pairs = load_pairs(mapping_semantics)
    lift = int(lift_start_frame)
    points = compute_linker_points(
        linker_model,
        linker_to_model_q(frames[lift : lift + 1]),
        pairs,
    )[0]
    indices = [mapping_semantics.index(name) for name in TIP_SEMANTICS]
    return np.mean(points[indices], axis=0).astype(np.float32)


def object_bbox_center(entry: dict, data: dict, position: int) -> np.ndarray:
    """恢复某条数据轨迹对应物体的世界包围盒中心。

    输入：manifest物体条目、已对齐候选字典和候选内位置。
    输出：摆到PhysX初始姿态后的三维AABB中心。
    内部逻辑：用共享几何函数应用数据缩放、旋转和离地间隙，再取顶点极值中点。
    作用：避免用模型原点或网格顶点均值代表物体位置，保持与重放初始化一致。
    """
    object_dir = Path(entry["object_asset_path"])
    vertices = transformed_object_vertices(
        object_dir,
        np.asarray(data["obj_scale"])[position],
        np.asarray(data["obj_rotmat"])[position],
    )
    return ((vertices.min(axis=0) + vertices.max(axis=0)) * 0.5).astype(np.float32)


def refine_manifest(
    manifest: dict,
    input_dir: Path,
    output_dir: Path,
    max_advance_m: float,
) -> dict:
    """对manifest内所有Linker轨迹应用同一个物体中心修正规则。

    输入：manifest、完整或小集候选目录、输出目录和全局距离上限。
    输出：批次摘要，同时写出与manifest严格对齐的每物体npy。
    内部逻辑：按源索引切片候选，逐轨迹恢复物体中心和目标指尖中心，再平滑修正。
    作用：生成可直接进入统一PhysX评测的单方法候选，不做逐轨迹候选并集。
    """
    if max_advance_m <= 0:
        raise ValueError("max_advance_m必须为正数")
    output_dir.mkdir(parents=True, exist_ok=True)
    _, linker_model = build_models("coupled6")
    records = []
    for entry in manifest.get("entries", []):
        object_name = str(entry["object_name"])
        input_path = input_dir / f"{object_name}.npy"
        if not input_path.is_file():
            raise FileNotFoundError(input_path)
        full = np.load(input_path, allow_pickle=True).item()
        indices = [int(value) for value in entry["trajectory_indices"]]
        data = slice_candidate(full, indices, 12)
        frames = np.asarray(data["grasp_seqs"], dtype=np.float32)
        phases = data.get("squeeze_phase_metadata")
        if not isinstance(phases, list) or len(phases) != len(indices):
            raise ValueError(f"{object_name}缺少对齐的squeeze阶段元数据")
        semantics = [str(value) for value in data.get("mapping_semantics", [])]
        outputs, audits = [], []
        for position, source_index in enumerate(indices):
            phase = phases[position]
            lift = int(phase["lift_start_frame"])
            grasp_center = target_grasp_center(
                linker_model, frames[position], lift, semantics
            )
            center = object_bbox_center(entry, data, position)
            output, audit = apply_object_centric_advance(
                frames[position],
                int(phase["close_start_frame"]),
                lift,
                grasp_center,
                center,
                max_advance_m,
            )
            audit["source_trajectory_index"] = source_index
            outputs.append(output)
            audits.append(audit)
            records.append({"object_name": object_name, **audit})
        result = dict(data)
        result.update(
            {
                "grasp_seqs": np.stack(outputs).astype(np.float32),
                "method": "linker_object_centric_grasp_center_advance_v1",
                "object_centric_input": str(input_path.resolve()),
                "object_centric_max_advance_m": float(max_advance_m),
                "object_centric_audit": audits,
            }
        )
        np.save(output_dir / input_path.name, result, allow_pickle=True)
    expected = int(manifest.get("trajectory_count", len(records)))
    if len(records) != expected:
        raise ValueError(f"输出轨迹数{len(records)}与manifest声明{expected}不符")
    return {
        "method": "linker_object_centric_grasp_center_advance_v1",
        "manifest_purpose": manifest.get("purpose"),
        "input_dir": str(input_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "max_advance_m": float(max_advance_m),
        "trajectory_count": len(records),
        "mean_center_distance_before_m": float(
            np.mean([item["center_distance_before_m"] for item in records])
        ),
        "mean_actual_advance_m": float(
            np.mean([item["actual_advance_m"] for item in records])
        ),
        "mean_center_distance_after_m": float(
            np.mean([item["center_distance_after_m"] for item in records])
        ),
        "mean_original_max_translation_step_m": float(
            np.mean([item["original_max_translation_step_m"] for item in records])
        ),
        "mean_corrected_max_translation_step_m": float(
            np.mean([item["corrected_max_translation_step_m"] for item in records])
        ),
        "records": records,
    }


def main() -> None:
    """解析参数，生成物体中心指向候选并保存静态审计摘要。

    输入：`--manifest/--input-dir/--output-dir/--max-advance-mm`。
    输出：候选npy与`object_centric_summary.json`，不启动物理仿真。
    内部逻辑：毫米参数转成米后调用批处理函数，并打印中心距离与实际修正均值。
    作用：为3/6/9 mm小范围筛选提供可重复、秒级的轨迹生成入口。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-advance-mm", type=float, required=True)
    args = parser.parse_args()
    if args.max_advance_mm <= 0:
        parser.error("--max-advance-mm必须为正数")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    summary = refine_manifest(
        manifest,
        args.input_dir,
        args.output_dir,
        args.max_advance_mm / 1000.0,
    )
    summary_path = args.output_dir / "object_centric_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"trajectories={summary['trajectory_count']}")
    print(
        "mean_center_distance_mm="
        f"{1000 * summary['mean_center_distance_before_m']:.3f}->"
        f"{1000 * summary['mean_center_distance_after_m']:.3f}"
    )
    print(f"mean_actual_advance_mm={1000 * summary['mean_actual_advance_m']:.3f}")
    print(f"summary={summary_path.resolve()}")


if __name__ == "__main__":
    main()
