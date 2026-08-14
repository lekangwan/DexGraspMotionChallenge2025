#!/usr/bin/env python3
"""用已有XHand官方轨迹和接触轨迹生成动态抬升残差候选。

输入：小manifest、完整官方/接触候选目录、残差保留系数和输出目录。
输出：与manifest对齐的18维XHand候选及逐轨迹残差/裁剪审计元数据。
内部逻辑：闭合前沿用接触轨迹；抬升时恢复官方逐帧关节动态并叠加闭合末帧残差。
作用：修复旧接触方法在lift阶段硬冻结12关节的问题，同时完全复用既有昂贵优化结果。
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


def residual_factor(
    frame_index: int,
    lift_start_frame: int,
    retention: float,
    transition_frames: int,
) -> float:
    """计算抬升某帧保留闭合残差的比例。

    输入：当前帧、抬升首帧、最终保留系数和过渡帧数。
    输出：抬升首帧为1、随后线性过渡到`retention`的浮点系数。
    内部逻辑：过渡至少1帧；5帧时系数依次为1、插值三次、最终值。
    作用：避免闭合末帧到抬升首帧因直接缩小残差产生人为关节跳变。
    """
    if frame_index < lift_start_frame:
        return 1.0
    if transition_frames <= 1:
        return float(retention)
    progress = min(
        max(frame_index - lift_start_frame, 0) / float(transition_frames - 1),
        1.0,
    )
    return float(1.0 + progress * (retention - 1.0))


def blend_dynamic_residual(
    official: np.ndarray,
    contact: np.ndarray,
    lift_start_frame: int,
    retention: float,
    transition_frames: int,
    joint_lower: np.ndarray,
    joint_upper: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """合成一条“接触闭合+官方动态抬升”XHand轨迹。

    输入：两条`(70,18)`轨迹、抬升首帧、系数/过渡长度和12关节上下限。
    输出：新轨迹，以及残差范数、最大跳变和被关节限位裁剪数等审计字典。
    内部逻辑：以抬升前一帧的`contact-official`作为残差；lift段逐帧加到官方关节。
    作用：保留接触优化改变的抓形，又不丢失官方轨迹在抬升期的动态包覆能力。
    """
    official = np.asarray(official, dtype=np.float32)
    contact = np.asarray(contact, dtype=np.float32)
    if official.shape != (70, 18) or contact.shape != (70, 18):
        raise ValueError(
            f"XHand单轨迹应均为(70,18)，实际{official.shape}和{contact.shape}"
        )
    if not 1 <= int(lift_start_frame) < len(official):
        raise ValueError(f"lift_start_frame越界: {lift_start_frame}")
    if not 0.0 <= float(retention) <= 1.0:
        raise ValueError(f"残差保留系数必须在[0,1]，实际{retention}")
    if transition_frames < 1:
        raise ValueError("transition_frames必须至少为1")
    joint_lower = np.asarray(joint_lower, dtype=np.float32)
    joint_upper = np.asarray(joint_upper, dtype=np.float32)
    if joint_lower.shape != (12,) or joint_upper.shape != (12,):
        raise ValueError("XHand关节上下限必须各含12项")

    output = contact.copy()
    anchor = int(lift_start_frame) - 1
    residual = contact[anchor, 6:] - official[anchor, 6:]
    clipped_value_count = 0
    factors = []
    for frame_index in range(int(lift_start_frame), len(output)):
        factor = residual_factor(
            frame_index,
            int(lift_start_frame),
            retention,
            transition_frames,
        )
        factors.append(factor)
        output[frame_index, :6] = official[frame_index, :6]
        raw_joints = official[frame_index, 6:] + factor * residual
        clipped_joints = np.clip(raw_joints, joint_lower, joint_upper)
        clipped_value_count += int(np.count_nonzero(raw_joints != clipped_joints))
        output[frame_index, 6:] = clipped_joints
    if not np.isfinite(output).all():
        raise ValueError("动态残差结果含NaN或Inf")
    joint_steps = np.linalg.norm(np.diff(output[:, 6:], axis=0), axis=1)
    audit = {
        "lift_start_frame": int(lift_start_frame),
        "residual_anchor_frame": anchor,
        "anchor_residual_l2_rad": float(np.linalg.norm(residual)),
        "anchor_residual_max_abs_rad": float(np.max(np.abs(residual))),
        "lift_first_residual_factor": float(factors[0]),
        "lift_final_residual_factor": float(factors[-1]),
        "joint_limit_clipped_value_count": clipped_value_count,
        "max_joint_step_l2_rad": float(np.max(joint_steps)),
    }
    return output.astype(np.float32), audit


def load_xhand_joint_limits() -> tuple[np.ndarray, np.ndarray]:
    """从项目现用XHand运动学模型读取12个精确关节限位。

    输入：无命令参数；使用与官方/接触重定向相同的只读模型资产。
    输出：按保存轨迹关节顺序排列的上下限数组。
    内部逻辑：延迟导入几何模块并只构建一次模型，避免纯函数测试依赖模型环境。
    作用：防止“官方动态+接触残差”相加后越过URDF可执行范围。
    """
    evaluate_dir = RUN_DIR.parent / "evaluate"
    if str(evaluate_dir) not in sys.path:
        sys.path.insert(0, str(evaluate_dir))
    from evaluate_xhand_geometry import build_models

    _, xhand_model = build_models()
    lower = xhand_model.revolute_joints_q_lower[0].detach().cpu().numpy()
    upper = xhand_model.revolute_joints_q_upper[0].detach().cpu().numpy()
    if lower.shape != (12,) or upper.shape != (12,):
        raise ValueError(f"XHand模型关节数不是12: {lower.shape}, {upper.shape}")
    return lower.astype(np.float32), upper.astype(np.float32)


def load_aligned_candidate(
    directory: Path, entry: dict, dimension: int = 18
) -> tuple[Path, dict]:
    """加载完整候选并按小manifest源索引切出对应行。

    输入：候选目录、单个manifest条目和期望动作维度。
    输出：原文件路径与已对齐的候选字典。
    内部逻辑：委托通用切片函数按`source_trajectory_indices`查找，而非假定行号相同。
    作用：可直接读取正式1000条产物，不要求预先生成XHand A/B副本。
    """
    path = directory / f"{entry['object_name']}.npy"
    if not path.is_file():
        raise FileNotFoundError(f"XHand候选不存在: {path}")
    data = np.load(path, allow_pickle=True).item()
    indices = [int(value) for value in entry["trajectory_indices"]]
    return path, slice_candidate(data, indices, dimension)


def refine_manifest(
    manifest: dict,
    official_dir: Path,
    contact_dir: Path,
    output_dir: Path,
    retention: float,
    transition_frames: int,
    joint_lower: np.ndarray,
    joint_upper: np.ndarray,
) -> dict:
    """对小manifest全部物体应用同一个动态残差规则。

    输入：manifest、两种既有候选目录、输出目录、全局系数/过渡和关节限位。
    输出：含逐轨迹审计数据的运行摘要，同时写出每物体候选npy。
    内部逻辑：切出对应行，从接触候选读取lift帧，逐轨迹调用纯合成函数。
    作用：以秒级后处理生成统一候选，不针对物体或成败选择不同算法。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for entry in manifest.get("entries", []):
        official_path, official = load_aligned_candidate(official_dir, entry)
        contact_path, contact = load_aligned_candidate(contact_dir, entry)
        official_indices = np.asarray(official["source_trajectory_indices"])
        contact_indices = np.asarray(contact["source_trajectory_indices"])
        if not np.array_equal(official_indices, contact_indices):
            raise ValueError(f"{entry['object_name']}官方与接触候选索引不一致")
        phases = contact.get("phase_metadata")
        if not isinstance(phases, list) or len(phases) != len(official_indices):
            raise ValueError(f"{entry['object_name']}接触候选缺少逐轨迹phase_metadata")

        trajectories, trajectory_audits = [], []
        for position, source_index in enumerate(official_indices):
            phase = phases[position]
            frames, audit = blend_dynamic_residual(
                official["grasp_seqs"][position],
                contact["grasp_seqs"][position],
                int(phase["lift_start_frame"]),
                retention,
                transition_frames,
                joint_lower,
                joint_upper,
            )
            audit["source_trajectory_index"] = int(source_index)
            trajectories.append(frames)
            trajectory_audits.append(audit)

        output = dict(contact)
        output.update(
            {
                "grasp_seqs": np.stack(trajectories).astype(np.float32),
                "method": "xhand_contact_close_official_dynamic_lift_residual_v1",
                "official_candidate": str(official_path.resolve()),
                "contact_candidate": str(contact_path.resolve()),
                "lift_residual_retention": float(retention),
                "lift_residual_transition_frames": int(transition_frames),
                "dynamic_residual_audit": trajectory_audits,
            }
        )
        output_path = output_dir / f"{entry['object_name']}.npy"
        np.save(output_path, output, allow_pickle=True)
        records.extend(
            {
                "object_name": entry["object_name"],
                **audit,
            }
            for audit in trajectory_audits
        )
    trajectory_count = len(records)
    expected = int(manifest.get("trajectory_count", trajectory_count))
    if trajectory_count != expected:
        raise ValueError(f"输出轨迹数{trajectory_count}与manifest声明{expected}不符")
    return {
        "method": "xhand_contact_close_official_dynamic_lift_residual_v1",
        "manifest_purpose": manifest.get("purpose"),
        "official_candidate_dir": str(official_dir.resolve()),
        "contact_candidate_dir": str(contact_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "lift_residual_retention": float(retention),
        "lift_residual_transition_frames": int(transition_frames),
        "trajectory_count": trajectory_count,
        "joint_limit_clipped_value_count": sum(
            item["joint_limit_clipped_value_count"] for item in records
        ),
        "records": records,
    }


def main() -> None:
    """解析参数，加载一次关节限位并生成整个小样本候选目录。

    输入：manifest、官方/接触/输出目录，以及一个全局残差保留系数。
    输出：候选npy和`dynamic_residual_summary.json`。
    内部逻辑：所有物体共享同一系数和过渡长度，严禁按重放结果逐轨迹切换。
    作用：作为XHand A组三系数消融及B组确认的可复现命令行入口。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--official-dir", type=Path, required=True)
    parser.add_argument("--contact-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--retention", type=float, required=True)
    parser.add_argument("--transition-frames", type=int, default=5)
    args = parser.parse_args()
    if not 0.0 <= args.retention <= 1.0:
        raise ValueError("--retention必须在[0,1]")
    if args.transition_frames < 1:
        raise ValueError("--transition-frames必须至少为1")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    lower, upper = load_xhand_joint_limits()
    summary = refine_manifest(
        manifest,
        args.official_dir,
        args.contact_dir,
        args.output_dir,
        args.retention,
        args.transition_frames,
        lower,
        upper,
    )
    summary_path = args.output_dir / "dynamic_residual_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"trajectories={summary['trajectory_count']}")
    print(f"clipped_values={summary['joint_limit_clipped_value_count']}")
    print(f"summary={summary_path.resolve()}")


if __name__ == "__main__":
    main()
