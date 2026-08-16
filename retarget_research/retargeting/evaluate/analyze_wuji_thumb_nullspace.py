#!/usr/bin/env python3
"""汇总Wuji拇指零空间候选的手型变化和指尖保持误差。

输入：冻结manifest、候选目录和输出目录。
输出：全局/逐轨迹拇指末节角度、近90度比例、指尖偏移JSON及中文表格。
内部逻辑：按manifest严格核对每个npy的源索引与方法名，再读取保存顺序中
`finger1_joint4`及生成阶段独立重算的指尖偏移，不读取物理成功标签。
作用：把“拇指看起来是否仍长期折叠”变成可复现的数值门，并与PhysX成功率分开。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


METHOD = "point_baseline_thumb_tip_nullspace_v1"
SAVED_THUMB_JOINT4_INDEX = 9


def angle_statistics(values_deg):
    """输入角度数组，输出分位数、近90度比例和极值字典。"""
    values = np.asarray(values_deg, dtype=np.float64).reshape(-1)
    quantiles = np.quantile(values, [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    return {
        "count": int(len(values)),
        "minimum_deg": float(quantiles[0]),
        "p10_deg": float(quantiles[1]),
        "p25_deg": float(quantiles[2]),
        "median_deg": float(quantiles[3]),
        "p75_deg": float(quantiles[4]),
        "p90_deg": float(quantiles[5]),
        "maximum_deg": float(quantiles[6]),
        "near_85_to_95_ratio": float(np.mean((values >= 85.0) & (values <= 95.0))),
        "at_or_above_92_ratio": float(np.mean(values >= 92.0)),
    }


def displacement_statistics(values_m):
    """输入指尖偏移米数组，输出平均、95分位和最大毫米值。"""
    millimeters = np.asarray(values_m, dtype=np.float64).reshape(-1) * 1000.0
    return {
        "mean_mm": float(np.mean(millimeters)),
        "p95_mm": float(np.quantile(millimeters, 0.95)),
        "maximum_mm": float(np.max(millimeters)),
    }


def analyze(manifest_path, candidate_dir):
    """核对manifest中所有候选，返回全局和逐轨迹手型统计。"""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    all_angles, all_errors, rows = [], [], []
    for entry in manifest["entries"]:
        path = Path(candidate_dir) / f"{entry['object_name']}.npy"
        data = np.load(path, allow_pickle=True).item()
        expected = np.asarray(entry["trajectory_indices"], dtype=np.int64)
        actual = np.asarray(data["source_trajectory_indices"], dtype=np.int64)
        if not np.array_equal(actual, expected):
            raise ValueError(f"候选索引与manifest不一致: {entry['object_name']}")
        if data.get("retarget_method") != METHOD:
            raise ValueError(f"候选方法错误: {entry['object_name']}")
        frames = np.asarray(data["grasp_seqs"], dtype=np.float64)
        errors = np.asarray(data["thumb_tip_displacement_m_per_frame"], dtype=np.float64)
        if frames.shape != (len(expected), 70, 26) or errors.shape != (len(expected), 70):
            raise ValueError(f"候选或指尖偏移形状错误: {entry['object_name']}")
        for local_index, source_index in enumerate(expected):
            angles = np.degrees(frames[local_index, :, SAVED_THUMB_JOINT4_INDEX])
            item_errors = errors[local_index]
            rows.append({
                "object_name": entry["object_name"],
                "source_trajectory_index": int(source_index),
                "angles": angle_statistics(angles),
                "terminal_angles": angle_statistics(angles[-10:]),
                "thumb_tip_displacement": displacement_statistics(item_errors),
            })
            all_angles.append(angles)
            all_errors.append(item_errors)
    angles = np.concatenate(all_angles)
    errors = np.concatenate(all_errors)
    return {
        "schema_version": 1,
        "method": METHOD,
        "manifest": str(Path(manifest_path).resolve()),
        "candidate_dir": str(Path(candidate_dir).resolve()),
        "trajectory_count": len(rows),
        "all_frames": angle_statistics(angles),
        "terminal_10_frames": angle_statistics(
            np.concatenate([
                np.asarray(values)[-10:] for values in all_angles
            ])
        ),
        "thumb_tip_displacement": displacement_statistics(errors),
        "results": rows,
    }


def write_markdown(path, summary):
    """输入统计字典，输出便于审查的简明中文Markdown。"""
    all_frames = summary["all_frames"]
    terminal = summary["terminal_10_frames"]
    displacement = summary["thumb_tip_displacement"]
    lines = [
        "# Wuji拇指零空间手型审计",
        "",
        f"- 轨迹：{summary['trajectory_count']}",
        f"- 全帧joint4中位角：{all_frames['median_deg']:.2f}度",
        f"- 全帧85–95度比例：{all_frames['near_85_to_95_ratio']:.2%}",
        f"- 末10帧85–95度比例：{terminal['near_85_to_95_ratio']:.2%}",
        f"- 指尖偏移平均 / P95 / 最大：{displacement['mean_mm']:.3f} / {displacement['p95_mm']:.3f} / {displacement['maximum_mm']:.3f} mm",
        "",
        "|物体|源索引|joint4中位角|近90度比例|指尖最大偏移(mm)|",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary["results"]:
        lines.append(
            f"|{row['object_name']}|{row['source_trajectory_index']}|"
            f"{row['angles']['median_deg']:.2f}|"
            f"{row['angles']['near_85_to_95_ratio']:.2%}|"
            f"{row['thumb_tip_displacement']['maximum_mm']:.3f}|"
        )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    """解析路径，执行拇指手型审计并写入JSON/Markdown。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = analyze(args.manifest, args.candidate_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "wuji_thumb_nullspace_summary.json"
    md_path = args.output_dir / "WUJI_THUMB_NULLSPACE_RESULTS.md"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(md_path, summary)
    print(f"trajectories={summary['trajectory_count']}")
    print(f"median_joint4_deg={summary['all_frames']['median_deg']:.3f}")
    print(f"near_90_ratio={summary['all_frames']['near_85_to_95_ratio']:.4f}")
    print(f"max_tip_displacement_mm={summary['thumb_tip_displacement']['maximum_mm']:.3f}")
    print(f"output={args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
