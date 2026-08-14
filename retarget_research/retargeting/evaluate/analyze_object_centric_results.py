#!/usr/bin/env python3
"""联合几何修正审计与PhysX结果，生成物体中心方法的机制诊断报告。

输入：冻结manifest、物体中心候选目录、基线/候选评测摘要和输出前缀。
输出：逐轨迹JSON与便于实验报告引用的中文Markdown摘要。
内部逻辑：严格按物体名和源轨迹索引对齐几何审计与两套物理结果，统计配对
成败、抬升/接触变化、中心误差分组及几何量和物理变化的描述性相关系数。
作用：解释方法在哪些条件下有效或失效；只做离线分析，不改变候选和成功判据。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


EVALUATE_DIR = Path(__file__).resolve().parent
if str(EVALUATE_DIR) not in sys.path:
    sys.path.insert(0, str(EVALUATE_DIR))

from compare_manifest_methods import (  # noqa: E402
    compare_methods,
    manifest_keys,
    select_summary_results,
)


def load_candidate_audits(
    manifest: dict, candidate_dir: Path
) -> dict[tuple[str, int], dict]:
    """读取并索引所有候选中的物体中心修正审计记录。

    输入：冻结manifest和候选npy目录。
    输出：以`(物体名,源轨迹索引)`为键的唯一审计字典。
    内部逻辑：逐物体读取`object_centric_audit`，核对源索引、重复和完整覆盖。
    作用：防止把某条轨迹的修正方向错误关联到另一条物理结果。
    """
    audits = {}
    for entry in manifest.get("entries", []):
        object_name = str(entry["object_name"])
        path = candidate_dir / f"{object_name}.npy"
        if not path.is_file():
            raise FileNotFoundError(path)
        data = np.load(path, allow_pickle=True).item()
        records = data.get("object_centric_audit")
        if not isinstance(records, list):
            raise ValueError(f"{path}缺少object_centric_audit")
        for record in records:
            key = (object_name, int(record["source_trajectory_index"]))
            if key in audits:
                raise ValueError(f"重复的物体中心审计键: {key}")
            audits[key] = record
    expected = set(manifest_keys(manifest))
    if set(audits) != expected:
        missing = sorted(expected - set(audits))
        extra = sorted(set(audits) - expected)
        raise ValueError(f"审计与manifest不一致: missing={missing[:3]} extra={extra[:3]}")
    return audits


def pearson_correlation(x_values: list[float], y_values: list[float]) -> float | None:
    """计算两个等长序列的Pearson相关系数。

    输入：两个数值列表。
    输出：`[-1,1]`相关系数；样本少于2或任一序列无方差时返回None。
    内部逻辑：转成双精度数组后中心化并除以两个L2范数。
    作用：提供描述性趋势，避免NumPy在常量输入时产生NaN警告。
    """
    x = np.asarray(x_values, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError(f"相关输入必须是两个等长一维数组，实际{x.shape}与{y.shape}")
    if len(x) < 2:
        return None
    x = x - x.mean()
    y = y - y.mean()
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denominator <= 1e-15:
        return None
    return float(np.dot(x, y) / denominator)


def summarize_rows(rows: list[dict]) -> dict:
    """把逐轨迹联合记录汇总为分组结果和描述性相关性。

    输入：已经严格对齐且包含中心几何、基线和候选指标的记录列表。
    输出：距离四分组、方向分组、变化均值、相关系数和成败改变案例。
    内部逻辑：按原始中心误差排序等人数分四组；按方向Z分量分为向下、侧向、
    向上；所有物理delta统一定义为候选减基线。
    作用：区分“整体趋势”与“单条越过成功阈值”，为失败分析提供证据。
    """
    if not rows:
        raise ValueError("至少需要一条联合记录")

    def group_summary(items: list[dict]) -> dict:
        """汇总一个非空轨迹组的成功和连续指标变化。

        输入：同一距离或方向分组的联合记录。
        输出：样本数、两方法成功数及三项平均delta。
        内部逻辑：直接对候选减基线字段求均值。
        作用：为JSON和Markdown提供统一分组统计口径。
        """
        return {
            "trajectory_count": len(items),
            "baseline_success_count": sum(item["baseline_success"] for item in items),
            "candidate_success_count": sum(item["candidate_success"] for item in items),
            "mean_delta_max_lift_m": float(
                np.mean([item["delta_max_lift_m"] for item in items])
            ),
            "mean_delta_final_lift_m": float(
                np.mean([item["delta_final_lift_m"] for item in items])
            ),
            "mean_delta_contact_steps": float(
                np.mean([item["delta_contact_steps"] for item in items])
            ),
        }

    ordered = sorted(rows, key=lambda item: item["center_distance_before_m"])
    quartiles = {}
    for number, indices in enumerate(np.array_split(np.arange(len(ordered)), 4), 1):
        items = [ordered[int(index)] for index in indices]
        quartiles[f"Q{number}"] = {
            "center_distance_min_m": min(
                item["center_distance_before_m"] for item in items
            ),
            "center_distance_max_m": max(
                item["center_distance_before_m"] for item in items
            ),
            **group_summary(items),
        }

    direction_groups = {"downward": [], "mostly_lateral": [], "upward": []}
    for item in rows:
        z = item["advance_direction_xyz"][2]
        name = "downward" if z < -1 / 3 else "upward" if z > 1 / 3 else "mostly_lateral"
        direction_groups[name].append(item)

    correlations = {}
    for geometry_field in ("center_distance_before_m", "advance_direction_z"):
        for delta_field in (
            "delta_max_lift_m",
            "delta_final_lift_m",
            "delta_contact_steps",
        ):
            correlations[f"{geometry_field}_vs_{delta_field}"] = pearson_correlation(
                [item[geometry_field] for item in rows],
                [item[delta_field] for item in rows],
            )
    return {
        "overall": group_summary(rows),
        "center_distance_quartiles": quartiles,
        "direction_groups": {
            name: group_summary(items) if items else {"trajectory_count": 0}
            for name, items in direction_groups.items()
        },
        "descriptive_pearson_correlations": correlations,
        "changed_cases": [
            item
            for item in rows
            if item["baseline_success"] != item["candidate_success"]
        ],
    }


def build_analysis(
    manifest: dict,
    candidate_dir: Path,
    baseline_summary: dict,
    candidate_summary: dict,
) -> dict:
    """构建物体中心候选相对基线的完整联合分析。

    输入：manifest、候选目录和两份统一评测摘要。
    输出：配对统计、机制分组摘要及全部可审计逐轨迹记录。
    内部逻辑：按manifest键对齐三方数据，计算候选减基线的物理delta后汇总。
    作用：同一函数可复用A组开发结果和C组独立确认结果。
    """
    keys = manifest_keys(manifest)
    baseline = select_summary_results(baseline_summary, keys)
    candidate = select_summary_results(candidate_summary, keys)
    audits = load_candidate_audits(manifest, candidate_dir)
    rows = []
    for key, base, new in zip(keys, baseline, candidate):
        audit = audits[key]
        direction = [float(value) for value in audit["advance_direction_xyz"]]
        rows.append(
            {
                "object_name": key[0],
                "category": new.get("category"),
                "source_trajectory_index": key[1],
                "center_distance_before_m": float(
                    audit["center_distance_before_m"]
                ),
                "actual_advance_m": float(audit["actual_advance_m"]),
                "advance_direction_xyz": direction,
                "advance_direction_z": direction[2],
                "baseline_success": bool(base["success"]),
                "candidate_success": bool(new["success"]),
                "baseline_max_lift_m": float(base["max_lift_m"]),
                "candidate_max_lift_m": float(new["max_lift_m"]),
                "delta_max_lift_m": float(new["max_lift_m"] - base["max_lift_m"]),
                "baseline_final_lift_m": float(base["final_lift_m"]),
                "candidate_final_lift_m": float(new["final_lift_m"]),
                "delta_final_lift_m": float(
                    new["final_lift_m"] - base["final_lift_m"]
                ),
                "baseline_contact_steps": int(base["hand_object_contact_steps"]),
                "candidate_contact_steps": int(new["hand_object_contact_steps"]),
                "delta_contact_steps": int(
                    new["hand_object_contact_steps"]
                    - base["hand_object_contact_steps"]
                ),
            }
        )
    return {
        "manifest_purpose": manifest.get("purpose"),
        "trajectory_count": len(rows),
        "candidate_dir": str(candidate_dir.resolve()),
        "paired_comparison": compare_methods(
            manifest,
            [("linker_current", baseline_summary), ("object_centric", candidate_summary)],
        ),
        "mechanism_summary": summarize_rows(rows),
        "rows": rows,
        "interpretation_boundary": (
            "相关系数和分组只用于描述机制，不代表因果；最终方法结论以冻结C组的"
            "配对成功净变化为准。"
        ),
    }


def render_markdown(analysis: dict) -> str:
    """把联合分析转换成一页左右的中文Markdown。

    输入：`build_analysis`返回的完整字典。
    输出：含总体结果、连续变化、改变案例和距离分组的Markdown字符串。
    内部逻辑：从配对比较读取成功与精确检验，从机制摘要读取物理delta。
    作用：直接为1–2页实验报告提供可复核文本与表格素材。
    """
    comparison = analysis["paired_comparison"]
    base = comparison["methods"]["linker_current"]
    candidate = comparison["methods"]["object_centric"]
    paired = comparison["comparisons_to_baseline"]["object_centric"]
    mechanism = analysis["mechanism_summary"]
    overall = mechanism["overall"]
    lines = [
        "# 物体中心指向校准诊断",
        "",
        f"样本：{analysis['trajectory_count']}条；基线成功{base['success_count']}条，"
        f"候选成功{candidate['success_count']}条。配对新增{paired['added_success']}条、"
        f"丢失{paired['lost_success']}条、净变化{paired['net_success_change']:+d}条，"
        f"双侧精确检验p={paired['paired_exact_two_sided_p']:.6g}。",
        "",
        "连续指标的候选减基线均值：最大抬升"
        f"{1000 * overall['mean_delta_max_lift_m']:+.2f} mm，最终抬升"
        f"{1000 * overall['mean_delta_final_lift_m']:+.2f} mm，接触步数"
        f"{overall['mean_delta_contact_steps']:+.2f}。",
        "",
        "## 成败发生改变的轨迹",
        "",
        "| 类别 | 物体 | 源轨迹 | 变化 | 最大抬升变化/mm | 接触步变化 |",
        "|---|---|---:|---|---:|---:|",
    ]
    for item in mechanism["changed_cases"]:
        outcome = "新增成功" if item["candidate_success"] else "丢失成功"
        lines.append(
            f"| {item['category']} | {item['object_name']} | "
            f"{item['source_trajectory_index']} | {outcome} | "
            f"{1000 * item['delta_max_lift_m']:+.2f} | "
            f"{item['delta_contact_steps']:+d} |"
        )
    if not mechanism["changed_cases"]:
        lines.append("| — | — | — | 无变化 | — | — |")
    lines.extend(
        [
            "",
            "## 原始中心误差四分组",
            "",
            "| 分组 | 距离范围/mm | 基线成功 | 候选成功 | 最大抬升变化/mm |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, item in mechanism["center_distance_quartiles"].items():
        lines.append(
            f"| {name} | {1000 * item['center_distance_min_m']:.1f}–"
            f"{1000 * item['center_distance_max_m']:.1f} | "
            f"{item['baseline_success_count']}/{item['trajectory_count']} | "
            f"{item['candidate_success_count']}/{item['trajectory_count']} | "
            f"{1000 * item['mean_delta_max_lift_m']:+.2f} |"
        )
    lines.extend(
        [
            "",
            "注：分组和相关性是机制诊断，不用于选择单条轨迹或拼接候选；"
            "是否保留方法仍由冻结确认集的配对净变化决定。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    """解析输入，写出JSON机制审计和中文Markdown报告。

    输入：manifest、候选目录、两份评测摘要及JSON/Markdown输出路径。
    输出：两个报告文件，并打印成功配对和主要连续指标变化。
    内部逻辑：调用联合分析与渲染函数；不启动Isaac Gym或修改原实验文件。
    作用：在A/C物理任务结束后用同一命令快速生成报告素材。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--baseline-summary", type=Path, required=True)
    parser.add_argument("--candidate-summary", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    baseline = json.loads(args.baseline_summary.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate_summary.read_text(encoding="utf-8"))
    analysis = build_analysis(manifest, args.candidate_dir, baseline, candidate)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.output_markdown.write_text(render_markdown(analysis), encoding="utf-8")
    paired = analysis["paired_comparison"]["comparisons_to_baseline"][
        "object_centric"
    ]
    print(
        f"added={paired['added_success']} lost={paired['lost_success']} "
        f"net={paired['net_success_change']:+d} "
        f"exact_p={paired['paired_exact_two_sided_p']:.6g}"
    )
    print(f"json={args.output_json.resolve()}")
    print(f"markdown={args.output_markdown.resolve()}")


if __name__ == "__main__":
    main()
