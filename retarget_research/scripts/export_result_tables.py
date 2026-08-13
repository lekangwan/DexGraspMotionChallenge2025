#!/usr/bin/env python3
"""把多份专家或策略评测摘要导出为报告可粘贴的Markdown/CSV表。

输入：一个或多个`标签=summary.json`、主表Markdown/CSV和可选逐类别CSV路径。
输出：成功数、微平均、物体宏平均、类别宏平均、抬升指标及逐类别长表。
内部逻辑：自动兼容专家`manifest_evaluation_summary`和策略`policy_evaluation_summary`字段。
作用：避免1–2页报告最后人工转抄多个大JSON时把分母、宏平均或类别对应写错。
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_summary_argument(value):
    """解析`label=path`参数并拒绝空标签或不存在文件。"""
    if "=" not in value:
        raise ValueError(f"摘要参数必须为label=path: {value}")
    label, path_text = value.split("=", 1)
    path = Path(path_text).resolve()
    if not label.strip() or not path.is_file():
        raise ValueError(f"摘要标签为空或文件不存在: {value}")
    return label.strip(), path


def summary_row(label, path, summary):
    """把专家/策略摘要统一为一行主结果字段。"""
    results = summary.get("results", [])
    mean_max_lift = (
        summary.get("mean_max_lift_m")
        if "mean_max_lift_m" in summary
        else (sum(float(item["max_lift_m"]) for item in results) / len(results) if results else None)
    )
    mean_final_lift = (
        summary.get("mean_final_lift_m")
        if "mean_final_lift_m" in summary
        else (sum(float(item["final_lift_m"]) for item in results) / len(results) if results else None)
    )
    return {
        "label": label,
        "kind": "policy" if "policy_split" in summary else "expert_replay",
        "hand": summary.get("hand", ""),
        "model_type": (results[0].get("model_type", "") if results else ""),
        "trajectory_count": int(summary["trajectory_count"]),
        "success_count": int(summary["success_count"]),
        "trajectory_micro_success_rate": float(
            summary.get("trajectory_micro_success_rate", summary.get("success_rate"))
        ),
        "object_macro_success_rate": float(summary["object_macro_success_rate"]),
        "category_macro_success_rate": float(summary["category_macro_success_rate"]),
        "mean_max_lift_m": None if mean_max_lift is None else float(mean_max_lift),
        "mean_final_lift_m": None if mean_final_lift is None else float(mean_final_lift),
        "source_summary": str(path),
    }


def category_rows(label, summary):
    """把两种摘要中的逐类别成功率统一成长表记录。"""
    if "per_category_success_rate" in summary:
        values = summary["per_category_success_rate"]
        return [
            {"label": label, "category": category, "success_rate": float(rate)}
            for category, rate in sorted(values.items())
        ]
    return [
        {
            "label": label,
            "category": category,
            "success_rate": float(values["success_rate"]),
        }
        for category, values in sorted(summary.get("per_category", {}).items())
    ]


def write_csv(path, rows):
    """以首行字段顺序原子式写CSV。"""
    if not rows:
        raise ValueError(f"没有可写入{path}的记录")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def markdown_table(rows):
    """把主结果行转换成紧凑Markdown百分比表。"""
    lines = [
        "| 方法 | 手 | 模型 | 成功数 | 轨迹微平均 | 物体宏平均 | 类别宏平均 | 平均最大抬升/m |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {label} | {hand} | {model} | {success}/{total} | {micro:.2%} | "
            "{object_macro:.2%} | {category_macro:.2%} | {lift} |".format(
                label=row["label"], hand=row["hand"], model=row["model_type"] or "-",
                success=row["success_count"], total=row["trajectory_count"],
                micro=row["trajectory_micro_success_rate"],
                object_macro=row["object_macro_success_rate"],
                category_macro=row["category_macro_success_rate"],
                lift="-" if row["mean_max_lift_m"] is None else f"{row['mean_max_lift_m']:.4f}",
            )
        )
    return "\n".join(lines) + "\n"


def main():
    """读取全部摘要并写主表、Markdown和可选逐类别表。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", action="append", required=True, help="重复传入label=path")
    parser.add_argument("--output-markdown", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--per-category-csv", type=Path)
    args = parser.parse_args()
    rows, per_category = [], []
    for value in args.summary:
        label, path = parse_summary_argument(value)
        summary = json.loads(path.read_text(encoding="utf-8"))
        rows.append(summary_row(label, path, summary))
        per_category.extend(category_rows(label, summary))
    write_csv(args.output_csv, rows)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.write_text(markdown_table(rows), encoding="utf-8")
    if args.per_category_csv is not None:
        write_csv(args.per_category_csv, per_category)
    print(f"rows={len(rows)} category_rows={len(per_category)}")
    print(f"RESULT_TABLE={args.output_markdown.resolve()}")


if __name__ == "__main__":
    main()
