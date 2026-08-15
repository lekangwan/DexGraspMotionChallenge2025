#!/usr/bin/env python3
"""把三手正式1000条结果导出为最终Markdown和CSV表。

输入：项目内五份正式评测摘要与Linker新旧方法配对报告。
输出：总结果Markdown、总结果CSV和逐类别CSV。
内部逻辑：按冻结方法合同读取摘要，统一提取成功率、抬升和类别统计，
并在Markdown中明确区分最终方法、历史基线和负消融。
作用：避免Linker方法升级后手工维护三张表而产生数字或标签不一致。
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


METHOD_SPECS = (
    (
        "Linker 3 mm最终方法",
        "selected",
        "linker",
        "outputs/formal_1000/linker_object_centric_3mm_v1_evaluation/manifest_evaluation_summary.json",
    ),
    (
        "Linker渐进夹紧基线",
        "historical_baseline",
        "linker",
        "outputs/formal_1000/linker_o6_optimized_v2_evaluation/manifest_evaluation_summary.json",
    ),
    (
        "Wuji v1最终方法",
        "selected",
        "wuji",
        "outputs/formal_1000/wuji_v1_evaluation/manifest_evaluation_summary.json",
    ),
    (
        "XHand官方最终方法",
        "selected",
        "xhand",
        "outputs/formal_1000/xhand_official_evaluation/manifest_evaluation_summary.json",
    ),
    (
        "XHand指腹细化负消融",
        "rejected_ablation",
        "xhand",
        "outputs/formal_1000/xhand_phase_contact_v2_evaluation/manifest_evaluation_summary.json",
    ),
)


def load_method_rows(project_root: Path) -> list[dict]:
    """读取冻结清单中的五份正式摘要。

    输入：`retarget_research`目录。
    输出：带标签、角色、相对路径和原摘要的行字典列表。
    内部逻辑：逐一检查文件存在、轨迹数为1000且成功数合法。
    作用：让后续三种导出共享完全相同的数据源和方法顺序。
    """
    rows = []
    for label, role, hand, relative in METHOD_SPECS:
        path = project_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"正式摘要不存在: {path}")
        summary = json.loads(path.read_text(encoding="utf-8"))
        count = int(summary["trajectory_count"])
        success = int(summary["success_count"])
        if count != 1000 or not 0 <= success <= count:
            raise ValueError(f"正式摘要数量无效: {label} {success}/{count}")
        rows.append(
            {
                "label": label,
                "role": role,
                "hand": hand,
                "source_summary": str(Path("retarget_research") / relative),
                "summary": summary,
            }
        )
    return rows


def render_markdown(rows: list[dict], linker_pair: dict) -> str:
    """生成包含最终选择边界的Markdown总表。

    输入：统一方法行和Linker新旧配对报告。
    输出：完整UTF-8 Markdown文本。
    内部逻辑：渲染成功率/抬升表，并追加Linker新增、回退和显著性说明。
    作用：报告既呈现最好数值，也不会隐去小幅提升的不确定性。
    """
    lines = [
        "# 三手重定向正式1000轨迹结果",
        "",
        "| 方法 | 角色 | 手 | 成功数 | 轨迹微平均 | 物体宏平均 | 类别宏平均 | 平均最大/最终抬升 |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    role_names = {
        "selected": "最终方法",
        "historical_baseline": "历史基线",
        "rejected_ablation": "负消融",
    }
    for row in rows:
        summary = row["summary"]
        lines.append(
            "| {label} | {role} | {hand} | {success}/{count} | {micro:.2%} | "
            "{obj:.2%} | {cat:.2%} | {max_lift:.1f}/{final_lift:.1f} mm |".format(
                label=row["label"],
                role=role_names[row["role"]],
                hand=row["hand"],
                success=int(summary["success_count"]),
                count=int(summary["trajectory_count"]),
                micro=float(summary["trajectory_micro_success_rate"]),
                obj=float(summary["object_macro_success_rate"]),
                cat=float(summary["category_macro_success_rate"]),
                max_lift=1000 * float(summary["mean_max_lift_m"]),
                final_lift=1000 * float(summary["mean_final_lift_m"]),
            )
        )
    comparison = linker_pair["comparisons_to_baseline"]["object_centric_3mm"]
    lines.extend(
        [
            "",
            "最终提交的三只手方法是Linker 3 mm、Wuji v1和XHand官方参考。"
            "Linker 3 mm相对渐进夹紧基线从231/1000升至234/1000，"
            f"逐轨迹新增{comparison['added_success']}、丢失{comparison['lost_success']}，"
            f"双侧精确配对检验`p={comparison['paired_exact_two_sided_p']:.3f}`。",
            "",
            "因此Linker结果应表述为“独立小样本选择后，在正式1000条上保持小幅正净收益”，"
            "不能表述为统计显著提升。所有方法均使用同一50类、100物体、1000轨迹manifest和20 Hz CPU PhysX重放协议。",
        ]
    )
    return "\n".join(lines) + "\n"


def write_summary_csv(rows: list[dict], output: Path) -> None:
    """写出每种方法一行的机器可读总表。

    输入：统一方法行和目标CSV路径。
    输出：包含角色、三种成功率、抬升及来源的CSV。
    内部逻辑：使用`csv.DictWriter`固定字段顺序并创建父目录。
    作用：为后续绘图和报告数字核对提供稳定接口。
    """
    fields = (
        "label", "role", "hand", "trajectory_count", "success_count",
        "trajectory_micro_success_rate", "object_macro_success_rate",
        "category_macro_success_rate", "mean_max_lift_m", "mean_final_lift_m",
        "source_summary",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            summary = row["summary"]
            writer.writerow(
                {
                    "label": row["label"],
                    "role": row["role"],
                    "hand": row["hand"],
                    "trajectory_count": summary["trajectory_count"],
                    "success_count": summary["success_count"],
                    "trajectory_micro_success_rate": summary["trajectory_micro_success_rate"],
                    "object_macro_success_rate": summary["object_macro_success_rate"],
                    "category_macro_success_rate": summary["category_macro_success_rate"],
                    "mean_max_lift_m": summary["mean_max_lift_m"],
                    "mean_final_lift_m": summary["mean_final_lift_m"],
                    "source_summary": row["source_summary"],
                }
            )


def write_per_category_csv(rows: list[dict], output: Path) -> None:
    """写出五种方法的逐类别成功率表。

    输入：统一方法行和目标CSV路径。
    输出：每种方法50行，共250行的类别统计。
    内部逻辑：按方法顺序和类别字母序读取摘要`per_category`。
    作用：支持寻找不同手型和方法各自擅长或困难的物体类别。
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("label", "role", "hand", "category", "success_count", "trajectory_count", "success_rate"),
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            for category, values in sorted(row["summary"]["per_category"].items()):
                writer.writerow(
                    {
                        "label": row["label"],
                        "role": row["role"],
                        "hand": row["hand"],
                        "category": category,
                        "success_count": values["success_count"],
                        "trajectory_count": values["trajectory_count"],
                        "success_rate": values["success_rate"],
                    }
                )


def main() -> None:
    """解析输出路径并一次生成三份最终报告。

    输入：可选Markdown、总CSV和类别CSV路径。
    输出：三份报告及终端文件位置摘要。
    内部逻辑：从脚本位置定位项目目录，加载固定正式结果并调用三个导出函数。
    作用：作为Linker正式1000完成后的可重复报告入口。
    """
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-markdown", type=Path, default=project_root / "reports/formal_1000_results.md")
    parser.add_argument("--output-csv", type=Path, default=project_root / "reports/formal_1000_results.csv")
    parser.add_argument("--per-category-csv", type=Path, default=project_root / "reports/formal_1000_per_category.csv")
    args = parser.parse_args()
    rows = load_method_rows(project_root)
    pair_path = project_root / "outputs/formal_1000/linker_object_centric_3mm_vs_current.json"
    linker_pair = json.loads(pair_path.read_text(encoding="utf-8"))
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.write_text(render_markdown(rows, linker_pair), encoding="utf-8")
    write_summary_csv(rows, args.output_csv)
    write_per_category_csv(rows, args.per_category_csv)
    print(f"methods={len(rows)} categories={sum(len(row['summary']['per_category']) for row in rows)}")
    print(f"markdown={args.output_markdown.resolve()}")
    print(f"summary_csv={args.output_csv.resolve()}")
    print(f"per_category_csv={args.per_category_csv.resolve()}")


if __name__ == "__main__":
    main()
