#!/usr/bin/env python3
"""为三只手固定各20条稳定30 cm物理轨迹的视频审查清单。

输入：策略split和三手正式物理评测摘要。
输出：每手一个渲染选择JSON、总审计JSON和带预期视频名的中文Markdown索引。
内部逻辑：先从三手共同成功交集按质量分位和类别多样性选10条相同轨迹；再为
每只手从非共同成功集中选10条，覆盖脆弱/典型/稳定成功且尽量类别不重复。
作用：让人工审查既能同轨迹横向比较，也不会再混入10 cm后掉落的旧成功。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SPLIT = (
    PROJECT_ROOT
    / "retarget_research/advanced_policy/data/formal_v1/policy_split_seed20260813.json"
)
DEFAULT_REPORTS = {
    "linker": PROJECT_ROOT / "retarget_research/outputs/stable_success_audit_v2/linker_stable_audit.json",
    "xhand": PROJECT_ROOT / "retarget_research/outputs/stable_success_audit_v2/xhand_stable_audit.json",
    "wuji": PROJECT_ROOT / "retarget_research/outputs/stable_success_audit_v2/wuji_stable_audit.json",
}


def trajectory_key(item):
    """输入逐轨迹结果，输出稳定的`物体名+源索引`二元键。"""
    return item["object_name"], int(item["source_trajectory_index"])


def quality_key(item):
    """把轨迹按稳定30 cm质量从弱到强排序，并兼容旧测试数据。"""
    if "terminal_min_lift_m" in item:
        return (
            float(item["terminal_min_lift_m"]),
            -float(item["peak_to_final_drop_m"]),
            -float(item["max_palm_relative_translation_change_m"]),
            float(item["final_lift_m"]),
        )
    return (
        float(item["longest_sustained_lift_time_s"]),
        float(item["final_lift_m"]),
        int(item["hand_object_contact_steps"]),
    )


def stratified_diverse_take(items, count, score_key):
    """沿质量分位均匀取样，并优先保持类别不重复。

    输入：候选、目标数量和由弱到强的排序函数。
    输出：按质量升序的选中列表。
    内部逻辑：在排序数组的等距目标位置附近搜索尚未使用的新类别；类别不足时
    才允许重复。它不是挑最高分，而是有意覆盖成功边界到稳定成功。
    作用：避免视频全部来自容易物体或全部是最好看的轨迹。
    """
    ordered = sorted(items, key=score_key)
    if len(ordered) < count:
        raise ValueError(f"候选只有{len(ordered)}条，无法选择{count}条")
    targets = np.linspace(0, len(ordered) - 1, count)
    selected_indices = set()
    selected_categories = set()
    selected = []
    for target in targets:
        candidates = sorted(
            range(len(ordered)), key=lambda index: (abs(index - target), index)
        )
        chosen = next(
            (
                index
                for index in candidates
                if index not in selected_indices
                and ordered[index]["category"] not in selected_categories
            ),
            None,
        )
        if chosen is None:
            chosen = next(index for index in candidates if index not in selected_indices)
        selected_indices.add(chosen)
        selected_categories.add(ordered[chosen]["category"])
        selected.append(ordered[chosen])
    return sorted(selected, key=score_key)


def safe_name(value):
    """把组名/物体名转换为与渲染器一致的安全文件名片段。"""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def slim_item(item, reason):
    """保留渲染和人工判断所需字段，并附上选择原因与关键质量指标。"""
    fields = {
        "hand",
        "object_name",
        "category",
        "source_trajectory_index",
        "target_trajectory_index",
        "physics_report",
        "policy_trace",
        "max_lift_m",
        "final_lift_m",
        "max_xy_drift_m",
        "longest_sustained_lift_time_s",
        "hand_object_contact_steps",
        "max_simultaneous_hand_object_contacts",
        "success",
        "stable_physics_success",
        "transport_quality_success",
        "training_eligible",
        "anatomy_gate",
        "terminal_min_lift_m",
        "terminal_lift_range_m",
        "peak_to_final_drop_m",
        "terminal_contact_ratio",
        "max_palm_relative_translation_change_m",
        "max_palm_relative_rotation_change_deg",
    }
    result = {name: item[name] for name in fields if name in item}
    result["selection_reason"] = reason
    return result


def build_selections(policy_split, reports, success_field="success"):
    """构造三手共同10条和各手分层10条清单。

    输入：策略split字典和手到评测摘要字典。
    输出：每手分组、共同键及整体统计。
    内部逻辑：只允许`split=train`且指定质量字段为True；共同样本的质量由三手中最短
    持续时间优先，再看三手平均最终高度/接触，保证弱的一只手也确实成功。
    作用：严格审查训练监督本身，不查看对象级test或失败策略rollout。
    """
    train_keys = {
        (item["object_name"], int(item["source_trajectory_index"]))
        for item in policy_split["records"]
        if item["split"] == "train"
    }
    by_hand = {}
    for hand, summary in reports.items():
        values = {
            trajectory_key(item): item
            for item in summary["results"]
            if bool(item.get(success_field, False))
            and trajectory_key(item) in train_keys
        }
        by_hand[hand] = values
    common_keys = set.intersection(*(set(values) for values in by_hand.values()))
    common_candidates = []
    for key in common_keys:
        hand_items = [by_hand[hand][key] for hand in ("linker", "xhand", "wuji")]
        representative = dict(hand_items[0])
        if success_field == "success":
            representative["common_quality"] = (
                min(float(item["longest_sustained_lift_time_s"]) for item in hand_items),
                float(np.mean([item["final_lift_m"] for item in hand_items])),
                float(np.mean([item["hand_object_contact_steps"] for item in hand_items])),
            )
        else:
            representative["common_quality"] = (
                min(float(item["terminal_min_lift_m"]) for item in hand_items),
                -max(float(item["peak_to_final_drop_m"]) for item in hand_items),
                -max(
                    float(item["max_palm_relative_translation_change_m"])
                    for item in hand_items
                ),
                float(np.mean([item["final_lift_m"] for item in hand_items])),
            )
        common_candidates.append(representative)
    common_selected = stratified_diverse_take(
        common_candidates, 10, lambda item: item["common_quality"]
    )
    selected_common_keys = [trajectory_key(item) for item in common_selected]

    selections = {}
    for hand in ("linker", "xhand", "wuji"):
        common_items = [
            slim_item(by_hand[hand][key], "shared_success_cross_hand_quality_stratified")
            for key in selected_common_keys
        ]
        hand_only_candidates = [
            item for key, item in by_hand[hand].items() if key not in common_keys
        ]
        if len(hand_only_candidates) < 10:
            hand_only_candidates = [
                item
                for key, item in by_hand[hand].items()
                if key not in set(selected_common_keys)
            ]
        own = stratified_diverse_take(hand_only_candidates, 10, quality_key)
        selections[hand] = {
            "schema_version": 2 if success_field != "success" else 1,
            "summary_kind": "expert_replay",
            "selection_rule": (
                f"train_{success_field}_only; 10 shared cross-hand + 10 hand-specific; "
                "quality-stratified and category-diverse; no policy/test outcome"
            ),
            "groups": {
                "shared_success": common_items,
                "hand_fragile_success": [
                    slim_item(item, "hand_specific_low_quality_success") for item in own[:4]
                ],
                "hand_typical_success": [
                    slim_item(item, "hand_specific_middle_quality_success") for item in own[4:7]
                ],
                "hand_stable_success": [
                    slim_item(item, "hand_specific_high_quality_success") for item in own[7:]
                ],
            },
            "selected_count": 20,
        }
    return selections, selected_common_keys, {
        "per_hand": {hand: len(values) for hand, values in by_hand.items()},
        "common_pool_count": len(common_keys),
    }


def write_markdown(path, selections, video_root):
    """写包含60条指标和预期MP4路径的中文人工审查索引。"""
    lines = [
        "# 稳定30 cm物理轨迹视频审查索引",
        "",
        "全部样本来自策略train划分，并通过稳定30 cm末段保持与掌物运输防滑门。每手前10条`shared_success`使用相同轨迹键，便于三手横向比较；其余10条覆盖该手从边界到稳定的物理质量。",
        "",
        "注意：Wuji仍因远端关节反向弯曲被手型门整体隔离；这里的Wuji视频只用于诊断物理运输和手型，不能作为可训练专家。",
        "",
        "建议重点观察：手是否真正包覆而非撞飞；抬升是否主要由手腕穿过物体；接触是否只在单侧/单指；物体是否明显滑移；末端保持时是否仍稳定。",
        "",
    ]
    for hand, selection in selections.items():
        lines.extend([f"## {hand}", ""])
        for group, items in selection["groups"].items():
            lines.extend([f"### {group}", ""])
            for index, item in enumerate(items):
                stem = (
                    f"{group}_{index}_{safe_name(item['object_name'])}_"
                    f"source{item['source_trajectory_index']}"
                )
                video = (video_root / hand / f"{stem}.mp4").resolve()
                lines.append(
                    f"- `{item['category']}` / `{item['object_name']}` / source {item['source_trajectory_index']}："
                    f"持续 {float(item['longest_sustained_lift_time_s']):.3f}s，"
                    f"最大/最终抬升 {float(item['max_lift_m']):.3f}/{float(item['final_lift_m']):.3f}m，"
                    f"接触 {int(item['hand_object_contact_steps'])} 步；"
                    f"[打开视频](<{video}>)"
                )
            lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    """解析路径、生成三手选择文件、总审计和Markdown索引。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--linker-report", type=Path, default=DEFAULT_REPORTS["linker"])
    parser.add_argument("--xhand-report", type=Path, default=DEFAULT_REPORTS["xhand"])
    parser.add_argument("--wuji-report", type=Path, default=DEFAULT_REPORTS["wuji"])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument(
        "--success-field",
        default="transport_quality_success",
        help="逐轨迹必须为True的质量字段；v2默认要求稳定30 cm且运输不滑移",
    )
    args = parser.parse_args()
    policy_split = json.loads(args.policy_split.read_text(encoding="utf-8"))
    report_paths = {
        "linker": args.linker_report,
        "xhand": args.xhand_report,
        "wuji": args.wuji_report,
    }
    reports = {
        hand: json.loads(path.read_text(encoding="utf-8"))
        for hand, path in report_paths.items()
    }
    selections, common_keys, pool_stats = build_selections(
        policy_split, reports, args.success_field
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for hand, selection in selections.items():
        selection["source_summary"] = str(report_paths[hand].resolve())
        (args.output_dir / f"{hand}_selection.json").write_text(
            json.dumps(selection, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    audit = {
        "schema_version": 2,
        "required_success_field": args.success_field,
        "policy_split": str(args.policy_split.resolve()),
        "success_train_counts": pool_stats["per_hand"],
        "common_success_pool_count": pool_stats["common_pool_count"],
        "selected_common_keys": [list(key) for key in common_keys],
        "per_hand_selected_count": {hand: 20 for hand in selections},
        "video_root": str(args.video_root.resolve()),
    }
    (args.output_dir / "selection_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_markdown(args.output_dir / "VIDEO_REVIEW_INDEX.md", selections, args.video_root)
    print(f"common_selected={len(common_keys)}")
    print(f"per_hand_selected={{'linker': 20, 'xhand': 20, 'wuji': 20}}")
    print(f"EXPERT_VIDEO_SELECTION={args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
