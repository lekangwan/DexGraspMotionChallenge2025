#!/usr/bin/env python3
"""汇总类别ID配对消融，分别检查全部valid和缺失训练类别。"""

import argparse
import json
from pathlib import Path


def load_results(path):
    """读取评测摘要，按物体和源轨迹索引建立唯一结果表。"""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        (item["object_name"], int(item["source_trajectory_index"])): item
        for item in payload["results"]
    }


def summarize(first, second, categories=None, allowed_keys=None):
    """在相同任务上统计成功数及无ID相对有ID的新增/丢失。"""
    keys = sorted(set(first) & set(second))
    if categories is not None:
        keys = [key for key in keys if first[key]["category"] in categories]
    if allowed_keys is not None:
        keys = [key for key in keys if key in allowed_keys]
    with_lift = [float(first[key]["max_lift_m"]) for key in keys]
    without_lift = [float(second[key]["max_lift_m"]) for key in keys]
    return {
        "trajectory_count": len(keys),
        "with_id_success": sum(bool(first[key]["success"]) for key in keys),
        "without_id_success": sum(bool(second[key]["success"]) for key in keys),
        "without_id_added": sum(
            not first[key]["success"] and second[key]["success"] for key in keys
        ),
        "without_id_lost": sum(
            first[key]["success"] and not second[key]["success"] for key in keys
        ),
        "with_id_mean_max_lift_m": sum(with_lift) / len(keys) if keys else None,
        "without_id_mean_max_lift_m": sum(without_lift) / len(keys) if keys else None,
    }


def expert_success_keys(data_dir):
    """从valid数据中的专家成功标签恢复轨迹键。"""
    import numpy as np

    with np.load(data_dir / "valid.npz", allow_pickle=False) as archive:
        trajectory = archive["trajectory_id"].astype(np.int64)
        objects = archive["object_id"].astype(np.int64)
        sources = archive["source_trajectory_index"].astype(np.int64)
        success = archive["expert_replay_success"].astype(bool)
    mappings = json.loads((data_dir / "mappings.json").read_text(encoding="utf-8"))
    names = {int(value): name for name, value in mappings["object_to_id"].items()}
    keys = set()
    for trajectory_id in np.unique(trajectory):
        index = int(np.flatnonzero(trajectory == trajectory_id)[0])
        if success[index]:
            keys.add((names[int(objects[index])], int(sources[index])))
    return keys


def main():
    """解析两个实验根目录，输出JSON并打印三手配对结果。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--evaluation-name", default="closed_loop_valid")
    args = parser.parse_args()
    report = {}
    for hand in ("linker", "xhand_official", "wuji_old"):
        experiment = f"{hand}_phase_residual_v1"
        relative = Path(experiment) / args.evaluation_name / "policy_evaluation_summary.json"
        with_id = load_results(args.run_root / "with_id" / relative)
        without_id = load_results(args.run_root / "without_id" / relative)
        dataset = json.loads(
            (args.data_root / hand / "dataset_summary.json").read_text(encoding="utf-8")
        )
        missing = set(dataset["missing_successful_train_categories"])
        expert_keys = expert_success_keys(args.data_root / hand)
        report[hand] = {
            "all_valid": summarize(with_id, without_id),
            "expert_success_valid": summarize(
                with_id, without_id, allowed_keys=expert_keys
            ),
            "missing_train_categories": sorted(missing),
            "missing_category_valid": summarize(with_id, without_id, missing),
        }
        print(hand, json.dumps(report[hand], ensure_ascii=False))
    output = args.run_root / f"category_id_ablation_{args.evaluation_name}_summary.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"CATEGORY_ID_ABLATION={output.resolve()}")


if __name__ == "__main__":
    main()
