#!/usr/bin/env python3
"""从正式manifest中抽取某次独立审计失败的轨迹。"""

import argparse
import json
from pathlib import Path


METRICS = {
    "source": "legacy_success_from_source",
    "stable": "stable_physics_success",
    "transport": "transport_quality_success",
}


def failure_keys(audit, metric):
    """输入审计字典和指标名，输出失败轨迹的(object, source_index)集合。"""
    field = METRICS[metric]
    return {
        (row["object_name"], int(row["source_trajectory_index"]))
        for row in audit["results"]
        if not bool(row.get(field, False))
    }


def subset_manifest(manifest, failures, metric, audit_path, max_trajectories=None):
    """保留失败索引并维持原物体、类别、资产路径和轨迹顺序。"""
    entries = []
    selected_count = 0
    for entry in manifest["entries"]:
        indices = [
            int(index) for index in entry["trajectory_indices"]
            if (entry["object_name"], int(index)) in failures
        ]
        if max_trajectories is not None:
            remaining = max(0, max_trajectories - selected_count)
            indices = indices[:remaining]
        if not indices:
            continue
        item = dict(entry)
        item["trajectory_indices"] = indices
        item["calibration_indices"] = [
            int(index) for index in entry.get("calibration_indices", [])
            if int(index) in indices
        ]
        item["heldout_indices"] = [
            int(index) for index in entry.get("heldout_indices", [])
            if int(index) in indices
        ]
        entries.append(item)
        selected_count += len(indices)
    result = dict(manifest)
    result.update({
        "purpose": "success_only_failure_refinement",
        "parent_purpose": manifest.get("purpose"),
        "failure_metric": metric,
        "source_audit": str(audit_path.resolve()),
        "available_failure_trajectory_count": len(failures),
        "object_count": len(entries),
        "trajectory_count": sum(len(item["trajectory_indices"]) for item in entries),
        "entries": entries,
    })
    return result


def main():
    """读取完整manifest和审计，写出可直接交给CEM runner的失败子manifest。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--metric", choices=tuple(METRICS), default="transport")
    parser.add_argument("--max-trajectories", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    failures = failure_keys(audit, args.metric)
    if args.max_trajectories is not None and args.max_trajectories < 1:
        parser.error("--max-trajectories必须至少为1")
    result = subset_manifest(
        manifest, failures, args.metric, args.audit, args.max_trajectories
    )
    expected = (
        min(len(failures), args.max_trajectories)
        if args.max_trajectories is not None else len(failures)
    )
    if result["trajectory_count"] != expected:
        raise ValueError(
            f"预期选择数{expected}与manifest匹配数"
            f"{result['trajectory_count']}不一致"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "metric": args.metric,
        "object_count": result["object_count"],
        "failure_trajectory_count": result["trajectory_count"],
        "output": str(args.output.resolve()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
