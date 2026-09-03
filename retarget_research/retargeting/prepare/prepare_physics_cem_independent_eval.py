#!/usr/bin/env python3
"""把CEM逐条输出整理为统一评测器可读取的候选与manifest。"""

import argparse
import json
from pathlib import Path

import numpy as np


def subset_target(path, source_indices):
    data = np.load(path, allow_pickle=True).item()
    indices = np.asarray(data["source_trajectory_indices"], dtype=np.int64)
    rows = []
    for source_index in source_indices:
        matches = np.flatnonzero(indices == int(source_index))
        if len(matches) != 1:
            raise ValueError(f"{path}中源轨迹{source_index}匹配到{len(matches)}行")
        rows.append(int(matches[0]))
    count = len(indices)
    result = {}
    for key, value in data.items():
        if isinstance(value, np.ndarray) and value.ndim and len(value) == count:
            result[key] = value[rows].copy()
        else:
            result[key] = value
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--screen-summary", type=Path, required=True)
    parser.add_argument("--formal-manifest", type=Path, required=True)
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    args = parser.parse_args()

    screen = json.loads(args.screen_summary.read_text(encoding="utf-8"))
    formal = json.loads(args.formal_manifest.read_text(encoding="utf-8"))
    entries = {item["object_name"]: item for item in formal["entries"]}
    grouped = {}
    for item in screen["results"]:
        grouped.setdefault(item["object_name"], []).append(item)
    selected_entries = []
    args.target_dir.mkdir(parents=True, exist_ok=True)
    for object_name, items in grouped.items():
        source_indices = [int(item["source_trajectory_index"]) for item in items]
        outputs = {Path(item["output"]).resolve() for item in items}
        if len(outputs) != 1 and len(items) != 1:
            raise ValueError(f"{object_name}的多轨迹结果未累积到同一文件")
        target = subset_target(Path(items[-1]["output"]), source_indices)
        target["physics_cem_screen"] = [{
            "source_trajectory_index": int(item["source_trajectory_index"]),
            "baseline_metric": item["baseline_metric"],
            "refined_metric": item["refined_metric"],
            "parameters": item["parameters"],
        } for item in items]
        np.save(args.target_dir / f"{object_name}.npy", target, allow_pickle=True)
        entry = dict(entries[object_name])
        entry["trajectory_indices"] = source_indices
        entry["calibration_indices"] = source_indices
        entry["heldout_indices"] = []
        selected_entries.append(entry)

    manifest = {
        "schema_version": 1,
        "purpose": "independent replay of physics-CEM candidates",
        "hand": screen["hand"],
        "trajectory_count": sum(
            len(entry["trajectory_indices"]) for entry in selected_entries),
        "entries": selected_entries,
    }
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
