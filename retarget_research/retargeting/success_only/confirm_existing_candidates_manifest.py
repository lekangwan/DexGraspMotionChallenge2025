#!/usr/bin/env python3
"""对现有CEM改动逐条重复确认，失败时恢复上一阶段基线。"""

import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from run_confirmed_synergy_manifest import repeated_confirmation


def changed_rows(screen):
    """输入CEM screen summary，输出参数非零的真正修改项。"""
    return [
        row for row in screen["results"]
        if np.any(np.asarray(row["parameters"], dtype=np.float32) != 0.0)
    ]


def row_index(data, source_index):
    """根据保存的源轨迹索引定位目标NPY中的行。"""
    indices = np.asarray(data["source_trajectory_indices"], dtype=np.int64)
    rows = np.flatnonzero(indices == int(source_index))
    if len(rows) != 1:
        raise ValueError(f"source index {source_index}未唯一匹配")
    return int(rows[0])


def write_selected_row(output_path, baseline_path, candidate_path,
                       source_index, accepted):
    """接受时写候选行，拒绝时写基线行，其余轨迹保持不变。"""
    output = np.load(output_path, allow_pickle=True).item()
    baseline = np.load(baseline_path, allow_pickle=True).item()
    candidate = np.load(candidate_path, allow_pickle=True).item()
    output_row = row_index(output, source_index)
    source = candidate if accepted else baseline
    source_row = row_index(source, source_index)
    sequences = np.asarray(output["grasp_seqs"]).copy()
    sequences[output_row] = source["grasp_seqs"][source_row]
    output["grasp_seqs"] = sequences
    np.save(output_path, output, allow_pickle=True)


def main():
    """确认所有非零CEM改动，输出可直接接受完整独立评测的100对象目录。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--hand", choices=("linker", "xhand", "wuji"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--screen-summary", type=Path, required=True)
    parser.add_argument("--baseline-target-dir", type=Path, required=True)
    parser.add_argument("--candidate-target-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--confirmation-repeats", type=int, default=2)
    parser.add_argument("--selection-margin", type=float, default=1.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--object-name")
    parser.add_argument("--source-index", type=int)
    parser.add_argument("--max-changed", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.confirmation_repeats < 1:
        parser.error("--confirmation-repeats必须至少为1")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    entries = {entry["object_name"]: entry for entry in manifest["entries"]}
    screen = json.loads(args.screen_summary.read_text(encoding="utf-8"))
    rows = changed_rows(screen)
    if args.object_name is not None:
        rows = [row for row in rows if row["object_name"] == args.object_name]
    if args.source_index is not None:
        rows = [
            row for row in rows
            if int(row["source_trajectory_index"]) == args.source_index
        ]
    if args.max_changed is not None:
        if args.max_changed < 1:
            parser.error("--max-changed必须至少为1")
        rows = rows[:args.max_changed]
    print(json.dumps({
        "hand": args.hand,
        "trajectory_count": int(screen["trajectory_count"]),
        "changed_trajectory_count": len(rows),
        "confirmation_rollouts": 2 * args.confirmation_repeats * len(rows),
        "dry_run": args.dry_run,
    }, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for entry in manifest["entries"]:
        source = args.candidate_target_dir / f"{entry['object_name']}.npy"
        output = args.output_dir / f"{entry['object_name']}.npy"
        if not output.exists():
            shutil.copy2(source, output)

    results = []
    for number, row in enumerate(rows, 1):
        object_name = row["object_name"]
        source_index = int(row["source_trajectory_index"])
        entry = entries[object_name]
        baseline_path = args.baseline_target_dir / f"{object_name}.npy"
        candidate_path = args.candidate_target_dir / f"{object_name}.npy"
        output_path = args.output_dir / f"{object_name}.npy"
        report = (
            args.output_dir / "confirmation_reports" / object_name
            / f"source_{source_index}.json"
        )
        if report.exists():
            decision = json.loads(report.read_text(encoding="utf-8"))
        else:
            baseline = np.load(baseline_path, allow_pickle=True).item()
            candidate = np.load(candidate_path, allow_pickle=True).item()
            decision = repeated_confirmation(
                args.hand,
                entry,
                source_index,
                baseline,
                candidate,
                args.confirmation_repeats,
                args.selection_margin,
                args.device,
            )
            report.parent.mkdir(parents=True, exist_ok=True)
            report.write_text(
                json.dumps(decision, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        write_selected_row(
            output_path,
            baseline_path,
            candidate_path,
            source_index,
            bool(decision["accepted"]),
        )
        results.append({
            "category": row["category"],
            "object_name": object_name,
            "source_trajectory_index": source_index,
            "accepted": bool(decision["accepted"]),
            "confirmation_report": str(report.resolve()),
        })
        print(
            f"[{args.hand}] {number}/{len(rows)} accepted={decision['accepted']}",
            flush=True,
        )

    summary = {
        "schema_version": 1,
        "hand": args.hand,
        "method": "existing_candidate_repeated_confirmation_v1",
        "source_screen_summary": str(args.screen_summary.resolve()),
        "confirmation_repeats": args.confirmation_repeats,
        "selection_margin": args.selection_margin,
        "trajectory_count": int(screen["trajectory_count"]),
        "changed_trajectory_count": len(rows),
        "accepted_change_count": sum(row["accepted"] for row in results),
        "restored_baseline_count": sum(not row["accepted"] for row in results),
        "results": results,
    }
    (args.output_dir / "confirmation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
