#!/usr/bin/env python3
"""逐轨迹启动独立进程确认现有CEM候选，避免Isaac/PhysX内存累积。"""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


WORKER = Path(__file__).resolve().parent / "confirm_existing_candidates_manifest.py"


def changed_rows(screen):
    """返回screen summary中参数非零、需要重复确认的轨迹。"""
    return [
        row for row in screen["results"]
        if np.any(np.asarray(row["parameters"], dtype=np.float32) != 0.0)
    ]


def worker_command(args, row=None):
    """生成一次确认命令；row为空时汇总所有已经保存的确认报告。"""
    command = [
        sys.executable, "-u", str(WORKER),
        "--hand", args.hand,
        "--manifest", str(args.manifest),
        "--screen-summary", str(args.screen_summary),
        "--baseline-target-dir", str(args.baseline_target_dir),
        "--candidate-target-dir", str(args.candidate_target_dir),
        "--output-dir", str(args.output_dir),
        "--confirmation-repeats", str(args.confirmation_repeats),
        "--selection-margin", str(args.selection_margin),
        "--device", args.device,
    ]
    if row is not None:
        command.extend([
            "--object-name", row["object_name"],
            "--source-index", str(row["source_trajectory_index"]),
        ])
    return command


def main():
    """跳过已有报告，逐条隔离确认剩余候选，最后生成完整汇总。"""
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
    args = parser.parse_args()

    screen = json.loads(args.screen_summary.read_text(encoding="utf-8"))
    rows = changed_rows(screen)
    pending = []
    for row in rows:
        report = (
            args.output_dir / "confirmation_reports" / row["object_name"]
            / f"source_{int(row['source_trajectory_index'])}.json"
        )
        if not report.exists():
            pending.append(row)

    print(
        f"[{args.hand}] changed={len(rows)} completed={len(rows)-len(pending)} "
        f"pending={len(pending)}",
        flush=True,
    )
    for number, row in enumerate(pending, 1):
        subprocess.run(worker_command(args, row), check=True)
        print(
            f"[{args.hand}] isolated {number}/{len(pending)} "
            f"{row['object_name']} source={row['source_trajectory_index']}",
            flush=True,
        )

    # 所有逐轨迹报告齐全后，由原worker快速重放选择并生成完整summary。
    subprocess.run(worker_command(args), check=True)


if __name__ == "__main__":
    main()
