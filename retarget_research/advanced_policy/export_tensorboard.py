#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


def export_phase_residual_run(run_dir, writer):
    csv_path = run_dir / "metrics.csv"
    summary_path = run_dir / "training_summary.json"
    if not csv_path.is_file():
        return
    with open(csv_path) as handle:
        rows = list(csv.DictReader(handle))
    tag = str(run_dir.relative_to(run_dir.parent.parent))
    for row in rows:
        step = int(row["epoch"])
        for key, value in row.items():
            if key == "epoch":
                continue
            try:
                writer.add_scalar(f"{tag}/{key}", float(value), step)
            except ValueError:
                continue
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        for key in ("best_valid_loss", "last_epoch"):
            if key in summary:
                writer.add_scalar(f"{tag}/{key}", float(summary[key]), 0)


def export_rl_run(run_dir, writer):
    log_path = run_dir / "training_log.json"
    if not log_path.is_file():
        return
    log = json.loads(log_path.read_text(encoding="utf-8"))
    tag = str(run_dir.relative_to(run_dir.parent.parent))
    for row in log:
        step = int(row["iteration"])
        for key, value in row.items():
            if key == "iteration":
                continue
            if isinstance(value, (int, float)):
                writer.add_scalar(f"{tag}/{key}", float(value), step)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path,
                        default=Path(__file__).resolve().parent / "runs")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent / "runs" / "tensorboard")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(args.output))
    for run_dir in sorted(args.runs_root.rglob("training_summary.json")):
        export_phase_residual_run(run_dir.parent, writer)
    for run_dir in sorted(args.runs_root.rglob("training_log.json")):
        export_rl_run(run_dir.parent, writer)
    writer.close()
    print(f"TENSORBOARD_EXPORTED={args.output.resolve()}")


if __name__ == "__main__":
    main()
