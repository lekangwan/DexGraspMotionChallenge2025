#!/usr/bin/env python3
"""Run the physics CEM refinement on one frozen case per category."""

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[3]
REFINE = Path(__file__).with_name("physics_cem_refine.py")


def select_cases(audit_path, split, limit):
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    rows = sorted(
        (row for row in audit["results"] if row["evaluation_split"] == split),
        key=lambda row: (
            row["category"], row["object_name"],
            int(row["source_trajectory_index"]),
        ),
    )
    selected = {}
    for row in rows:
        selected.setdefault(row["category"], row)
    return list(selected.values())[:limit]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hand", choices=("linker", "xhand", "wuji"), required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="calibration")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--population", type=int, default=12)
    parser.add_argument("--elite", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    assets = {
        row["object_name"]: Path(row["object_asset_path"])
        for row in manifest["entries"]
    }
    cases = select_cases(args.audit, args.split, args.limit)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for number, row in enumerate(cases, 1):
        geometry = json.loads(Path(row["geometry_report"]).read_text(encoding="utf-8"))
        case_dir = args.output_dir / row["object_name"]
        output = case_dir / f"source_{int(row['source_trajectory_index'])}.npy"
        report = output.with_suffix(".json")
        if not report.exists():
            command = [
                sys.executable, "-u", str(REFINE),
                "--hand", args.hand,
                "--source", geometry["source"],
                "--target", geometry["target"],
                "--source-index", str(int(row["source_trajectory_index"])),
                "--object-dir", str(assets[row["object_name"]]),
                "--output", str(output),
                "--population", str(args.population),
                "--elite", str(args.elite),
                "--iterations", str(args.iterations),
                "--seed", str(args.seed + number),
                "--device", args.device,
                "--lift-threshold", "0.15",
            ]
            subprocess.run(command, cwd=ROOT, check=True)
        result = json.loads(report.read_text(encoding="utf-8"))
        results.append({
            "category": row["category"],
            "object_name": row["object_name"],
            "source_trajectory_index": int(row["source_trajectory_index"]),
            "output": str(output.resolve()),
            "baseline_metric": result["baseline_metric"],
            "refined_metric": result["metric"],
            "parameters": result["parameters"],
        })
        baseline = sum(item["baseline_metric"]["success"] for item in results)
        refined = sum(item["refined_metric"]["success"] for item in results)
        print(f"[{args.hand}] {number}/{len(cases)} baseline={baseline} refined={refined}", flush=True)

    summary = {
        "schema_version": 1,
        "hand": args.hand,
        "selection": f"first sorted case per category from {args.split}",
        "lift_threshold_m": 0.15,
        "population": args.population,
        "elite": args.elite,
        "iterations": args.iterations,
        "trajectory_count": len(results),
        "baseline_success_count": int(sum(
            item["baseline_metric"]["success"] for item in results
        )),
        "refined_success_count": int(sum(
            item["refined_metric"]["success"] for item in results
        )),
        "results": results,
    }
    (args.output_dir / "screen_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        key: value for key, value in summary.items() if key != "results"
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
