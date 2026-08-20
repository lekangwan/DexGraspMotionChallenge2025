#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import sys
import time

import numpy as np

from run_wuji_manifest import file_sha256, verify_entry
from retarget_xhand_joint_normalized import METHOD, retarget_file


class EntryArgs:
    pass


def run_entry(entry, output_dir, source_z_offset, resume):
    source = verify_entry(entry)
    output = output_dir / f"{entry['object_name']}.npy"
    if resume and output.is_file():
        data = np.load(output, allow_pickle=True).item()
        if (
            data.get("retarget_method") == METHOD
            and float(data.get("source_z_offset", 0.0)) == float(source_z_offset)
            and list(data["source_trajectory_indices"]) == entry["trajectory_indices"]
        ):
            return {"object_name": entry["object_name"], "success": True, "skipped": True}
    args = EntryArgs()
    args.source = source
    args.output = output
    args.trajectory_indices = entry["trajectory_indices"]
    args.source_z_offset = source_z_offset
    started = time.perf_counter()
    retarget_file(args)
    return {
        "object_name": entry["object_name"],
        "success": output.is_file(),
        "skipped": False,
        "elapsed_seconds": time.perf_counter() - started,
        "output": str(output.resolve()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--source-z-offset", type=float, default=0.4)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    for entry in manifest["entries"]:
        verify_entry(entry)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                run_entry, entry, args.output_dir, args.source_z_offset, args.resume
            ): entry
            for entry in manifest["entries"]
        }
        results = [future.result() for future in as_completed(futures)]
    results.sort(key=lambda item: item["object_name"])
    summary = {
        "hand": "xhand",
        "retarget_method": METHOD,
        "manifest": str(args.manifest.resolve()),
        "object_count": len(results),
        "trajectory_count": sum(len(e["trajectory_indices"]) for e in manifest["entries"]),
        "workers": args.workers,
        "source_z_offset": args.source_z_offset,
        "wall_time_seconds": time.perf_counter() - started,
        "all_successful": all(item["success"] for item in results),
        "results": results,
    }
    summary_path = args.output_dir / "manifest_run_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"all_successful={summary['all_successful']}")
    print(f"output={summary_path}")
    if not summary["all_successful"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
