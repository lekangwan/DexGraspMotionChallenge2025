#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import sys
import time

from run_linker_manifest import file_sha256, verify_entry
from retarget_linker_joint_normalized import METHOD, retarget_file


def run_entry(entry, output_dir, source_z_offset, flex_mode, resume):
    """验证manifest的一项并生成该物体的候选轨迹。"""
    source = verify_entry(entry)
    output = output_dir / f"{entry['object_name']}.npy"
    if resume and output.is_file():
        import numpy as np
        data = np.load(output, allow_pickle=True).item()
        if (
            data.get("retarget_method") == METHOD
            and data.get("source_z_offset") == source_z_offset
            and data.get("flex_mode") == flex_mode
            and list(data["source_trajectory_indices"]) == entry["trajectory_indices"]
        ):
            return {"object_name": entry["object_name"], "success": True, "skipped": True}
    started = time.perf_counter()
    retarget_file(source, output, entry["trajectory_indices"], source_z_offset, flex_mode)
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
    parser.add_argument("--flex-mode", choices=("mean", "max"), default="mean")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    for entry in manifest["entries"]:
        verify_entry(entry)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                run_entry,
                entry,
                args.output_dir,
                args.source_z_offset,
                args.flex_mode,
                args.resume,
            ): entry
            for entry in manifest["entries"]
        }
        results = [future.result() for future in as_completed(futures)]
    results.sort(key=lambda item: item["object_name"])
    summary = {
        "hand": "linker",
        "retarget_method": METHOD,
        "manifest": str(args.manifest.resolve()),
        "object_count": len(results),
        "trajectory_count": sum(len(e["trajectory_indices"]) for e in manifest["entries"]),
        "workers": args.workers,
        "source_z_offset": args.source_z_offset,
        "flex_mode": args.flex_mode,
        "wall_time_seconds": time.perf_counter() - started,
        "all_successful": all(item["success"] for item in results),
        "results": results,
    }
    summary_path = args.output_dir / "manifest_run_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(f"all_successful={summary['all_successful']}")
    print(f"output={summary_path}")
    if not summary["all_successful"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
