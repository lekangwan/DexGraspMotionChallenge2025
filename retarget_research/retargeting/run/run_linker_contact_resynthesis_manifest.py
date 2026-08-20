#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys
import time

from run_linker_manifest import verify_entry


RUN_DIR = Path(__file__).resolve().parent
SCRIPT = RUN_DIR / "retarget_linker_contact_resynthesis.py"


def run_entry(entry, args):
    source = verify_entry(entry)
    baseline = args.baseline_dir / f"{entry['object_name']}.npy"
    output = args.output_dir / f"{entry['object_name']}.npy"
    object_dir = Path(entry.get("object_asset_path", args.object_root / entry["object_name"]))
    command = [
        sys.executable, str(SCRIPT),
        "--source", str(source), "--baseline", str(baseline), "--output", str(output),
        "--object-dir", str(object_dir), "--trajectory-indices",
        *[str(i) for i in entry["trajectory_indices"]],
        "--pad-config", str(args.pad_config), "--maxeval", str(args.maxeval),
        "--phase-contact-weight", str(args.phase_contact_weight),
        "--phase-normal-weight", str(args.phase_normal_weight),
        "--phase-joint-hold-weight", str(args.phase_joint_hold_weight),
        "--phase-joint-prior-weight", str(args.phase_joint_prior_weight),
        "--keypoint-weight", str(args.keypoint_weight),
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.resume and output.is_file():
        return {"object_name": entry["object_name"], "success": True, "skipped": True}
    started = time.perf_counter()
    process = subprocess.run(command, text=True, capture_output=True, check=False)
    return {
        "object_name": entry["object_name"],
        "success": process.returncode == 0 and output.is_file(),
        "skipped": False,
        "elapsed_seconds": time.perf_counter() - started,
        "stdout": process.stdout[-4000:],
        "stderr": process.stderr[-4000:],
        "output": str(output.resolve()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pad-config", type=Path, default=RUN_DIR.parent / "configs" / "linker_contact_pads_v1.json")
    parser.add_argument("--object-root", type=Path, default=RUN_DIR.parents[1] / "reference" / "HandRetargetTask2026" / "scripts" / "data" / "sorting" / "object_41")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--maxeval", type=int, default=60)
    parser.add_argument("--phase-contact-weight", type=float, default=5.0)
    parser.add_argument("--phase-normal-weight", type=float, default=0.05)
    parser.add_argument("--phase-joint-hold-weight", type=float, default=0.05)
    parser.add_argument("--phase-joint-prior-weight", type=float, default=0.01)
    parser.add_argument("--keypoint-weight", type=float, default=1e-5)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_entry, entry, args) for entry in manifest["entries"]]
        results = [future.result() for future in as_completed(futures)]
    results.sort(key=lambda x: x["object_name"])
    summary = {
        "hand": "linker",
        "retarget_method": "linker_contact_resynthesis_v1",
        "manifest": str(args.manifest.resolve()),
        "baseline_dir": str(args.baseline_dir.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "maxeval": args.maxeval,
        "phase_contact_weight": args.phase_contact_weight,
        "phase_normal_weight": args.phase_normal_weight,
        "phase_joint_hold_weight": args.phase_joint_hold_weight,
        "phase_joint_prior_weight": args.phase_joint_prior_weight,
        "wall_time_seconds": time.perf_counter() - started,
        "all_successful": all(x["success"] for x in results),
        "results": results,
    }
    path = args.output_dir / "manifest_run_summary.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"all_successful={summary['all_successful']}")
    print(f"output={path}")
    if not summary["all_successful"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
