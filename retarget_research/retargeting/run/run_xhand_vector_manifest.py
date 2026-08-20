#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

from run_wuji_manifest import file_sha256, run_streaming_command, verify_entry
from retarget_xhand_vectors import METHOD


RUN_DIR = Path(__file__).resolve().parent
RETARGET_ROOT = RUN_DIR.parent
VECTOR_SCRIPT = RUN_DIR / "retarget_xhand_vectors.py"
DEFAULT_VECTOR_CONFIG = RETARGET_ROOT / "configs" / "xhand_anydex_vectors_v1.json"


def build_command(entry, source, output, args):
    command = [
        sys.executable,
        str(VECTOR_SCRIPT),
        "--source",
        str(source),
        "--output",
        str(output),
        "--trajectory-indices",
        *[str(index) for index in entry["trajectory_indices"]],
        "--maxeval",
        str(args.maxeval),
        "--translation-bound",
        str(args.translation_bound),
        "--source-z-offset",
        str(args.source_z_offset),
        "--vector-config",
        str(args.vector_config),
    ]
    if args.contact_weight > 0 or args.grip_flexion_weight > 0:
        command.extend([
            "--contact-weight", str(args.contact_weight),
            "--contact-threshold", str(args.contact_threshold),
            "--lift-delta", str(args.lift_delta),
            "--object-root", str(args.object_root),
        ])
    if args.grip_flexion_weight > 0:
        command.extend(["--grip-flexion-weight", str(args.grip_flexion_weight)])
    if args.contact_weight > 0 or args.grip_flexion_weight > 0:
        command.extend(["--contact-fallback", str(args.contact_fallback)])
    if args.warm_start_dir is not None:
        command.extend(["--warm-start-dir", str(args.warm_start_dir)])
    return command


def existing_output_matches(output, entry, args):
    if not output.is_file():
        return False
    try:
        data = np.load(output, allow_pickle=True).item()
        vector_raw = args.vector_config.read_bytes()
        return bool(
            data.get("retarget_method") == METHOD
            and Path(str(data["vector_config"])).resolve() == args.vector_config.resolve()
            and data.get("vector_config_sha256") == hashlib.sha256(vector_raw).hexdigest()
            and np.array_equal(
                np.asarray(data["source_trajectory_indices"]),
                np.asarray(entry["trajectory_indices"]),
            )
            and np.asarray(data["grasp_seqs"]).shape
            == (len(entry["trajectory_indices"]), 70, 18)
            and int(data["maxeval"]) == int(args.maxeval)
            and float(data["source_z_offset"]) == float(args.source_z_offset)
            and float(data.get("contact_weight", 0.0)) == float(args.contact_weight)
            and float(data.get("contact_threshold", 0.02)) == float(args.contact_threshold)
            and float(data.get("lift_delta", 0.03)) == float(args.lift_delta)
            and float(data.get("grip_flexion_weight", 0.0)) == float(args.grip_flexion_weight)
        )
    except (KeyError, OSError, TypeError, ValueError):
        return False


def run_entry(entry, args):
    source = verify_entry(entry)
    output = args.output_dir / f"{entry['object_name']}.npy"
    output.parent.mkdir(parents=True, exist_ok=True)
    command = build_command(entry, source, output, args)
    if args.resume and existing_output_matches(output, entry, args):
        return {
            "object_name": entry["object_name"],
            "trajectory_indices": entry["trajectory_indices"],
            "trajectory_count": len(entry["trajectory_indices"]),
            "output": str(output.resolve()),
            "command": command,
            "elapsed_seconds": 0.0,
            "return_code": 0,
            "stdout": "skipped: matching vector output already exists",
            "success": True,
            "skipped_existing": True,
        }
    started = time.perf_counter()
    return_code, output_text = run_streaming_command(command, entry["object_name"])
    return {
        "object_name": entry["object_name"],
        "trajectory_indices": entry["trajectory_indices"],
        "trajectory_count": len(entry["trajectory_indices"]),
        "output": str(output.resolve()),
        "command": command,
        "elapsed_seconds": time.perf_counter() - started,
        "return_code": return_code,
        "stdout": output_text,
        "success": return_code == 0 and output.is_file(),
        "skipped_existing": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--maxeval", type=int, default=50)
    parser.add_argument("--translation-bound", type=float, default=2.0)
    parser.add_argument("--source-z-offset", type=float, default=0.4)
    parser.add_argument("--vector-config", type=Path, default=DEFAULT_VECTOR_CONFIG)
    parser.add_argument("--contact-weight", type=float, default=0.0)
    parser.add_argument("--contact-threshold", type=float, default=0.02)
    parser.add_argument("--lift-delta", type=float, default=0.03)
    parser.add_argument("--object-root", type=Path,
                        default=RETARGET_ROOT.parent / "reference" / "HandRetargetTask2026"
                                / "scripts" / "data" / "sorting" / "object_41")
    parser.add_argument("--grip-flexion-weight", type=float, default=0.0)
    parser.add_argument("--contact-fallback", choices=("error", "nearest"), default="nearest")
    parser.add_argument("--warm-start-dir", type=Path)
    args = parser.parse_args()
    args.vector_config = args.vector_config.resolve()
    args.object_root = args.object_root.resolve()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    for entry in manifest["entries"]:
        verify_entry(entry)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_entry, entry, args): entry for entry in manifest["entries"]}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(f"{result['object_name']}: success={result['success']} time={result['elapsed_seconds']:.2f}s", flush=True)
    results.sort(key=lambda item: item["object_name"])
    summary = {
        "hand": "xhand",
        "retarget_method": METHOD,
        "manifest": str(args.manifest.resolve()),
        "object_count": len(results),
        "trajectory_count": sum(item["trajectory_count"] for item in results),
        "workers": args.workers,
        "wall_time_seconds": time.perf_counter() - started,
        "all_successful": all(item["success"] for item in results),
        "method": {
            "vector_config": str(args.vector_config),
            "vector_config_sha256": file_sha256(args.vector_config),
            "maxeval": args.maxeval,
            "translation_bound": args.translation_bound,
            "source_z_offset": args.source_z_offset,
        },
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
