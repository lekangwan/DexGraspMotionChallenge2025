#!/usr/bin/env python3
"""按冻结manifest批量生成Wuji功能向量候选。

输入：train-only manifest、向量/解剖配置、SLSQP预算和输出目录。
输出：每物体一个标准Wuji npy，以及命令、耗时、续跑和方法哈希摘要。
内部逻辑：复用源文件哈希验证；续跑时核对向量配置、解剖配置、索引和形状，
不允许把绝对关键点候选误当成功能向量候选。
作用：在相同20类轨迹上公平比较底层方法，并支持长任务中断后安全继续。
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from run_wuji_manifest import file_sha256, run_streaming_command, verify_entry


RUN_DIR = Path(__file__).resolve().parent
RETARGET_ROOT = RUN_DIR.parent
VECTOR_SCRIPT = RUN_DIR / "retarget_wuji_anydex_vectors.py"
DEFAULT_VECTOR_CONFIG = RETARGET_ROOT / "configs" / "wuji_anydex_vectors_v1.json"


def build_command(entry, source, output, args):
    """把一个manifest条目和冻结参数转换为单文件向量重定向命令。"""
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
    if args.anatomy_config is not None:
        command.extend(["--anatomy-config", str(args.anatomy_config)])
    if args.grip_flexion_weight > 0 or args.contact_weight > 0:
        command.extend([
            "--grip-flexion-weight", str(args.grip_flexion_weight),
            "--contact-threshold", str(args.contact_threshold),
            "--lift-delta", str(args.lift_delta),
            "--object-root", str(args.object_root),
            "--contact-fallback", str(args.contact_fallback),
        ])
    if args.contact_weight > 0:
        command.extend(["--contact-weight", str(args.contact_weight)])
    if args.warm_start_dir is not None:
        command.extend(["--warm-start-dir", str(args.warm_start_dir)])
    return command


def existing_output_matches(output, entry, args):
    """检查已有npy是否与本次向量方法、索引、配置哈希和数值参数完全一致。"""
    if not output.is_file():
        return False
    try:
        data = np.load(output, allow_pickle=True).item()
        vector_raw = args.vector_config.read_bytes()
        vector_matches = (
            data.get("retarget_method") == "anydex_style_segment_vectors_v1"
            and Path(str(data["vector_config"])).resolve() == args.vector_config.resolve()
            and data.get("vector_config_sha256") == hashlib.sha256(vector_raw).hexdigest()
        )
        anatomy_matches = args.anatomy_config is None and data.get("anatomy_config") is None
        if args.anatomy_config is not None:
            anatomy_raw = args.anatomy_config.read_bytes()
            anatomy_matches = bool(
                data.get("anatomy_config")
                and Path(str(data["anatomy_config"])).resolve() == args.anatomy_config.resolve()
                and data.get("anatomy_config_sha256") == hashlib.sha256(anatomy_raw).hexdigest()
            )
        return bool(
            vector_matches
            and anatomy_matches
            and float(data.get("grip_flexion_weight", 0.0)) == float(args.grip_flexion_weight)
            and float(data.get("contact_weight", 0.0)) == float(args.contact_weight)
            and np.array_equal(
                np.asarray(data["source_trajectory_indices"]),
                np.asarray(entry["trajectory_indices"]),
            )
            and np.asarray(data["grasp_seqs"]).shape
            == (len(entry["trajectory_indices"]), 70, 26)
            and int(data["maxeval"]) == int(args.maxeval)
            and float(data["source_z_offset"]) == float(args.source_z_offset)
        )
    except (KeyError, OSError, TypeError, ValueError):
        return False


def run_entry(entry, args):
    """验证并生成一个物体的向量候选，或安全复用完全匹配的已有文件。"""
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
    """解析批处理参数，并行生成、汇总且以非零退出码暴露任何失败。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--maxeval", type=int, default=50)
    parser.add_argument("--translation-bound", type=float, default=2.0)
    parser.add_argument("--source-z-offset", type=float, default=0.4)
    parser.add_argument("--vector-config", type=Path, default=DEFAULT_VECTOR_CONFIG)
    parser.add_argument("--anatomy-config", type=Path)
    parser.add_argument("--grip-flexion-weight", type=float, default=0.0)
    parser.add_argument("--contact-weight", type=float, default=0.0)
    parser.add_argument("--contact-threshold", type=float, default=0.02)
    parser.add_argument("--lift-delta", type=float, default=0.03)
    parser.add_argument("--object-root", type=Path,
                        default=RETARGET_ROOT.parent / "reference" / "HandRetargetTask2026"
                                / "scripts" / "data" / "sorting" / "object_41")
    parser.add_argument("--contact-fallback", choices=("error", "nearest"), default="nearest")
    parser.add_argument("--warm-start-dir", type=Path)
    args = parser.parse_args()
    args.vector_config = args.vector_config.resolve()
    if args.anatomy_config is not None:
        args.anatomy_config = args.anatomy_config.resolve()
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
        "hand": "wuji",
        "retarget_method": "anydex_style_segment_vectors_v1",
        "manifest": str(args.manifest.resolve()),
        "object_count": len(results),
        "trajectory_count": sum(item["trajectory_count"] for item in results),
        "workers": args.workers,
        "wall_time_seconds": time.perf_counter() - started,
        "all_successful": all(item["success"] for item in results),
        "method": {
            "vector_config": str(args.vector_config),
            "vector_config_sha256": file_sha256(args.vector_config),
            "anatomy_config": None if args.anatomy_config is None else str(args.anatomy_config),
            "anatomy_config_sha256": None if args.anatomy_config is None else file_sha256(args.anatomy_config),
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
