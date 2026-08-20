#!/usr/bin/env python3
"""按冻结manifest批量修正Wuji点法候选的拇指冗余姿态。

输入：manifest、点法初始候选目录、三项loss权重、worker数和输出目录。
输出：每物体唯一的拇指零空间候选及安全续跑摘要。
内部逻辑：核对初始候选源索引，逐物体调用4自由度细化；续跑比较初始文件
SHA、索引、方法名、形状和全部参数。
作用：用统一规则修正拇指，不按物理成功与否为单条轨迹选择旧/新候选。
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

from refine_wuji_thumb_nullspace import METHOD
from run_wuji_manifest import run_streaming_command, verify_entry


RUN_DIR = Path(__file__).resolve().parent
REFINE_SCRIPT = RUN_DIR / "refine_wuji_thumb_nullspace.py"


def sha256(path):
    """输入文件路径，输出SHA-256用于初始候选身份核对。"""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def output_matches(path, initial, source, object_dir, indices, args):
    """检查旧输出是否与本轮输入、索引和参数完全相同。"""
    if not path.is_file():
        return False
    try:
        data = np.load(path, allow_pickle=True).item()
        return bool(
            data.get("retarget_method") == METHOD
            and Path(str(data["source"])).resolve() == source.resolve()
            and data.get("source_sha256") == sha256(source)
            and Path(str(data["object_dir"])).resolve() == object_dir.resolve()
            and Path(str(data["initial_target"])).resolve() == initial.resolve()
            and data.get("initial_target_sha256") == sha256(initial)
            and np.array_equal(data["source_trajectory_indices"], indices)
            and np.asarray(data["grasp_seqs"]).shape == (len(indices), 70, 26)
            and int(data["maxeval"]) == args.maxeval
            and float(data["tip_weight"]) == args.tip_weight
            and float(data["neutral_weight"]) == args.neutral_weight
            and float(data["temporal_weight"]) == args.temporal_weight
            and float(data["source_z_offset"]) == args.source_z_offset
            and float(data["contact_threshold"]) == args.contact_threshold
            and int(data["min_contact_tips"]) == args.min_contact_tips
            and float(data["lift_delta"]) == args.lift_delta
            and float(data["object_clearance"]) == args.object_clearance
            and int(data["close_lead_frames"]) == args.close_lead_frames
            and int(data["grasp_settle_frames"]) == args.grasp_settle_frames
        )
    except (KeyError, OSError, TypeError, ValueError):
        return False


def run_entry(entry, args):
    """对一个物体验证点法基线并生成拇指零空间候选。"""
    source = verify_entry(entry).resolve()
    object_dir = Path(entry["object_asset_path"]).resolve()
    if not object_dir.is_dir():
        raise FileNotFoundError(f"缺少物体网格目录: {object_dir}")
    indices = np.asarray(entry["trajectory_indices"], dtype=np.int64)
    initial = (args.initial_dir / f"{entry['object_name']}.npy").resolve()
    if not initial.is_file():
        raise FileNotFoundError(f"缺少点法初始候选: {initial}")
    initial_data = np.load(initial, allow_pickle=True).item()
    if not np.array_equal(initial_data["source_trajectory_indices"], indices):
        raise ValueError(f"初始候选索引错误: {entry['object_name']}")
    output = args.output_dir / f"{entry['object_name']}.npy"
    command = [
        sys.executable, str(REFINE_SCRIPT),
        "--initial-target", str(initial),
        "--source", str(source),
        "--object-dir", str(object_dir),
        "--output", str(output),
        "--trajectory-indices", *[str(value) for value in indices],
        "--maxeval", str(args.maxeval),
        "--tip-weight", str(args.tip_weight),
        "--neutral-weight", str(args.neutral_weight),
        "--temporal-weight", str(args.temporal_weight),
        "--source-z-offset", str(args.source_z_offset),
        "--contact-threshold", str(args.contact_threshold),
        "--min-contact-tips", str(args.min_contact_tips),
        "--lift-delta", str(args.lift_delta),
        "--object-clearance", str(args.object_clearance),
        "--close-lead-frames", str(args.close_lead_frames),
        "--grasp-settle-frames", str(args.grasp_settle_frames),
    ]
    if args.resume and output_matches(output, initial, source, object_dir, indices, args):
        return {
            "object_name": entry["object_name"], "trajectory_indices": indices.tolist(),
            "trajectory_count": len(indices), "output": str(output.resolve()),
            "command": command, "elapsed_seconds": 0.0, "return_code": 0,
            "stdout": "skipped: matching thumb-nullspace output exists",
            "success": True, "skipped_existing": True,
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    code, text = run_streaming_command(command, entry["object_name"])
    return {
        "object_name": entry["object_name"], "trajectory_indices": indices.tolist(),
        "trajectory_count": len(indices), "output": str(output.resolve()),
        "command": command, "elapsed_seconds": time.perf_counter() - started,
        "return_code": code, "stdout": text,
        "success": code == 0 and output.is_file(), "skipped_existing": False,
    }


def main():
    """解析批处理参数，并行执行并写入可审计摘要。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--initial-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--maxeval", type=int, default=80)
    parser.add_argument("--tip-weight", type=float, default=1.0)
    parser.add_argument("--neutral-weight", type=float, default=0.05)
    parser.add_argument("--temporal-weight", type=float, default=0.01)
    parser.add_argument("--source-z-offset", type=float, default=0.4)
    parser.add_argument("--contact-threshold", type=float, default=0.02)
    parser.add_argument("--min-contact-tips", type=int, default=2)
    parser.add_argument("--lift-delta", type=float, default=0.03)
    parser.add_argument("--object-clearance", type=float, default=0.005)
    parser.add_argument("--close-lead-frames", type=int, default=6)
    parser.add_argument("--grasp-settle-frames", type=int, default=3)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers必须为正整数")
    args.initial_dir = args.initial_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    started, results = time.perf_counter(), []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_entry, entry, args): entry for entry in manifest["entries"]}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                f"{result['object_name']}: success={result['success']} "
                f"time={result['elapsed_seconds']:.2f}s",
                flush=True,
            )
    results.sort(key=lambda item: item["object_name"])
    summary = {
        "hand": "wuji", "retarget_method": METHOD,
        "manifest": str(args.manifest.resolve()), "initial_dir": str(args.initial_dir),
        "object_count": len(results),
        "trajectory_count": sum(item["trajectory_count"] for item in results),
        "workers": args.workers, "wall_time_seconds": time.perf_counter() - started,
        "all_successful": all(item["success"] for item in results),
        "parameters": {
            "maxeval": args.maxeval, "tip_weight": args.tip_weight,
            "neutral_weight": args.neutral_weight, "temporal_weight": args.temporal_weight,
            "source_z_offset": args.source_z_offset,
            "contact_threshold": args.contact_threshold,
            "min_contact_tips": args.min_contact_tips,
            "lift_delta": args.lift_delta,
            "object_clearance": args.object_clearance,
            "close_lead_frames": args.close_lead_frames,
            "grasp_settle_frames": args.grasp_settle_frames,
        },
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
