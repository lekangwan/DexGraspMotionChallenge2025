#!/usr/bin/env python3
"""按冻结manifest批量细化Wuji功能向量轨迹的物体表面接触。

输入：manifest、纯向量候选目录、指腹配置、接触参数和输出目录。
输出：逐物体一个混合候选，以及完整命令、耗时、配置哈希和续跑摘要。
内部逻辑：先核对源数据与初始候选索引；每个物体独立调用细化脚本；续跑时
比较方法、输入绝对路径、索引、关键参数和指腹SHA，拒绝误用旧结果。
作用：让20条筛选和未来1000条生成使用同一个确定性入口，不进行结果取并集。
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

from run_wuji_manifest import run_streaming_command, verify_entry
from refine_wuji_vector_contacts import METHOD


RUN_DIR = Path(__file__).resolve().parent
RETARGET_ROOT = RUN_DIR.parent
REFINE_SCRIPT = RUN_DIR / "refine_wuji_vector_contacts.py"
DEFAULT_PAD_CONFIG = RETARGET_ROOT / "configs" / "wuji_contact_pads_v1.json"
FLOAT_ARGUMENTS = (
    "object_clearance", "vector_weight", "contact_weight", "normal_weight",
    "penetration_weight", "joint_prior_weight", "contact_offset",
    "min_signed_distance", "contact_threshold", "lift_delta",
    "opposition_distance_scale", "opposition_weight", "pad_alignment_weight",
    "friction_stability_weight", "friction_coefficient", "max_reachable_distance",
)
INT_ARGUMENTS = (
    "maxeval", "min_contact_tips", "region_neighbors",
    "opposition_candidate_neighbors", "opposition_refine_frames",
    "min_opposing_fingers", "friction_cone_edges",
)


def file_sha256(path):
    """输入文件路径，输出SHA-256十六进制串，用于配置身份核对。"""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_command(entry, source, initial, output, args):
    """把manifest条目及冻结超参数转换成单个物体的细化子命令。"""
    command = [
        sys.executable, str(REFINE_SCRIPT),
        "--source", str(source),
        "--initial-target", str(initial),
        "--output", str(output),
        "--object-name", entry["object_name"],
        "--contact-pad-config", str(args.contact_pad_config),
        "--trajectory-indices", *[str(value) for value in entry["trajectory_indices"]],
    ]
    if args.object_root is not None:
        command.extend(["--object-root", str(args.object_root)])
    for name in (*FLOAT_ARGUMENTS, *INT_ARGUMENTS):
        command.extend(["--" + name.replace("_", "-"), str(getattr(args, name))])
    return command


def existing_output_matches(path, entry, initial, args):
    """核对已有输出的方法、输入、索引、指腹SHA和全部关键参数。"""
    if not path.is_file():
        return False
    try:
        data = np.load(path, allow_pickle=True).item()
        if data.get("retarget_method") != METHOD:
            return False
        if Path(str(data["initial_target"])).resolve() != initial.resolve():
            return False
        if data.get("contact_pad_config_sha256") != file_sha256(args.contact_pad_config):
            return False
        if not np.array_equal(
            np.asarray(data["source_trajectory_indices"]),
            np.asarray(entry["trajectory_indices"]),
        ):
            return False
        if np.asarray(data["grasp_seqs"]).shape != (
            len(entry["trajectory_indices"]), 70, 26
        ):
            return False
        for name in FLOAT_ARGUMENTS:
            if float(data[name]) != float(getattr(args, name)):
                return False
        for name in INT_ARGUMENTS:
            if int(data[name]) != int(getattr(args, name)):
                return False
        return True
    except (KeyError, OSError, TypeError, ValueError):
        return False


def run_entry(entry, args):
    """验证一个物体的源/基线并执行细化，或安全跳过匹配输出。"""
    source = verify_entry(entry)
    initial = (args.initial_dir / f"{entry['object_name']}.npy").resolve()
    if not initial.is_file():
        raise FileNotFoundError(f"缺少纯向量初始候选: {initial}")
    initial_data = np.load(initial, allow_pickle=True).item()
    if not np.array_equal(
        np.asarray(initial_data["source_trajectory_indices"]),
        np.asarray(entry["trajectory_indices"]),
    ):
        raise ValueError(f"初始候选索引与manifest不一致: {entry['object_name']}")
    output = args.output_dir / f"{entry['object_name']}.npy"
    output.parent.mkdir(parents=True, exist_ok=True)
    command = build_command(entry, source, initial, output, args)
    if args.resume and existing_output_matches(output, entry, initial, args):
        return {
            "object_name": entry["object_name"],
            "trajectory_indices": entry["trajectory_indices"],
            "trajectory_count": len(entry["trajectory_indices"]),
            "output": str(output.resolve()),
            "command": command,
            "elapsed_seconds": 0.0,
            "return_code": 0,
            "stdout": "skipped: matching vector-contact output already exists",
            "success": True,
            "skipped_existing": True,
        }
    started = time.perf_counter()
    return_code, text = run_streaming_command(command, entry["object_name"])
    return {
        "object_name": entry["object_name"],
        "trajectory_indices": entry["trajectory_indices"],
        "trajectory_count": len(entry["trajectory_indices"]),
        "output": str(output.resolve()),
        "command": command,
        "elapsed_seconds": time.perf_counter() - started,
        "return_code": return_code,
        "stdout": text,
        "success": return_code == 0 and output.is_file(),
        "skipped_existing": False,
    }


def main():
    """解析批处理参数，并行运行后写入可审计汇总。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--initial-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--object-root", type=Path)
    parser.add_argument("--contact-pad-config", type=Path, default=DEFAULT_PAD_CONFIG)
    parser.add_argument("--object-clearance", type=float, default=0.005)
    parser.add_argument("--maxeval", type=int, default=40)
    parser.add_argument("--vector-weight", type=float, default=1.0)
    parser.add_argument("--contact-weight", type=float, default=5.0)
    parser.add_argument("--normal-weight", type=float, default=0.05)
    parser.add_argument("--penetration-weight", type=float, default=1.0)
    parser.add_argument("--joint-prior-weight", type=float, default=1.0)
    parser.add_argument("--contact-offset", type=float, default=-0.002)
    parser.add_argument("--min-signed-distance", type=float, default=-0.005)
    parser.add_argument("--contact-threshold", type=float, default=0.02)
    parser.add_argument("--min-contact-tips", type=int, default=2)
    parser.add_argument("--lift-delta", type=float, default=0.03)
    parser.add_argument("--region-neighbors", type=int, default=32)
    parser.add_argument("--opposition-candidate-neighbors", type=int, default=24)
    parser.add_argument("--opposition-distance-scale", type=float, default=0.03)
    parser.add_argument("--opposition-weight", type=float, default=3.0)
    parser.add_argument("--opposition-refine-frames", type=int, default=5)
    parser.add_argument("--pad-alignment-weight", type=float, default=1.0)
    parser.add_argument("--min-opposing-fingers", type=int, default=2)
    parser.add_argument("--friction-stability-weight", type=float, default=0.0)
    parser.add_argument("--friction-coefficient", type=float, default=1.0)
    parser.add_argument("--friction-cone-edges", type=int, default=4)
    parser.add_argument("--max-reachable-distance", type=float, default=0.03)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers必须为正整数")
    args.initial_dir = args.initial_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.contact_pad_config = args.contact_pad_config.resolve()
    if args.object_root is not None:
        args.object_root = args.object_root.resolve()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    results = []
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
        "hand": "wuji",
        "retarget_method": METHOD,
        "manifest": str(args.manifest.resolve()),
        "initial_dir": str(args.initial_dir),
        "object_count": len(results),
        "trajectory_count": sum(item["trajectory_count"] for item in results),
        "workers": args.workers,
        "wall_time_seconds": time.perf_counter() - started,
        "all_successful": all(item["success"] for item in results),
        "contact_pad_config": str(args.contact_pad_config),
        "contact_pad_config_sha256": file_sha256(args.contact_pad_config),
        "parameters": {
            name: getattr(args, name) for name in (*FLOAT_ARGUMENTS, *INT_ARGUMENTS)
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
