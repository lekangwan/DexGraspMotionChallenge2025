#!/usr/bin/env python3
"""按冻结manifest批量细化已有XHand官方基线。

输入：manifest、官方基线目录、输出目录、指腹配置、并行数和固定接触参数。
输出：每物体单候选18维轨迹及可复现命令/耗时/状态汇总JSON。
内部逻辑：验证源哈希与基线索引后，并行调用`retarget_xhand_contact.py`。
作用：在不按物理结果选择轨迹的前提下，对冻结开发集统一应用同一接触方法。
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import sys
import time

import numpy as np

from run_wuji_manifest import run_streaming_command, verify_entry


RUN_DIR = Path(__file__).resolve().parent
RETARGET_SCRIPT = RUN_DIR / "retarget_xhand_contact.py"


def verify_baseline(path, entry):
    """核对官方基线文件的18维形状和源索引。

    输入：基线npy路径和manifest条目。
    输出：已解析Path；不匹配时抛出明确异常。
    内部逻辑：要求索引逐项相等且形状为`(冻结数量,70,18)`。
    作用：防止接触细化套到同物体的错误轨迹行上。
    """
    if not path.is_file():
        raise FileNotFoundError(f"XHand官方基线不存在: {path}")
    data = np.load(path, allow_pickle=True).item()
    expected = np.asarray(entry["trajectory_indices"], dtype=np.int64)
    actual = np.asarray(data["source_trajectory_indices"], dtype=np.int64)
    if not np.array_equal(actual, expected):
        raise ValueError(f"基线源索引不一致: {path} {actual} vs {expected}")
    shape = np.asarray(data["grasp_seqs"]).shape
    if shape != (len(expected), 70, 18):
        raise ValueError(f"基线形状错误: {path} {shape}")
    return path


def build_command(entry, source, baseline, output, args):
    """构造一个物体的XHand接触细化子命令。

    输入：manifest条目、源/基线/输出路径和全部方法参数。
    输出：可直接执行的参数列表。
    内部逻辑：显式写入冻结索引、对象名、指腹配置和每个阶段/损失参数。
    作用：使摘要中的单条命令足以完整复现实验。
    """
    object_asset = Path(
        entry.get(
            "object_asset_path",
            RETARGET_SCRIPT.parents[2]
            / "reference"
            / "HandRetargetTask2026"
            / "scripts"
            / "data"
            / "sorting"
            / "object_41"
            / entry["object_name"],
        )
    )
    return [
        sys.executable,
        str(RETARGET_SCRIPT),
        "--source", str(source),
        "--initial-target", str(baseline),
        "--output", str(output),
        "--trajectory-indices", *[str(i) for i in entry["trajectory_indices"]],
        "--object-name", object_asset.name,
        "--object-root", str(object_asset.parent),
        "--contact-pad-config", str(args.contact_pad_config),
        "--maxeval", str(args.maxeval),
        "--contact-weight", str(args.contact_weight),
        "--normal-weight", str(args.normal_weight),
        "--penetration-weight", str(args.penetration_weight),
        "--joint-prior-weight", str(args.joint_prior_weight),
        "--contact-threshold", str(args.contact_threshold),
        "--min-contact-tips", str(args.min_contact_tips),
        "--lift-delta", str(args.lift_delta),
        "--contact-fallback", args.contact_fallback,
        "--region-neighbors", str(args.region_neighbors),
        "--contact-offset", str(args.contact_offset),
        "--min-signed-distance", str(args.min_signed_distance),
    ]


def existing_output_matches(output, entry, baseline, args):
    """检查已有细化候选是否与本次索引、基线和参数完全一致。

    输入：输出、manifest条目、官方基线路径和当前参数。
    输出：可安全续跑跳过时为True。
    内部逻辑：核对18维形状、源索引、方法名、输入绝对路径和全部数值参数。
    作用：避免`--resume`误用早期弱参数或不同官方基线的文件。
    """
    if not output.is_file():
        return False
    try:
        data = np.load(output, allow_pickle=True).item()
        expected_values = {
            "maxeval": args.maxeval,
            "contact_weight": args.contact_weight,
            "normal_weight": args.normal_weight,
            "penetration_weight": args.penetration_weight,
            "joint_prior_weight": args.joint_prior_weight,
            "contact_offset": args.contact_offset,
            "min_signed_distance": args.min_signed_distance,
            "contact_threshold": args.contact_threshold,
            "min_contact_tips": args.min_contact_tips,
            "lift_delta": args.lift_delta,
            "region_neighbors": args.region_neighbors,
        }
        return bool(
            data["method"] == "xhand_official_baseline_phase_contact_refinement_v1"
            and np.array_equal(
                np.asarray(data["source_trajectory_indices"]),
                np.asarray(entry["trajectory_indices"]),
            )
            and np.asarray(data["grasp_seqs"]).shape
            == (len(entry["trajectory_indices"]), 70, 18)
            and Path(data["initial_target"]).resolve() == baseline.resolve()
            and Path(data["contact_pad_config"]).resolve()
            == args.contact_pad_config.resolve()
            and data.get("contact_fallback", "error") == args.contact_fallback
            and all(float(data[name]) == float(value) for name, value in expected_values.items())
        )
    except (KeyError, OSError, TypeError, ValueError):
        return False


def output_fallback_count(output):
    """读取一个XHand细化文件的阶段回退数。

    输入：已成功生成的XHand Numpy输出路径。
    输出：使用最近多指帧回退的轨迹整数。
    内部逻辑：遍历`phase_metadata`并累加布尔审计字段。
    作用：让批处理摘要直接报告边界轨迹数，无需再遍历100个文件。
    """
    data = np.load(output, allow_pickle=True).item()
    return sum(
        int(phase.get("contact_fallback_used", False))
        for phase in data.get("phase_metadata", [])
    )


def run_entry(entry, args):
    """验证并执行一个物体的全部冻结XHand轨迹。

    输入：manifest条目和批处理参数。
    输出：命令、日志、耗时、退出码和候选路径状态字典。
    内部逻辑：先验证源与基线，再安全续跑或启动独立子进程。
    作用：形成可由线程池并行、但不共享PyTorch/NLopt状态的任务单元。
    """
    source = verify_entry(entry)
    baseline = verify_baseline(
        args.baseline_dir / f"{entry['object_name']}.npy", entry
    )
    output = args.output_dir / f"{entry['object_name']}.npy"
    output.parent.mkdir(parents=True, exist_ok=True)
    command = build_command(entry, source, baseline, output, args)
    if args.resume and existing_output_matches(output, entry, baseline, args):
        return {
            "object_name": entry["object_name"],
            "trajectory_indices": entry["trajectory_indices"],
            "trajectory_count": len(entry["trajectory_indices"]),
            "output": str(output.resolve()),
            "command": command,
            "elapsed_seconds": 0.0,
            "return_code": 0,
            "stdout": "skipped: matching output already exists",
            "success": True,
            "skipped_existing": True,
            "contact_fallback_trajectory_count": output_fallback_count(output),
        }
    started = time.perf_counter()
    return_code, output_text = run_streaming_command(command, entry["object_name"])
    success = return_code == 0 and output.is_file()
    return {
        "object_name": entry["object_name"],
        "trajectory_indices": entry["trajectory_indices"],
        "trajectory_count": len(entry["trajectory_indices"]),
        "output": str(output.resolve()),
        "command": command,
        "elapsed_seconds": time.perf_counter() - started,
        "return_code": return_code,
        "stdout": output_text,
        "success": success,
        "skipped_existing": False,
        "contact_fallback_trajectory_count": (
            output_fallback_count(output) if success else 0
        ),
    }


def main():
    """并行运行冻结manifest并保存XHand细化摘要。

    输入：manifest/基线/输出目录、worker数、指腹配置和固定强参数。
    输出：5个物体候选及`manifest_run_summary.json`；任一失败则非零退出。
    内部逻辑：运行前验证全部源/基线，再并行执行并按物体名排序汇总。
    作用：作为XHand物体感知开发集实验的标准run入口。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--contact-pad-config", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--maxeval", type=int, default=20)
    parser.add_argument("--contact-weight", type=float, default=5.0)
    parser.add_argument("--normal-weight", type=float, default=0.05)
    parser.add_argument("--penetration-weight", type=float, default=1.0)
    parser.add_argument("--joint-prior-weight", type=float, default=2.0)
    parser.add_argument("--contact-threshold", type=float, default=0.02)
    parser.add_argument("--min-contact-tips", type=int, default=2)
    parser.add_argument("--lift-delta", type=float, default=0.03)
    parser.add_argument(
        "--contact-fallback",
        choices=("error", "nearest"),
        default="error",
    )
    parser.add_argument("--region-neighbors", type=int, default=32)
    parser.add_argument("--contact-offset", type=float, default=-0.003)
    parser.add_argument("--min-signed-distance", type=float, default=-0.006)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    entries = manifest["entries"]
    for entry in entries:
        verify_entry(entry)
        verify_baseline(args.baseline_dir / f"{entry['object_name']}.npy", entry)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_entry, entry, args): entry for entry in entries}
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
        "hand": "xhand",
        "method_name": "official_baseline_phase_contact_refinement_v1",
        "manifest": str(args.manifest.resolve()),
        "baseline_directory": str(args.baseline_dir.resolve()),
        "object_count": len(results),
        "trajectory_count": sum(item["trajectory_count"] for item in results),
        "workers": args.workers,
        "wall_time_seconds": time.perf_counter() - started,
        "all_successful": all(item["success"] for item in results),
        "contact_fallback_trajectory_count": sum(
            item["contact_fallback_trajectory_count"] for item in results
        ),
        "parameters": {
            "contact_pad_config": str(args.contact_pad_config.resolve()),
            "maxeval": args.maxeval,
            "contact_weight": args.contact_weight,
            "normal_weight": args.normal_weight,
            "penetration_weight": args.penetration_weight,
            "joint_prior_weight": args.joint_prior_weight,
            "contact_threshold": args.contact_threshold,
            "min_contact_tips": args.min_contact_tips,
            "lift_delta": args.lift_delta,
            "contact_fallback": args.contact_fallback,
            "region_neighbors": args.region_neighbors,
            "contact_offset": args.contact_offset,
            "min_signed_distance": args.min_signed_distance,
        },
        "results": results,
    }
    summary_path = args.output_dir / "manifest_run_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"all_successful={summary['all_successful']}")
    print(f"output={summary_path}")
    if not summary["all_successful"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
