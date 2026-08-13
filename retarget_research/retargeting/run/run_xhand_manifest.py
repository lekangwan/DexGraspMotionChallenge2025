#!/usr/bin/env python3
"""按冻结manifest批量调用参考XHand重定向包装器。

输入：manifest、输出目录、并行数和固定参考优化配置。
输出：每物体XHand候选及命令、耗时、日志和状态汇总JSON。
内部逻辑：复用Wuji批处理的manifest哈希验证，再并行调用本项目XHand单文件包装器。
作用：让官方参考算法在与Linker/Wuji相同的未见轨迹上接受公平评测。
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from run_wuji_manifest import run_streaming_command, verify_entry


RUN_DIR = Path(__file__).resolve().parent
RETARGET_SCRIPT = RUN_DIR / "retarget_xhand_reference.py"


def build_command(entry, source, output, args):
    """构造一个物体的参考XHand子命令。

    输入：manifest条目、源/输出路径和参考优化参数。
    输出：可直接执行的参数列表。
    内部逻辑：显式传入冻结索引和所有影响参考候选的数值参数。
    作用：保证批处理摘要足以复现实验。
    """
    return [
        sys.executable,
        str(RETARGET_SCRIPT),
        "--source",
        str(source),
        "--output",
        str(output),
        "--trajectory-indices",
        *[str(index) for index in entry["trajectory_indices"]],
        "--object-name",
        entry["object_name"],
        "--iter-num",
        str(args.iter_num),
        "--sample-frame-num",
        str(args.sample_frame_num),
        "--trans-lr",
        str(args.trans_lr),
        "--ang-lr",
        str(args.ang_lr),
        "--trans-bound",
        str(args.trans_bound),
        "--enlarge-scale",
        str(args.enlarge_scale),
        "--device",
        args.device,
    ]


def existing_output_matches(output, entry, args):
    """检查已有XHand候选是否与本次参考配置完全一致。

    输入：输出路径、manifest条目和方法参数。
    输出：索引、形状和所有参考参数一致时为True。
    内部逻辑：只读取候选元数据，不重新加载运动学模型。
    作用：允许安全续跑并拒绝旧的无索引参考输出。
    """
    if not output.is_file():
        return False
    try:
        data = np.load(output, allow_pickle=True).item()
        return bool(
            np.array_equal(
                np.asarray(data["source_trajectory_indices"]),
                np.asarray(entry["trajectory_indices"]),
            )
            and np.asarray(data["grasp_seqs"]).shape
            == (len(entry["trajectory_indices"]), 70, 18)
            and int(data["iter_num"]) == args.iter_num
            and int(data["sample_frame_num"]) == args.sample_frame_num
            and float(data["trans_lr"]) == args.trans_lr
            and float(data["ang_lr"]) == args.ang_lr
            and float(data["trans_bound"]) == args.trans_bound
            and float(data["enlarge_scale"]) == args.enlarge_scale
        )
    except (KeyError, OSError, TypeError, ValueError):
        return False


def run_entry(entry, args):
    """执行一个物体的全部冻结XHand轨迹。

    输入：manifest条目和批处理参数。
    输出：命令、耗时、日志、退出码与候选状态字典。
    内部逻辑：验证源文件，安全续跑或启动隔离子进程。
    作用：构成可并行调度且不共享NLopt/PyTorch状态的最小任务。
    """
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
            "stdout": "skipped: matching output already exists",
            "stderr": "",
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
        "stderr": "",
        "success": return_code == 0 and output.is_file(),
        "skipped_existing": False,
    }


def main():
    """解析参数、并行运行XHand冻结manifest并保存摘要。

    输入：manifest、输出目录、worker数和参考优化配置。
    输出：每物体候选及`manifest_run_summary.json`；任一失败则非零退出。
    内部逻辑：先验证全部源数据，再并行执行、排序并汇总。
    作用：作为XHand开发集候选生成的标准批处理入口。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--iter-num", type=int, default=100)
    parser.add_argument("--sample-frame-num", type=int, default=5)
    parser.add_argument("--trans-lr", type=float, default=5e-3)
    parser.add_argument("--ang-lr", type=float, default=1e-2)
    parser.add_argument("--trans-bound", type=float, default=2.0)
    parser.add_argument("--enlarge-scale", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    entries = manifest["entries"]
    for entry in entries:
        verify_entry(entry)
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
        "manifest": str(args.manifest.resolve()),
        "manifest_purpose": manifest.get("purpose"),
        "object_count": len(results),
        "trajectory_count": sum(item["trajectory_count"] for item in results),
        "workers": args.workers,
        "wall_time_seconds": time.perf_counter() - started,
        "all_successful": all(item["success"] for item in results),
        "method": {
            "reference_script": str(RETARGET_SCRIPT.resolve()),
            "iter_num": args.iter_num,
            "sample_frame_num": args.sample_frame_num,
            "trans_lr": args.trans_lr,
            "ang_lr": args.ang_lr,
            "trans_bound": args.trans_bound,
            "enlarge_scale": args.enlarge_scale,
            "device": args.device,
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
