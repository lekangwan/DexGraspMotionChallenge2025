#!/usr/bin/env python3
"""按冻结manifest批量生成Wuji v2重定向候选。

输入：manifest、输出目录、并行数和固定的Wuji优化配置。
输出：每物体一个候选`.npy`及命令、耗时、日志和状态汇总JSON。
内部逻辑：先验证源SHA-256，再为每个物体调用单文件Wuji入口；续跑时核对全部方法元数据。
作用：在未见开发轨迹上公平测试Wuji，而不是继续选择Camera单条案例。
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


RUN_DIR = Path(__file__).resolve().parent
RETARGET_ROOT = RUN_DIR.parent
RETARGET_SCRIPT = RUN_DIR / "retarget_wuji_keypoints.py"
DEFAULT_MAPPING = RETARGET_ROOT / "configs" / "wuji_keypoint_map_v2.json"


def file_sha256(path):
    """计算源文件的SHA-256。

    输入：文件路径。
    输出：64字符十六进制摘要。
    内部逻辑：按1 MiB分块读入哈希器。
    作用：保证运行数据与冻结manifest记录完全一致。
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def verify_entry(entry):
    """验证一个manifest条目的路径、哈希和轨迹索引。

    输入：单个manifest条目字典。
    输出：已解析源Path；错误时抛出异常。
    内部逻辑：检查文件存在、SHA-256一致，以及索引位于冻结数量内。
    作用：在耗时优化前阻止数据漂移和越界索引。
    """
    source = Path(entry["source_path"])
    if not source.is_file():
        raise FileNotFoundError(f"源文件不存在: {source}")
    if file_sha256(source) != entry["source_sha256"]:
        raise ValueError(f"源文件哈希变化: {source}")
    indices = [int(index) for index in entry["trajectory_indices"]]
    available = int(entry["available_trajectory_count"])
    if not indices or min(indices) < 0 or max(indices) >= available:
        raise ValueError(f"轨迹索引越界: {entry['object_name']} {indices}")
    return source


def build_command(entry, source, output, args):
    """构造单个物体的Wuji重定向子命令。

    输入：manifest条目、源/输出路径和方法参数。
    输出：可交给`subprocess.run`的参数列表。
    内部逻辑：显式写入固定索引、v2映射、SLSQP次数和三类时序权重。
    作用：让每个候选文件都能从批处理摘要完整复现。
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
        "--maxeval",
        str(args.maxeval),
        "--translation-bound",
        str(args.translation_bound),
        "--source-z-offset",
        str(args.source_z_offset),
        "--mapping-config",
        str(args.mapping_config),
        "--joint-temporal-weight",
        str(args.joint_temporal_weight),
        "--translation-temporal-weight",
        str(args.translation_temporal_weight),
        "--rotation-temporal-weight",
        str(args.rotation_temporal_weight),
    ]


def run_streaming_command(command, label):
    """执行子命令并实时转发其合并输出。

    输入：子进程参数列表和用于区分并行任务的标签。
    输出：子进程退出码和完整合并日志文本。
    内部逻辑：把stderr合并到stdout，逐行加标签打印并同时累积。
    作用：避免长优化期间终端长时间无任何进度信息。
    """
    process = subprocess.Popen(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    lines = []
    if process.stdout is None:
        raise RuntimeError("无法读取子进程输出")
    for line in process.stdout:
        lines.append(line)
        print(f"[{label}] {line}", end="", flush=True)
    return process.wait(), "".join(lines)


def existing_output_matches(output, entry, args):
    """判断已有Wuji候选是否与本次冻结配置完全匹配。

    输入：候选路径、manifest条目和当前方法参数。
    输出：索引、形状、映射与所有数值参数都一致时为True。
    内部逻辑：只读取npy元数据，不重新计算轨迹。
    作用：实现安全续跑，避免误把旧v1或其他权重的结果当成v2。
    """
    if not output.is_file():
        return False
    try:
        data = np.load(output, allow_pickle=True).item()
        mapping = Path(str(data["mapping_config"])).resolve()
        return bool(
            np.array_equal(
                np.asarray(data["source_trajectory_indices"]),
                np.asarray(entry["trajectory_indices"]),
            )
            and np.asarray(data["grasp_seqs"]).shape
            == (len(entry["trajectory_indices"]), 70, 26)
            and mapping == args.mapping_config.resolve()
            and int(data["maxeval"]) == args.maxeval
            and float(data["source_z_offset"]) == args.source_z_offset
            and float(data["joint_temporal_weight"])
            == args.joint_temporal_weight
            and float(data["translation_temporal_weight"])
            == args.translation_temporal_weight
            and float(data["rotation_temporal_weight"])
            == args.rotation_temporal_weight
        )
    except (KeyError, OSError, TypeError, ValueError):
        return False


def run_entry(entry, args):
    """运行一个物体的全部冻结Wuji轨迹。

    输入：一个manifest条目和批处理参数。
    输出：命令、输出、耗时、日志、退出码与成功状态字典。
    内部逻辑：先验证源数据，匹配则续跑跳过，否则同步执行独立子进程。
    作用：形成线程池可并行调度且互不共享优化器状态的最小任务。
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
    """解析参数、并行运行manifest并保存Wuji批处理摘要。

    输入：manifest、输出目录、worker数和完整Wuji方法配置。
    输出：每物体候选及`manifest_run_summary.json`。
    内部逻辑：先验证全部条目，再并发执行并按物体名排序；任一失败则非零退出。
    作用：作为冻结开发集从Shadow轨迹到Wuji候选的标准入口。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--maxeval", type=int, default=100)
    parser.add_argument("--translation-bound", type=float, default=2.0)
    parser.add_argument("--source-z-offset", type=float, default=0.4)
    parser.add_argument("--mapping-config", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--joint-temporal-weight", type=float, default=0.0)
    parser.add_argument("--translation-temporal-weight", type=float, default=0.0)
    parser.add_argument("--rotation-temporal-weight", type=float, default=0.0)
    args = parser.parse_args()
    args.mapping_config = args.mapping_config.resolve()

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
        "hand": "wuji",
        "manifest": str(args.manifest.resolve()),
        "manifest_purpose": manifest.get("purpose"),
        "object_count": len(results),
        "trajectory_count": sum(item["trajectory_count"] for item in results),
        "workers": args.workers,
        "wall_time_seconds": time.perf_counter() - started,
        "all_successful": all(item["success"] for item in results),
        "method": {
            "mapping_config": str(args.mapping_config),
            "maxeval": args.maxeval,
            "translation_bound": args.translation_bound,
            "source_z_offset": args.source_z_offset,
            "joint_temporal_weight": args.joint_temporal_weight,
            "translation_temporal_weight": args.translation_temporal_weight,
            "rotation_temporal_weight": args.rotation_temporal_weight,
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
