#!/usr/bin/env python3
"""在对象级test split上批量运行目标手策略闭环评测。

输入：手、正式manifest、策略split、候选目录、checkpoint、数据目录和输出目录。
输出：逐轨迹Isaac JSON与微平均、物体宏平均、类别宏平均成功率摘要。
内部逻辑：只选择split中test记录，逐条调用单轨迹入口，已有匹配报告可续跑跳过。
作用：生成进阶报告的最终测试成功率，严格避免把训练物体混入测试统计。
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


ROLLOUT_SCRIPT = Path(__file__).resolve().parent / "evaluate_policy_isaac.py"


def build_tasks(manifest, policy_split, target_dir, split_name="test"):
    """把指定split记录解析为带候选内部索引的闭环任务。

    输入：manifest、策略split、候选目录和`train/valid/test`名称。
    输出：任务字典列表。
    内部逻辑：只保留指定划分，按物体加载一次候选源索引并查找唯一位置。
    作用：允许先用valid选择策略、最后只用test报告泛化，同时防止索引错配。
    """
    if split_name not in {"train", "valid", "test"}:
        raise ValueError(f"未知策略划分: {split_name}")
    entries = {item["object_name"]: item for item in manifest["entries"]}
    records = [
        item for item in policy_split["records"] if item["split"] == split_name
    ]
    tasks = []
    cache = {}
    for record in records:
        name = record["object_name"]
        entry = entries.get(name)
        if entry is None:
            raise ValueError(f"策略test含manifest外物体: {name}")
        target = target_dir / f"{name}.npy"
        if name not in cache:
            data = np.load(target, allow_pickle=True).item()
            cache[name] = np.asarray(data["source_trajectory_indices"], dtype=np.int64)
        source_index = int(record["source_trajectory_index"])
        matches = np.flatnonzero(cache[name] == source_index)
        if len(matches) != 1:
            raise ValueError(f"{target}中源索引{source_index}匹配数={len(matches)}")
        tasks.append(
            {
                "object_name": name,
                "category": record["category"],
                "source_index": source_index,
                "target_index": int(matches[0]),
                "source": Path(entry["source_path"]),
                "object_dir": Path(entry["object_asset_path"]),
                "target": target,
            }
        )
    return sorted(tasks, key=lambda item: (item["object_name"], item["source_index"]))


def run_task(task, args):
    """运行或复用一条闭环任务并返回精简摘要。

    输入：任务和公共参数。
    输出：不含逐步大数组的成功摘要。
    内部逻辑：报告存在且checkpoint一致时跳过，否则子进程执行并检查退出码。
    作用：支持数百条长评测中断后安全续跑，同时控制总summary体积。
    """
    output = args.output_dir / task["object_name"] / f"source_{task['source_index']}.json"
    reusable = False
    if args.resume and output.is_file():
        previous = json.loads(output.read_text(encoding="utf-8"))
        reusable = (
            previous.get("hand") == args.hand
            and previous.get("object_name") == task["object_name"]
            and int(previous.get("source_trajectory_index", -1)) == task["source_index"]
            and Path(previous.get("checkpoint", "")).resolve() == args.checkpoint.resolve()
        )
    if not reusable:
        command = [
            sys.executable, str(ROLLOUT_SCRIPT), "--hand", args.hand,
            "--source", str(task["source"]), "--target", str(task["target"]),
            "--object-dir", str(task["object_dir"]), "--object-name", task["object_name"],
            "--category", task["category"], "--source-index", str(task["source_index"]),
            "--target-index", str(task["target_index"]), "--checkpoint", str(args.checkpoint),
            "--data-dir", str(args.data_dir), "--output", str(output),
            "--device", args.device, "--diffusion-execute-steps", str(args.diffusion_execute_steps),
            "--normalized-action-clip", str(args.normalized_action_clip),
        ]
        process = subprocess.run(command, text=True, capture_output=True, check=False)
        if process.returncode != 0:
            raise RuntimeError(
                f"{task['object_name']}:{task['source_index']}失败\n"
                f"stdout:\n{process.stdout}\nstderr:\n{process.stderr}"
            )
    report = json.loads(output.read_text(encoding="utf-8"))
    return {
        "hand": args.hand,
        "model_type": report["model_type"],
        "object_name": task["object_name"],
        "category": task["category"],
        "source_trajectory_index": task["source_index"],
        "success": bool(report["success"]),
        "max_lift_m": float(report["max_lift_m"]),
        "final_lift_m": float(report["final_lift_m"]),
        "longest_sustained_lift_time_s": float(report["longest_sustained_lift_time_s"]),
        "hand_object_contact_steps": int(report["hand_object_contact_steps"]),
        "report": str(output.resolve()),
        "reused": reusable,
    }


def summarize(results):
    """计算test结果的轨迹微平均、物体宏平均和类别宏平均。"""
    by_object = defaultdict(list)
    by_category = defaultdict(list)
    for item in results:
        by_object[item["object_name"]].append(float(item["success"]))
        by_category[item["category"]].append(float(item["success"]))
    object_rates = {name: float(np.mean(values)) for name, values in sorted(by_object.items())}
    category_rates = {name: float(np.mean(values)) for name, values in sorted(by_category.items())}
    success_count = sum(bool(item["success"]) for item in results)
    return {
        "trajectory_count": len(results),
        "success_count": success_count,
        "trajectory_micro_success_rate": success_count / len(results),
        "object_macro_success_rate": float(np.mean(list(object_rates.values()))),
        "category_macro_success_rate": float(np.mean(list(category_rates.values()))),
        "per_object_success_rate": object_rates,
        "per_category_success_rate": category_rates,
    }


def main():
    """解析批量参数、执行test闭环并保存汇总。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--hand", choices=["linker", "xhand", "wuji"], required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--policy-split", type=Path, required=True)
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--split", choices=["train", "valid", "test"], default="test")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--diffusion-execute-steps", type=int, default=2)
    parser.add_argument("--normalized-action-clip", type=float, default=5.0)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    policy_split = json.loads(args.policy_split.read_text(encoding="utf-8"))
    tasks = build_tasks(manifest, policy_split, args.target_dir, args.split)
    if not tasks:
        raise ValueError(f"策略split没有{args.split}任务")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_task, task, args): task for task in tasks}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                f"{result['object_name']}[{result['source_trajectory_index']}]: "
                f"success={result['success']} lift={result['max_lift_m']:.4f}",
                flush=True,
            )
    results.sort(key=lambda item: (item["object_name"], item["source_trajectory_index"]))
    boundaries = {
        "train": "training trajectories on training objects",
        "valid": "held-out validation trajectories on training objects",
        "test": "object-level unseen test objects only",
    }
    summary = {
        "status": "complete",
        "hand": args.hand,
        "split": args.split,
        "checkpoint": str(args.checkpoint.resolve()),
        "manifest": str(args.manifest.resolve()),
        "policy_split": str(args.policy_split.resolve()),
        "evaluation_boundary": boundaries[args.split],
        "wall_time_seconds": time.perf_counter() - started,
        **summarize(results),
        "results": results,
    }
    output = args.output_dir / "policy_evaluation_summary.json"
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"success={summary['success_count']}/{summary['trajectory_count']}")
    print(f"POLICY_EVALUATION={output.resolve()}")


if __name__ == "__main__":
    main()
