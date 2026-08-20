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
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


ROLLOUT_SCRIPT = Path(__file__).resolve().parent / "evaluate_policy_isaac.py"


def stable_task_seed(base_seed, task):
    """由固定总seed和轨迹键生成跨进程稳定的单轨迹seed。

    输入：总seed及含物体名、源轨迹索引的任务。
    输出：PyTorch/NumPy均可接受的31位正整数。
    内部逻辑：对字符串键做SHA-256而不使用进程随机化的Python `hash`。
    作用：使Diffusion同一轨迹重跑完全可复现，同时不同轨迹不共享同一噪声样本。
    """
    key = f"{int(base_seed)}:{task['object_name']}:{int(task['source_index'])}"
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "little") % (2 ** 31)


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


def load_expert_success_keys(data_dir, split_name):
    """从物化策略NPZ恢复该split中通过专家重放筛选的轨迹键。

    输入：单手数据目录和split名。
    输出：`(object_name, source_index)`集合。
    内部逻辑：使用NPZ的轨迹边界、object id和映射文件恢复，不读取策略结果。
    作用：只用于小规模故障诊断，正式valid/test仍必须报告完整预定split。
    """
    data_dir = Path(data_dir)
    with np.load(data_dir / f"{split_name}.npz", allow_pickle=False) as archive:
        trajectory_ids = archive["trajectory_id"].astype(np.int64)
        object_ids = archive["object_id"].astype(np.int64)
        source_indices = archive["source_trajectory_index"].astype(np.int64)
    mappings = json.loads((data_dir / "mappings.json").read_text(encoding="utf-8"))
    object_names = {int(value): name for name, value in mappings["object_to_id"].items()}
    keys = set()
    for trajectory_id in np.unique(trajectory_ids):
        index = int(np.flatnonzero(trajectory_ids == trajectory_id)[0])
        keys.add((object_names[int(object_ids[index])], int(source_indices[index])))
    return keys


def run_task(task, args):
    """运行或复用一条闭环任务并返回精简摘要。

    输入：任务和公共参数。
    输出：不含逐步大数组的成功摘要。
    内部逻辑：报告存在且checkpoint一致时跳过，否则子进程执行并检查退出码。
    作用：支持数百条长评测中断后安全续跑，同时控制总summary体积。
    """
    output = args.output_dir / task["object_name"] / f"source_{task['source_index']}.json"
    online_output = (
        None
        if args.online_data_dir is None
        else args.online_data_dir / task["object_name"] / f"source_{task['source_index']}.npz"
    )
    task_seed = stable_task_seed(args.seed, task)
    reusable = False
    if args.resume and output.is_file():
        previous = json.loads(output.read_text(encoding="utf-8"))
        reusable = (
            previous.get("hand") == args.hand
            and previous.get("object_name") == task["object_name"]
            and int(previous.get("source_trajectory_index", -1)) == task["source_index"]
            and Path(previous.get("checkpoint", "")).resolve() == args.checkpoint.resolve()
            and int(previous.get("evaluation_seed", -1)) == task_seed
            and int(previous.get("diffusion_execute_steps", -1)) == args.diffusion_execute_steps
            and float(previous.get("normalized_action_clip", -1.0)) == args.normalized_action_clip
            and float(previous.get("action_rate_limit_scale", -1.0)) == args.action_rate_limit_scale
            and bool(previous.get("expert_wrist", False)) == bool(args.expert_wrist)
            and (
                args.residual_rl_checkpoint is None
                or Path(previous.get("residual_rl_checkpoint", "")).resolve()
                == args.residual_rl_checkpoint.resolve()
            )
            and (
                args.teacher_checkpoint is None
                or (
                    Path(previous.get("teacher_checkpoint", "")).resolve()
                    == args.teacher_checkpoint.resolve()
                    and online_output is not None
                    and online_output.is_file()
                )
            )
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
            "--action-rate-limit-scale", str(args.action_rate_limit_scale),
            "--seed", str(task_seed),
        ]
        if args.teacher_checkpoint is not None:
            command.extend(
                [
                    "--teacher-checkpoint",
                    str(args.teacher_checkpoint),
                    "--online-output",
                    str(online_output),
                ]
            )
        if args.expert_wrist:
            command.append("--expert-wrist")
        if args.residual_rl_checkpoint is not None:
            command.extend(["--residual-rl-checkpoint", str(args.residual_rl_checkpoint)])
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
        "mean_max_lift_m": float(np.mean([item["max_lift_m"] for item in results])),
        "mean_final_lift_m": float(np.mean([item["final_lift_m"] for item in results])),
        "mean_hand_object_contact_steps": float(
            np.mean([item["hand_object_contact_steps"] for item in results])
        ),
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
    parser.add_argument("--teacher-checkpoint", type=Path)
    parser.add_argument("--online-data-dir", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--split", choices=["train", "valid", "test"], default="test")
    parser.add_argument(
        "--max-tasks-per-category",
        type=int,
        default=0,
        help="0表示完整split；正数按已排序轨迹对每类取前N条，供均衡Online-R1采集",
    )
    parser.add_argument(
        "--expert-success-only",
        action="store_true",
        help="仅用于诊断：只评测已进入该split策略NPZ的成功专家轨迹",
    )
    parser.add_argument("--expert-wrist", action="store_true")
    parser.add_argument("--residual-rl-checkpoint", type=Path)
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=0,
        help="0表示不截断；正数在所有过滤后按稳定排序取前N条，仅用于小规模screen",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--diffusion-execute-steps", type=int, default=2)
    parser.add_argument("--normalized-action-clip", type=float, default=5.0)
    parser.add_argument("--action-rate-limit-scale", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260813)
    args = parser.parse_args()
    if (args.teacher_checkpoint is None) != (args.online_data_dir is None):
        raise ValueError("在线DAgger采集必须同时提供teacher-checkpoint和online-data-dir")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    policy_split = json.loads(args.policy_split.read_text(encoding="utf-8"))
    tasks = build_tasks(manifest, policy_split, args.target_dir, args.split)
    if args.expert_success_only:
        successful_keys = load_expert_success_keys(args.data_dir, args.split)
        tasks = [
            task
            for task in tasks
            if (task["object_name"], task["source_index"]) in successful_keys
        ]
    if args.max_tasks_per_category < 0:
        raise ValueError("max-tasks-per-category不能为负数")
    if args.max_tasks_per_category > 0:
        category_counts = defaultdict(int)
        selected = []
        for task in tasks:
            if category_counts[task["category"]] >= args.max_tasks_per_category:
                continue
            selected.append(task)
            category_counts[task["category"]] += 1
        tasks = selected
    if args.max_tasks < 0:
        raise ValueError("max-tasks不能为负数")
    if args.max_tasks > 0:
        tasks = tasks[: args.max_tasks]
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
        "max_tasks_per_category": int(args.max_tasks_per_category),
        "expert_success_only": bool(args.expert_success_only),
        "max_tasks": int(args.max_tasks),
        "evaluation_seed": int(args.seed),
        "diffusion_execute_steps": int(args.diffusion_execute_steps),
        "normalized_action_clip": float(args.normalized_action_clip),
        "action_rate_limit_scale": float(args.action_rate_limit_scale),
        "expert_wrist": bool(args.expert_wrist),
        "residual_rl_checkpoint": (
            None
            if args.residual_rl_checkpoint is None
            else str(args.residual_rl_checkpoint.resolve())
        ),
        "teacher_checkpoint": (
            None
            if args.teacher_checkpoint is None
            else str(args.teacher_checkpoint.resolve())
        ),
        "online_data_dir": (
            None
            if args.online_data_dir is None
            else str(args.online_data_dir.resolve())
        ),
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
