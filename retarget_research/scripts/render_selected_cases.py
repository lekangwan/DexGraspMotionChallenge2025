#!/usr/bin/env python3
"""把自动选出的少量专家/策略案例重跑并录制MP4。

输入：案例选择JSON、正式manifest、视频输出目录、可选执行开关和策略设备。
输出：可审计的逐案例命令与MP4；默认只打印计划，`--execute`才真正运行。
内部逻辑：专家案例读取原physics report还原源/目标/索引；策略案例读取原rollout report还原checkpoint。
作用：报告视频可复现且只重跑少量代表轨迹，不需要给1000条评测全部开启相机。
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERT_SCRIPTS = {
    "linker": PROJECT_ROOT / "retarget_research/retargeting/evaluate/replay_linker_isaac.py",
    "xhand": PROJECT_ROOT / "retarget_research/retargeting/evaluate/replay_xhand_isaac.py",
    "wuji": PROJECT_ROOT / "retarget_research/retargeting/evaluate/replay_wuji_isaac.py",
}
POLICY_SCRIPT = PROJECT_ROOT / "retarget_research/advanced_policy/evaluate_policy_isaac.py"
SOFTWARE_SCRIPT = PROJECT_ROOT / "retarget_research/scripts/render_software_replay.py"


def safe_name(value):
    """把任意物体/分组名转为不含路径分隔符的文件名片段。"""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def flatten_cases(selection):
    """按分组名和列表顺序展开选择文件中的案例。"""
    return [
        (group, index, item)
        for group, items in selection["groups"].items()
        for index, item in enumerate(items)
    ]


def expert_command(group, index, item, manifest_entries, output_dir):
    """根据原专家物理报告构造同参数录像命令。"""
    report_path = Path(item["physics_report"])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    hand_value = report["hand"]
    hand = "linker" if hand_value.startswith("linker") else hand_value
    entry = manifest_entries[item["object_name"]]
    stem = f"{group}_{index}_{safe_name(item['object_name'])}_source{item['source_trajectory_index']}"
    command = [
        sys.executable, str(EXPERT_SCRIPTS[hand]),
        "--source", report["source"], "--target", report["target"],
        "--object-dir", entry["object_asset_path"], "--object-name", item["object_name"],
        "--source-index", str(report["source_trajectory_index"]),
        "--target-index", str(report["target_trajectory_index"]),
        "--output", str(output_dir / f"{stem}.json"),
        "--video-output", str(output_dir / f"{stem}.mp4"),
    ]
    if hand == "linker":
        command.extend(
            [
                "--finger-stiffness", str(report.get("finger_stiffness", 120.0)),
                "--finger-damping", str(report.get("finger_damping", 5.0)),
                "--mimic-stiffness", str(report.get("mimic_stiffness", 120.0)),
                "--mimic-damping", str(report.get("mimic_damping", 5.0)),
            ]
        )
    return command, output_dir / f"{stem}.mp4"


def policy_command(group, index, item, output_dir, device, diffusion_execute_steps):
    """根据原闭环报告构造同checkpoint策略录像命令。"""
    report_path = Path(item["report"])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    required = {"source", "target", "object_dir", "data_dir", "checkpoint"}
    missing = required - set(report)
    if missing:
        raise ValueError(f"旧策略报告缺少可复现字段{sorted(missing)}，需用新评测器重跑")
    stem = f"{group}_{index}_{safe_name(item['object_name'])}_source{item['source_trajectory_index']}"
    command = [
        sys.executable, str(POLICY_SCRIPT), "--hand", report["hand"],
        "--source", report["source"], "--target", report["target"],
        "--object-dir", report["object_dir"], "--object-name", report["object_name"],
        "--category", report["category"],
        "--source-index", str(report["source_trajectory_index"]),
        "--target-index", str(report["target_trajectory_index"]),
        "--checkpoint", report["checkpoint"], "--data-dir", report["data_dir"],
        "--output", str(output_dir / f"{stem}.json"),
        "--video-output", str(output_dir / f"{stem}.mp4"),
        "--device", device,
        "--diffusion-execute-steps", str(diffusion_execute_steps),
    ]
    return command, output_dir / f"{stem}.mp4"


def software_command(group, index, item, manifest_entries, output_dir, summary_kind):
    """由已保存状态构造完全CPU的软件骨架+物体网格渲染命令。"""
    if summary_kind == "policy":
        state_path = Path(item["report"])
        report = json.loads(state_path.read_text(encoding="utf-8"))
        object_dir = report.get("object_dir")
    else:
        state_value = item.get("policy_trace")
        if not state_value:
            raise ValueError("专家案例没有policy_trace，无法软件渲染；可改用--renderer isaac")
        state_path = Path(state_value)
        object_dir = manifest_entries[item["object_name"]]["object_asset_path"]
    if not object_dir:
        raise ValueError("案例缺少物体资产路径")
    stem = f"{group}_{index}_{safe_name(item['object_name'])}_source{item['source_trajectory_index']}"
    video = output_dir / f"{stem}.mp4"
    command = [
        sys.executable, str(SOFTWARE_SCRIPT), "--state", str(state_path),
        "--object-dir", str(object_dir), "--output", str(video),
        "--title", f"{group}: {item['object_name']} [{item['source_trajectory_index']}]",
    ]
    return command, video


def main():
    """生成渲染计划，可选顺序执行并保存每条退出状态。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--renderer", choices=["auto", "software", "isaac"], default="auto")
    parser.add_argument("--diffusion-execute-steps", type=int, default=2)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    entries = {item["object_name"]: item for item in manifest["entries"]}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    render_cache = Path("/tmp/retarget-render-cache")
    render_cache.mkdir(parents=True, exist_ok=True)
    render_environment = os.environ.copy()
    render_environment["XDG_CACHE_HOME"] = str(render_cache)
    render_environment["MESA_SHADER_CACHE_DIR"] = str(render_cache / "mesa")
    (render_cache / "mesa").mkdir(parents=True, exist_ok=True)
    records = []
    for group, index, item in flatten_cases(selection):
        use_software = args.renderer == "software" or (
            args.renderer == "auto"
            and (selection["summary_kind"] == "policy" or bool(item.get("policy_trace")))
        )
        if use_software:
            command, video = software_command(
                group, index, item, entries, args.output_dir, selection["summary_kind"]
            )
            renderer = "software"
        elif selection["summary_kind"] == "policy":
            command, video = policy_command(
                group, index, item, args.output_dir, args.device,
                args.diffusion_execute_steps,
            )
            renderer = "isaac"
        else:
            command, video = expert_command(group, index, item, entries, args.output_dir)
            renderer = "isaac"
        record = {
            "group": group,
            "object_name": item["object_name"],
            "source_trajectory_index": item["source_trajectory_index"],
            "video": str(video.resolve()),
            "command": command,
            "renderer": renderer,
            "status": "planned",
        }
        print(" ".join(command), flush=True)
        if args.execute:
            process = subprocess.run(
                command, check=False, env=render_environment
            )
            record["returncode"] = process.returncode
            record["status"] = "complete" if process.returncode == 0 else "failed"
            if process.returncode != 0:
                records.append(record)
                break
        records.append(record)
    plan = {
        "schema_version": 1,
        "selection": str(args.selection.resolve()),
        "executed": bool(args.execute),
        "all_successful": (
            None
            if not args.execute
            else bool(records) and all(item["status"] == "complete" for item in records)
        ),
        "records": records,
    }
    output = args.output_dir / "render_plan.json"
    output.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"RENDER_PLAN={output.resolve()}")
    if args.execute and not plan["all_successful"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
