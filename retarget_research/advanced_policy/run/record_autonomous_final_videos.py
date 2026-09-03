#!/usr/bin/env python3
"""从冻结test结果重跑三手成功/失败案例并录制Isaac Gym视频。"""

import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[3]
ROLLOUT = ROOT / "retarget_research/advanced_policy/evaluate_policy_isaac.py"
TEST = ROOT / "retarget_research/advanced_policy/runs/autonomous_parametric_final_test_v1"
OUTPUT = ROOT / "retarget_research/advanced_policy/videos/autonomous_parametric_final_test_v1"

CASES = {
    "linker_success": TEST / "linker/sem-USBStick-56f49e3cd0d16824b2bf4f773fe0e622/source_30.json",
    "linker_slip": TEST / "linker/sem-Ipad-805a9bf3d265c408d9869786ff7f6a46/source_13.json",
    "xhand_success": TEST / "xhand/core-can-70172e6afe6aff7847f90c1ac631b97f/source_29.json",
    "xhand_slip": TEST / "xhand/sem-Battery-62733b55e76a3b718c9d9ab13336021b/source_24.json",
    "wuji_success": TEST / "wuji/core-can-70172e6afe6aff7847f90c1ac631b97f/source_17.json",
    "wuji_slip": TEST / "wuji/sem-CerealBox-2ee85d45fe615a734322eb6f7ad3b3a2/source_34.json",
}


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    selection = []
    for name, report_path in CASES.items():
        source = json.loads(report_path.read_text(encoding="utf-8"))
        output_json = OUTPUT / f"{name}.json"
        output_video = OUTPUT / f"{name}.mp4"
        command = [
            sys.executable, str(ROLLOUT),
            "--hand", source["hand"],
            "--source", source["source"],
            "--target", source["target"],
            "--object-dir", source["object_dir"],
            "--object-name", source["object_name"],
            "--category", source["category"],
            "--source-index", str(source["source_trajectory_index"]),
            "--target-index", str(source["target_trajectory_index"]),
            "--checkpoint", source["checkpoint"],
            "--data-dir", source["data_dir"],
            "--output", str(output_json),
            "--video-output", str(output_video),
            "--device", "cuda",
            "--seed", str(source["evaluation_seed"]),
            "--autonomous-only",
        ]
        subprocess.run(command, check=True)
        recorded = json.loads(output_json.read_text(encoding="utf-8"))
        selection.append({
            "name": name,
            "object_name": source["object_name"],
            "source_index": source["source_trajectory_index"],
            "expected_success": source["success"],
            "recorded_success": recorded["success"],
            "max_lift_m": recorded["max_lift_m"],
            "final_lift_m": recorded["final_lift_m"],
            "video": str(output_video.resolve()),
        })
    (OUTPUT / "selection.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    rows = [
        "# 最终自主策略视频索引",
        "",
        "以下案例来自对象隔离的500条最终测试集；录像时重新执行相同checkpoint，",
        "策略只从重定向轨迹首帧取得手腕初态，之后不读取任何未来专家动作。",
        "",
        "| 目标手 | 类型 | 物体 | 最大/最终抬升 | 视频 |",
        "|---|---|---|---:|---|",
    ]
    for item in selection:
        hand, kind = item["name"].split("_", 1)
        label = "稳定成功" if kind == "success" else "抬升后滑落"
        rows.append(
            f'| {hand.capitalize()} | {label} | {item["object_name"]} | '
            f'{item["max_lift_m"]:.3f}/{item["final_lift_m"]:.3f} m | '
            f'[{item["name"]}.mp4]({item["name"]}.mp4) |'
        )
    (OUTPUT / "VIDEO_REVIEW_INDEX.md").write_text(
        "\n".join(rows) + "\n", encoding="utf-8"
    )
    print(f"AUTONOMOUS_FINAL_VIDEOS={len(selection)}")


if __name__ == "__main__":
    main()
