#!/usr/bin/env python3
"""把两份PDF、最终12段视频和复现说明整理为本地提交压缩包。"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import zipfile


ROOT = Path(__file__).resolve().parents[1]
DELIVERY_ROOT = ROOT / "deliverables"
BUNDLE = DELIVERY_ROOT / "dexterous_hand_retargeting_submission"


def copy_render_plan(source_dir: Path, target_dir: Path) -> int:
    """只复制当前render plan列出的最终视频，不混入目录中的历史素材。"""
    plan = json.loads((source_dir / "render_plan.json").read_text(encoding="utf-8"))
    target_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for record in plan["records"]:
        source = Path(record["video"])
        if not source.is_file() or source.stat().st_size <= 1024:
            raise FileNotFoundError(f"最终视频缺失或为空: {source}")
        shutil.copy2(source, target_dir / source.name)
        count += 1
    return count


def main():
    """构造可独立解压观看的目录并生成ZIP。"""
    report_dir = BUNDLE / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    for name in ("basic_retargeting_report.pdf", "advanced_policy_report.pdf"):
        shutil.copy2(ROOT / "reports/pdf" / name, report_dir / name)
    shutil.copy2(
        ROOT / "reports/FINAL_EXPERIMENT_METADATA.json",
        BUNDLE / "FINAL_EXPERIMENT_METADATA.json",
    )
    basic_count = copy_render_plan(
        ROOT / "advanced_policy/videos/final_basic_isaac_state_v1",
        BUNDLE / "advanced_policy/videos/final_basic_isaac_state_v1",
    )
    advanced_count = copy_render_plan(
        ROOT / "advanced_policy/videos/final_residual_ppo_isaac_state_v1",
        BUNDLE / "advanced_policy/videos/final_residual_ppo_isaac_state_v1",
    )
    readme = (
        "灵巧手重定向研究提交材料\n\n"
        "1. reports/basic_retargeting_report.pdf：基础任务，两页。\n"
        "2. reports/advanced_policy_report.pdf：进阶任务，两页。\n"
        f"3. advanced_policy/videos/：基础{basic_count}段、进阶{advanced_count}段Isaac Gym视频。\n"
        "4. FINAL_EXPERIMENT_METADATA.json：环境、随机种子、checkpoint与SHA256。\n\n"
        "请保持解压后的目录结构不变，PDF中的本地视频链接才能正确定位。"
        "若PDF阅读器禁止本地链接，可直接进入advanced_policy/videos目录观看。\n"
        "代码：https://github.com/lekangwan/DexGraspMotionChallenge2025/tree/hand-retargeting-research\n"
    )
    (BUNDLE / "README.txt").write_text(readme, encoding="utf-8")
    DELIVERY_ROOT.mkdir(parents=True, exist_ok=True)
    archive = DELIVERY_ROOT / "dexterous_hand_retargeting_submission.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as output:
        for path in sorted(BUNDLE.rglob("*")):
            if path.is_file():
                output.write(path, path.relative_to(BUNDLE.parent))
    print(f"BUNDLE={BUNDLE.resolve()}")
    print(f"ZIP={archive.resolve()}")
    print(f"VIDEOS={basic_count + advanced_count}")


if __name__ == "__main__":
    main()
