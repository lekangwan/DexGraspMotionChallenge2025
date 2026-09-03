#!/usr/bin/env bash
set -euo pipefail

# 输入：三手最终案例选择JSON与冻结manifest。
# 输出：每只手各1段稳定成功、到达后不稳、未到达的Isaac Gym MP4。
# 作用：只在GPU训练结束后执行，避免录像与策略训练争用显存。

ROOT=/home/lekangwan/projects/DexGraspMotionChallenge2025-final-release
PYTHON=/home/lekangwan/miniconda3/envs/hand-retarget/bin/python
FINAL=$ROOT/retarget_research/outputs/reboot_synergy_rank5_formal1000_v1/postconfirmed_rank5_v1

cd "$ROOT"
for hand in linker xhand wuji; do
  "$PYTHON" retarget_research/scripts/render_selected_cases.py \
    --selection "retarget_research/reports/video_cases/final_retargeting_${hand}.json" \
    --manifest "$FINAL/manifests/${hand}.json" \
    --output-dir "retarget_research/reports/videos/final_retargeting/${hand}" \
    --renderer isaac \
    --execute \
    --resume
done

