#!/usr/bin/env bash
set -euo pipefail

# 先构造初始15点手物关系，再并行训练；本阶段只比较离线系数误差。
ROOT=/home/lekangwan/projects/DexGraspMotionChallenge2025-final-release
PYTHON=/home/lekangwan/miniconda3/envs/dexgrasp/bin/python

cd "$ROOT"
"$PYTHON" retarget_research/advanced_policy_v2/prepare/build_initial_interaction_pca_configs.py
for hand in linker xhand wuji; do
  "$PYTHON" retarget_research/advanced_policy_v2/prepare/prepare_initial_interaction.py \
    --hand "$hand"
done

pids=()
for hand in linker xhand wuji; do
  MPLCONFIGDIR=/tmp/matplotlib-retarget "$PYTHON" -u \
    retarget_research/advanced_policy_v2/train_geometry_pca.py \
    --config "retarget_research/advanced_policy_v2/configs/generated/${hand}_geometry_pca_initial_interaction.json" \
    --device cuda 2>&1 | sed -u "s/^/[${hand}-initial-interaction] /" &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
exit "$status"
