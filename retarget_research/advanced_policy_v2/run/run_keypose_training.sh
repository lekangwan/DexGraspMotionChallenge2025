#!/usr/bin/env bash
set -euo pipefail

# 三手统一训练关键状态策略，本阶段不自动启动PhysX评测。
ROOT=/home/lekangwan/projects/DexGraspMotionChallenge2025-final-release
PYTHON=/home/lekangwan/miniconda3/envs/dexgrasp/bin/python

cd "$ROOT"
"$PYTHON" retarget_research/advanced_policy_v2/prepare/build_keypose_configs.py
pids=()
for hand in linker xhand wuji; do
  MPLCONFIGDIR=/tmp/matplotlib-retarget "$PYTHON" -u \
    retarget_research/advanced_policy_v2/train_keypose_policy.py \
    --config "retarget_research/advanced_policy_v2/configs/generated/${hand}_geometry_keypose_lift.json" \
    --device cuda 2>&1 | sed -u "s/^/[${hand}-keypose] /" &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
exit "$status"
