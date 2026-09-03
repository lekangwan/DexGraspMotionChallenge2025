#!/usr/bin/env bash
set -euo pipefail

# 三只手各取valid50中最接近成功的两条失败轨迹，只验证物理优化上限。
ROOT=/home/lekangwan/projects/DexGraspMotionChallenge2025-final-release
PYTHON=/home/lekangwan/miniconda3/envs/dexgrasp/bin/python
OUT="$ROOT/retarget_research/advanced_policy_v2/results/pca_cem_feasibility_v1"

cd "$ROOT"
mkdir -p "$OUT"
pids=()
for hand in linker xhand wuji; do
  MPLCONFIGDIR=/tmp/matplotlib-retarget "$PYTHON" -u \
    retarget_research/advanced_policy_v2/prepare/optimize_pca_rollouts_cem.py \
    --hand "$hand" --cases 2 --population 8 --elite 3 --iterations 3 \
    --device cuda --output "$OUT/$hand.json" 2>&1 \
    | sed -u "s/^/[$hand] /" | tee "$OUT/$hand.log" &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
exit "$status"
