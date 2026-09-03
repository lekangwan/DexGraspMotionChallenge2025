#!/usr/bin/env bash
# 三只手并行训练：测试时只使用初始观测和相位，不读取未来专家动作。
set -euo pipefail

ROOT="/home/lekangwan/projects/DexGraspMotionChallenge2025-final-release"
PY="/home/lekangwan/miniconda3/envs/hand-retarget/bin/python"
INDEX="$ROOT/retarget_research/advanced_policy/configs/generated/autonomous_initial_phase_v1/config_index.json"
RUNNER="$ROOT/retarget_research/advanced_policy/run_training_matrix.py"

cd "$ROOT"
export MPLCONFIGDIR=/tmp/matplotlib-retarget
pids=()

for hand in linker xhand_official wuji_old; do
  "$PY" "$RUNNER" --index "$INDEX" --filter "$hand" --device cuda \
    2>&1 | sed -u "s/^/[${hand}] /" &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  wait "$pid"
done
echo "AUTONOMOUS_INITIAL_PHASE_TRAINING=COMPLETE"
