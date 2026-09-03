#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/lekangwan/projects/DexGraspMotionChallenge2025-final-release"
PY="/home/lekangwan/miniconda3/envs/hand-retarget/bin/python"
DATA="$ROOT/retarget_research/advanced_policy/data/formal_final_30cm"
INDEX="$ROOT/retarget_research/advanced_policy/configs/generated/autonomous_initial_phase_delta_v1/config_index.json"

cd "$ROOT"
"$PY" retarget_research/advanced_policy/prepare/add_initial_delta_stats.py \
  "$DATA/linker" "$DATA/xhand_official" "$DATA/wuji_old"
pids=()
for hand in linker xhand_official wuji_old; do
  "$PY" retarget_research/advanced_policy/run_training_matrix.py \
    --index "$INDEX" --filter "$hand" --device cuda \
    2>&1 | sed -u "s/^/[${hand}] /" & pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done
echo "AUTONOMOUS_INITIAL_PHASE_DELTA_TRAINING=COMPLETE"
