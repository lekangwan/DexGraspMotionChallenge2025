#!/usr/bin/env bash
# 同时训练有ID和无ID两条链；每条链内部依次训练Linker、XHand、Wuji。
set -euo pipefail

ROOT="/home/lekangwan/projects/DexGraspMotionChallenge2025-final-release"
PY="/home/lekangwan/miniconda3/envs/hand-retarget/bin/python"
RUNNER="$ROOT/retarget_research/advanced_policy/run_training_matrix.py"
CONFIG_ROOT="$ROOT/retarget_research/advanced_policy/configs/generated/category_id_ablation_30cm"
DATA_ROOT="$ROOT/retarget_research/advanced_policy/data/formal_final_30cm"
ADD_STATS="$ROOT/retarget_research/advanced_policy/prepare/add_residual_action_stats.py"
ADD_PHASE_LIMITS="$ROOT/retarget_research/advanced_policy/prepare/add_phase_delta_limits.py"

cd "$ROOT"
export MPLCONFIGDIR=/tmp/matplotlib-retarget

for hand in linker xhand_official wuji_old; do
  "$PY" "$ADD_STATS" --data-dir "$DATA_ROOT/$hand"
  "$PY" "$ADD_PHASE_LIMITS" --data-dir "$DATA_ROOT/$hand"
done

"$PY" "$RUNNER" --index "$CONFIG_ROOT/with_id/config_index.json" --device cuda \
  2>&1 | sed -u 's/^/[with-id] /' &
with_pid=$!
"$PY" "$RUNNER" --index "$CONFIG_ROOT/without_id/config_index.json" --device cuda \
  2>&1 | sed -u 's/^/[without-id] /' &
without_pid=$!

wait "$with_pid"
wait "$without_pid"
echo "CATEGORY_ID_ABLATION_TRAINING=COMPLETE"
