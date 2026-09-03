#!/usr/bin/env bash
# 三个XHand Huber候选并行训练并在同一valid50评测。
set -euo pipefail

ROOT="/home/lekangwan/projects/DexGraspMotionChallenge2025-final-release"
PY="/home/lekangwan/miniconda3/envs/hand-retarget/bin/python"
PREP="$ROOT/retarget_research/advanced_policy/prepare/prepare_xhand_huber_tuning.py"
INDEX="$ROOT/retarget_research/advanced_policy/configs/generated/autonomous_xhand_huber_tuning_v1/config_index.json"
EVAL="$ROOT/retarget_research/advanced_policy/evaluate_policy_manifest.py"
MANIFEST="$ROOT/retarget_research/manifests/formal_50c_100o_1000t_seed20260808.json"
SPLIT="$ROOT/retarget_research/advanced_policy/data/formal_v1/policy_split_seed20260813.json"
DATA="$ROOT/retarget_research/advanced_policy/data/formal_final_30cm/xhand_official"
TARGET="$ROOT/retarget_research/outputs/formal_1000/xhand_official"
RUN="$ROOT/retarget_research/advanced_policy/runs/autonomous_xhand_huber_tuning_v1"

cd "$ROOT"
"$PY" "$PREP"

pipeline() {
  local name="$1"
  "$PY" retarget_research/advanced_policy/run_training_matrix.py \
    --index "$INDEX" --filter "$name" --device cuda
  "$PY" "$EVAL" --hand xhand --manifest "$MANIFEST" \
    --policy-split "$SPLIT" --split valid --max-tasks-per-category 1 \
    --target-dir "$TARGET" --checkpoint "$RUN/$name/best.pt" \
    --data-dir "$DATA" --output-dir "$RUN/$name/closed_loop_valid50" \
    --device cuda --workers 2 --autonomous-only --resume
}

pids=()
for name in xhand_huber_beta05_v1 xhand_huber_beta20_v1 xhand_huber_warm_v1; do
  pipeline "$name" 2>&1 | sed -u "s/^/[${name}] /" & pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done
echo "AUTONOMOUS_XHAND_HUBER_TUNING=COMPLETE"
