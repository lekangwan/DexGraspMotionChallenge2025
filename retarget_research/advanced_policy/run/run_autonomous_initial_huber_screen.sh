#!/usr/bin/env bash
# 三手Huber初态相位策略：并行训练后在统一valid50评测。
set -euo pipefail

ROOT="/home/lekangwan/projects/DexGraspMotionChallenge2025-final-release"
PY="/home/lekangwan/miniconda3/envs/hand-retarget/bin/python"
INDEX="$ROOT/retarget_research/advanced_policy/configs/generated/autonomous_initial_phase_huber_v1/config_index.json"
EVAL="$ROOT/retarget_research/advanced_policy/evaluate_policy_manifest.py"
MANIFEST="$ROOT/retarget_research/manifests/formal_50c_100o_1000t_seed20260808.json"
SPLIT="$ROOT/retarget_research/advanced_policy/data/formal_v1/policy_split_seed20260813.json"
DATA="$ROOT/retarget_research/advanced_policy/data/formal_final_30cm"
RUN="$ROOT/retarget_research/advanced_policy/runs/autonomous_initial_phase_huber_v1"

cd "$ROOT"

pipeline() {
  local label="$1" hand="$2" target="$3"
  local name="${label}_initial_phase_huber_v1"
  "$PY" retarget_research/advanced_policy/run_training_matrix.py \
    --index "$INDEX" --filter "$name" --device cuda
  "$PY" "$EVAL" --hand "$hand" --manifest "$MANIFEST" \
    --policy-split "$SPLIT" --split valid --max-tasks-per-category 1 \
    --target-dir "$target" --checkpoint "$RUN/$name/best.pt" \
    --data-dir "$DATA/$label" --output-dir "$RUN/$name/closed_loop_valid50" \
    --device cuda --workers 2 --autonomous-only --resume
}

pids=()
pipeline linker linker \
  "$ROOT/retarget_research/outputs/formal_1000/linker_vector_v2alpha_c3g8_v1" \
  2>&1 | sed -u 's/^/[linker] /' & pids+=("$!")
pipeline xhand_official xhand \
  "$ROOT/retarget_research/outputs/formal_1000/xhand_official" \
  2>&1 | sed -u 's/^/[xhand] /' & pids+=("$!")
pipeline wuji_old wuji \
  "$ROOT/retarget_research/outputs/formal_1000/wuji_v1" \
  2>&1 | sed -u 's/^/[wuji] /' & pids+=("$!")

for pid in "${pids[@]}"; do wait "$pid"; done
echo "AUTONOMOUS_INITIAL_HUBER_SCREEN=COMPLETE"
