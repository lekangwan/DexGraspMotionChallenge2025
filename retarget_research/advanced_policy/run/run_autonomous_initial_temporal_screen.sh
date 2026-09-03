#!/usr/bin/env bash
# 冻结合法初态MLP，训练三帧闭环反馈并评测同一50条valid。
set -euo pipefail

ROOT="/home/lekangwan/projects/DexGraspMotionChallenge2025-final-release"
PY="/home/lekangwan/miniconda3/envs/hand-retarget/bin/python"
TRAIN="$ROOT/retarget_research/advanced_policy/train.py"
EVAL="$ROOT/retarget_research/advanced_policy/evaluate_policy_manifest.py"
CONFIG="$ROOT/retarget_research/advanced_policy/configs/generated/autonomous_initial_temporal_feedback_v1"
RUN="$ROOT/retarget_research/advanced_policy/runs/autonomous_initial_temporal_feedback_v1"
MANIFEST="$ROOT/retarget_research/manifests/formal_50c_100o_1000t_seed20260808.json"
SPLIT="$ROOT/retarget_research/advanced_policy/data/formal_v1/policy_split_seed20260813.json"
DATA="$ROOT/retarget_research/advanced_policy/data/formal_final_30cm"

cd "$ROOT"
pids=()
for label in linker xhand_official wuji_old; do
  "$PY" "$TRAIN" --config "$CONFIG/${label}_initial_temporal_feedback_v1.json" --device cuda \
    2>&1 | sed -u "s/^/[train-$label] /" & pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done

evaluate() {
  local label="$1" hand="$2" target="$3"
  "$PY" "$EVAL" --hand "$hand" --manifest "$MANIFEST" --policy-split "$SPLIT" \
    --split valid --max-tasks-per-category 1 --target-dir "$target" \
    --checkpoint "$RUN/${label}_initial_temporal_feedback_v1/best.pt" \
    --data-dir "$DATA/$label" --output-dir "$RUN/${label}_initial_temporal_feedback_v1/closed_loop_valid50" \
    --device cuda --workers 2 --autonomous-only --resume
}
pids=()
evaluate linker linker "$ROOT/retarget_research/outputs/formal_1000/linker_vector_v2alpha_c3g8_v1" 2>&1 | sed -u 's/^/[eval-linker] /' & pids+=("$!")
evaluate xhand_official xhand "$ROOT/retarget_research/outputs/formal_1000/xhand_official" 2>&1 | sed -u 's/^/[eval-xhand] /' & pids+=("$!")
evaluate wuji_old wuji "$ROOT/retarget_research/outputs/formal_1000/wuji_v1" 2>&1 | sed -u 's/^/[eval-wuji] /' & pids+=("$!")
for pid in "${pids[@]}"; do wait "$pid"; done
echo "AUTONOMOUS_INITIAL_TEMPORAL_SCREEN=COMPLETE"
