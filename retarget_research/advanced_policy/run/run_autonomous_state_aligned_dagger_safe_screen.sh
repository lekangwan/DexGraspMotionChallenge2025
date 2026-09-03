#!/usr/bin/env bash
# 状态对齐DAgger安全门消融：反馈幅度0.5降至0.1。
set -euo pipefail

ROOT="/home/lekangwan/projects/DexGraspMotionChallenge2025-final-release"
PY="/home/lekangwan/miniconda3/envs/hand-retarget/bin/python"
SCALE="$ROOT/retarget_research/advanced_policy/prepare/scale_temporal_feedback.py"
EVAL="$ROOT/retarget_research/advanced_policy/evaluate_policy_manifest.py"
MANIFEST="$ROOT/retarget_research/manifests/formal_50c_100o_1000t_seed20260808.json"
SPLIT="$ROOT/retarget_research/advanced_policy/data/formal_v1/policy_split_seed20260813.json"
DATA="$ROOT/retarget_research/advanced_policy/data/formal_final_30cm"
SOURCE="$ROOT/retarget_research/advanced_policy/runs/autonomous_state_aligned_dagger_v1"
RUN="$ROOT/retarget_research/advanced_policy/runs/autonomous_state_aligned_dagger_safe_v1"

cd "$ROOT"
for label in linker xhand_official wuji_old; do
  "$PY" "$SCALE" --input "$SOURCE/${label}_state_aligned_dagger_v1/best.pt" \
    --output "$RUN/${label}_state_aligned_dagger_safe_v1/best.pt" --feedback-limit 0.1
done

evaluate() {
  local label="$1" hand="$2" target="$3"
  "$PY" "$EVAL" --hand "$hand" --manifest "$MANIFEST" --policy-split "$SPLIT" \
    --split valid --max-tasks-per-category 1 --target-dir "$target" \
    --checkpoint "$RUN/${label}_state_aligned_dagger_safe_v1/best.pt" \
    --data-dir "$DATA/$label" --output-dir "$RUN/${label}_state_aligned_dagger_safe_v1/closed_loop_valid50" \
    --device cuda --workers 2 --autonomous-only --resume
}
pids=()
evaluate linker linker "$ROOT/retarget_research/outputs/formal_1000/linker_vector_v2alpha_c3g8_v1" 2>&1 | sed -u 's/^/[linker] /' & pids+=("$!")
evaluate xhand_official xhand "$ROOT/retarget_research/outputs/formal_1000/xhand_official" 2>&1 | sed -u 's/^/[xhand] /' & pids+=("$!")
evaluate wuji_old wuji "$ROOT/retarget_research/outputs/formal_1000/wuji_v1" 2>&1 | sed -u 's/^/[wuji] /' & pids+=("$!")
for pid in "${pids[@]}"; do wait "$pid"; done
echo "AUTONOMOUS_STATE_ALIGNED_DAGGER_SAFE_SCREEN=COMPLETE"
