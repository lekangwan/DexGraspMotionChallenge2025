#!/usr/bin/env bash
# Wuji状态对齐DAgger反馈强度搜索；0.1已有结果，只测试其两侧。
set -euo pipefail

ROOT="/home/lekangwan/projects/DexGraspMotionChallenge2025-final-release"
PY="/home/lekangwan/miniconda3/envs/hand-retarget/bin/python"
SCALE="$ROOT/retarget_research/advanced_policy/prepare/scale_temporal_feedback.py"
EVAL="$ROOT/retarget_research/advanced_policy/evaluate_policy_manifest.py"
SOURCE="$ROOT/retarget_research/advanced_policy/runs/autonomous_state_aligned_dagger_v1/wuji_old_state_aligned_dagger_v1/best.pt"
RUN="$ROOT/retarget_research/advanced_policy/runs/autonomous_wuji_state_aligned_feedback_sweep_v1"
MANIFEST="$ROOT/retarget_research/manifests/formal_50c_100o_1000t_seed20260808.json"
SPLIT="$ROOT/retarget_research/advanced_policy/data/formal_v1/policy_split_seed20260813.json"
DATA="$ROOT/retarget_research/advanced_policy/data/formal_final_30cm/wuji_old"
TARGET="$ROOT/retarget_research/outputs/formal_1000/wuji_v1"

cd "$ROOT"
for tag in 005 015 020; do
  case "$tag" in
    005) limit=0.05 ;;
    015) limit=0.15 ;;
    020) limit=0.20 ;;
  esac
  "$PY" "$SCALE" --input "$SOURCE" --output "$RUN/limit_${tag}/best.pt" \
    --feedback-limit "$limit"
done

evaluate() {
  local tag="$1"
  "$PY" "$EVAL" --hand wuji --manifest "$MANIFEST" --policy-split "$SPLIT" \
    --split valid --max-tasks-per-category 1 --target-dir "$TARGET" \
    --checkpoint "$RUN/limit_${tag}/best.pt" --data-dir "$DATA" \
    --output-dir "$RUN/limit_${tag}/closed_loop_valid50" \
    --device cuda --workers 2 --autonomous-only --resume
}

pids=()
for tag in 005 015 020; do
  evaluate "$tag" 2>&1 | sed -u "s/^/[limit_${tag}] /" & pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done
echo "WUJI_STATE_ALIGNED_FEEDBACK_SWEEP=COMPLETE"
