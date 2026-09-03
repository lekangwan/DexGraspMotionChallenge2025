#!/usr/bin/env bash
# 在每类两条valid轨迹上比较有ID/无ID；同一只手的两个模型并行评测。
set -euo pipefail

ROOT="/home/lekangwan/projects/DexGraspMotionChallenge2025-final-release"
PY="/home/lekangwan/miniconda3/envs/hand-retarget/bin/python"
EVAL="$ROOT/retarget_research/advanced_policy/evaluate_policy_manifest.py"
MANIFEST="$ROOT/retarget_research/manifests/formal_50c_100o_1000t_seed20260808.json"
SPLIT="$ROOT/retarget_research/advanced_policy/data/formal_v1/policy_split_seed20260813.json"
RUN_ROOT="$ROOT/retarget_research/advanced_policy/runs/category_id_ablation_30cm"
DATA_ROOT="$ROOT/retarget_research/advanced_policy/data/formal_final_30cm"
EXPERT_WRIST="${EXPERT_WRIST:-0}"
EVALUATION_NAME="${EVALUATION_NAME:-closed_loop_valid}"

cd "$ROOT"
export MPLCONFIGDIR=/tmp/matplotlib-retarget

evaluate_one() {
  local variant="$1" label="$2" hand="$3" target="$4"
  local experiment="${label}_phase_residual_v1"
  local extra=()
  if [ "$EXPERT_WRIST" = "1" ]; then
    extra+=(--expert-wrist)
  fi
  "$PY" "$EVAL" \
    --hand "$hand" --manifest "$MANIFEST" --policy-split "$SPLIT" --split valid \
    --target-dir "$target" \
    --checkpoint "$RUN_ROOT/$variant/$experiment/best.pt" \
    --data-dir "$DATA_ROOT/$label" \
    --output-dir "$RUN_ROOT/$variant/$experiment/$EVALUATION_NAME" \
    --device cuda --workers 3 --resume "${extra[@]}"
}

for label in linker xhand_official wuji_old; do
  case "$label" in
    linker)
      hand="linker"
      target="$ROOT/retarget_research/outputs/formal_1000/linker_vector_v2alpha_c3g8_v1"
      ;;
    xhand_official)
      hand="xhand"
      target="$ROOT/retarget_research/outputs/formal_1000/xhand_official"
      ;;
    wuji_old)
      hand="wuji"
      target="$ROOT/retarget_research/outputs/formal_1000/wuji_v1"
      ;;
  esac
  evaluate_one with_id "$label" "$hand" "$target" 2>&1 | sed -u "s/^/[${label}-with] /" &
  with_pid=$!
  evaluate_one without_id "$label" "$hand" "$target" 2>&1 | sed -u "s/^/[${label}-without] /" &
  without_pid=$!
  wait "$with_pid"
  wait "$without_pid"
done

"$PY" "$ROOT/retarget_research/advanced_policy/compare_category_id_ablation.py" \
  --run-root "$RUN_ROOT" --data-root "$DATA_ROOT" --evaluation-name "$EVALUATION_NAME"
