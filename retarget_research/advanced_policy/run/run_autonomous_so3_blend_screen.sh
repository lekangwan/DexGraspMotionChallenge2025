#!/usr/bin/env bash
# 只评测SO(3)有效的XHand和Wuji，并沿用各自已筛出的融合比例。
set -euo pipefail

ROOT="/home/lekangwan/projects/DexGraspMotionChallenge2025-final-release"
PY="/home/lekangwan/miniconda3/envs/hand-retarget/bin/python"
BUILD="$ROOT/retarget_research/advanced_policy/prepare/build_trajectory_se3_blend.py"
EVAL="$ROOT/retarget_research/advanced_policy/evaluate_policy_manifest.py"
MANIFEST="$ROOT/retarget_research/manifests/formal_50c_100o_1000t_seed20260808.json"
SPLIT="$ROOT/retarget_research/advanced_policy/data/formal_v1/policy_split_seed20260813.json"
DATA="$ROOT/retarget_research/advanced_policy/data/formal_final_30cm"
SO3="$ROOT/retarget_research/advanced_policy/runs/autonomous_so3_knn5_v1"
MLP="$ROOT/retarget_research/advanced_policy/runs/autonomous_initial_phase_delta_v1"
RUN="$ROOT/retarget_research/advanced_policy/runs/autonomous_so3_blend_v1"

cd "$ROOT"
for spec in "xhand_official:0.50" "wuji_old:0.25"; do
  label="${spec%%:*}"; alpha="${spec##*:}"
  output="$RUN/${label}_so3_blend_v1/best.pt"
  if [[ ! -f "$output" ]]; then
    "$PY" "$BUILD" --retrieval "$SO3/${label}_so3_knn5_v1/best.pt" \
      --learned "$MLP/${label}_initial_phase_delta_v1/best.pt" \
      --alpha "$alpha" --output "$output"
  fi
done

evaluate() {
  local label="$1" hand="$2" target="$3"
  "$PY" "$EVAL" --hand "$hand" --manifest "$MANIFEST" --policy-split "$SPLIT" \
    --split valid --max-tasks-per-category 1 --target-dir "$target" \
    --checkpoint "$RUN/${label}_so3_blend_v1/best.pt" --data-dir "$DATA/$label" \
    --output-dir "$RUN/${label}_so3_blend_v1/closed_loop_valid50" \
    --device cuda --workers 2 --autonomous-only --resume
}

pids=()
evaluate xhand_official xhand "$ROOT/retarget_research/outputs/formal_1000/xhand_official" 2>&1 | sed -u 's/^/[xhand] /' & pids+=("$!")
evaluate wuji_old wuji "$ROOT/retarget_research/outputs/formal_1000/wuji_v1" 2>&1 | sed -u 's/^/[wuji] /' & pids+=("$!")
for pid in "${pids[@]}"; do wait "$pid"; done
echo "AUTONOMOUS_SO3_BLEND_SCREEN=COMPLETE"
