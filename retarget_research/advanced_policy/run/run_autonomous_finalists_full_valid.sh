#!/usr/bin/env bash
# 将50条screen胜者补齐为每类2条、共100条valid；已有结果通过resume复用。
set -euo pipefail

ROOT="/home/lekangwan/projects/DexGraspMotionChallenge2025-final-release"
PY="/home/lekangwan/miniconda3/envs/hand-retarget/bin/python"
EVAL="$ROOT/retarget_research/advanced_policy/evaluate_policy_manifest.py"
MANIFEST="$ROOT/retarget_research/manifests/formal_50c_100o_1000t_seed20260808.json"
SPLIT="$ROOT/retarget_research/advanced_policy/data/formal_v1/policy_split_seed20260813.json"
DATA="$ROOT/retarget_research/advanced_policy/data/formal_final_30cm"

cd "$ROOT"
evaluate() {
  local tag="$1" checkpoint="$2" screen="$3" label="$4" hand="$5" target="$6"
  local output="${screen%50}"
  mkdir -p "$output"
  cp -a "$screen/." "$output/"
  "$PY" "$EVAL" --hand "$hand" --manifest "$MANIFEST" --policy-split "$SPLIT" \
    --split valid --target-dir "$target" --checkpoint "$checkpoint" \
    --data-dir "$DATA/$label" --output-dir "$output" --device cuda \
    --workers 2 --autonomous-only --resume 2>&1 | sed -u "s/^/[$tag] /"
}

pids=()
evaluate xhand-ridge \
  "$ROOT/retarget_research/advanced_policy/runs/autonomous_local_ridge_v1/xhand_official_local_ridge_v1/best.pt" \
  "$ROOT/retarget_research/advanced_policy/runs/autonomous_local_ridge_v1/xhand_official_local_ridge_v1/closed_loop_valid50" \
  xhand_official xhand "$ROOT/retarget_research/outputs/formal_1000/xhand_official" & pids+=("$!")
evaluate xhand-so3 \
  "$ROOT/retarget_research/advanced_policy/runs/autonomous_so3_knn5_v1/xhand_official_so3_knn5_v1/best.pt" \
  "$ROOT/retarget_research/advanced_policy/runs/autonomous_so3_knn5_v1/xhand_official_so3_knn5_v1/closed_loop_valid50" \
  xhand_official xhand "$ROOT/retarget_research/outputs/formal_1000/xhand_official" & pids+=("$!")
evaluate xhand-finger1 \
  "$ROOT/retarget_research/advanced_policy/runs/autonomous_wrist5_finger1_v1/xhand_official_wrist5_finger1_v1/best.pt" \
  "$ROOT/retarget_research/advanced_policy/runs/autonomous_wrist5_finger1_v1/xhand_official_wrist5_finger1_v1/closed_loop_valid50" \
  xhand_official xhand "$ROOT/retarget_research/outputs/formal_1000/xhand_official" & pids+=("$!")
evaluate wuji-blend25 \
  "$ROOT/retarget_research/advanced_policy/runs/autonomous_sequence_candidates_v1/wuji_old_knn5_mlp_a25_v1/best.pt" \
  "$ROOT/retarget_research/advanced_policy/runs/autonomous_sequence_candidates_v1/wuji_old_knn5_mlp_a25_v1/closed_loop_valid50" \
  wuji_old wuji "$ROOT/retarget_research/outputs/formal_1000/wuji_v1" & pids+=("$!")

for pid in "${pids[@]}"; do wait "$pid"; done
echo "AUTONOMOUS_FINALISTS_FULL_VALID=COMPLETE"
