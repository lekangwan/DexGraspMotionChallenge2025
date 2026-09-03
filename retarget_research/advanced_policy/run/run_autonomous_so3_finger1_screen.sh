#!/usr/bin/env bash
# XHand/Wuji：世界5NN平移、SO(3)旋转、1NN完整抓形。
set -euo pipefail

ROOT="/home/lekangwan/projects/DexGraspMotionChallenge2025-final-release"
PY="/home/lekangwan/miniconda3/envs/hand-retarget/bin/python"
BUILD="$ROOT/retarget_research/advanced_policy/prepare/build_trajectory_se3_retrieval.py"
EVAL="$ROOT/retarget_research/advanced_policy/evaluate_policy_manifest.py"
MANIFEST="$ROOT/retarget_research/manifests/formal_50c_100o_1000t_seed20260808.json"
SPLIT="$ROOT/retarget_research/advanced_policy/data/formal_v1/policy_split_seed20260813.json"
DATA="$ROOT/retarget_research/advanced_policy/data/formal_final_30cm"
RUN="$ROOT/retarget_research/advanced_policy/runs/autonomous_so3_finger1_v1"

cd "$ROOT"
for label in xhand_official wuji_old; do
  output="$RUN/${label}_so3_finger1_v1/best.pt"
  if [[ ! -f "$output" ]]; then
    "$PY" "$BUILD" --data-dir "$DATA/$label" --output "$output" --k 5 \
      --finger-k 1 --translation-frame world
  fi
done

evaluate() {
  local label="$1" hand="$2" target="$3"
  "$PY" "$EVAL" --hand "$hand" --manifest "$MANIFEST" --policy-split "$SPLIT" \
    --split valid --max-tasks-per-category 1 --target-dir "$target" \
    --checkpoint "$RUN/${label}_so3_finger1_v1/best.pt" --data-dir "$DATA/$label" \
    --output-dir "$RUN/${label}_so3_finger1_v1/closed_loop_valid50" \
    --device cuda --workers 2 --autonomous-only --resume
}

pids=()
evaluate xhand_official xhand "$ROOT/retarget_research/outputs/formal_1000/xhand_official" 2>&1 | sed -u 's/^/[xhand] /' & pids+=("$!")
evaluate wuji_old wuji "$ROOT/retarget_research/outputs/formal_1000/wuji_v1" 2>&1 | sed -u 's/^/[wuji] /' & pids+=("$!")
for pid in "${pids[@]}"; do wait "$pid"; done
echo "AUTONOMOUS_SO3_FINGER1_SCREEN=COMPLETE"
