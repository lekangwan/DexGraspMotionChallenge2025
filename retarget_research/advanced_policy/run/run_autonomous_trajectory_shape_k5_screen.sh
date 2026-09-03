#!/usr/bin/env bash
# 固定5NN，只测试XHand/Wuji是否应把14维物体形状加入检索距离。
set -euo pipefail

ROOT="/home/lekangwan/projects/DexGraspMotionChallenge2025-final-release"
PY="/home/lekangwan/miniconda3/envs/hand-retarget/bin/python"
EVAL="$ROOT/retarget_research/advanced_policy/evaluate_policy_manifest.py"
MANIFEST="$ROOT/retarget_research/manifests/formal_50c_100o_1000t_seed20260808.json"
SPLIT="$ROOT/retarget_research/advanced_policy/data/formal_v1/policy_split_seed20260813.json"
DATA="$ROOT/retarget_research/advanced_policy/data/formal_final_30cm"
RUN="$ROOT/retarget_research/advanced_policy/runs/autonomous_trajectory_retrieval_v1"

cd "$ROOT"
build_and_evaluate() {
  local label="$1" hand="$2" target="$3"
  local experiment="${label}_trajectory_wrist_shape_knn5_v1"
  if [[ ! -f "$RUN/$experiment/best.pt" ]]; then
    "$PY" retarget_research/advanced_policy/prepare/build_trajectory_retrieval.py \
      --data-dir "$DATA/$label" --output "$RUN/$experiment/best.pt" \
      --k 5 --features wrist_shape
  fi
  "$PY" "$EVAL" --hand "$hand" --manifest "$MANIFEST" --policy-split "$SPLIT" \
    --split valid --max-tasks-per-category 1 --target-dir "$target" \
    --checkpoint "$RUN/$experiment/best.pt" --data-dir "$DATA/$label" \
    --output-dir "$RUN/$experiment/closed_loop_valid50" \
    --device cuda --workers 2 --autonomous-only --resume
}
pids=()
build_and_evaluate xhand_official xhand "$ROOT/retarget_research/outputs/formal_1000/xhand_official" 2>&1 | sed -u 's/^/[xhand] /' & pids+=("$!")
build_and_evaluate wuji_old wuji "$ROOT/retarget_research/outputs/formal_1000/wuji_v1" 2>&1 | sed -u 's/^/[wuji] /' & pids+=("$!")
for pid in "${pids[@]}"; do wait "$pid"; done
echo "AUTONOMOUS_TRAJECTORY_SHAPE_K5_SCREEN=COMPLETE"
