#!/usr/bin/env bash
# 构造1NN完整复制和5NN平滑混合，并在每类1条、共50条valid上筛选。
set -euo pipefail

ROOT="/home/lekangwan/projects/DexGraspMotionChallenge2025-final-release"
PY="/home/lekangwan/miniconda3/envs/hand-retarget/bin/python"
EVAL="$ROOT/retarget_research/advanced_policy/evaluate_policy_manifest.py"
MANIFEST="$ROOT/retarget_research/manifests/formal_50c_100o_1000t_seed20260808.json"
SPLIT="$ROOT/retarget_research/advanced_policy/data/formal_v1/policy_split_seed20260813.json"
DATA="$ROOT/retarget_research/advanced_policy/data/formal_final_30cm"
RUN="$ROOT/retarget_research/advanced_policy/runs/autonomous_trajectory_retrieval_v1"

cd "$ROOT"
for label in linker xhand_official wuji_old; do
  for k in 1 5; do
    output="$RUN/${label}_trajectory_knn${k}_v1/best.pt"
    if [[ ! -f "$output" ]]; then
      "$PY" retarget_research/advanced_policy/prepare/build_trajectory_retrieval.py \
        --data-dir "$DATA/$label" --output "$output" --k "$k"
    fi
  done
done

evaluate() {
  local label="$1" hand="$2" target="$3" k="$4"
  "$PY" "$EVAL" --hand "$hand" --manifest "$MANIFEST" --policy-split "$SPLIT" \
    --split valid --max-tasks-per-category 1 --target-dir "$target" \
    --checkpoint "$RUN/${label}_trajectory_knn${k}_v1/best.pt" \
    --data-dir "$DATA/$label" \
    --output-dir "$RUN/${label}_trajectory_knn${k}_v1/closed_loop_valid50" \
    --device cuda --workers 1 --autonomous-only --resume
}
pids=()
for k in 1 5; do
  evaluate linker linker "$ROOT/retarget_research/outputs/formal_1000/linker_vector_v2alpha_c3g8_v1" "$k" 2>&1 | sed -u "s/^/[linker-k${k}] /" & pids+=("$!")
  evaluate xhand_official xhand "$ROOT/retarget_research/outputs/formal_1000/xhand_official" "$k" 2>&1 | sed -u "s/^/[xhand-k${k}] /" & pids+=("$!")
  evaluate wuji_old wuji "$ROOT/retarget_research/outputs/formal_1000/wuji_v1" "$k" 2>&1 | sed -u "s/^/[wuji-k${k}] /" & pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done
echo "AUTONOMOUS_TRAJECTORY_RETRIEVAL_SCREEN=COMPLETE"
