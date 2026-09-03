#!/usr/bin/env bash
# 在相同50条valid上筛选5NN-MLP融合与PCA-RBF整段生成。
set -euo pipefail

ROOT="/home/lekangwan/projects/DexGraspMotionChallenge2025-final-release"
PY="/home/lekangwan/miniconda3/envs/hand-retarget/bin/python"
EVAL="$ROOT/retarget_research/advanced_policy/evaluate_policy_manifest.py"
MANIFEST="$ROOT/retarget_research/manifests/formal_50c_100o_1000t_seed20260808.json"
SPLIT="$ROOT/retarget_research/advanced_policy/data/formal_v1/policy_split_seed20260813.json"
DATA="$ROOT/retarget_research/advanced_policy/data/formal_final_30cm"
KNN="$ROOT/retarget_research/advanced_policy/runs/autonomous_trajectory_retrieval_v1"
MLP="$ROOT/retarget_research/advanced_policy/runs/autonomous_initial_phase_delta_v1"
RUN="$ROOT/retarget_research/advanced_policy/runs/autonomous_sequence_candidates_v1"

cd "$ROOT"
for label in linker xhand_official wuji_old; do
  for code in 25 50; do
    alpha="0.$code"
    output="$RUN/${label}_knn5_mlp_a${code}_v1/best.pt"
    if [[ ! -f "$output" ]]; then
      "$PY" retarget_research/advanced_policy/prepare/build_trajectory_blend.py \
        --retrieval "$KNN/${label}_trajectory_knn5_v1/best.pt" \
        --learned "$MLP/${label}_initial_phase_delta_v1/best.pt" \
        --alpha "$alpha" --output "$output"
    fi
  done
  pca="$RUN/${label}_pca_rbf_v1/best.pt"
  if [[ ! -f "$pca" ]]; then
    "$PY" retarget_research/advanced_policy/prepare/build_trajectory_pca_rbf.py \
      --data-dir "$DATA/$label" --output "$pca"
  fi
done

evaluate() {
  local label="$1" hand="$2" target="$3" experiment="$4"
  "$PY" "$EVAL" --hand "$hand" --manifest "$MANIFEST" --policy-split "$SPLIT" \
    --split valid --max-tasks-per-category 1 --target-dir "$target" \
    --checkpoint "$RUN/${label}_${experiment}/best.pt" --data-dir "$DATA/$label" \
    --output-dir "$RUN/${label}_${experiment}/closed_loop_valid50" \
    --device cuda --workers 1 --autonomous-only --resume
}

run_group() {
  local experiment="$1"
  pids=()
  evaluate linker linker "$ROOT/retarget_research/outputs/formal_1000/linker_vector_v2alpha_c3g8_v1" "$experiment" 2>&1 | sed -u "s/^/[linker-${experiment}] /" & pids+=("$!")
  evaluate xhand_official xhand "$ROOT/retarget_research/outputs/formal_1000/xhand_official" "$experiment" 2>&1 | sed -u "s/^/[xhand-${experiment}] /" & pids+=("$!")
  evaluate wuji_old wuji "$ROOT/retarget_research/outputs/formal_1000/wuji_v1" "$experiment" 2>&1 | sed -u "s/^/[wuji-${experiment}] /" & pids+=("$!")
  for pid in "${pids[@]}"; do wait "$pid"; done
}

run_group knn5_mlp_a25_v1
run_group knn5_mlp_a50_v1
run_group pca_rbf_v1
echo "AUTONOMOUS_SEQUENCE_CANDIDATES_SCREEN=COMPLETE"
