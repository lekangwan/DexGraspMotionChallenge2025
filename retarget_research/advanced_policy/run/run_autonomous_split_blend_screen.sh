#!/usr/bin/env bash
# 将每手最佳全动作融合比例拆成“只融合手腕”和“只融合手指”。
set -euo pipefail

ROOT="/home/lekangwan/projects/DexGraspMotionChallenge2025-final-release"
PY="/home/lekangwan/miniconda3/envs/hand-retarget/bin/python"
EVAL="$ROOT/retarget_research/advanced_policy/evaluate_policy_manifest.py"
MANIFEST="$ROOT/retarget_research/manifests/formal_50c_100o_1000t_seed20260808.json"
SPLIT="$ROOT/retarget_research/advanced_policy/data/formal_v1/policy_split_seed20260813.json"
DATA="$ROOT/retarget_research/advanced_policy/data/formal_final_30cm"
KNN="$ROOT/retarget_research/advanced_policy/runs/autonomous_trajectory_retrieval_v1"
MLP="$ROOT/retarget_research/advanced_policy/runs/autonomous_initial_phase_delta_v1"
RUN="$ROOT/retarget_research/advanced_policy/runs/autonomous_split_blend_v1"

cd "$ROOT"
alpha_for() {
  [[ "$1" == "xhand_official" ]] && echo 0.50 || echo 0.25
}
for label in linker xhand_official wuji_old; do
  alpha="$(alpha_for "$label")"
  for part in wrist finger; do
    wrist=0; finger=0
    [[ "$part" == "wrist" ]] && wrist="$alpha"
    [[ "$part" == "finger" ]] && finger="$alpha"
    output="$RUN/${label}_${part}_only_v1/best.pt"
    if [[ ! -f "$output" ]]; then
      "$PY" retarget_research/advanced_policy/prepare/build_trajectory_blend.py \
        --retrieval "$KNN/${label}_trajectory_knn5_v1/best.pt" \
        --learned "$MLP/${label}_initial_phase_delta_v1/best.pt" \
        --wrist-alpha "$wrist" --finger-alpha "$finger" --output "$output"
    fi
  done
done

evaluate() {
  local label="$1" hand="$2" target="$3" part="$4"
  "$PY" "$EVAL" --hand "$hand" --manifest "$MANIFEST" --policy-split "$SPLIT" \
    --split valid --max-tasks-per-category 1 --target-dir "$target" \
    --checkpoint "$RUN/${label}_${part}_only_v1/best.pt" --data-dir "$DATA/$label" \
    --output-dir "$RUN/${label}_${part}_only_v1/closed_loop_valid50" \
    --device cuda --workers 1 --autonomous-only --resume
}
for part in wrist finger; do
  pids=()
  evaluate linker linker "$ROOT/retarget_research/outputs/formal_1000/linker_vector_v2alpha_c3g8_v1" "$part" 2>&1 | sed -u "s/^/[linker-${part}] /" & pids+=("$!")
  evaluate xhand_official xhand "$ROOT/retarget_research/outputs/formal_1000/xhand_official" "$part" 2>&1 | sed -u "s/^/[xhand-${part}] /" & pids+=("$!")
  evaluate wuji_old wuji "$ROOT/retarget_research/outputs/formal_1000/wuji_v1" "$part" 2>&1 | sed -u "s/^/[wuji-${part}] /" & pids+=("$!")
  for pid in "${pids[@]}"; do wait "$pid"; done
done
echo "AUTONOMOUS_SPLIT_BLEND_SCREEN=COMPLETE"
