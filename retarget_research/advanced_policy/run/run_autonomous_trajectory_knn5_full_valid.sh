#!/usr/bin/env bash
# 冻结三手wrist-5NN，复用已有50条报告并补齐完整100条valid。
set -euo pipefail

ROOT="/home/lekangwan/projects/DexGraspMotionChallenge2025-final-release"
PY="/home/lekangwan/miniconda3/envs/hand-retarget/bin/python"
EVAL="$ROOT/retarget_research/advanced_policy/evaluate_policy_manifest.py"
MANIFEST="$ROOT/retarget_research/manifests/formal_50c_100o_1000t_seed20260808.json"
SPLIT="$ROOT/retarget_research/advanced_policy/data/formal_v1/policy_split_seed20260813.json"
DATA="$ROOT/retarget_research/advanced_policy/data/formal_final_30cm"
RUN="$ROOT/retarget_research/advanced_policy/runs/autonomous_trajectory_retrieval_v1"

cd "$ROOT"
evaluate() {
  local label="$1" hand="$2" target="$3"
  local experiment="${label}_trajectory_knn5_v1"
  mkdir -p "$RUN/$experiment/closed_loop_valid"
  cp -a "$RUN/$experiment/closed_loop_valid50/." "$RUN/$experiment/closed_loop_valid/"
  "$PY" "$EVAL" --hand "$hand" --manifest "$MANIFEST" --policy-split "$SPLIT" \
    --split valid --target-dir "$target" --checkpoint "$RUN/$experiment/best.pt" \
    --data-dir "$DATA/$label" --output-dir "$RUN/$experiment/closed_loop_valid" \
    --device cuda --workers 2 --autonomous-only --resume
}
pids=()
evaluate linker linker "$ROOT/retarget_research/outputs/formal_1000/linker_vector_v2alpha_c3g8_v1" 2>&1 | sed -u 's/^/[linker] /' & pids+=("$!")
evaluate xhand_official xhand "$ROOT/retarget_research/outputs/formal_1000/xhand_official" 2>&1 | sed -u 's/^/[xhand] /' & pids+=("$!")
evaluate wuji_old wuji "$ROOT/retarget_research/outputs/formal_1000/wuji_v1" 2>&1 | sed -u 's/^/[wuji] /' & pids+=("$!")
for pid in "${pids[@]}"; do wait "$pid"; done
"$PY" retarget_research/advanced_policy/summarize_autonomous_initial_phase.py \
  --run-root "$RUN" --split valid --experiment-suffix trajectory_knn5_v1
echo "AUTONOMOUS_TRAJECTORY_KNN5_FULL_VALID=COMPLETE"
