#!/usr/bin/env bash
# 冻结三手最终方法，在500条对象隔离test上各评测一次。
set -euo pipefail

echo "ABORTED: 该脚本中的检索/局部回归checkpoint携带训练轨迹库，不符合纯参数化自主策略要求。" >&2
exit 2

ROOT="/home/lekangwan/projects/DexGraspMotionChallenge2025-final-release"
PY="/home/lekangwan/miniconda3/envs/hand-retarget/bin/python"
EVAL="$ROOT/retarget_research/advanced_policy/evaluate_policy_manifest.py"
MANIFEST="$ROOT/retarget_research/manifests/formal_50c_100o_1000t_seed20260808.json"
SPLIT="$ROOT/retarget_research/advanced_policy/data/formal_v1/policy_split_seed20260813.json"
DATA="$ROOT/retarget_research/advanced_policy/data/formal_final_30cm"
RUN="$ROOT/retarget_research/advanced_policy/runs/autonomous_final_test_v1"

cd "$ROOT"
evaluate() {
  local tag="$1" label="$2" hand="$3" checkpoint="$4" target="$5"
  "$PY" "$EVAL" --hand "$hand" --manifest "$MANIFEST" --policy-split "$SPLIT" \
    --split test --target-dir "$target" --checkpoint "$checkpoint" \
    --data-dir "$DATA/$label" --output-dir "$RUN/$tag" --device cuda \
    --workers 2 --autonomous-only --resume 2>&1 | sed -u "s/^/[$tag] /"
}

pids=()
evaluate linker linker linker \
  "$ROOT/retarget_research/advanced_policy/runs/autonomous_trajectory_retrieval_v1/linker_trajectory_knn5_v1/best.pt" \
  "$ROOT/retarget_research/outputs/formal_1000/linker_vector_v2alpha_c3g8_v1" & pids+=("$!")
evaluate xhand xhand_official xhand \
  "$ROOT/retarget_research/advanced_policy/runs/autonomous_local_ridge_v1/xhand_official_local_ridge_v1/best.pt" \
  "$ROOT/retarget_research/outputs/formal_1000/xhand_official" & pids+=("$!")
evaluate wuji wuji_old wuji \
  "$ROOT/retarget_research/advanced_policy/runs/autonomous_sequence_candidates_v1/wuji_old_knn5_mlp_a25_v1/best.pt" \
  "$ROOT/retarget_research/outputs/formal_1000/wuji_v1" & pids+=("$!")

for pid in "${pids[@]}"; do wait "$pid"; done
echo "AUTONOMOUS_FINAL_TEST=COMPLETE"
