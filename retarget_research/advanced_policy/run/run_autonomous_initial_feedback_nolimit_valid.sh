#!/usr/bin/env bash
# 同一DAgger反馈checkpoint关闭动作限速，只做100条valid因果消融。
set -euo pipefail

ROOT="/home/lekangwan/projects/DexGraspMotionChallenge2025-final-release"
PY="/home/lekangwan/miniconda3/envs/hand-retarget/bin/python"
EVAL="$ROOT/retarget_research/advanced_policy/evaluate_policy_manifest.py"
MANIFEST="$ROOT/retarget_research/manifests/formal_50c_100o_1000t_seed20260808.json"
SPLIT="$ROOT/retarget_research/advanced_policy/data/formal_v1/policy_split_seed20260813.json"
DATA="$ROOT/retarget_research/advanced_policy/data/formal_final_30cm"
RUN="$ROOT/retarget_research/advanced_policy/runs/autonomous_initial_feedback_dagger_v1"

cd "$ROOT"
validate() {
  local label="$1" hand="$2" target="$3"
  "$PY" "$EVAL" --hand "$hand" --manifest "$MANIFEST" --policy-split "$SPLIT" \
    --split valid --target-dir "$target" \
    --checkpoint "$RUN/${label}_initial_phase_feedback_v1/best.pt" \
    --data-dir "$DATA/$label" \
    --output-dir "$RUN/${label}_initial_phase_feedback_v1/closed_loop_valid_nolimit" \
    --device cuda --workers 2 --autonomous-only --action-rate-limit-scale 0 --resume
}
pids=()
validate linker linker "$ROOT/retarget_research/outputs/formal_1000/linker_vector_v2alpha_c3g8_v1" 2>&1 | sed -u 's/^/[linker] /' & pids+=("$!")
validate xhand_official xhand "$ROOT/retarget_research/outputs/formal_1000/xhand_official" 2>&1 | sed -u 's/^/[xhand] /' & pids+=("$!")
validate wuji_old wuji "$ROOT/retarget_research/outputs/formal_1000/wuji_v1" 2>&1 | sed -u 's/^/[wuji] /' & pids+=("$!")
for pid in "${pids[@]}"; do wait "$pid"; done
echo "AUTONOMOUS_INITIAL_FEEDBACK_NOLIMIT_VALID=COMPLETE"
