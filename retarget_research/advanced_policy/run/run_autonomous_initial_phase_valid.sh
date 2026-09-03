#!/usr/bin/env bash
# 三只手并行评测完整100条valid；禁止expert-wrist和residual-RL checkpoint。
set -euo pipefail

ROOT="/home/lekangwan/projects/DexGraspMotionChallenge2025-final-release"
PY="/home/lekangwan/miniconda3/envs/hand-retarget/bin/python"
EVAL="$ROOT/retarget_research/advanced_policy/evaluate_policy_manifest.py"
MANIFEST="$ROOT/retarget_research/manifests/formal_50c_100o_1000t_seed20260808.json"
SPLIT="$ROOT/retarget_research/advanced_policy/data/formal_v1/policy_split_seed20260813.json"
RUN_ROOT="$ROOT/retarget_research/advanced_policy/runs/autonomous_initial_phase_v1"
DATA_ROOT="$ROOT/retarget_research/advanced_policy/data/formal_final_30cm"

cd "$ROOT"
export MPLCONFIGDIR=/tmp/matplotlib-retarget
pids=()

run_hand() {
  local label="$1" hand="$2" target="$3"
  local experiment="${label}_initial_phase_v1"
  "$PY" "$EVAL" \
    --hand "$hand" --manifest "$MANIFEST" --policy-split "$SPLIT" --split valid \
    --target-dir "$target" --checkpoint "$RUN_ROOT/$experiment/best.pt" \
    --data-dir "$DATA_ROOT/$label" --output-dir "$RUN_ROOT/$experiment/closed_loop_valid" \
    --device cuda --workers 2 --autonomous-only --resume
}

run_hand linker linker "$ROOT/retarget_research/outputs/formal_1000/linker_vector_v2alpha_c3g8_v1" \
  2>&1 | sed -u 's/^/[linker] /' & pids+=("$!")
run_hand xhand_official xhand "$ROOT/retarget_research/outputs/formal_1000/xhand_official" \
  2>&1 | sed -u 's/^/[xhand] /' & pids+=("$!")
run_hand wuji_old wuji "$ROOT/retarget_research/outputs/formal_1000/wuji_v1" \
  2>&1 | sed -u 's/^/[wuji] /' & pids+=("$!")

for pid in "${pids[@]}"; do
  wait "$pid"
done
"$PY" "$ROOT/retarget_research/advanced_policy/summarize_autonomous_initial_phase.py" \
  --run-root "$RUN_ROOT"
echo "AUTONOMOUS_INITIAL_PHASE_VALID=COMPLETE"
