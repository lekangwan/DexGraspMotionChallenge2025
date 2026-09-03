#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/lekangwan/projects/DexGraspMotionChallenge2025-final-release"
PY="/home/lekangwan/miniconda3/envs/hand-retarget/bin/python"
EVAL="$ROOT/retarget_research/advanced_policy/evaluate_policy_manifest.py"
MANIFEST="$ROOT/retarget_research/manifests/formal_50c_100o_1000t_seed20260808.json"
SPLIT="$ROOT/retarget_research/advanced_policy/data/formal_v1/policy_split_seed20260813.json"
RUN_ROOT="$ROOT/retarget_research/advanced_policy/runs/autonomous_initial_phase_delta_v1"
DATA="$ROOT/retarget_research/advanced_policy/data/formal_final_30cm"

cd "$ROOT"
pids=()
run_hand() {
  local label="$1" hand="$2" target="$3"
  "$PY" "$EVAL" --hand "$hand" --manifest "$MANIFEST" --policy-split "$SPLIT" \
    --split valid --target-dir "$target" \
    --checkpoint "$RUN_ROOT/${label}_initial_phase_delta_v1/best.pt" \
    --data-dir "$DATA/$label" \
    --output-dir "$RUN_ROOT/${label}_initial_phase_delta_v1/closed_loop_valid" \
    --device cuda --workers 2 --autonomous-only --resume
}
run_hand linker linker "$ROOT/retarget_research/outputs/formal_1000/linker_vector_v2alpha_c3g8_v1" 2>&1 | sed -u 's/^/[linker] /' & pids+=("$!")
run_hand xhand_official xhand "$ROOT/retarget_research/outputs/formal_1000/xhand_official" 2>&1 | sed -u 's/^/[xhand] /' & pids+=("$!")
run_hand wuji_old wuji "$ROOT/retarget_research/outputs/formal_1000/wuji_v1" 2>&1 | sed -u 's/^/[wuji] /' & pids+=("$!")
for pid in "${pids[@]}"; do wait "$pid"; done
"$PY" retarget_research/advanced_policy/summarize_autonomous_initial_phase.py \
  --run-root "$RUN_ROOT" --split valid --experiment-suffix initial_phase_delta_v1
echo "AUTONOMOUS_INITIAL_PHASE_DELTA_VALID=COMPLETE"
