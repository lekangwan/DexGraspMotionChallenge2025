#!/usr/bin/env bash
# train成功专家轨迹上采集DAgger-R1，训练有界反馈策略，再评测100条valid。
set -euo pipefail

ROOT="/home/lekangwan/projects/DexGraspMotionChallenge2025-final-release"
PY="/home/lekangwan/miniconda3/envs/hand-retarget/bin/python"
EVAL="$ROOT/retarget_research/advanced_policy/evaluate_policy_manifest.py"
MANIFEST="$ROOT/retarget_research/manifests/formal_50c_100o_1000t_seed20260808.json"
SPLIT="$ROOT/retarget_research/advanced_policy/data/formal_v1/policy_split_seed20260813.json"
DATA="$ROOT/retarget_research/advanced_policy/data/formal_final_30cm"
BASE="$ROOT/retarget_research/advanced_policy/runs/autonomous_initial_phase_delta_v1"
RUN="$ROOT/retarget_research/advanced_policy/runs/autonomous_initial_feedback_dagger_v1"
INDEX="$ROOT/retarget_research/advanced_policy/configs/generated/autonomous_initial_feedback_dagger_v1/config_index.json"

cd "$ROOT"
mkdir -p "$RUN/online_data"

collect() {
  local label="$1" hand="$2" target="$3"
  "$PY" "$EVAL" --hand "$hand" --manifest "$MANIFEST" --policy-split "$SPLIT" \
    --split train --expert-success-only --max-tasks-per-category 1 \
    --target-dir "$target" --checkpoint "$BASE/${label}_initial_phase_delta_v1/best.pt" \
    --data-dir "$DATA/$label" --output-dir "$RUN/${label}_dagger_r1_collection" \
    --teacher-checkpoint phase_expert --online-data-dir "$RUN/${label}_dagger_r1_raw" \
    --device cuda --workers 2 --resume
}
pids=()
collect linker linker "$ROOT/retarget_research/outputs/formal_1000/linker_vector_v2alpha_c3g8_v1" 2>&1 | sed -u 's/^/[collect-linker] /' & pids+=("$!")
collect xhand_official xhand "$ROOT/retarget_research/outputs/formal_1000/xhand_official" 2>&1 | sed -u 's/^/[collect-xhand] /' & pids+=("$!")
collect wuji_old wuji "$ROOT/retarget_research/outputs/formal_1000/wuji_v1" 2>&1 | sed -u 's/^/[collect-wuji] /' & pids+=("$!")
for pid in "${pids[@]}"; do wait "$pid"; done

for label in linker xhand_official wuji_old; do
  output="$RUN/online_data/${label}_r1.npz"
  if [[ ! -f "$output" ]]; then
    "$PY" retarget_research/advanced_policy/prepare/aggregate_online_data.py \
      --online-dir "$RUN/${label}_dagger_r1_raw" --data-dir "$DATA/$label" --output "$output"
  fi
done
"$PY" retarget_research/advanced_policy/prepare/prepare_initial_feedback_configs.py --project-root "$ROOT"

pids=()
for label in linker xhand_official wuji_old; do
  "$PY" retarget_research/advanced_policy/run_training_matrix.py --index "$INDEX" \
    --filter "$label" --device cuda 2>&1 | sed -u "s/^/[train-${label}] /" & pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done

validate() {
  local label="$1" hand="$2" target="$3"
  "$PY" "$EVAL" --hand "$hand" --manifest "$MANIFEST" --policy-split "$SPLIT" \
    --split valid --target-dir "$target" \
    --checkpoint "$RUN/${label}_initial_phase_feedback_v1/best.pt" \
    --data-dir "$DATA/$label" --output-dir "$RUN/${label}_initial_phase_feedback_v1/closed_loop_valid" \
    --device cuda --workers 2 --autonomous-only --action-rate-limit-scale 2.0 --resume
}
pids=()
validate linker linker "$ROOT/retarget_research/outputs/formal_1000/linker_vector_v2alpha_c3g8_v1" 2>&1 | sed -u 's/^/[valid-linker] /' & pids+=("$!")
validate xhand_official xhand "$ROOT/retarget_research/outputs/formal_1000/xhand_official" 2>&1 | sed -u 's/^/[valid-xhand] /' & pids+=("$!")
validate wuji_old wuji "$ROOT/retarget_research/outputs/formal_1000/wuji_v1" 2>&1 | sed -u 's/^/[valid-wuji] /' & pids+=("$!")
for pid in "${pids[@]}"; do wait "$pid"; done

"$PY" retarget_research/advanced_policy/summarize_autonomous_initial_phase.py \
  --run-root "$RUN" --split valid --experiment-suffix initial_phase_feedback_v1
echo "AUTONOMOUS_INITIAL_FEEDBACK_DAGGER_V1=COMPLETE"
