#!/usr/bin/env bash
# train成功任务采集状态对齐DAgger-R1，训练三帧反馈并评测50条valid。
set -euo pipefail

ROOT="/home/lekangwan/projects/DexGraspMotionChallenge2025-final-release"
PY="/home/lekangwan/miniconda3/envs/hand-retarget/bin/python"
EVAL="$ROOT/retarget_research/advanced_policy/evaluate_policy_manifest.py"
TRAIN="$ROOT/retarget_research/advanced_policy/train.py"
MANIFEST="$ROOT/retarget_research/manifests/formal_50c_100o_1000t_seed20260808.json"
SPLIT="$ROOT/retarget_research/advanced_policy/data/formal_v1/policy_split_seed20260813.json"
DATA="$ROOT/retarget_research/advanced_policy/data/formal_final_30cm"
BASE="$ROOT/retarget_research/advanced_policy/runs/autonomous_initial_phase_delta_v1"
RUN="$ROOT/retarget_research/advanced_policy/runs/autonomous_state_aligned_dagger_v1"
CONFIG="$ROOT/retarget_research/advanced_policy/configs/generated/autonomous_state_aligned_dagger_v1"

cd "$ROOT"
collect() {
  local label="$1" hand="$2" target="$3"
  "$PY" "$EVAL" --hand "$hand" --manifest "$MANIFEST" --policy-split "$SPLIT" \
    --split train --expert-success-only --max-tasks-per-category 1 \
    --target-dir "$target" --checkpoint "$BASE/${label}_initial_phase_delta_v1/best.pt" \
    --data-dir "$DATA/$label" --output-dir "$RUN/${label}_collection" \
    --teacher-checkpoint state_aligned_expert --online-data-dir "$RUN/${label}_raw" \
    --device cuda --workers 2 --resume
}
pids=()
collect linker linker "$ROOT/retarget_research/outputs/formal_1000/linker_vector_v2alpha_c3g8_v1" 2>&1 | sed -u 's/^/[collect-linker] /' & pids+=("$!")
collect xhand_official xhand "$ROOT/retarget_research/outputs/formal_1000/xhand_official" 2>&1 | sed -u 's/^/[collect-xhand] /' & pids+=("$!")
collect wuji_old wuji "$ROOT/retarget_research/outputs/formal_1000/wuji_v1" 2>&1 | sed -u 's/^/[collect-wuji] /' & pids+=("$!")
for pid in "${pids[@]}"; do wait "$pid"; done

mkdir -p "$RUN/online_data"
for label in linker xhand_official wuji_old; do
  output="$RUN/online_data/${label}_r1.npz"
  if [[ ! -f "$output" ]]; then
    "$PY" retarget_research/advanced_policy/prepare/aggregate_online_data.py \
      --online-dir "$RUN/${label}_raw" --data-dir "$DATA/$label" --output "$output"
  fi
done

pids=()
for label in linker xhand_official wuji_old; do
  "$PY" "$TRAIN" --config "$CONFIG/${label}_state_aligned_dagger_v1.json" --device cuda \
    2>&1 | sed -u "s/^/[train-$label] /" & pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done

validate() {
  local label="$1" hand="$2" target="$3"
  "$PY" "$EVAL" --hand "$hand" --manifest "$MANIFEST" --policy-split "$SPLIT" \
    --split valid --max-tasks-per-category 1 --target-dir "$target" \
    --checkpoint "$RUN/${label}_state_aligned_dagger_v1/best.pt" \
    --data-dir "$DATA/$label" --output-dir "$RUN/${label}_state_aligned_dagger_v1/closed_loop_valid50" \
    --device cuda --workers 2 --autonomous-only --resume
}
pids=()
validate linker linker "$ROOT/retarget_research/outputs/formal_1000/linker_vector_v2alpha_c3g8_v1" 2>&1 | sed -u 's/^/[valid-linker] /' & pids+=("$!")
validate xhand_official xhand "$ROOT/retarget_research/outputs/formal_1000/xhand_official" 2>&1 | sed -u 's/^/[valid-xhand] /' & pids+=("$!")
validate wuji_old wuji "$ROOT/retarget_research/outputs/formal_1000/wuji_v1" 2>&1 | sed -u 's/^/[valid-wuji] /' & pids+=("$!")
for pid in "${pids[@]}"; do wait "$pid"; done
echo "AUTONOMOUS_STATE_ALIGNED_DAGGER_SCREEN=COMPLETE"
