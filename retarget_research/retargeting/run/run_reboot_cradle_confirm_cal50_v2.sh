#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/lekangwan/projects/DexGraspMotionChallenge2025-final-release
PYTHON=/home/lekangwan/miniconda3/envs/hand-retarget/bin/python
CONFIRM=$ROOT/retarget_research/retargeting/run/confirm_physics_cem_manifest.py
PREPARE=$ROOT/retarget_research/retargeting/prepare/prepare_physics_cem_independent_eval.py
EVALUATOR=$ROOT/retarget_research/retargeting/evaluate/evaluate_hand_manifest.py
AUDITOR=$ROOT/retarget_research/retargeting/evaluate/audit_stable_success.py
PROTOCOL=$ROOT/retarget_research/retargeting/configs/stable_success_protocol_v3.json
BASE=$ROOT/retarget_research/outputs/reboot_phase_cem_cal50_v2
CANDIDATE=$ROOT/retarget_research/outputs/reboot_lift_cradle_cem_cal50_v1
EXP=$ROOT/retarget_research/outputs/reboot_lift_cradle_cem_cal50_v2_confirmed

mkdir -p "$EXP/raw" "$EXP/independent_targets" "$EXP/manifests" \
  "$EXP/logs" "$EXP/independent_evaluation" "$EXP/independent_traces"
cd "$ROOT"

confirm_hand() {
  local hand=$1
  MPLCONFIGDIR=/tmp/matplotlib-retarget OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  "$PYTHON" -u "$CONFIRM" --hand "$hand" \
    --manifest "$BASE/manifests/${hand}_cal50.json" \
    --baseline-target-dir "$BASE/independent_targets/$hand" \
    --candidate-summary "$CANDIDATE/raw/$hand/screen_summary.json" \
    --output-dir "$EXP/raw/$hand" --selection-margin 1.0 --device cpu \
    > "$EXP/logs/${hand}_confirm.log" 2>&1
}

confirm_hand linker & p1=$!
confirm_hand xhand & p2=$!
confirm_hand wuji & p3=$!
status=0
wait "$p1" || status=1
wait "$p2" || status=1
wait "$p3" || status=1
if [[ "$status" -ne 0 ]]; then
  for hand in linker xhand wuji; do tail -n 25 "$EXP/logs/${hand}_confirm.log"; done
  exit "$status"
fi

for hand in linker xhand wuji; do
  "$PYTHON" "$PREPARE" --screen-summary "$EXP/raw/$hand/screen_summary.json" \
    --formal-manifest "$BASE/manifests/${hand}_cal50.json" \
    --target-dir "$EXP/independent_targets/$hand" \
    --manifest-output "$EXP/manifests/${hand}_cal50.json"
done

evaluate_hand() {
  local hand=$1
  MPLCONFIGDIR=/tmp/matplotlib-retarget OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  "$PYTHON" -u "$EVALUATOR" --hand "$hand" \
    --manifest "$EXP/manifests/${hand}_cal50.json" \
    --target-dir "$EXP/independent_targets/$hand" \
    --output-dir "$EXP/independent_evaluation/$hand" \
    --policy-trace-dir "$EXP/independent_traces/$hand" \
    --workers 1 --resume --steps-per-frame 3 --hold-steps 30 \
    > "$EXP/logs/${hand}_evaluate.log" 2>&1
}

evaluate_hand linker & p1=$!
evaluate_hand xhand & p2=$!
evaluate_hand wuji & p3=$!
status=0
wait "$p1" || status=1
wait "$p2" || status=1
wait "$p3" || status=1
if [[ "$status" -ne 0 ]]; then
  for hand in linker xhand wuji; do tail -n 25 "$EXP/logs/${hand}_evaluate.log"; done
  exit "$status"
fi

"$PYTHON" -u "$AUDITOR" --config "$PROTOCOL" \
  --linker-report "$EXP/independent_evaluation/linker/manifest_evaluation_summary.json" \
  --xhand-report "$EXP/independent_evaluation/xhand/manifest_evaluation_summary.json" \
  --wuji-report "$EXP/independent_evaluation/wuji/manifest_evaluation_summary.json" \
  --output-dir "$EXP/stable_audit_v3"

for hand in linker xhand wuji; do
  echo "===== $hand ====="
  tail -n 12 "$EXP/logs/${hand}_evaluate.log"
done
