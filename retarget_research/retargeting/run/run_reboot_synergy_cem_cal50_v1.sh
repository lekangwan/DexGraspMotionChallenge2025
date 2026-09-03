#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/lekangwan/projects/DexGraspMotionChallenge2025-final-release
PYTHON=/home/lekangwan/miniconda3/envs/hand-retarget/bin/python
RUNNER=$ROOT/retarget_research/retargeting/run/run_synergy_cem_manifest.py
PREPARE=$ROOT/retarget_research/retargeting/prepare/prepare_physics_cem_independent_eval.py
EVALUATOR=$ROOT/retarget_research/retargeting/evaluate/evaluate_hand_manifest.py
AUDITOR=$ROOT/retarget_research/retargeting/evaluate/audit_stable_success.py
PROTOCOL=$ROOT/retarget_research/retargeting/configs/stable_success_protocol_v3.json
EXP_NAME=${EXP_NAME:-reboot_synergy_cem_cal50_v1}
RANK=${RANK:-3}
EXP=$ROOT/retarget_research/outputs/$EXP_NAME
LINKER_BASE=$ROOT/retarget_research/outputs/reboot_physics_cem_transport_cal50_v2
OTHER_BASE=$ROOT/retarget_research/outputs/reboot_physics_cem_cal50_v1

mkdir -p "$EXP/raw" "$EXP/independent_targets" "$EXP/manifests" \
  "$EXP/logs" "$EXP/independent_evaluation" "$EXP/independent_traces"
cd "$ROOT"

run_cem() {
  local hand=$1
  local base=$2
  MPLCONFIGDIR=/tmp/matplotlib-retarget OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  "$PYTHON" -u "$RUNNER" \
    --hand "$hand" --manifest "$base/manifests/${hand}_cal50.json" \
    --target-dir "$base/independent_targets/$hand" \
    --output-dir "$EXP/raw/$hand" --rank "$RANK" \
    --population 8 --elite 2 --iterations 2 --selection-margin 1.0 \
    --seed 20260828 --device cpu > "$EXP/logs/${hand}_generate.log" 2>&1
}

run_cem linker "$LINKER_BASE" & p1=$!
run_cem xhand "$OTHER_BASE" & p2=$!
run_cem wuji "$OTHER_BASE" & p3=$!
status=0
wait "$p1" || status=1
wait "$p2" || status=1
wait "$p3" || status=1
if [[ "$status" -ne 0 ]]; then
  for hand in linker xhand wuji; do tail -n 25 "$EXP/logs/${hand}_generate.log"; done
  exit "$status"
fi

"$PYTHON" "$PREPARE" --screen-summary "$EXP/raw/linker/screen_summary.json" \
  --formal-manifest "$LINKER_BASE/manifests/linker_cal50.json" \
  --target-dir "$EXP/independent_targets/linker" \
  --manifest-output "$EXP/manifests/linker_cal50.json"
for hand in xhand wuji; do
  "$PYTHON" "$PREPARE" --screen-summary "$EXP/raw/$hand/screen_summary.json" \
    --formal-manifest "$OTHER_BASE/manifests/${hand}_cal50.json" \
    --target-dir "$EXP/independent_targets/$hand" \
    --manifest-output "$EXP/manifests/${hand}_cal50.json"
done

evaluate_hand() {
  local hand=$1
  MPLCONFIGDIR=/tmp/matplotlib-retarget OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  "$PYTHON" -u "$EVALUATOR" \
    --hand "$hand" --manifest "$EXP/manifests/${hand}_cal50.json" \
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
