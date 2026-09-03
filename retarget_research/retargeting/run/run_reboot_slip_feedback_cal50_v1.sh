#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/lekangwan/projects/DexGraspMotionChallenge2025-final-release
PYTHON=/home/lekangwan/miniconda3/envs/hand-retarget/bin/python
METHOD=$ROOT/retarget_research/retargeting/run/physics_slip_feedback_refine.py
EVALUATOR=$ROOT/retarget_research/retargeting/evaluate/evaluate_hand_manifest.py
AUDITOR=$ROOT/retarget_research/retargeting/evaluate/audit_stable_success.py
PROTOCOL=$ROOT/retarget_research/retargeting/configs/stable_success_protocol_v3.json
EXP=$ROOT/retarget_research/outputs/reboot_slip_feedback_cal50_v1
LINKER_BASE=$ROOT/retarget_research/outputs/reboot_physics_cem_transport_cal50_v2
OTHER_BASE=$ROOT/retarget_research/outputs/reboot_physics_cem_cal50_v1

mkdir -p "$EXP/targets" "$EXP/manifests" "$EXP/logs" \
  "$EXP/evaluation" "$EXP/traces"
cd "$ROOT"

generate_hand() {
  local hand=$1
  local base=$2
  MPLCONFIGDIR=/tmp/matplotlib-retarget OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  "$PYTHON" -u "$METHOD" \
    --hand "$hand" --manifest "$base/manifests/${hand}_cal50.json" \
    --target-dir "$base/independent_targets/$hand" \
    --output-dir "$EXP/targets/$hand" \
    --output-manifest "$EXP/manifests/${hand}_cal50.json" \
    --batch-size 10 --slip-threshold 0.005 --min-contact-impulse 0.02 \
    --tighten-per-frame 0.015 --max-residual 0.25 --score-margin 1.0 \
    > "$EXP/logs/${hand}_generate.log" 2>&1
}

generate_hand linker "$LINKER_BASE" & p1=$!
generate_hand xhand "$OTHER_BASE" & p2=$!
generate_hand wuji "$OTHER_BASE" & p3=$!
status=0
wait "$p1" || status=1
wait "$p2" || status=1
wait "$p3" || status=1
if [[ "$status" -ne 0 ]]; then
  for hand in linker xhand wuji; do tail -n 25 "$EXP/logs/${hand}_generate.log"; done
  exit "$status"
fi

evaluate_hand() {
  local hand=$1
  MPLCONFIGDIR=/tmp/matplotlib-retarget OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  "$PYTHON" -u "$EVALUATOR" \
    --hand "$hand" --manifest "$EXP/manifests/${hand}_cal50.json" \
    --target-dir "$EXP/targets/$hand" --output-dir "$EXP/evaluation/$hand" \
    --policy-trace-dir "$EXP/traces/$hand" --workers 1 --resume \
    --steps-per-frame 3 --hold-steps 30 \
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
  --linker-report "$EXP/evaluation/linker/manifest_evaluation_summary.json" \
  --xhand-report "$EXP/evaluation/xhand/manifest_evaluation_summary.json" \
  --wuji-report "$EXP/evaluation/wuji/manifest_evaluation_summary.json" \
  --output-dir "$EXP/stable_audit_v3"

for hand in linker xhand wuji; do
  echo "===== $hand ====="
  tail -n 12 "$EXP/logs/${hand}_evaluate.log"
done
