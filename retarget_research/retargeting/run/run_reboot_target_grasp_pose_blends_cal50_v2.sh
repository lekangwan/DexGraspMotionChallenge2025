#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/lekangwan/projects/DexGraspMotionChallenge2025-final-release
PYTHON=/home/lekangwan/miniconda3/envs/hand-retarget/bin/python
BLENDER=$ROOT/retarget_research/retargeting/prepare/blend_target_grasp_pose.py
EVALUATOR=$ROOT/retarget_research/retargeting/evaluate/evaluate_hand_manifest.py
AUDITOR=$ROOT/retarget_research/retargeting/evaluate/audit_stable_success.py
PROTOCOL=$ROOT/retarget_research/retargeting/configs/stable_success_protocol_v3.json
BASE=$ROOT/retarget_research/outputs/reboot_lift_cradle_cem_cal50_v2_confirmed
CANDIDATE=$ROOT/retarget_research/outputs/reboot_target_grasp_pose_cal50_v1

cd "$ROOT"

run_experiment() {
  local tag=$1
  local scale=$2
  local exp=$ROOT/retarget_research/outputs/reboot_target_grasp_pose_blend_${tag}_cal50_v2
  mkdir -p "$exp/independent_targets" "$exp/manifests" "$exp/logs" \
    "$exp/independent_evaluation" "$exp/independent_traces"
  for hand in linker xhand wuji; do
    "$PYTHON" "$BLENDER" \
      --manifest "$BASE/manifests/${hand}_cal50.json" \
      --baseline-dir "$BASE/independent_targets/$hand" \
      --candidate-dir "$CANDIDATE/independent_targets/$hand" \
      --output-dir "$exp/independent_targets/$hand" --scale "$scale"
    cp "$BASE/manifests/${hand}_cal50.json" "$exp/manifests/${hand}_cal50.json"
  done

  evaluate_hand() {
    local hand=$1
    MPLCONFIGDIR=/tmp/matplotlib-retarget OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    "$PYTHON" -u "$EVALUATOR" \
      --hand "$hand" --manifest "$exp/manifests/${hand}_cal50.json" \
      --target-dir "$exp/independent_targets/$hand" \
      --output-dir "$exp/independent_evaluation/$hand" \
      --policy-trace-dir "$exp/independent_traces/$hand" \
      --workers 1 --resume --steps-per-frame 3 --hold-steps 30 \
      > "$exp/logs/${hand}_evaluate.log" 2>&1
  }

  evaluate_hand linker & local p1=$!
  evaluate_hand xhand & local p2=$!
  evaluate_hand wuji & local p3=$!
  local status=0
  wait "$p1" || status=1
  wait "$p2" || status=1
  wait "$p3" || status=1
  if [[ "$status" -ne 0 ]]; then
    for hand in linker xhand wuji; do tail -n 30 "$exp/logs/${hand}_evaluate.log"; done
    return "$status"
  fi
  "$PYTHON" -u "$AUDITOR" --config "$PROTOCOL" \
    --linker-report "$exp/independent_evaluation/linker/manifest_evaluation_summary.json" \
    --xhand-report "$exp/independent_evaluation/xhand/manifest_evaluation_summary.json" \
    --wuji-report "$exp/independent_evaluation/wuji/manifest_evaluation_summary.json" \
    --output-dir "$exp/stable_audit_v3" \
    > "$exp/logs/audit.log" 2>&1
}

run_experiment scale025 0.25 & p1=$!
run_experiment scale050 0.50 & p2=$!
status=0
wait "$p1" || status=1
wait "$p2" || status=1
if [[ "$status" -ne 0 ]]; then
  exit "$status"
fi

for tag in scale025 scale050; do
  echo "===== $tag ====="
  "$PYTHON" - <<PY
import json
from pathlib import Path
p = Path("$ROOT/retarget_research/outputs/reboot_target_grasp_pose_blend_${tag}_cal50_v2/stable_audit_v3/three_hand_stable_audit_summary.json")
d = json.loads(p.read_text())
for hand in ("linker", "xhand", "wuji"):
    print(hand, d[hand]["stable_physics_success_count"], d[hand]["transport_quality_success_count"])
PY
done
