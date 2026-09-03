#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/lekangwan/projects/DexGraspMotionChallenge2025-final-release
PYTHON=/home/lekangwan/miniconda3/envs/hand-retarget/bin/python
RUNNER=$ROOT/retarget_research/retargeting/run/run_physics_cem_screen.py
MANIFEST=$ROOT/retarget_research/manifests/formal_50c_100o_1000t_seed20260808.json
AUDIT_ROOT=$ROOT/retarget_research/outputs/formal_1000/selected_methods_audit_v3
OUTPUT_ROOT=$ROOT/retarget_research/outputs/reboot_physics_cem_cal50_v1

mkdir -p "$OUTPUT_ROOT/logs"
cd "$ROOT"

run_hand() {
  local hand=$1
  MPLCONFIGDIR=/tmp/matplotlib-retarget \
  OMP_NUM_THREADS=4 \
  MKL_NUM_THREADS=4 \
  "$PYTHON" -u "$RUNNER" \
    --hand "$hand" \
    --audit "$AUDIT_ROOT/${hand}_stable_audit.json" \
    --manifest "$MANIFEST" \
    --output-dir "$OUTPUT_ROOT/$hand" \
    --split calibration \
    --limit 50 \
    --population 8 \
    --elite 2 \
    --iterations 2 \
    --seed 20260827 \
    --device cpu \
    > "$OUTPUT_ROOT/logs/${hand}.log" 2>&1
}

run_hand linker &
LINKER_PID=$!
run_hand xhand &
XHAND_PID=$!
run_hand wuji &
WUJI_PID=$!

STATUS=0
wait "$LINKER_PID" || STATUS=1
wait "$XHAND_PID" || STATUS=1
wait "$WUJI_PID" || STATUS=1

for hand in linker xhand wuji; do
  echo "===== $hand ====="
  tail -n 15 "$OUTPUT_ROOT/logs/${hand}.log"
done

exit "$STATUS"
