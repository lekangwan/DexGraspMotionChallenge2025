#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/lekangwan/projects/DexGraspMotionChallenge2025-final-release
PYTHON=/home/lekangwan/miniconda3/envs/hand-retarget/bin/python
RUNNER=$ROOT/retarget_research/retargeting/run/run_physics_cem_screen.py
V1=$ROOT/retarget_research/outputs/reboot_physics_cem_cal50_v1
V2=$ROOT/retarget_research/outputs/reboot_physics_cem_transport_cal50_v2

mkdir -p "$V2/logs"
cd "$ROOT"

run_hand() {
  local hand=$1
  MPLCONFIGDIR=/tmp/matplotlib-retarget \
  OMP_NUM_THREADS=4 \
  MKL_NUM_THREADS=4 \
  "$PYTHON" -u "$RUNNER" \
    --hand "$hand" \
    --audit "$V1/independent_evaluation/$hand/manifest_evaluation_summary.json" \
    --manifest "$V1/manifests/${hand}_cal50.json" \
    --output-dir "$V2/$hand" \
    --split calibration \
    --limit 50 \
    --population 8 \
    --elite 2 \
    --iterations 2 \
    --seed 20260828 \
    --device cpu \
    > "$V2/logs/${hand}.log" 2>&1
}

run_hand linker &
LINKER_PID=$!
run_hand xhand &
XHAND_PID=$!
run_hand wuji &
WUJI_PID=$!

echo "运输感知CEM第二轮已启动：Linker PID=$LINKER_PID，XHand PID=$XHAND_PID，Wuji PID=$WUJI_PID"
echo "实时日志目录：$V2/logs"

STATUS=0
wait "$LINKER_PID" || STATUS=1
wait "$XHAND_PID" || STATUS=1
wait "$WUJI_PID" || STATUS=1

for hand in linker xhand wuji; do
  echo "===== $hand ====="
  tail -n 15 "$V2/logs/${hand}.log"
done

exit "$STATUS"
