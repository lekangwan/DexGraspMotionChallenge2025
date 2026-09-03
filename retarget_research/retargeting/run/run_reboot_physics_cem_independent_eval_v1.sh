#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/lekangwan/projects/DexGraspMotionChallenge2025-final-release
PYTHON=/home/lekangwan/miniconda3/envs/hand-retarget/bin/python
EVALUATOR=$ROOT/retarget_research/retargeting/evaluate/evaluate_hand_manifest.py
EXP=$ROOT/retarget_research/outputs/reboot_physics_cem_cal50_v1

mkdir -p "$EXP/independent_logs" "$EXP/independent_traces"
cd "$ROOT"

run_hand() {
  local hand=$1
  MPLCONFIGDIR=/tmp/matplotlib-retarget \
  OMP_NUM_THREADS=4 \
  MKL_NUM_THREADS=4 \
  "$PYTHON" -u "$EVALUATOR" \
    --hand "$hand" \
    --manifest "$EXP/manifests/${hand}_cal50.json" \
    --target-dir "$EXP/independent_targets/$hand" \
    --output-dir "$EXP/independent_evaluation/$hand" \
    --policy-trace-dir "$EXP/independent_traces/$hand" \
    --workers 1 \
    --resume \
    --steps-per-frame 3 \
    --hold-steps 30 \
    > "$EXP/independent_logs/${hand}.log" 2>&1
}

run_hand linker &
LINKER_PID=$!
run_hand xhand &
XHAND_PID=$!
run_hand wuji &
WUJI_PID=$!

echo "三手独立重放已启动：Linker PID=$LINKER_PID，XHand PID=$XHAND_PID，Wuji PID=$WUJI_PID"
echo "实时日志目录：$EXP/independent_logs"
echo "主终端将在三只手全部结束后打印汇总。"

STATUS=0
wait "$LINKER_PID" || STATUS=1
wait "$XHAND_PID" || STATUS=1
wait "$WUJI_PID" || STATUS=1

for hand in linker xhand wuji; do
  echo "===== $hand ====="
  tail -n 12 "$EXP/independent_logs/${hand}.log"
done

exit "$STATUS"
