#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/lekangwan/projects/DexGraspMotionChallenge2025-final-release
PYTHON=/home/lekangwan/miniconda3/envs/hand-retarget/bin/python
TRAINER=$ROOT/retarget_research/retargeting/run/train_trajectory_residual_ppo.py
EVALUATOR=$ROOT/retarget_research/retargeting/evaluate/evaluate_hand_manifest.py
AUDITOR=$ROOT/retarget_research/retargeting/evaluate/audit_stable_success.py
PROTOCOL=$ROOT/retarget_research/retargeting/configs/stable_success_protocol_v3.json
BASE=$ROOT/retarget_research/outputs/reboot_lift_cradle_cem_cal50_v2_confirmed
EXP=$ROOT/retarget_research/outputs/reboot_trajectory_residual_ppo_cal50_v1
ITERATIONS=${ITERATIONS:-60}
RESIDUAL_SCALE=${RESIDUAL_SCALE:-0.20}

if [[ -f "$EXP/stable_audit_v3/three_hand_stable_audit_summary.json" ]]; then
  echo "该实验已经完整结束，不重复覆盖：$EXP"
  exit 0
fi

mkdir -p "$EXP/runs" "$EXP/manifests" "$EXP/logs" \
  "$EXP/independent_evaluation" "$EXP/independent_traces"
cd "$ROOT"

train_hand() {
  local hand=$1
  cp "$BASE/manifests/${hand}_cal50.json" "$EXP/manifests/${hand}_cal50.json"
  echo "[$hand] 开始残差PPO训练：${ITERATIONS}轮"
  MPLCONFIGDIR=/tmp/matplotlib-retarget OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  "$PYTHON" -u "$TRAINER" \
    --hand "$hand" \
    --manifest "$EXP/manifests/${hand}_cal50.json" \
    --target-dir "$BASE/independent_targets/$hand" \
    --run-dir "$EXP/runs/$hand" \
    --iterations "$ITERATIONS" --num-envs 50 \
    --residual-scale "$RESIDUAL_SCALE" --evaluation-interval 10 \
    --device cuda --seed 20260829 \
    2>&1 | sed -u "s/^/[$hand] /" | tee "$EXP/logs/${hand}_train.log"
}

train_hand linker & p1=$!
train_hand xhand & p2=$!
train_hand wuji & p3=$!
status=0
wait "$p1" || status=1
wait "$p2" || status=1
wait "$p3" || status=1
if [[ "$status" -ne 0 ]]; then
  for hand in linker xhand wuji; do
    echo "===== $hand train ====="
    tail -n 40 "$EXP/logs/${hand}_train.log" 2>/dev/null || true
  done
  exit "$status"
fi
echo "三只手训练完成，开始独立回放导出的70帧轨迹。"

evaluate_hand() {
  local hand=$1
  MPLCONFIGDIR=/tmp/matplotlib-retarget OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  "$PYTHON" -u "$EVALUATOR" \
    --hand "$hand" --manifest "$EXP/manifests/${hand}_cal50.json" \
    --target-dir "$EXP/runs/$hand/independent_targets" \
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
  for hand in linker xhand wuji; do
    echo "===== $hand evaluate ====="
    tail -n 40 "$EXP/logs/${hand}_evaluate.log" 2>/dev/null || true
  done
  exit "$status"
fi
echo "三只手独立回放完成，开始按冻结协议v3审计。"

"$PYTHON" -u "$AUDITOR" --config "$PROTOCOL" \
  --linker-report "$EXP/independent_evaluation/linker/manifest_evaluation_summary.json" \
  --xhand-report "$EXP/independent_evaluation/xhand/manifest_evaluation_summary.json" \
  --wuji-report "$EXP/independent_evaluation/wuji/manifest_evaluation_summary.json" \
  --output-dir "$EXP/stable_audit_v3" \
  > "$EXP/logs/audit.log" 2>&1

"$PYTHON" - <<PY
import json
from pathlib import Path
p = Path("$EXP/stable_audit_v3/three_hand_stable_audit_summary.json")
d = json.loads(p.read_text())
for hand in ("linker", "xhand", "wuji"):
    row = d[hand]
    print(hand, "stable", row["stable_physics_success_count"],
          "transport", row["transport_quality_success_count"])
PY
