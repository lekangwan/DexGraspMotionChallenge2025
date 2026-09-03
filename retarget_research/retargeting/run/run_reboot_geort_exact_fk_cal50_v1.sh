#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/lekangwan/projects/DexGraspMotionChallenge2025-final-release
PYTHON=/home/lekangwan/miniconda3/envs/hand-retarget/bin/python
METHOD=$ROOT/retarget_research/retargeting/run/geort_exact_fk.py
EVALUATOR=$ROOT/retarget_research/retargeting/evaluate/evaluate_hand_manifest.py
AUDITOR=$ROOT/retarget_research/retargeting/evaluate/audit_stable_success.py
FORMAL=$ROOT/retarget_research/manifests/formal_50c_100o_1000t_seed20260808.json
OLD_AUDIT=$ROOT/retarget_research/outputs/formal_1000/selected_methods_audit_v3
EXP=$ROOT/retarget_research/outputs/reboot_geort_exact_fk_v1

mkdir -p "$EXP/checkpoints" "$EXP/cache" "$EXP/targets" "$EXP/manifests" \
  "$EXP/evaluation" "$EXP/traces"
cd "$ROOT"

for hand in linker xhand wuji; do
  echo "===== 训练 $hand GeoRT exact-FK ====="
  MPLCONFIGDIR=/tmp/matplotlib-retarget OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  "$PYTHON" -u "$METHOD" train \
    --hand "$hand" \
    --audit "$OLD_AUDIT/${hand}_stable_audit.json" \
    --source-cache "$EXP/cache/${hand}_source_tips.npy" \
    --checkpoint "$EXP/checkpoints/${hand}.pt" \
    --device cuda \
    --workspace-samples 20000 \
    --epoch-samples 20000 \
    --epochs 50 \
    --batch-size 256 \
    --hidden 128 \
    --learning-rate 0.0001

  echo "===== 生成 $hand 固定50条候选 ====="
  "$PYTHON" -u "$METHOD" apply \
    --hand "$hand" \
    --audit "$OLD_AUDIT/${hand}_stable_audit.json" \
    --checkpoint "$EXP/checkpoints/${hand}.pt" \
    --device cuda \
    --split calibration --one-per-category \
    --formal-manifest "$FORMAL" \
    --output-dir "$EXP/targets/$hand" \
    --manifest-output "$EXP/manifests/${hand}_cal50.json"
done

run_eval() {
  local hand=$1
  MPLCONFIGDIR=/tmp/matplotlib-retarget OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  "$PYTHON" -u "$EVALUATOR" \
    --hand "$hand" \
    --manifest "$EXP/manifests/${hand}_cal50.json" \
    --target-dir "$EXP/targets/$hand" \
    --output-dir "$EXP/evaluation/$hand" \
    --policy-trace-dir "$EXP/traces/$hand" \
    --workers 1 --resume --steps-per-frame 3 --hold-steps 30 2>&1 | sed "s/^/[$hand] /"
}

echo "===== 三手并行独立PhysX重放 ====="
run_eval linker & p1=$!
run_eval xhand & p2=$!
run_eval wuji & p3=$!
status=0
wait "$p1" || status=1
wait "$p2" || status=1
wait "$p3" || status=1
if [[ "$status" -ne 0 ]]; then
  exit "$status"
fi

"$PYTHON" -u "$AUDITOR" \
  --linker-report "$EXP/evaluation/linker/manifest_evaluation_summary.json" \
  --xhand-report "$EXP/evaluation/xhand/manifest_evaluation_summary.json" \
  --wuji-report "$EXP/evaluation/wuji/manifest_evaluation_summary.json" \
  --output-dir "$EXP/stable_audit"

