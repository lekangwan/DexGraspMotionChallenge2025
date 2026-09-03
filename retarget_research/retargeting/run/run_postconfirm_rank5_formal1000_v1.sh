#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/lekangwan/projects/DexGraspMotionChallenge2025-final-release
PYTHON=/home/lekangwan/miniconda3/envs/hand-retarget/bin/python
CONFIRM=$ROOT/retarget_research/retargeting/success_only/confirm_existing_candidates_isolated.py
EVALUATOR=$ROOT/retarget_research/retargeting/evaluate/evaluate_hand_manifest.py
AUDITOR=$ROOT/retarget_research/retargeting/evaluate/audit_stable_success.py
PROTOCOL=$ROOT/retarget_research/retargeting/configs/stable_success_protocol_v3_selected_methods.json
FORMAL=$ROOT/retarget_research/outputs/reboot_synergy_rank5_formal1000_v1
FINAL=$FORMAL/final_synergy_rank5
OUT=$FORMAL/postconfirmed_rank5_v1

if [[ -f "$OUT/audit/three_hand_stable_audit_summary.json" ]]; then
  echo "重复确认、独立评测和稳定审计均已完成，不重复覆盖：$OUT"
  exit 0
fi

mkdir -p "$OUT/logs" "$OUT/targets" "$OUT/evaluation" "$OUT/traces" "$OUT/audit"
cd "$ROOT"

confirm_hand() {
  local hand=$1
  local baseline=$2
  MPLCONFIGDIR=/tmp/matplotlib-retarget OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  "$PYTHON" -u "$CONFIRM" \
    --hand "$hand" \
    --manifest "$FINAL/manifests/${hand}.json" \
    --screen-summary "$FINAL/raw/$hand/screen_summary.json" \
    --baseline-target-dir "$baseline" \
    --candidate-target-dir "$FINAL/targets/$hand" \
    --output-dir "$OUT/targets/$hand" \
    --confirmation-repeats 2 \
    --selection-margin 1 \
    --device cpu > "$OUT/logs/confirm_${hand}.log" 2>&1
}

echo "[1/3] 三只手并行重复确认：只复验Rank-5真正修改过的轨迹。"
confirm_hand linker "$FORMAL/base_linker_global2/targets/linker" & p1=$!
confirm_hand xhand "$FORMAL/base_global1/targets/xhand" & p2=$!
confirm_hand wuji "$FORMAL/base_global1/targets/wuji" & p3=$!
status=0; wait "$p1" || status=1; wait "$p2" || status=1; wait "$p3" || status=1
[[ "$status" -eq 0 ]] || { tail -n 60 "$OUT"/logs/confirm_*.log; exit 1; }

evaluate_hand() {
  local hand=$1
  MPLCONFIGDIR=/tmp/matplotlib-retarget OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  "$PYTHON" -u "$EVALUATOR" \
    --hand "$hand" \
    --manifest "$FINAL/manifests/${hand}.json" \
    --target-dir "$OUT/targets/$hand" \
    --output-dir "$OUT/evaluation/$hand" \
    --policy-trace-dir "$OUT/traces/$hand" \
    --workers 1 --resume --steps-per-frame 3 --hold-steps 30 \
    > "$OUT/logs/evaluate_${hand}.log" 2>&1
}

echo "[2/3] 对确认后的三套完整1000条轨迹做独立回放。"
evaluate_hand linker & p1=$!
evaluate_hand xhand & p2=$!
evaluate_hand wuji & p3=$!
status=0; wait "$p1" || status=1; wait "$p2" || status=1; wait "$p3" || status=1
[[ "$status" -eq 0 ]] || { tail -n 60 "$OUT"/logs/evaluate_*.log; exit 1; }

echo "[3/3] 按冻结的参考口径和稳定运输口径重新审计。"
"$PYTHON" -u "$AUDITOR" --config "$PROTOCOL" \
  --linker-report "$OUT/evaluation/linker/manifest_evaluation_summary.json" \
  --xhand-report "$OUT/evaluation/xhand/manifest_evaluation_summary.json" \
  --wuji-report "$OUT/evaluation/wuji/manifest_evaluation_summary.json" \
  --output-dir "$OUT/audit" | tee "$OUT/logs/audit.log"
