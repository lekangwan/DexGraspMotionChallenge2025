#!/usr/bin/env bash
# 输入：冻结且从未查看物理结果的C组50类50轨迹manifest。
# 输出：当前Linker主方法、唯一3毫米候选、两套PhysX摘要及严格配对比较。
# 内部逻辑：先续跑无时序关键点优化，再生成渐进夹紧和几何修正；最后并行重放两种方法。
# 作用：用一次不可回头调参的独立确认，判断A组的+1是否能泛化到新物体/新轨迹。

set -euo pipefail

PROJECT_ROOT="/home/lekangwan/projects/DexGraspMotionChallenge2025-final-release"
PYTHON_BIN="/home/lekangwan/miniconda3/envs/hand-retarget/bin/python"
MANIFEST="$PROJECT_ROOT/retarget_research/manifests/formal_confirmation_c_50c_50t_seed20260814.json"
RUN_DIR="$PROJECT_ROOT/retarget_research/retargeting/run"
EVALUATE_DIR="$PROJECT_ROOT/retarget_research/retargeting/evaluate"
OUTPUT_ROOT="$PROJECT_ROOT/retarget_research/outputs/object_centric_advance_search_v1/c"
BASELINE_DIR="$OUTPUT_ROOT/linker_no_temporal_baseline"
CURRENT_DIR="$OUTPUT_ROOT/linker_current_squeeze"
CANDIDATE_DIR="$OUTPUT_ROOT/linker_advance_3mm"
CURRENT_EVAL="$OUTPUT_ROOT/linker_current_squeeze_evaluation"
CANDIDATE_EVAL="$OUTPUT_ROOT/linker_advance_3mm_evaluation"
LOG_DIR="$OUTPUT_ROOT/logs"

mkdir -p "$LOG_DIR"
cd "$PROJECT_ROOT"
export MPLCONFIGDIR=/tmp/matplotlib-retarget

echo "[1/5] Generate or resume 50 Linker no-temporal retargeted trajectories."
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 "$PYTHON_BIN" \
  "$RUN_DIR/run_linker_manifest.py" \
  --manifest "$MANIFEST" \
  --output-dir "$BASELINE_DIR" \
  --workers 2 --resume --maxeval 100 --include-thumb-middle \
  --joint-temporal-weight 0 \
  --translation-temporal-weight 0 \
  --rotation-temporal-weight 0

echo "[2/5] Apply the frozen Linker dynamic squeeze."
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 "$PYTHON_BIN" \
  "$RUN_DIR/refine_linker_squeeze.py" \
  --manifest "$MANIFEST" \
  --baseline-dir "$BASELINE_DIR" \
  --output-dir "$CURRENT_DIR" \
  --method-name linker_o6_no_temporal_dynamic_squeeze_v2 \
  --thumb-yaw-delta 0.075 \
  --thumb-pitch-delta 0.1875 \
  --finger-delta 0.425 \
  --contact-threshold 0.02 \
  --min-contact-tips 2 \
  --lift-delta 0.03 \
  --contact-fallback nearest

echo "[3/5] Apply the uniquely selected 3 mm object-centric correction."
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$PYTHON_BIN" \
  "$RUN_DIR/refine_linker_object_centric_advance.py" \
  --manifest "$MANIFEST" \
  --input-dir "$CURRENT_DIR" \
  --output-dir "$CANDIDATE_DIR" \
  --max-advance-mm 3

echo "[4/5] Replay current and candidate methods in parallel."
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$PYTHON_BIN" \
  "$EVALUATE_DIR/evaluate_hand_manifest.py" \
  --hand linker \
  --manifest "$MANIFEST" \
  --target-dir "$CURRENT_DIR" \
  --output-dir "$CURRENT_EVAL" \
  --workers 1 --resume \
  --linker-finger-stiffness 120 --linker-finger-damping 5 \
  --linker-mimic-stiffness 120 --linker-mimic-damping 5 \
  >"$LOG_DIR/linker_current.log" 2>&1 &
pid_current=$!

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$PYTHON_BIN" \
  "$EVALUATE_DIR/evaluate_hand_manifest.py" \
  --hand linker \
  --manifest "$MANIFEST" \
  --target-dir "$CANDIDATE_DIR" \
  --output-dir "$CANDIDATE_EVAL" \
  --workers 1 --resume \
  --linker-finger-stiffness 120 --linker-finger-damping 5 \
  --linker-mimic-stiffness 120 --linker-mimic-damping 5 \
  >"$LOG_DIR/linker_advance_3mm.log" 2>&1 &
pid_candidate=$!

pids=("$pid_current" "$pid_candidate")
while true; do
  running=0
  for pid in "${pids[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      running=$((running + 1))
    fi
  done
  if [[ "$running" -eq 0 ]]; then
    break
  fi
  echo "Linker confirmation C running: $running process(es); logs: $LOG_DIR"
  sleep 30
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if [[ "$failed" -ne 0 ]]; then
  echo "At least one C evaluation failed; inspect $LOG_DIR" >&2
  exit 1
fi

echo "[5/5] Build the frozen paired comparison."
"$PYTHON_BIN" "$EVALUATE_DIR/compare_manifest_methods.py" \
  --manifest "$MANIFEST" \
  --summary linker_current "$CURRENT_EVAL/manifest_evaluation_summary.json" \
  --summary advance_3mm "$CANDIDATE_EVAL/manifest_evaluation_summary.json" \
  --output "$OUTPUT_ROOT/paired_comparison.json"

echo "Linker object-centric confirmation C completed."
