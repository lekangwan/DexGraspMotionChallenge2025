#!/usr/bin/env bash
# 输入：已生成的Linker A组3/6/9毫米物体中心指向修正候选。
# 输出：三套统一PhysX摘要、独立日志和相对正式Linker主方法的配对比较JSON。
# 内部逻辑：三个单worker评测并行运行，每30秒报告存活数；全部成功后统一比较。
# 作用：由用户终端一次执行全部超过3分钟的A组筛选，避免逐候选等待和手工统计。

set -uo pipefail

PROJECT_ROOT="/home/lekangwan/projects/DexGraspMotionChallenge2025-final-release"
PYTHON_BIN="/home/lekangwan/miniconda3/envs/hand-retarget/bin/python"
EVALUATOR="$PROJECT_ROOT/retarget_research/retargeting/evaluate/evaluate_hand_manifest.py"
COMPARER="$PROJECT_ROOT/retarget_research/retargeting/evaluate/compare_manifest_methods.py"
MANIFEST="$PROJECT_ROOT/retarget_research/manifests/formal_method_selection_a_50c_50t_seed20260814.json"
BASELINE_SUMMARY="$PROJECT_ROOT/retarget_research/outputs/formal_1000/linker_o6_optimized_v2_evaluation/manifest_evaluation_summary.json"
OUTPUT_ROOT="$PROJECT_ROOT/retarget_research/outputs/object_centric_advance_search_v1/a"
LOG_DIR="$OUTPUT_ROOT/logs"

mkdir -p "$LOG_DIR"
cd "$PROJECT_ROOT" || exit 1
export MPLCONFIGDIR=/tmp/matplotlib-retarget
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

run_candidate() {
  local advance_mm="$1"
  "$PYTHON_BIN" "$EVALUATOR" \
    --hand linker \
    --manifest "$MANIFEST" \
    --target-dir "$OUTPUT_ROOT/linker_advance_${advance_mm}mm" \
    --output-dir "$OUTPUT_ROOT/linker_advance_${advance_mm}mm_evaluation" \
    --workers 1 --resume \
    --linker-finger-stiffness 120 --linker-finger-damping 5 \
    --linker-mimic-stiffness 120 --linker-mimic-damping 5 \
    >"$LOG_DIR/linker_advance_${advance_mm}mm.log" 2>&1
}

run_candidate 3 &
pid_3=$!
run_candidate 6 &
pid_6=$!
run_candidate 9 &
pid_9=$!
pids=("$pid_3" "$pid_6" "$pid_9")

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
  echo "Linker object-centric A running: $running process(es); logs: $LOG_DIR"
  sleep 30
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if [[ "$failed" -ne 0 ]]; then
  echo "At least one evaluation failed; inspect $LOG_DIR" >&2
  exit 1
fi

"$PYTHON_BIN" "$COMPARER" \
  --manifest "$MANIFEST" \
  --summary linker_current "$BASELINE_SUMMARY" \
  --summary advance_3mm "$OUTPUT_ROOT/linker_advance_3mm_evaluation/manifest_evaluation_summary.json" \
  --summary advance_6mm "$OUTPUT_ROOT/linker_advance_6mm_evaluation/manifest_evaluation_summary.json" \
  --summary advance_9mm "$OUTPUT_ROOT/linker_advance_9mm_evaluation/manifest_evaluation_summary.json" \
  --output "$OUTPUT_ROOT/paired_comparison.json"

echo "All three Linker object-centric A evaluations and paired comparison completed."
