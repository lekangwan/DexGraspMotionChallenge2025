#!/usr/bin/env bash
# 输入：已生成的Linker A组2/4/6帧阶段重定时候选。
# 输出：三套统一PhysX摘要和独立日志。
# 内部逻辑：三个单worker评测并行运行，每30秒报告存活进程数，支持严格续跑。
# 作用：由用户终端一次执行全部超过3分钟的A组筛选任务。

set -uo pipefail

PROJECT_ROOT="/home/lekangwan/projects/DexGraspMotionChallenge2025-final-release"
PYTHON_BIN="/home/lekangwan/miniconda3/envs/hand-retarget/bin/python"
EVALUATOR="$PROJECT_ROOT/retarget_research/retargeting/evaluate/evaluate_hand_manifest.py"
MANIFEST="$PROJECT_ROOT/retarget_research/manifests/formal_method_selection_a_50c_50t_seed20260814.json"
OUTPUT_ROOT="$PROJECT_ROOT/retarget_research/outputs/phase_retiming_search_v1/a"
LOG_DIR="$OUTPUT_ROOT/logs"

mkdir -p "$LOG_DIR"
cd "$PROJECT_ROOT" || exit 1
export MPLCONFIGDIR=/tmp/matplotlib-retarget
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

run_candidate() {
  local settle="$1"
  "$PYTHON_BIN" "$EVALUATOR" \
    --hand linker \
    --manifest "$MANIFEST" \
    --target-dir "$OUTPUT_ROOT/linker_settle_${settle}" \
    --output-dir "$OUTPUT_ROOT/linker_settle_${settle}_evaluation" \
    --workers 1 --resume \
    --linker-finger-stiffness 120 --linker-finger-damping 5 \
    --linker-mimic-stiffness 120 --linker-mimic-damping 5 \
    >"$LOG_DIR/linker_settle_${settle}.log" 2>&1
}

run_candidate 2 &
pid_2=$!
run_candidate 4 &
pid_4=$!
run_candidate 6 &
pid_6=$!
pids=("$pid_2" "$pid_4" "$pid_6")

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
  echo "Linker phase-retiming A running: $running process(es); logs: $LOG_DIR"
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
echo "All three Linker phase-retiming A evaluations completed."
