#!/usr/bin/env bash
# 输入：已生成的A/B manifest和轻量候选目录。
# 输出：三个XHand A组摘要、两个Linker无夹紧A/B摘要及独立日志。
# 内部逻辑：先并行运行3个XHand候选，再并行运行2个Linker基线，任一失败即非零退出。
# 作用：把五条超过3分钟的PhysX命令交给用户终端一次执行，并限制最高并发为3。

set -uo pipefail

PROJECT_ROOT="/home/lekangwan/projects/DexGraspMotionChallenge2025-final-release"
PYTHON_BIN="/home/lekangwan/miniconda3/envs/hand-retarget/bin/python"
EVALUATOR="$PROJECT_ROOT/retarget_research/retargeting/evaluate/evaluate_hand_manifest.py"
MANIFEST_A="$PROJECT_ROOT/retarget_research/manifests/formal_method_selection_a_50c_50t_seed20260814.json"
MANIFEST_B="$PROJECT_ROOT/retarget_research/manifests/formal_method_selection_b_50c_50t_seed20260814.json"
OUTPUT_ROOT="$PROJECT_ROOT/retarget_research/outputs/method_selection_ab"
LOG_DIR="$OUTPUT_ROOT/logs"

mkdir -p "$LOG_DIR"
cd "$PROJECT_ROOT" || exit 1
export MPLCONFIGDIR=/tmp/matplotlib-retarget
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

run_xhand() {
  local candidate="$1"
  "$PYTHON_BIN" "$EVALUATOR" \
    --hand xhand \
    --manifest "$MANIFEST_A" \
    --target-dir "$OUTPUT_ROOT/a/$candidate" \
    --output-dir "$OUTPUT_ROOT/a/${candidate}_evaluation" \
    --workers 1 --resume \
    >"$LOG_DIR/${candidate}.log" 2>&1
}

run_linker() {
  local split="$1"
  local manifest="$2"
  "$PYTHON_BIN" "$EVALUATOR" \
    --hand linker \
    --manifest "$manifest" \
    --target-dir "$OUTPUT_ROOT/$split/linker_baseline" \
    --output-dir "$OUTPUT_ROOT/$split/linker_baseline_evaluation" \
    --workers 1 --resume \
    --linker-finger-stiffness 120 --linker-finger-damping 5 \
    --linker-mimic-stiffness 120 --linker-mimic-damping 5 \
    >"$LOG_DIR/linker_baseline_${split}.log" 2>&1
}

wait_group() {
  local stage="$1"
  shift
  local failed=0
  local pid
  local running
  while true; do
    running=0
    for pid in "$@"; do
      if kill -0 "$pid" 2>/dev/null; then
        running=$((running + 1))
      fi
    done
    if [[ "$running" -eq 0 ]]; then
      break
    fi
    echo "$stage running: $running process(es); logs: $LOG_DIR"
    sleep 30
  done
  for pid in "$@"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  if [[ "$failed" -ne 0 ]]; then
    echo "$stage failed; inspect $LOG_DIR" >&2
    return 1
  fi
  echo "$stage completed"
}

echo "Stage 1/2: three XHand A candidates (3 parallel processes)"
run_xhand xhand_dynamic_r0 &
pid_r0=$!
run_xhand xhand_dynamic_r05 &
pid_r05=$!
run_xhand xhand_dynamic_r1 &
pid_r1=$!
wait_group "XHand A" "$pid_r0" "$pid_r05" "$pid_r1" || exit 1

echo "Stage 2/2: Linker baseline A/B (2 parallel processes)"
run_linker a "$MANIFEST_A" &
pid_linker_a=$!
run_linker b "$MANIFEST_B" &
pid_linker_b=$!
wait_group "Linker A/B" "$pid_linker_a" "$pid_linker_b" || exit 1

echo "All five evaluations completed. Logs: $LOG_DIR"
