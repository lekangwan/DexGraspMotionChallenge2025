#!/usr/bin/env bash
# 输入：A组50类50轨迹及三只手当前唯一候选，速度档为30/15/10 Hz。
# 输出：9套PhysX摘要、独立日志和三份相对20 Hz当前方法的严格配对JSON。
# 内部逻辑：每只手一波并行评测三个速度；动作值、PD、末端保持和成功判据不变。
# 作用：判断失败来自重定向几何，还是目标手没有足够时间跟踪整段连续轨迹。

set -uo pipefail

PROJECT_ROOT="/home/lekangwan/projects/DexGraspMotionChallenge2025-final-release"
PYTHON_BIN="/home/lekangwan/miniconda3/envs/hand-retarget/bin/python"
EVALUATOR="$PROJECT_ROOT/retarget_research/retargeting/evaluate/evaluate_hand_manifest.py"
COMPARER="$PROJECT_ROOT/retarget_research/retargeting/evaluate/compare_manifest_methods.py"
MANIFEST="$PROJECT_ROOT/retarget_research/manifests/formal_method_selection_a_50c_50t_seed20260814.json"
OUTPUT_ROOT="$PROJECT_ROOT/retarget_research/outputs/replay_frequency_search_v1/a"
LOG_DIR="$OUTPUT_ROOT/logs"

LINKER_TARGET="$PROJECT_ROOT/retarget_research/outputs/object_centric_advance_search_v1/a/linker_advance_3mm"
XHAND_TARGET="$PROJECT_ROOT/retarget_research/outputs/method_selection_ab/a/xhand_official"
WUJI_TARGET="$PROJECT_ROOT/retarget_research/outputs/shared_grasp_center_search_v1/a/wuji_current"

mkdir -p "$LOG_DIR"
cd "$PROJECT_ROOT" || exit 1
export MPLCONFIGDIR=/tmp/matplotlib-retarget
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# 输入：目标手、速度标签、当前候选目录、每轨迹帧物理步数和额外PD参数。
# 输出：50条几何/物理报告和汇总；完整终端输出进入独立日志。
# 内部逻辑：调用统一评测器，60 Hz仿真下2/4/6步分别为30/15/10 Hz。
# 作用：让同一动作轨迹只改变连续执行速度，不混入新的几何或控制增益。
run_eval() {
  local hand="$1"
  local name="$2"
  local target_dir="$3"
  local steps="$4"
  shift 4
  "$PYTHON_BIN" "$EVALUATOR" \
    --hand "$hand" --manifest "$MANIFEST" \
    --target-dir "$target_dir" \
    --output-dir "$OUTPUT_ROOT/${name}_evaluation" \
    --steps-per-frame "$steps" --hold-steps 30 \
    --workers 1 --resume "$@" >"$LOG_DIR/${name}.log" 2>&1
}

# 输入：阶段名和本波全部后台PID。
# 输出：每30秒存活数，并在任一任务失败时返回非零。
# 内部逻辑：先轮询存活状态，全部结束后逐一wait收集真正退出码。
# 作用：三档速度并行节约墙钟，同时不静默吞掉某个候选的错误。
wait_group() {
  local stage="$1"
  shift
  local running failed=0 pid
  while true; do
    running=0
    for pid in "$@"; do
      if kill -0 "$pid" 2>/dev/null; then running=$((running + 1)); fi
    done
    if [[ "$running" -eq 0 ]]; then break; fi
    echo "$stage: $running process(es) running; logs: $LOG_DIR"
    sleep 30
  done
  for pid in "$@"; do if ! wait "$pid"; then failed=1; fi; done
  if [[ "$failed" -ne 0 ]]; then
    echo "$stage failed; inspect $LOG_DIR" >&2
    return 1
  fi
  echo "$stage completed"
}

echo "Wave 1/3: Linker replay frequency"
run_eval linker linker_30hz "$LINKER_TARGET" 2 \
  --linker-finger-stiffness 120 --linker-finger-damping 5 \
  --linker-mimic-stiffness 120 --linker-mimic-damping 5 & p1=$!
run_eval linker linker_15hz "$LINKER_TARGET" 4 \
  --linker-finger-stiffness 120 --linker-finger-damping 5 \
  --linker-mimic-stiffness 120 --linker-mimic-damping 5 & p2=$!
run_eval linker linker_10hz "$LINKER_TARGET" 6 \
  --linker-finger-stiffness 120 --linker-finger-damping 5 \
  --linker-mimic-stiffness 120 --linker-mimic-damping 5 & p3=$!
wait_group "Wave 1/3" "$p1" "$p2" "$p3" || exit 1

echo "Wave 2/3: XHand replay frequency"
run_eval xhand xhand_30hz "$XHAND_TARGET" 2 & p1=$!
run_eval xhand xhand_15hz "$XHAND_TARGET" 4 & p2=$!
run_eval xhand xhand_10hz "$XHAND_TARGET" 6 & p3=$!
wait_group "Wave 2/3" "$p1" "$p2" "$p3" || exit 1

echo "Wave 3/3: Wuji replay frequency"
run_eval wuji wuji_30hz "$WUJI_TARGET" 2 & p1=$!
run_eval wuji wuji_15hz "$WUJI_TARGET" 4 & p2=$!
run_eval wuji wuji_10hz "$WUJI_TARGET" 6 & p3=$!
wait_group "Wave 3/3" "$p1" "$p2" "$p3" || exit 1

echo "Build strict paired comparisons against current 20 Hz summaries."
"$PYTHON_BIN" "$COMPARER" \
  --manifest "$MANIFEST" \
  --summary linker_20hz "$PROJECT_ROOT/retarget_research/outputs/object_centric_advance_search_v1/a/linker_advance_3mm_evaluation/manifest_evaluation_summary.json" \
  --summary linker_30hz "$OUTPUT_ROOT/linker_30hz_evaluation/manifest_evaluation_summary.json" \
  --summary linker_15hz "$OUTPUT_ROOT/linker_15hz_evaluation/manifest_evaluation_summary.json" \
  --summary linker_10hz "$OUTPUT_ROOT/linker_10hz_evaluation/manifest_evaluation_summary.json" \
  --output "$OUTPUT_ROOT/linker_paired_comparison.json"

"$PYTHON_BIN" "$COMPARER" \
  --manifest "$MANIFEST" \
  --summary xhand_20hz "$PROJECT_ROOT/retarget_research/outputs/formal_1000/xhand_official_evaluation/manifest_evaluation_summary.json" \
  --summary xhand_30hz "$OUTPUT_ROOT/xhand_30hz_evaluation/manifest_evaluation_summary.json" \
  --summary xhand_15hz "$OUTPUT_ROOT/xhand_15hz_evaluation/manifest_evaluation_summary.json" \
  --summary xhand_10hz "$OUTPUT_ROOT/xhand_10hz_evaluation/manifest_evaluation_summary.json" \
  --output "$OUTPUT_ROOT/xhand_paired_comparison.json"

"$PYTHON_BIN" "$COMPARER" \
  --manifest "$MANIFEST" \
  --summary wuji_20hz "$PROJECT_ROOT/retarget_research/outputs/formal_1000/wuji_v1_evaluation/manifest_evaluation_summary.json" \
  --summary wuji_30hz "$OUTPUT_ROOT/wuji_30hz_evaluation/manifest_evaluation_summary.json" \
  --summary wuji_15hz "$OUTPUT_ROOT/wuji_15hz_evaluation/manifest_evaluation_summary.json" \
  --summary wuji_10hz "$OUTPUT_ROOT/wuji_10hz_evaluation/manifest_evaluation_summary.json" \
  --output "$OUTPUT_ROOT/wuji_paired_comparison.json"

echo "All 9 replay-frequency A evaluations and paired comparisons completed."
