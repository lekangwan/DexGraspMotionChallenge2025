#!/usr/bin/env bash
# 输入：A组已生成的XHand/Wuji各2套时序候选与三手各2套分指缺口候选。
# 输出：10套PhysX摘要、独立日志和三份相对当前方法的严格配对JSON。
# 内部逻辑：分3波、每波最多4个单worker进程；候选全部独立与基线比较。
# 作用：同时检验“接触稳定时间不足”和“特定手指缺接触”两个新假设，不进行方法并集。

set -uo pipefail

PROJECT_ROOT="/home/lekangwan/projects/DexGraspMotionChallenge2025-final-release"
PYTHON_BIN="/home/lekangwan/miniconda3/envs/hand-retarget/bin/python"
EVALUATOR="$PROJECT_ROOT/retarget_research/retargeting/evaluate/evaluate_hand_manifest.py"
COMPARER="$PROJECT_ROOT/retarget_research/retargeting/evaluate/compare_manifest_methods.py"
MANIFEST="$PROJECT_ROOT/retarget_research/manifests/formal_method_selection_a_50c_50t_seed20260814.json"
TIMING_ROOT="$PROJECT_ROOT/retarget_research/outputs/phase_retiming_three_hand_v2/a"
FINGER_ROOT="$PROJECT_ROOT/retarget_research/outputs/adaptive_finger_gap_search_v1/a"
LOG_DIR="$PROJECT_ROOT/retarget_research/outputs/timing_and_finger_gap_screen_a_v1/logs"

mkdir -p "$LOG_DIR"
cd "$PROJECT_ROOT" || exit 1
export MPLCONFIGDIR=/tmp/matplotlib-retarget
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# 输入：手名、候选名、候选目录和评测目录。
# 输出：50条物理报告与摘要，详细终端信息写入独立日志。
# 内部逻辑：调用三手共用评测器，单worker严格续跑，Linker显式使用已冻结PD。
# 作用：保证所有候选使用相同成功阈值、物体几何和轨迹键。
run_eval() {
  local hand="$1"
  local name="$2"
  local target_dir="$3"
  local output_dir="$4"
  shift 4
  "$PYTHON_BIN" "$EVALUATOR" \
    --hand "$hand" --manifest "$MANIFEST" \
    --target-dir "$target_dir" --output-dir "$output_dir" \
    --workers 1 --resume "$@" >"$LOG_DIR/${name}.log" 2>&1
}

# 输入：阶段名和当前波的所有PID。
# 输出：每30秒存活数；任一子进程失败则整体返回非零。
# 内部逻辑：轮询`kill -0`，全部结束后再逐个`wait`回收真实退出码。
# 作用：控制并发数并防止一个失败的候选被静默忽略。
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

echo "Wave 1/3: XHand and Wuji pre-lift settle"
run_eval xhand xhand_settle2 "$TIMING_ROOT/xhand_settle2" "$TIMING_ROOT/xhand_settle2_evaluation" & p1=$!
run_eval xhand xhand_settle4 "$TIMING_ROOT/xhand_settle4" "$TIMING_ROOT/xhand_settle4_evaluation" & p2=$!
run_eval wuji wuji_settle2 "$TIMING_ROOT/wuji_settle2" "$TIMING_ROOT/wuji_settle2_evaluation" & p3=$!
run_eval wuji wuji_settle4 "$TIMING_ROOT/wuji_settle4" "$TIMING_ROOT/wuji_settle4_evaluation" & p4=$!
wait_group "Wave 1/3" "$p1" "$p2" "$p3" "$p4" || exit 1

echo "Wave 2/3: Linker and XHand adaptive finger-gap recovery"
run_eval linker linker_gap005 "$FINGER_ROOT/linker_delta0.05" "$FINGER_ROOT/linker_delta0.05_evaluation" \
  --linker-finger-stiffness 120 --linker-finger-damping 5 --linker-mimic-stiffness 120 --linker-mimic-damping 5 & p1=$!
run_eval linker linker_gap010 "$FINGER_ROOT/linker_delta0.10" "$FINGER_ROOT/linker_delta0.10_evaluation" \
  --linker-finger-stiffness 120 --linker-finger-damping 5 --linker-mimic-stiffness 120 --linker-mimic-damping 5 & p2=$!
run_eval xhand xhand_gap005 "$FINGER_ROOT/xhand_delta0.05" "$FINGER_ROOT/xhand_delta0.05_evaluation" & p3=$!
run_eval xhand xhand_gap010 "$FINGER_ROOT/xhand_delta0.10" "$FINGER_ROOT/xhand_delta0.10_evaluation" & p4=$!
wait_group "Wave 2/3" "$p1" "$p2" "$p3" "$p4" || exit 1

echo "Wave 3/3: Wuji adaptive finger-gap recovery"
run_eval wuji wuji_gap005 "$FINGER_ROOT/wuji_delta0.05" "$FINGER_ROOT/wuji_delta0.05_evaluation" & p1=$!
run_eval wuji wuji_gap010 "$FINGER_ROOT/wuji_delta0.10" "$FINGER_ROOT/wuji_delta0.10_evaluation" & p2=$!
wait_group "Wave 3/3" "$p1" "$p2" || exit 1

echo "Build strict paired comparisons."
"$PYTHON_BIN" "$COMPARER" \
  --manifest "$MANIFEST" \
  --summary linker_current "$PROJECT_ROOT/retarget_research/outputs/object_centric_advance_search_v1/a/linker_advance_3mm_evaluation/manifest_evaluation_summary.json" \
  --summary linker_gap005 "$FINGER_ROOT/linker_delta0.05_evaluation/manifest_evaluation_summary.json" \
  --summary linker_gap010 "$FINGER_ROOT/linker_delta0.10_evaluation/manifest_evaluation_summary.json" \
  --output "$FINGER_ROOT/linker_paired_comparison.json"

"$PYTHON_BIN" "$COMPARER" \
  --manifest "$MANIFEST" \
  --summary xhand_current "$PROJECT_ROOT/retarget_research/outputs/formal_1000/xhand_official_evaluation/manifest_evaluation_summary.json" \
  --summary xhand_settle2 "$TIMING_ROOT/xhand_settle2_evaluation/manifest_evaluation_summary.json" \
  --summary xhand_settle4 "$TIMING_ROOT/xhand_settle4_evaluation/manifest_evaluation_summary.json" \
  --summary xhand_gap005 "$FINGER_ROOT/xhand_delta0.05_evaluation/manifest_evaluation_summary.json" \
  --summary xhand_gap010 "$FINGER_ROOT/xhand_delta0.10_evaluation/manifest_evaluation_summary.json" \
  --output "$FINGER_ROOT/xhand_paired_comparison.json"

"$PYTHON_BIN" "$COMPARER" \
  --manifest "$MANIFEST" \
  --summary wuji_current "$PROJECT_ROOT/retarget_research/outputs/formal_1000/wuji_v1_evaluation/manifest_evaluation_summary.json" \
  --summary wuji_settle2 "$TIMING_ROOT/wuji_settle2_evaluation/manifest_evaluation_summary.json" \
  --summary wuji_settle4 "$TIMING_ROOT/wuji_settle4_evaluation/manifest_evaluation_summary.json" \
  --summary wuji_gap005 "$FINGER_ROOT/wuji_delta0.05_evaluation/manifest_evaluation_summary.json" \
  --summary wuji_gap010 "$FINGER_ROOT/wuji_delta0.10_evaluation/manifest_evaluation_summary.json" \
  --output "$FINGER_ROOT/wuji_paired_comparison.json"

echo "All 10 timing/finger-gap A evaluations and paired comparisons completed."
