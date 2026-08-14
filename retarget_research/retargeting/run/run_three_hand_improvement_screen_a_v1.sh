#!/usr/bin/env bash
# 输入：冻结A组50类50轨迹、已生成的10套抓取中心候选和XHand/Wuji当前轨迹。
# 输出：14套PhysX评测摘要、独立日志及按手严格配对的比较JSON。
# 内部逻辑：分四波、最多4进程并行；先评测几何条件腕部校准，同时屏幕指PD软/硬两档。
# 作用：一次给三只手各至少一类新改进，但严禁按单条轨迹拼接方法并集。

set -uo pipefail

PROJECT_ROOT="/home/lekangwan/projects/DexGraspMotionChallenge2025-final-release"
PYTHON_BIN="/home/lekangwan/miniconda3/envs/hand-retarget/bin/python"
EVALUATOR="$PROJECT_ROOT/retarget_research/retargeting/evaluate/evaluate_hand_manifest.py"
COMPARER="$PROJECT_ROOT/retarget_research/retargeting/evaluate/compare_manifest_methods.py"
MANIFEST="$PROJECT_ROOT/retarget_research/manifests/formal_method_selection_a_50c_50t_seed20260814.json"
CENTER_ROOT="$PROJECT_ROOT/retarget_research/outputs/shared_grasp_center_search_v1/a"
PD_ROOT="$PROJECT_ROOT/retarget_research/outputs/finger_pd_search_v1/a"
LOG_DIR="$PROJECT_ROOT/retarget_research/outputs/three_hand_improvement_screen_a_v1/logs"

mkdir -p "$LOG_DIR" "$PD_ROOT"
cd "$PROJECT_ROOT" || exit 1
export MPLCONFIGDIR=/tmp/matplotlib-retarget
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# 输入：手名、候选名、候选目录、评测目录，以及可选PD参数。
# 输出：该候选的50条物理报告和一份摘要；终端详细输出写入独立日志。
# 内部逻辑：全部使用单worker避免一个候选抢占过多CPU，`--resume`可在中断后严格续跑。
# 作用：为不同改进统一调用同一几何、成功判定和统计入口。
run_eval() {
  local hand="$1"
  local name="$2"
  local target_dir="$3"
  local output_dir="$4"
  shift 4
  "$PYTHON_BIN" "$EVALUATOR" \
    --hand "$hand" \
    --manifest "$MANIFEST" \
    --target-dir "$target_dir" \
    --output-dir "$output_dir" \
    --workers 1 --resume "$@" \
    >"$LOG_DIR/${name}.log" 2>&1
}

# 输入：阶段名和该波全部子进程PID。
# 输出：每30秒的存活数；全部成功时返回0，任一失败返回1。
# 内部逻辑：先非阻塞轮询存活数，结束后逐个`wait`收集真实退出码。
# 作用：既保留并行节省时间，又不会因为后台任务失败而继续生成错误比较。
wait_group() {
  local stage="$1"
  shift
  local running
  local failed=0
  local pid
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
    echo "$stage: $running process(es) running; logs: $LOG_DIR"
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

echo "Wave 1/4: Linker expert-center and XHand expert-center"
run_eval linker linker_shadow_1p5 "$CENTER_ROOT/linker_shadow_tips_1.5mm" "$CENTER_ROOT/linker_shadow_tips_1.5mm_evaluation" \
  --linker-finger-stiffness 120 --linker-finger-damping 5 --linker-mimic-stiffness 120 --linker-mimic-damping 5 &
p1=$!
run_eval linker linker_shadow_3 "$CENTER_ROOT/linker_shadow_tips_3mm" "$CENTER_ROOT/linker_shadow_tips_3mm_evaluation" \
  --linker-finger-stiffness 120 --linker-finger-damping 5 --linker-mimic-stiffness 120 --linker-mimic-damping 5 &
p2=$!
run_eval xhand xhand_shadow_1 "$CENTER_ROOT/xhand_shadow_tips_1mm" "$CENTER_ROOT/xhand_shadow_tips_1mm_evaluation" &
p3=$!
run_eval xhand xhand_shadow_2 "$CENTER_ROOT/xhand_shadow_tips_2mm" "$CENTER_ROOT/xhand_shadow_tips_2mm_evaluation" &
p4=$!
wait_group "Wave 1/4" "$p1" "$p2" "$p3" "$p4" || exit 1

echo "Wave 2/4: XHand object-center and Wuji expert-center"
run_eval xhand xhand_object_1p5 "$CENTER_ROOT/xhand_object_bbox_1.5mm" "$CENTER_ROOT/xhand_object_bbox_1.5mm_evaluation" &
p1=$!
run_eval xhand xhand_object_3 "$CENTER_ROOT/xhand_object_bbox_3mm" "$CENTER_ROOT/xhand_object_bbox_3mm_evaluation" &
p2=$!
run_eval wuji wuji_shadow_1p5 "$CENTER_ROOT/wuji_shadow_tips_1.5mm" "$CENTER_ROOT/wuji_shadow_tips_1.5mm_evaluation" &
p3=$!
run_eval wuji wuji_shadow_3 "$CENTER_ROOT/wuji_shadow_tips_3mm" "$CENTER_ROOT/wuji_shadow_tips_3mm_evaluation" &
p4=$!
wait_group "Wave 2/4" "$p1" "$p2" "$p3" "$p4" || exit 1

echo "Wave 3/4: Wuji object-center and XHand controller gains"
run_eval wuji wuji_object_1p5 "$CENTER_ROOT/wuji_object_bbox_1.5mm" "$CENTER_ROOT/wuji_object_bbox_1.5mm_evaluation" &
p1=$!
run_eval wuji wuji_object_3 "$CENTER_ROOT/wuji_object_bbox_3mm" "$CENTER_ROOT/wuji_object_bbox_3mm_evaluation" &
p2=$!
run_eval xhand xhand_pd_soft "$PROJECT_ROOT/retarget_research/outputs/method_selection_ab/a/xhand_official" "$PD_ROOT/xhand_soft_evaluation" \
  --xhand-finger-stiffness 60 --xhand-finger-damping 3 &
p3=$!
run_eval xhand xhand_pd_stiff "$PROJECT_ROOT/retarget_research/outputs/method_selection_ab/a/xhand_official" "$PD_ROOT/xhand_stiff_evaluation" \
  --xhand-finger-stiffness 240 --xhand-finger-damping 10 &
p4=$!
wait_group "Wave 3/4" "$p1" "$p2" "$p3" "$p4" || exit 1

echo "Wave 4/4: Wuji controller gains"
run_eval wuji wuji_pd_soft "$CENTER_ROOT/wuji_current" "$PD_ROOT/wuji_soft_evaluation" \
  --wuji-finger-stiffness 60 --wuji-finger-damping 3 &
p1=$!
run_eval wuji wuji_pd_stiff "$CENTER_ROOT/wuji_current" "$PD_ROOT/wuji_stiff_evaluation" \
  --wuji-finger-stiffness 240 --wuji-finger-damping 10 &
p2=$!
wait_group "Wave 4/4" "$p1" "$p2" || exit 1

echo "Build paired comparisons against each hand's current method."
"$PYTHON_BIN" "$COMPARER" \
  --manifest "$MANIFEST" \
  --summary linker_current_object3 "$PROJECT_ROOT/retarget_research/outputs/object_centric_advance_search_v1/a/linker_advance_3mm_evaluation/manifest_evaluation_summary.json" \
  --summary linker_shadow1p5 "$CENTER_ROOT/linker_shadow_tips_1.5mm_evaluation/manifest_evaluation_summary.json" \
  --summary linker_shadow3 "$CENTER_ROOT/linker_shadow_tips_3mm_evaluation/manifest_evaluation_summary.json" \
  --output "$CENTER_ROOT/linker_paired_comparison.json"

"$PYTHON_BIN" "$COMPARER" \
  --manifest "$MANIFEST" \
  --summary xhand_current "$PROJECT_ROOT/retarget_research/outputs/formal_1000/xhand_official_evaluation/manifest_evaluation_summary.json" \
  --summary xhand_shadow1 "$CENTER_ROOT/xhand_shadow_tips_1mm_evaluation/manifest_evaluation_summary.json" \
  --summary xhand_shadow2 "$CENTER_ROOT/xhand_shadow_tips_2mm_evaluation/manifest_evaluation_summary.json" \
  --summary xhand_object1p5 "$CENTER_ROOT/xhand_object_bbox_1.5mm_evaluation/manifest_evaluation_summary.json" \
  --summary xhand_object3 "$CENTER_ROOT/xhand_object_bbox_3mm_evaluation/manifest_evaluation_summary.json" \
  --summary xhand_pd_soft "$PD_ROOT/xhand_soft_evaluation/manifest_evaluation_summary.json" \
  --summary xhand_pd_stiff "$PD_ROOT/xhand_stiff_evaluation/manifest_evaluation_summary.json" \
  --output "$CENTER_ROOT/xhand_paired_comparison.json"

"$PYTHON_BIN" "$COMPARER" \
  --manifest "$MANIFEST" \
  --summary wuji_current "$PROJECT_ROOT/retarget_research/outputs/formal_1000/wuji_v1_evaluation/manifest_evaluation_summary.json" \
  --summary wuji_shadow1p5 "$CENTER_ROOT/wuji_shadow_tips_1.5mm_evaluation/manifest_evaluation_summary.json" \
  --summary wuji_shadow3 "$CENTER_ROOT/wuji_shadow_tips_3mm_evaluation/manifest_evaluation_summary.json" \
  --summary wuji_object1p5 "$CENTER_ROOT/wuji_object_bbox_1.5mm_evaluation/manifest_evaluation_summary.json" \
  --summary wuji_object3 "$CENTER_ROOT/wuji_object_bbox_3mm_evaluation/manifest_evaluation_summary.json" \
  --summary wuji_pd_soft "$PD_ROOT/wuji_soft_evaluation/manifest_evaluation_summary.json" \
  --summary wuji_pd_stiff "$PD_ROOT/wuji_stiff_evaluation/manifest_evaluation_summary.json" \
  --output "$CENTER_ROOT/wuji_paired_comparison.json"

echo "All 14 A-screen evaluations and three paired comparisons completed."
