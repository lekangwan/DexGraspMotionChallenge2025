#!/usr/bin/env bash
# 输入：A组50类50轨迹、三只手当前候选和冻结的0.05/0.10 rad专家接触区域规则。
# 输出：六套候选、PhysX摘要、日志及三份相对当前方法的严格配对JSON。
# 内部逻辑：先确定性生成六套语义表面候选，再分两波、最多三进程完成物理重放。
# 作用：检验“到Shadow同指接触区域”能否避免任意最近表面造成的错误侧接触。

set -euo pipefail

PROJECT_ROOT="/home/lekangwan/projects/DexGraspMotionChallenge2025-final-release"
PYTHON_BIN="/home/lekangwan/miniconda3/envs/hand-retarget/bin/python"
REFINER="$PROJECT_ROOT/retarget_research/retargeting/run/refine_adaptive_finger_gap.py"
EVALUATOR="$PROJECT_ROOT/retarget_research/retargeting/evaluate/evaluate_hand_manifest.py"
COMPARER="$PROJECT_ROOT/retarget_research/retargeting/evaluate/compare_manifest_methods.py"
MANIFEST="$PROJECT_ROOT/retarget_research/manifests/formal_method_selection_a_50c_50t_seed20260814.json"
OUTPUT_ROOT="$PROJECT_ROOT/retarget_research/outputs/expert_contact_region_search_v1/a"
LOG_DIR="$OUTPUT_ROOT/logs"

LINKER_TARGET="$PROJECT_ROOT/retarget_research/outputs/object_centric_advance_search_v1/a/linker_advance_3mm"
XHAND_TARGET="$PROJECT_ROOT/retarget_research/outputs/method_selection_ab/a/xhand_official"
WUJI_TARGET="$PROJECT_ROOT/retarget_research/outputs/shared_grasp_center_search_v1/a/wuji_current"

mkdir -p "$LOG_DIR"
cd "$PROJECT_ROOT"
export MPLCONFIGDIR=/tmp/matplotlib-retarget
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# 输入：手名、当前候选目录、残差上限和输出目录。
# 输出：50条专家接触区域候选及完整静态审计JSON。
# 内部逻辑：固定32点语义区域、20 mm专家接触门和3 mm缺口门，不读取物理结果。
# 作用：确保两档候选除残差上限外完全同口径，并可在物理前核对是否真实移动。
generate_candidate() {
  local hand="$1" input_dir="$2" delta="$3" output_dir="$4"
  "$PYTHON_BIN" "$REFINER" \
    --hand "$hand" --manifest "$MANIFEST" \
    --input-dir "$input_dir" --output-dir "$output_dir" \
    --max-delta-rad "$delta" \
    --target-mode expert_contact_region --region-neighbors 32 \
    --contact-threshold 0.02 --mismatch-margin 0.003 --epsilon-rad 0.01
}

# 输入：手名、候选标签、候选目录及可选PD参数。
# 输出：50条物理报告和汇总；详细终端输出进入独立日志。
# 内部逻辑：使用当前20 Hz统一评测器和严格续跑，不改变执行速度或成功判据。
# 作用：让唯一自变量保持为分指接触目标及其残差上限。
run_eval() {
  local hand="$1" name="$2" target_dir="$3"
  shift 3
  "$PYTHON_BIN" "$EVALUATOR" \
    --hand "$hand" --manifest "$MANIFEST" \
    --target-dir "$target_dir" --output-dir "$OUTPUT_ROOT/${name}_evaluation" \
    --steps-per-frame 3 --hold-steps 30 --workers 1 --resume "$@" \
    >"$LOG_DIR/${name}.log" 2>&1
}

# 输入：阶段名和全部后台PID。
# 输出：每30秒存活数；任一任务失败时整体非零退出。
# 内部逻辑：轮询存活进程，全部结束后逐个wait回收真实退出码。
# 作用：安全并行三套单worker物理任务，避免静默漏掉候选。
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

echo "[1/4] Regenerate six deterministic expert-region candidates."
generate_candidate linker "$LINKER_TARGET" 0.05 "$OUTPUT_ROOT/linker_delta0.05"
generate_candidate linker "$LINKER_TARGET" 0.10 "$OUTPUT_ROOT/linker_delta0.10"
generate_candidate xhand "$XHAND_TARGET" 0.05 "$OUTPUT_ROOT/xhand_delta0.05"
generate_candidate xhand "$XHAND_TARGET" 0.10 "$OUTPUT_ROOT/xhand_delta0.10"
generate_candidate wuji "$WUJI_TARGET" 0.05 "$OUTPUT_ROOT/wuji_delta0.05"
generate_candidate wuji "$WUJI_TARGET" 0.10 "$OUTPUT_ROOT/wuji_delta0.10"

echo "[2/4] Wave 1: 0.05 rad for all three hands."
run_eval linker linker_delta0.05 "$OUTPUT_ROOT/linker_delta0.05" \
  --linker-finger-stiffness 120 --linker-finger-damping 5 \
  --linker-mimic-stiffness 120 --linker-mimic-damping 5 & p1=$!
run_eval xhand xhand_delta0.05 "$OUTPUT_ROOT/xhand_delta0.05" & p2=$!
run_eval wuji wuji_delta0.05 "$OUTPUT_ROOT/wuji_delta0.05" & p3=$!
wait_group "Wave 1/2" "$p1" "$p2" "$p3" || exit 1

echo "[3/4] Wave 2: 0.10 rad for all three hands."
run_eval linker linker_delta0.10 "$OUTPUT_ROOT/linker_delta0.10" \
  --linker-finger-stiffness 120 --linker-finger-damping 5 \
  --linker-mimic-stiffness 120 --linker-mimic-damping 5 & p1=$!
run_eval xhand xhand_delta0.10 "$OUTPUT_ROOT/xhand_delta0.10" & p2=$!
run_eval wuji wuji_delta0.10 "$OUTPUT_ROOT/wuji_delta0.10" & p3=$!
wait_group "Wave 2/2" "$p1" "$p2" "$p3" || exit 1

echo "[4/4] Build strict paired comparisons against current 20 Hz methods."
"$PYTHON_BIN" "$COMPARER" \
  --manifest "$MANIFEST" \
  --summary linker_current "$PROJECT_ROOT/retarget_research/outputs/object_centric_advance_search_v1/a/linker_advance_3mm_evaluation/manifest_evaluation_summary.json" \
  --summary linker_region005 "$OUTPUT_ROOT/linker_delta0.05_evaluation/manifest_evaluation_summary.json" \
  --summary linker_region010 "$OUTPUT_ROOT/linker_delta0.10_evaluation/manifest_evaluation_summary.json" \
  --output "$OUTPUT_ROOT/linker_paired_comparison.json"

"$PYTHON_BIN" "$COMPARER" \
  --manifest "$MANIFEST" \
  --summary xhand_current "$PROJECT_ROOT/retarget_research/outputs/formal_1000/xhand_official_evaluation/manifest_evaluation_summary.json" \
  --summary xhand_region005 "$OUTPUT_ROOT/xhand_delta0.05_evaluation/manifest_evaluation_summary.json" \
  --summary xhand_region010 "$OUTPUT_ROOT/xhand_delta0.10_evaluation/manifest_evaluation_summary.json" \
  --output "$OUTPUT_ROOT/xhand_paired_comparison.json"

"$PYTHON_BIN" "$COMPARER" \
  --manifest "$MANIFEST" \
  --summary wuji_current "$PROJECT_ROOT/retarget_research/outputs/formal_1000/wuji_v1_evaluation/manifest_evaluation_summary.json" \
  --summary wuji_region005 "$OUTPUT_ROOT/wuji_delta0.05_evaluation/manifest_evaluation_summary.json" \
  --summary wuji_region010 "$OUTPUT_ROOT/wuji_delta0.10_evaluation/manifest_evaluation_summary.json" \
  --output "$OUTPUT_ROOT/wuji_paired_comparison.json"

echo "All six expert-contact-region A evaluations and paired comparisons completed."
