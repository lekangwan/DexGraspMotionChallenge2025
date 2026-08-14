#!/usr/bin/env bash
# 输入：已生成的正式1000条Linker 3毫米候选、正式manifest和策略对象级split。
# 输出：1000条PhysX结果与新trace、配对/机制报告、Linker策略数据和三模型冒烟产物。
# 内部逻辑：续跑正式回放并保存执行前状态trace；完成后与旧主方法严格配对，
# 再物化最终Linker策略数据并运行限量BC/Temporal3/Diffusion CPU冒烟。
# 作用：一次长命令同时完成基本任务规模化结果和进阶任务Linker数据准备，避免重放两次。

set -euo pipefail

PROJECT_ROOT="/home/lekangwan/projects/DexGraspMotionChallenge2025-final-release"
PYTHON_BIN="/home/lekangwan/miniconda3/envs/hand-retarget/bin/python"
MANIFEST="$PROJECT_ROOT/retarget_research/manifests/formal_50c_100o_1000t_seed20260808.json"
TARGET_DIR="$PROJECT_ROOT/retarget_research/outputs/formal_1000/linker_object_centric_3mm_v1"
EVAL_DIR="$PROJECT_ROOT/retarget_research/outputs/formal_1000/linker_object_centric_3mm_v1_evaluation"
TRACE_DIR="$PROJECT_ROOT/retarget_research/advanced_policy/traces/formal_v2/linker_object_centric_3mm"
BASELINE_SUMMARY="$PROJECT_ROOT/retarget_research/outputs/formal_1000/linker_o6_optimized_v2_evaluation/manifest_evaluation_summary.json"
NEW_SUMMARY="$EVAL_DIR/manifest_evaluation_summary.json"
DATA_ROOT="$PROJECT_ROOT/retarget_research/advanced_policy/data/formal_v1"
POLICY_SPLIT="$DATA_ROOT/policy_split_seed20260813.json"
LOG_DIR="$PROJECT_ROOT/retarget_research/outputs/formal_1000/logs"
LOG_FILE="$LOG_DIR/linker_object_centric_3mm_v1.log"

mkdir -p "$LOG_DIR"
cd "$PROJECT_ROOT"
export MPLCONFIGDIR=/tmp/matplotlib-retarget

echo "[1/5] Replay 1000 trajectories and save policy-aligned traces."
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "$PYTHON_BIN" \
  retarget_research/retargeting/evaluate/evaluate_hand_manifest.py \
  --hand linker \
  --manifest "$MANIFEST" \
  --target-dir "$TARGET_DIR" \
  --output-dir "$EVAL_DIR" \
  --policy-trace-dir "$TRACE_DIR" \
  --workers 2 --resume \
  --linker-finger-stiffness 120 --linker-finger-damping 5 \
  --linker-mimic-stiffness 120 --linker-mimic-damping 5 \
  2>&1 | tee "$LOG_FILE"

echo "[2/5] Compare the promoted method with the historical Linker baseline."
"$PYTHON_BIN" retarget_research/retargeting/evaluate/compare_manifest_methods.py \
  --manifest "$MANIFEST" \
  --summary linker_current "$BASELINE_SUMMARY" \
  --summary object_centric_3mm "$NEW_SUMMARY" \
  --output retarget_research/outputs/formal_1000/linker_object_centric_3mm_vs_current.json

echo "[3/5] Generate the full mechanism analysis."
"$PYTHON_BIN" retarget_research/retargeting/evaluate/analyze_object_centric_results.py \
  --manifest "$MANIFEST" \
  --candidate-dir "$TARGET_DIR" \
  --baseline-summary "$BASELINE_SUMMARY" \
  --candidate-summary "$NEW_SUMMARY" \
  --output-json retarget_research/outputs/formal_1000/linker_object_centric_3mm_mechanism.json \
  --output-markdown retarget_research/outputs/formal_1000/linker_object_centric_3mm_mechanism.md

echo "[4/5] Materialize the final Linker policy dataset."
"$PYTHON_BIN" retarget_research/advanced_policy/prepare/prepare_policy_dataset.py \
  --manifest "$MANIFEST" \
  --policy-split "$POLICY_SPLIT" \
  --hand linker \
  --trace-dir "$TRACE_DIR" \
  --evaluation-summary "$NEW_SUMMARY" \
  --output-dir "$DATA_ROOT/linker" \
  --hand-specs retarget_research/advanced_policy/configs/hand_data_specs_v3.json

echo "[5/5] Run the three limited Linker CPU smoke trainings."
"$PYTHON_BIN" retarget_research/advanced_policy/prepare/generate_training_configs.py \
  --matrix retarget_research/advanced_policy/configs/training_matrix_smoke_v1.json \
  --output-dir retarget_research/advanced_policy/configs/generated/smoke_v1
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$PYTHON_BIN" \
  retarget_research/advanced_policy/run_training_matrix.py \
  --index retarget_research/advanced_policy/configs/generated/smoke_v1/config_index.json \
  --filter linker --device cpu

echo "Linker formal 1000, final policy data, and smoke training completed."
