#!/usr/bin/env bash
# XHand 正式1000：向量+α r1（50集胜者 32/50）。生成 → 物理重放+trace → 完成标记。
set -euo pipefail

PROJECT_ROOT="/home/lekangwan/projects/DexGraspMotionChallenge2025-final-release"
PYTHON_BIN="/home/lekangwan/miniconda3/envs/hand-retarget/bin/python"
MANIFEST="$PROJECT_ROOT/retarget_research/manifests/formal_50c_100o_1000t_seed20260808.json"
RUN_DIR="$PROJECT_ROOT/retarget_research/retargeting/run"
EVAL_DIR="$PROJECT_ROOT/retarget_research/retargeting/evaluate"
OUTPUT_ROOT="$PROJECT_ROOT/retarget_research/outputs/formal_1000"
CANDIDATE_DIR="$OUTPUT_ROOT/xhand_vector_alpha_v1"
EVAL_DIR_OUT="$OUTPUT_ROOT/xhand_vector_alpha_v1_evaluation"
TRACE_DIR="$PROJECT_ROOT/retarget_research/advanced_policy/traces/formal_v3/xhand_vector_alpha"

mkdir -p "$TRACE_DIR"
cd "$PROJECT_ROOT"
export MPLCONFIGDIR=/tmp/matplotlib-retarget

echo "[1/2] 生成 1000 条向量+α 候选"
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 "$PYTHON_BIN" \
  "$RUN_DIR/run_xhand_vector_manifest.py" \
  --manifest "$MANIFEST" \
  --output-dir "$CANDIDATE_DIR" \
  --workers 6 --resume --maxeval 50

echo "[2/2] 物理重放 + 专家 trace"
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 "$PYTHON_BIN" \
  "$EVAL_DIR/evaluate_hand_manifest.py" \
  --hand xhand \
  --manifest "$MANIFEST" \
  --target-dir "$CANDIDATE_DIR" \
  --output-dir "$EVAL_DIR_OUT" \
  --policy-trace-dir "$TRACE_DIR" \
  --workers 6 --resume

echo "ALL_STAGES_COMPLETE"
