#!/usr/bin/env bash
# Linker 正式1000：向量v2+α+接触3+抓握8+warm start（50集胜者 13/50）。
# warm start 复用已冻结的正式夹紧候选 linker_o6_optimized_v2。
set -euo pipefail

PROJECT_ROOT="/home/lekangwan/projects/DexGraspMotionChallenge2025-final-release"
PYTHON_BIN="/home/lekangwan/miniconda3/envs/hand-retarget/bin/python"
MANIFEST="$PROJECT_ROOT/retarget_research/manifests/formal_50c_100o_1000t_seed20260808.json"
RUN_DIR="$PROJECT_ROOT/retarget_research/retargeting/run"
EVAL_DIR="$PROJECT_ROOT/retarget_research/retargeting/evaluate"
OUTPUT_ROOT="$PROJECT_ROOT/retarget_research/outputs/formal_1000"
CANDIDATE_DIR="$OUTPUT_ROOT/linker_vector_v2alpha_c3g8_v1"
EVAL_DIR_OUT="$OUTPUT_ROOT/linker_vector_v2alpha_c3g8_v1_evaluation"
TRACE_DIR="$PROJECT_ROOT/retarget_research/advanced_policy/traces/formal_v3/linker_vector_v2alpha_c3g8"
WARM_DIR="$OUTPUT_ROOT/linker_o6_optimized_v2"
MESH_ROOT="/home/lekangwan/projects/DexGraspMotionChallenge2025/external_data/meshdata"

mkdir -p "$TRACE_DIR"
cd "$PROJECT_ROOT"
export MPLCONFIGDIR=/tmp/matplotlib-retarget

if [ ! -d "$WARM_DIR" ]; then
  echo "缺少 warm start 候选目录: $WARM_DIR"; exit 1
fi

echo "[1/2] 生成 1000 条向量v2+α+接触3+抓握8 候选"
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 "$PYTHON_BIN" \
  "$RUN_DIR/run_linker_vector_manifest.py" \
  --manifest "$MANIFEST" \
  --output-dir "$CANDIDATE_DIR" \
  --workers 6 --resume --maxeval 50 \
  --vector-config "$PROJECT_ROOT/retarget_research/retargeting/configs/linker_anydex_vectors_v2.json" \
  --contact-weight 3 --grip-flexion-weight 8 \
  --warm-start-dir "$WARM_DIR" \
  --contact-fallback nearest \
  --object-root "$MESH_ROOT"

echo "[2/2] 物理重放 + 专家 trace"
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 "$PYTHON_BIN" \
  "$EVAL_DIR/evaluate_hand_manifest.py" \
  --hand linker \
  --manifest "$MANIFEST" \
  --target-dir "$CANDIDATE_DIR" \
  --output-dir "$EVAL_DIR_OUT" \
  --policy-trace-dir "$TRACE_DIR" \
  --workers 6 --resume

echo "ALL_STAGES_COMPLETE"
