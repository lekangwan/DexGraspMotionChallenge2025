#!/usr/bin/env bash
# 在 confirm_e 50集上生成 Linker 现任正式方法（无时序关键点 → 动态夹紧 → 3mm 物体中心校准）
# 三阶段均 --resume 安全续跑；物理评测由助手在本地执行。
set -euo pipefail

PROJECT_ROOT="/home/lekangwan/projects/DexGraspMotionChallenge2025-final-release"
PYTHON_BIN="/home/lekangwan/miniconda3/envs/hand-retarget/bin/python"
MANIFEST="$PROJECT_ROOT/retarget_research/manifests/formal_confirmation_e_50c_50t_seed20260816.json"
RUN_DIR="$PROJECT_ROOT/retarget_research/retargeting/run"
OUTPUT_ROOT="$PROJECT_ROOT/retarget_research/outputs/linker_incumbent_confirm_e"
BASELINE_DIR="$OUTPUT_ROOT/no_temporal_baseline"
SQUEEZE_DIR="$OUTPUT_ROOT/dynamic_squeeze"
ADVANCE_DIR="$OUTPUT_ROOT/advance_3mm"
LOG_DIR="$OUTPUT_ROOT/logs"

mkdir -p "$LOG_DIR"
cd "$PROJECT_ROOT"
export MPLCONFIGDIR=/tmp/matplotlib-retarget

echo "[1/3] 无时序关键点基线"
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 "$PYTHON_BIN" \
  "$RUN_DIR/run_linker_manifest.py" \
  --manifest "$MANIFEST" \
  --output-dir "$BASELINE_DIR" \
  --workers 4 --resume --maxeval 100 --include-thumb-middle \
  --joint-temporal-weight 0 \
  --translation-temporal-weight 0 \
  --rotation-temporal-weight 0

echo "[2/3] 动态夹紧"
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 "$PYTHON_BIN" \
  "$RUN_DIR/refine_linker_squeeze.py" \
  --manifest "$MANIFEST" \
  --baseline-dir "$BASELINE_DIR" \
  --output-dir "$SQUEEZE_DIR" \
  --method-name linker_o6_no_temporal_dynamic_squeeze_v2 \
  --thumb-yaw-delta 0.075 \
  --thumb-pitch-delta 0.1875 \
  --finger-delta 0.425 \
  --contact-threshold 0.02 \
  --min-contact-tips 2 \
  --lift-delta 0.03 \
  --contact-fallback nearest

echo "[3/3] 3mm 物体中心校准"
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 "$PYTHON_BIN" \
  "$RUN_DIR/refine_linker_object_centric_advance.py" \
  --manifest "$MANIFEST" \
  --input-dir "$SQUEEZE_DIR" \
  --output-dir "$ADVANCE_DIR" \
  --max-advance-mm 3

echo "ALL_STAGES_COMPLETE"
