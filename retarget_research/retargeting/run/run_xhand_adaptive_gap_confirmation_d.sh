#!/usr/bin/env bash
# 输入：物理结果从未被查看的D组50类50轨迹，以及A组唯一入选的XHand 0.10 rad分指规则。
# 输出：D组官方XHand基线、分指候选、两套PhysX摘要和严格配对JSON。
# 内部逻辑：先为D组生成一次官方轨迹，再做确定性分指后处理；两方物理并行重放。
# 作用：不再调整0.10 rad或门控阈值，只检验A组的+1/-0能否在未见物体/轨迹上复现。

set -euo pipefail

PROJECT_ROOT="/home/lekangwan/projects/DexGraspMotionChallenge2025-final-release"
PYTHON_BIN="/home/lekangwan/miniconda3/envs/hand-retarget/bin/python"
MANIFEST="$PROJECT_ROOT/retarget_research/manifests/formal_confirmation_d_50c_50t_seed20260815.json"
RUN_DIR="$PROJECT_ROOT/retarget_research/retargeting/run"
EVAL_DIR="$PROJECT_ROOT/retarget_research/retargeting/evaluate"
OUTPUT_ROOT="$PROJECT_ROOT/retarget_research/outputs/adaptive_finger_gap_search_v1/d"
BASELINE_DIR="$OUTPUT_ROOT/xhand_official"
CANDIDATE_DIR="$OUTPUT_ROOT/xhand_delta0.10"
BASELINE_EVAL="$OUTPUT_ROOT/xhand_official_evaluation"
CANDIDATE_EVAL="$OUTPUT_ROOT/xhand_delta0.10_evaluation"
LOG_DIR="$OUTPUT_ROOT/logs"

mkdir -p "$LOG_DIR"
cd "$PROJECT_ROOT"
export MPLCONFIGDIR=/tmp/matplotlib-retarget

echo "[1/4] Generate or resume the official XHand trajectories on frozen D."
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "$PYTHON_BIN" \
  "$RUN_DIR/run_xhand_manifest.py" \
  --manifest "$MANIFEST" --output-dir "$BASELINE_DIR" \
  --workers 2 --resume \
  --iter-num 100 --sample-frame-num 5 \
  --trans-lr 0.005 --ang-lr 0.01 \
  --trans-bound 2.0 --enlarge-scale 1.0 --device cpu

echo "[2/4] Apply the frozen 0.10 rad expert-gated per-finger correction."
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$PYTHON_BIN" \
  "$RUN_DIR/refine_adaptive_finger_gap.py" \
  --hand xhand --manifest "$MANIFEST" \
  --input-dir "$BASELINE_DIR" --output-dir "$CANDIDATE_DIR" \
  --max-delta-rad 0.10 \
  --contact-threshold 0.02 --mismatch-margin 0.003 --epsilon-rad 0.01

echo "[3/4] Replay baseline and the sole candidate in parallel."
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$PYTHON_BIN" \
  "$EVAL_DIR/evaluate_hand_manifest.py" \
  --hand xhand --manifest "$MANIFEST" \
  --target-dir "$BASELINE_DIR" --output-dir "$BASELINE_EVAL" \
  --workers 1 --resume >"$LOG_DIR/xhand_official.log" 2>&1 &
pid_baseline=$!

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$PYTHON_BIN" \
  "$EVAL_DIR/evaluate_hand_manifest.py" \
  --hand xhand --manifest "$MANIFEST" \
  --target-dir "$CANDIDATE_DIR" --output-dir "$CANDIDATE_EVAL" \
  --workers 1 --resume >"$LOG_DIR/xhand_delta0.10.log" 2>&1 &
pid_candidate=$!

pids=("$pid_baseline" "$pid_candidate")
while true; do
  running=0
  for pid in "${pids[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then running=$((running + 1)); fi
  done
  if [[ "$running" -eq 0 ]]; then break; fi
  echo "XHand confirmation D: $running process(es) running; logs: $LOG_DIR"
  sleep 30
done

failed=0
for pid in "${pids[@]}"; do if ! wait "$pid"; then failed=1; fi; done
if [[ "$failed" -ne 0 ]]; then
  echo "D evaluation failed; inspect $LOG_DIR" >&2
  exit 1
fi

echo "[4/4] Build the frozen paired comparison."
"$PYTHON_BIN" "$EVAL_DIR/compare_manifest_methods.py" \
  --manifest "$MANIFEST" \
  --summary xhand_official "$BASELINE_EVAL/manifest_evaluation_summary.json" \
  --summary xhand_adaptive_gap_0.10 "$CANDIDATE_EVAL/manifest_evaluation_summary.json" \
  --output "$OUTPUT_ROOT/paired_comparison.json"

echo "XHand adaptive finger-gap confirmation D completed."
