#!/usr/bin/env bash
# 新方法开发集首筛：AnyDex风格向量 + 物体接触锚（Linker/XHand 各两个变体）
# 数据集：linker_independent_validation_v1.json（10物体×2轨迹=20条，已用于方法选择的开发集）
# 基线：Linker夹紧v1、XHand指腹细化v1（重放为当前30cm稳定协议，与旧10cm历史结果并存）
# 所有阶段带 --resume，中断后重跑本脚本即可安全续跑。
# 日志: retarget_research/outputs/vector_methods_dev_screen/run.log
set -euo pipefail

cd "$(cd "$(dirname "$0")" && pwd)/../../.."

export MPLCONFIGDIR=/tmp/matplotlib-retarget
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

PY=/home/lekangwan/miniconda3/envs/hand-retarget/bin/python
RUN=retarget_research/retargeting/run
EVAL=retarget_research/retargeting/evaluate
MANIFEST=retarget_research/retargeting/configs/linker_independent_validation_v1.json
OUT=retarget_research/outputs/vector_methods_dev_screen
LOG=$OUT/run.log

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
run_stage() {
    local name="$1"; shift
    log "--- $name ---"
    if ! "$@" 2>&1 | tee -a "$LOG"; then
        log "STAGE_FAILED: $name"; exit 1
    fi
    log "STAGE_OK: $name"
}

mkdir -p "$OUT"

# ---- Stage 1: 生成四个新方法候选 ----
run_stage "gen_xhand_vec" "$PY" "$RUN/run_xhand_vector_manifest.py" \
    --manifest "$MANIFEST" --output-dir "$OUT/xhand_vec" \
    --workers 4 --resume --maxeval 50

run_stage "gen_xhand_vec_contact" "$PY" "$RUN/run_xhand_vector_manifest.py" \
    --manifest "$MANIFEST" --output-dir "$OUT/xhand_vec_contact" \
    --workers 4 --resume --maxeval 50 --contact-weight 5.0

run_stage "gen_linker_vec" "$PY" "$RUN/run_linker_vector_manifest.py" \
    --manifest "$MANIFEST" --output-dir "$OUT/linker_vec" \
    --workers 4 --resume --maxeval 50

run_stage "gen_linker_vec_contact" "$PY" "$RUN/run_linker_vector_manifest.py" \
    --manifest "$MANIFEST" --output-dir "$OUT/linker_vec_contact" \
    --workers 4 --resume --maxeval 50 --contact-weight 5.0

# ---- Stage 2: 基线重放（30cm协议）+ 四个新方法物理重放 ----
run_stage "eval_xhand_pad_baseline" "$PY" "$EVAL/evaluate_hand_manifest.py" \
    --hand xhand --manifest "$MANIFEST" \
    --target-dir retarget_research/outputs/xhand_independent_validation_contact_v1 \
    --output-dir "$OUT/xhand_pad_baseline_evaluation" --workers 4 --resume

run_stage "eval_linker_squeeze_baseline" "$PY" "$EVAL/evaluate_hand_manifest.py" \
    --hand linker --manifest "$MANIFEST" \
    --target-dir retarget_research/outputs/linker_keypoint15_squeeze_independent_v1 \
    --output-dir "$OUT/linker_squeeze_baseline_evaluation" --workers 4 --resume

run_stage "eval_xhand_vec" "$PY" "$EVAL/evaluate_hand_manifest.py" \
    --hand xhand --manifest "$MANIFEST" \
    --target-dir "$OUT/xhand_vec" --output-dir "$OUT/xhand_vec_evaluation" --workers 4 --resume

run_stage "eval_xhand_vec_contact" "$PY" "$EVAL/evaluate_hand_manifest.py" \
    --hand xhand --manifest "$MANIFEST" \
    --target-dir "$OUT/xhand_vec_contact" --output-dir "$OUT/xhand_vec_contact_evaluation" \
    --workers 4 --resume

run_stage "eval_linker_vec" "$PY" "$EVAL/evaluate_hand_manifest.py" \
    --hand linker --manifest "$MANIFEST" \
    --target-dir "$OUT/linker_vec" --output-dir "$OUT/linker_vec_evaluation" --workers 4 --resume

run_stage "eval_linker_vec_contact" "$PY" "$EVAL/evaluate_hand_manifest.py" \
    --hand linker --manifest "$MANIFEST" \
    --target-dir "$OUT/linker_vec_contact" --output-dir "$OUT/linker_vec_contact_evaluation" \
    --workers 4 --resume

# ---- Stage 3: 配对比较 ----
run_stage "compare_xhand" "$PY" "$EVAL/compare_manifest_methods.py" \
    --manifest "$MANIFEST" \
    --summary xhand_pad_v1 "$OUT/xhand_pad_baseline_evaluation/manifest_evaluation_summary.json" \
    --summary xhand_vec "$OUT/xhand_vec_evaluation/manifest_evaluation_summary.json" \
    --summary xhand_vec_contact "$OUT/xhand_vec_contact_evaluation/manifest_evaluation_summary.json" \
    --output "$OUT/xhand_three_way_comparison.json"

run_stage "compare_linker" "$PY" "$EVAL/compare_manifest_methods.py" \
    --manifest "$MANIFEST" \
    --summary linker_squeeze_v1 "$OUT/linker_squeeze_baseline_evaluation/manifest_evaluation_summary.json" \
    --summary linker_vec "$OUT/linker_vec_evaluation/manifest_evaluation_summary.json" \
    --summary linker_vec_contact "$OUT/linker_vec_contact_evaluation/manifest_evaluation_summary.json" \
    --output "$OUT/linker_three_way_comparison.json"

log "=== ALL_STAGES_COMPLETE ==="
