#!/usr/bin/env bash
# Linker 向量法第1轮优化：闭合/抬升期抓握屈曲偏置（grip-flexion-weight 5.0）
# 基线复用 vector_methods_dev_screen/linker_squeeze_baseline_evaluation（30cm协议）
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
LOG=$OUT/grip_round1.log

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
run_stage() {
    local name="$1"; shift
    log "--- $name ---"
    if ! "$@" 2>&1 | tee -a "$LOG"; then
        log "STAGE_FAILED: $name"; exit 1
    fi
    log "STAGE_OK: $name"
}

run_stage "gen_linker_vec_grip5" "$PY" "$RUN/run_linker_vector_manifest.py" \
    --manifest "$MANIFEST" --output-dir "$OUT/linker_vec_grip5" \
    --workers 4 --resume --maxeval 50 --grip-flexion-weight 5.0

run_stage "gen_linker_vec_contact_grip5" "$PY" "$RUN/run_linker_vector_manifest.py" \
    --manifest "$MANIFEST" --output-dir "$OUT/linker_vec_contact_grip5" \
    --workers 4 --resume --maxeval 50 --contact-weight 5.0 --grip-flexion-weight 5.0

run_stage "eval_linker_vec_grip5" "$PY" "$EVAL/evaluate_hand_manifest.py" \
    --hand linker --manifest "$MANIFEST" \
    --target-dir "$OUT/linker_vec_grip5" --output-dir "$OUT/linker_vec_grip5_evaluation" \
    --workers 4 --resume

run_stage "eval_linker_vec_contact_grip5" "$PY" "$EVAL/evaluate_hand_manifest.py" \
    --hand linker --manifest "$MANIFEST" \
    --target-dir "$OUT/linker_vec_contact_grip5" --output-dir "$OUT/linker_vec_contact_grip5_evaluation" \
    --workers 4 --resume

run_stage "compare_linker_grip5" "$PY" "$EVAL/compare_manifest_methods.py" \
    --manifest "$MANIFEST" \
    --summary linker_squeeze_v1 "$OUT/linker_squeeze_baseline_evaluation/manifest_evaluation_summary.json" \
    --summary linker_vec "$OUT/linker_vec_evaluation/manifest_evaluation_summary.json" \
    --summary linker_vec_grip5 "$OUT/linker_vec_grip5_evaluation/manifest_evaluation_summary.json" \
    --summary linker_vec_contact_grip5 "$OUT/linker_vec_contact_grip5_evaluation/manifest_evaluation_summary.json" \
    --output "$OUT/linker_grip5_comparison.json"

log "=== ALL_STAGES_COMPLETE ==="
