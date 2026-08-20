#!/usr/bin/env bash
# Wuji 正式 1000 三阶段一键脚本
# A: coupled_flexion 解剖点法初始候选 -> B: v1 拇指零空间细化 -> C: 物理重放 + 策略 trace
# 所有阶段带 --resume，中断后重跑本脚本即可安全续跑。
# 日志: retarget_research/outputs/formal_1000/formal_1000_v1_run.log
set -euo pipefail

cd "$(cd "$(dirname "$0")" && pwd)/../../.."

export MPLCONFIGDIR=/tmp/matplotlib-retarget
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

PY=/home/lekangwan/miniconda3/envs/hand-retarget/bin/python
LOG=retarget_research/outputs/formal_1000/formal_1000_v1_run.log
MANIFEST=retarget_research/manifests/formal_50c_100o_1000t_seed20260808.json

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

log "=== 开始 Wuji 正式 1000 三阶段 ==="

log "--- Stage A: coupled_flexion 解剖点法初始候选 ---"
if ! "$PY" retarget_research/retargeting/run/run_wuji_manifest.py \
    --manifest "$MANIFEST" \
    --output-dir retarget_research/outputs/formal_1000/wuji_anatomy_coupled_v2 \
    --workers 6 --resume \
    --maxeval 50 \
    --anatomy-config retarget_research/retargeting/configs/wuji_anatomy_coupled_v1.json \
    2>&1 | tee -a "$LOG"; then
    log "STAGE_A_FAILED"; exit 1
fi
log "STAGE_A_OK"

log "--- Stage B: v1 拇指零空间细化 ---"
if ! "$PY" retarget_research/outputs/formal_1000/v1runner/run/run_wuji_thumb_nullspace_manifest.py \
    --manifest "$MANIFEST" \
    --initial-dir retarget_research/outputs/formal_1000/wuji_anatomy_coupled_v2 \
    --output-dir retarget_research/outputs/formal_1000/wuji_thumb_nullspace_v1 \
    --workers 6 --resume \
    --maxeval 80 --tip-weight 1.0 --neutral-weight 0.05 --temporal-weight 0.01 \
    2>&1 | tee -a "$LOG"; then
    log "STAGE_B_FAILED"; exit 1
fi
log "STAGE_B_OK"

log "--- Stage C: 物理重放 + 策略 trace ---"
if ! "$PY" retarget_research/retargeting/evaluate/evaluate_hand_manifest.py \
    --hand wuji \
    --manifest "$MANIFEST" \
    --target-dir retarget_research/outputs/formal_1000/wuji_thumb_nullspace_v1 \
    --output-dir retarget_research/outputs/formal_1000/wuji_thumb_nullspace_v1_evaluation \
    --policy-trace-dir retarget_research/outputs/formal_1000/wuji_thumb_nullspace_v1_traces \
    --workers 6 --resume \
    2>&1 | tee -a "$LOG"; then
    log "STAGE_C_FAILED"; exit 1
fi
log "STAGE_C_OK"

log "=== ALL_STAGES_COMPLETE ==="
"$PY" -c "
import json
s = json.load(open('retarget_research/outputs/formal_1000/wuji_thumb_nullspace_v1_evaluation/manifest_evaluation_summary.json'))
print('formal 1000 summary: success=%s/%s rate=%.1f%% wall=%ss' % (
    s['success_count'], s['trajectory_count'], 100 * s['success_rate'], round(s['wall_time_seconds'], 1)))
" | tee -a "$LOG"
