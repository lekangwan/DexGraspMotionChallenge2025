#!/usr/bin/env bash
# 12个已训练模型的统一闭环评测：每模型500条对象级test轨迹，CPU PhysX。
set -euo pipefail

PROJECT_ROOT="/home/lekangwan/projects/DexGraspMotionChallenge2025-final-release"
PYTHON_BIN="/home/lekangwan/miniconda3/envs/hand-retarget/bin/python"
MANIFEST="$PROJECT_ROOT/retarget_research/manifests/formal_50c_100o_1000t_seed20260808.json"
SPLIT="$PROJECT_ROOT/retarget_research/advanced_policy/data/formal_v1/policy_split_seed20260813.json"
EVAL="$PROJECT_ROOT/retarget_research/advanced_policy/evaluate_policy_manifest.py"
RUNS="$PROJECT_ROOT/retarget_research/advanced_policy/runs/formal_final"
DATA="$PROJECT_ROOT/retarget_research/advanced_policy/data/formal_final"
LOG="$PROJECT_ROOT/retarget_research/advanced_policy/runs/formal_final/policy_eval_matrix.log"

cd "$PROJECT_ROOT"
export MPLCONFIGDIR=/tmp/matplotlib-retarget

declare -A TARGETS=(
  [linker]="$PROJECT_ROOT/retarget_research/outputs/formal_1000/linker_vector_v2alpha_c3g8_v1"
  [xhand_official]="$PROJECT_ROOT/retarget_research/outputs/formal_1000/xhand_official"
  [wuji_old]="$PROJECT_ROOT/retarget_research/outputs/formal_1000/wuji_v1"
  [wuji_n005]="$PROJECT_ROOT/retarget_research/outputs/formal_1000/wuji_thumb_nullspace_v1"
)
declare -A HAND_OF=( [linker]=linker [xhand_official]=xhand [wuji_old]=wuji [wuji_n005]=wuji )

for exp in linker_bc_v1 linker_temporal3_v1 linker_diffusion_v1 \
           xhand_official_bc_v1 xhand_official_temporal3_v1 xhand_official_diffusion_v1 \
           wuji_old_bc_v1 wuji_old_temporal3_v1 wuji_old_diffusion_v1 \
           wuji_n005_bc_v1 wuji_n005_temporal3_v1 wuji_n005_diffusion_v1; do
  pool="${exp%_bc_v1}"; pool="${pool%_temporal3_v1}"; pool="${pool%_diffusion_v1}"
  hand="${HAND_OF[$pool]}"
  echo "[$(date '+%F %T')] eval $exp (hand=$hand pool=$pool)" | tee -a "$LOG"
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 "$PYTHON_BIN" "$EVAL" \
    --hand "$hand" \
    --manifest "$MANIFEST" \
    --policy-split "$SPLIT" \
    --target-dir "${TARGETS[$pool]}" \
    --checkpoint "$RUNS/$exp/best.pt" \
    --data-dir "$DATA/$pool" \
    --output-dir "$RUNS/${exp}_policy_eval" \
    --device cpu --workers 6 --split test 2>&1 | tee -a "$LOG"
done

echo "ALL_POLICY_EVALS_COMPLETE" | tee -a "$LOG"
