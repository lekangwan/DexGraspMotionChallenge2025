#!/usr/bin/env bash
set -euo pipefail

# 三手并行训练相同结构的PCA动态交互残差，再在固定valid50闭环首筛。
ROOT=/home/lekangwan/projects/DexGraspMotionChallenge2025-final-release
PYTHON=/home/lekangwan/miniconda3/envs/dexgrasp/bin/python
TRAIN="$ROOT/retarget_research/advanced_policy_v2/train_interaction_residual.py"

cd "$ROOT"
"$PYTHON" retarget_research/advanced_policy_v2/prepare/build_interaction_configs.py
pids=()
for hand in linker xhand wuji; do
  MPLCONFIGDIR=/tmp/matplotlib-retarget "$PYTHON" -u "$TRAIN" \
    --config "retarget_research/advanced_policy_v2/configs/generated/${hand}_geometry_pca_interaction.json" \
    --device cuda 2>&1 | sed -u "s/^/[${hand}-interaction] /" &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done

MPLCONFIGDIR=/tmp/matplotlib-retarget "$PYTHON" -u \
  retarget_research/advanced_policy_v2/evaluate/run_candidate_valid50.py \
  --models geometry_pca_interaction \
  --hands linker xhand wuji \
  --device cuda \
  --workers 3
