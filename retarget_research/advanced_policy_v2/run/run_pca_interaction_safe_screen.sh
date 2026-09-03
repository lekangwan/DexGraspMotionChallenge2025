#!/usr/bin/env bash
set -euo pipefail

# 不重训网络，只比较冻结PCA手腕后的25%/50%手指残差。
ROOT=/home/lekangwan/projects/DexGraspMotionChallenge2025-final-release
PYTHON=/home/lekangwan/miniconda3/envs/dexgrasp/bin/python

cd "$ROOT"
"$PYTHON" retarget_research/advanced_policy_v2/prepare/build_interaction_safe_checkpoints.py
MPLCONFIGDIR=/tmp/matplotlib-retarget "$PYTHON" -u \
  retarget_research/advanced_policy_v2/evaluate/run_candidate_valid50.py \
  --models geometry_pca_interaction_finger025 geometry_pca_interaction_finger050 \
  --hands linker xhand wuji \
  --device cuda \
  --workers 3
