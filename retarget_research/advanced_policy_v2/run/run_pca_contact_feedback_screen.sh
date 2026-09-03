#!/usr/bin/env bash
set -euo pipefail

# 三只手使用完全相同的接触阈值和收紧幅度；只包装自主PCA，不重新训练。
ROOT=/home/lekangwan/projects/DexGraspMotionChallenge2025-final-release
PYTHON=/home/lekangwan/miniconda3/envs/dexgrasp/bin/python

cd "$ROOT"
"$PYTHON" retarget_research/advanced_policy_v2/prepare/build_contact_feedback_checkpoints.py
MPLCONFIGDIR=/tmp/matplotlib-retarget "$PYTHON" -u \
  retarget_research/advanced_policy_v2/evaluate/run_candidate_valid50.py \
  --models geometry_pca_contact_feedback \
  --hands linker xhand wuji \
  --device cuda \
  --workers 3
