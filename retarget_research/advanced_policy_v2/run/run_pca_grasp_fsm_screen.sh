#!/usr/bin/env bash
set -euo pipefail

# 抬升前最多暂停20个物理步，三只手统一等待稳定对向接触。
ROOT=/home/lekangwan/projects/DexGraspMotionChallenge2025-final-release
PYTHON=/home/lekangwan/miniconda3/envs/dexgrasp/bin/python

cd "$ROOT"
"$PYTHON" retarget_research/advanced_policy_v2/prepare/build_contact_feedback_checkpoints.py
MPLCONFIGDIR=/tmp/matplotlib-retarget "$PYTHON" -u \
  retarget_research/advanced_policy_v2/evaluate/run_candidate_valid50.py \
  --models geometry_pca_grasp_fsm \
  --hands linker xhand wuji \
  --device cuda \
  --workers 3
