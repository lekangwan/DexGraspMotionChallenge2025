#!/usr/bin/env bash
set -euo pipefail

# 关键状态模型通过离线门后，才运行固定valid50闭环评测。
ROOT=/home/lekangwan/projects/DexGraspMotionChallenge2025-final-release
PYTHON=/home/lekangwan/miniconda3/envs/dexgrasp/bin/python

cd "$ROOT"
MPLCONFIGDIR=/tmp/matplotlib-retarget "$PYTHON" -u \
  retarget_research/advanced_policy_v2/evaluate/run_candidate_valid50.py \
  --models geometry_keypose_lift \
  --hands linker xhand wuji \
  --device cuda \
  --workers 3
