#!/usr/bin/env bash
set -euo pipefail

# 先训练三手Initial-Geometry+Phase Chunk8，再在固定valid50做自主闭环评测。
ROOT=/home/lekangwan/projects/DexGraspMotionChallenge2025-final-release
PYTHON=/home/lekangwan/miniconda3/envs/dexgrasp/bin/python

cd "$ROOT"
"$PYTHON" -u retarget_research/advanced_policy_v2/run/run_candidate_training_matrix.py \
  --config-dir retarget_research/advanced_policy_v2/configs/generated \
  --models geometry_plan_chunk \
  --hands linker xhand wuji \
  --device cuda

"$PYTHON" -u retarget_research/advanced_policy_v2/evaluate/run_candidate_valid50.py \
  --models geometry_plan_chunk \
  --hands linker xhand wuji \
  --device cuda \
  --workers 3

