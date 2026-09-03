#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/lekangwan/projects/DexGraspMotionChallenge2025-final-release"
PYTHON="/home/lekangwan/miniconda3/envs/hand-retarget/bin/python"
cd "$ROOT"

"$PYTHON" -u retarget_research/advanced_policy_v2/evaluate/run_candidate_valid50.py \
  --models direct_interaction_temporal3 \
  --hands linker xhand wuji \
  --device cuda \
  --workers 1 \
  --parallel-hands
