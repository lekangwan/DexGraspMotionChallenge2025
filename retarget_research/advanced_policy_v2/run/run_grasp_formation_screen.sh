#!/usr/bin/env bash
set -euo pipefail

# 生成三种自主组合策略，并在同一valid50上评测。
ROOT=/home/lekangwan/projects/DexGraspMotionChallenge2025-final-release
PYTHON=/home/lekangwan/miniconda3/envs/dexgrasp/bin/python

cd "$ROOT"
"$PYTHON" retarget_research/advanced_policy_v2/prepare/build_composite_candidates.py
"$PYTHON" -u retarget_research/advanced_policy_v2/evaluate/run_candidate_valid50.py \
  --models phase_lead05 phase_lead10 phase_feedback_fingers \
  --hands linker xhand wuji \
  --device cuda \
  --workers 3
