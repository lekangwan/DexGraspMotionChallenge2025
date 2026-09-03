#!/usr/bin/env bash
set -euo pipefail

# 顺序训练两档紧凑Phase；训练完成后在固定valid50闭环评测。
ROOT=/home/lekangwan/projects/DexGraspMotionChallenge2025-final-release
PYTHON=/home/lekangwan/miniconda3/envs/dexgrasp/bin/python

cd "$ROOT"
"$PYTHON" retarget_research/advanced_policy_v2/prepare/build_compact_phase_configs.py
"$PYTHON" -u retarget_research/advanced_policy_v2/run/run_candidate_training_matrix.py \
  --config-dir retarget_research/advanced_policy_v2/configs/generated \
  --models phase_compact192 phase_compact96 \
  --hands linker xhand wuji \
  --device cuda
"$PYTHON" -u retarget_research/advanced_policy_v2/evaluate/run_candidate_valid50.py \
  --models phase_compact192 phase_compact96 \
  --hands linker xhand wuji \
  --device cuda \
  --workers 3
