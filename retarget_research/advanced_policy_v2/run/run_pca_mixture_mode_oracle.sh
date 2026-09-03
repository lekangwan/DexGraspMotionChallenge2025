#!/usr/bin/env bash
set -euo pipefail

# 固定执行四个生成模式；最后只统计并集上限，不把它当作可部署策略成绩。
ROOT=/home/lekangwan/projects/DexGraspMotionChallenge2025-final-release
PYTHON=/home/lekangwan/miniconda3/envs/dexgrasp/bin/python

cd "$ROOT"
"$PYTHON" retarget_research/advanced_policy_v2/prepare/build_mixture_mode_checkpoints.py
"$PYTHON" -u retarget_research/advanced_policy_v2/evaluate/run_candidate_valid50.py \
  --models geometry_mixture_mode0 geometry_mixture_mode1 geometry_mixture_mode2 geometry_mixture_mode3 \
  --hands linker xhand wuji \
  --device cuda \
  --workers 3
