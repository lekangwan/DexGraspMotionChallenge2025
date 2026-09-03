#!/usr/bin/env bash
# 先并行训练三只手，成功后自动进入100条valid，不运行500条test。
set -euo pipefail

ROOT="/home/lekangwan/projects/DexGraspMotionChallenge2025-final-release"
cd "$ROOT"
bash retarget_research/advanced_policy/run/run_autonomous_initial_phase_delta_train.sh
bash retarget_research/advanced_policy/run/run_autonomous_initial_phase_delta_valid.sh
