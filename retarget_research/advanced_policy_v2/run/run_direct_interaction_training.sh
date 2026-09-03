#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/lekangwan/projects/DexGraspMotionChallenge2025-final-release"
PYTHON="/home/lekangwan/miniconda3/envs/hand-retarget/bin/python"
cd "$ROOT"

for hand in linker xhand wuji; do
  "$PYTHON" -m retarget_research.advanced_policy_v2.prepare.prepare_direct_interaction \
    --hand "$hand"
done
"$PYTHON" -m retarget_research.advanced_policy_v2.prepare.build_direct_interaction_configs

for hand in linker xhand wuji; do
  "$PYTHON" -u -m retarget_research.advanced_policy_v2.train_direct_interaction \
    --config "retarget_research/advanced_policy_v2/configs/generated/${hand}_direct_interaction_temporal3.json" \
    --device cuda 2>&1 | sed -u "s/^/[$hand] /" &
done
wait

echo "三只手的无PCA Temporal3直接策略训练完成。"
