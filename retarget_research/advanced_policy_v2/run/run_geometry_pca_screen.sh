#!/usr/bin/env bash
set -euo pipefail

# 每档PCA秩内三只手并行训练，随后在固定valid50评测。
ROOT=/home/lekangwan/projects/DexGraspMotionChallenge2025-final-release
PYTHON=/home/lekangwan/miniconda3/envs/dexgrasp/bin/python
TRAIN="$ROOT/retarget_research/advanced_policy_v2/train_geometry_pca.py"

cd "$ROOT"
"$PYTHON" retarget_research/advanced_policy_v2/prepare/build_geometry_pca_configs.py
for rank in 16 32; do
  pids=()
  for hand in linker xhand wuji; do
    "$PYTHON" -u "$TRAIN" \
      --config "retarget_research/advanced_policy_v2/configs/generated/${hand}_geometry_pca${rank}.json" \
      --device cuda 2>&1 | sed -u "s/^/[${hand}-pca${rank}] /" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
done

"$PYTHON" -u retarget_research/advanced_policy_v2/evaluate/run_candidate_valid50.py \
  --models geometry_pca16 geometry_pca32 \
  --hands linker xhand wuji \
  --device cuda \
  --workers 3
