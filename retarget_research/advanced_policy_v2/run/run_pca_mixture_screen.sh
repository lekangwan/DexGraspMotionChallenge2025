#!/usr/bin/env bash
set -euo pipefail

# 三手并行训练同结构的四候选生成器与质量判别器，再评测两种选择规则。
ROOT=/home/lekangwan/projects/DexGraspMotionChallenge2025-final-release
PYTHON=/home/lekangwan/miniconda3/envs/dexgrasp/bin/python
TRAIN="$ROOT/retarget_research/advanced_policy_v2/train_pca_mixture.py"

cd "$ROOT"
"$PYTHON" retarget_research/advanced_policy_v2/prepare/build_mixture_configs.py
pids=()
for hand in linker xhand wuji; do
  "$PYTHON" -u "$TRAIN" \
    --config "retarget_research/advanced_policy_v2/configs/generated/${hand}_geometry_pca_mixture.json" \
    --device cuda 2>&1 | sed -u "s/^/[${hand}-mixture] /" &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done

"$PYTHON" -u retarget_research/advanced_policy_v2/evaluate/run_candidate_valid50.py \
  --models geometry_mixture_gate geometry_mixture_critic \
  --hands linker xhand wuji \
  --device cuda \
  --workers 3
