#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/lekangwan/projects/DexGraspMotionChallenge2025-final-release"
PYTHON="/home/lekangwan/miniconda3/envs/hand-retarget/bin/python"
V2="$ROOT/retarget_research/advanced_policy_v2"

"$PYTHON" "$V2/prepare/build_pca_latent_diffusion_configs.py"
for hand in linker xhand wuji; do
  output="$V2/runs/candidates_v1/$hand/geometry_pca_latent_diffusion"
  if [[ -f "$output/training_summary.json" ]]; then
    echo "[$hand] 已有完整训练摘要，跳过以免覆盖。"
    continue
  fi
  echo "[$hand] PCA潜空间Diffusion训练"
  "$PYTHON" -u "$V2/train_pca_latent_diffusion.py" \
    --config "$V2/configs/generated/${hand}_geometry_pca_latent_diffusion.json" \
    --device cuda 2>&1 | sed -u "s/^/[$hand] /" &
done
wait

echo "三只手训练完成；先审查training_summary.json，不自动启动PhysX。"
