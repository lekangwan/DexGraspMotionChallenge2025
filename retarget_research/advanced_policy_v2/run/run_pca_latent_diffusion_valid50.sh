#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/lekangwan/projects/DexGraspMotionChallenge2025-final-release"
PYTHON="/home/lekangwan/miniconda3/envs/hand-retarget/bin/python"
V2="$ROOT/retarget_research/advanced_policy_v2"

"$PYTHON" "$V2/prepare/calibrate_pca_latent_diffusion.py"
"$PYTHON" "$V2/evaluate/run_candidate_valid50.py" \
  --models geometry_pca_latent_diffusion_calibrated \
  --hands linker xhand wuji \
  --device cuda \
  --workers 1
