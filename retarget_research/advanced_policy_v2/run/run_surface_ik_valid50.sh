#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="/home/lekangwan/miniconda3/envs/hand-retarget/bin/python"

cd "$ROOT"
CUDA_VISIBLE_DEVICES=0 "$PYTHON" -u \
  retarget_research/advanced_policy_v2/evaluate/run_candidate_valid50.py \
  --models geometry_pca_surface_ik \
  --hands linker xhand wuji \
  --device cuda \
  --workers 1 \
  --parallel-hands

echo "三只手的 surface-IK valid50 均已完成。"
