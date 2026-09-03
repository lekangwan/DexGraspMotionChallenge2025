#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/lekangwan/projects/DexGraspMotionChallenge2025-final-release"
PYTHON="/home/lekangwan/miniconda3/envs/hand-retarget/bin/python"
V2="$ROOT/retarget_research/advanced_policy_v2"
FORMAL="$ROOT/retarget_research/outputs/reboot_synergy_rank5_formal1000_v1/postconfirmed_rank5_v1"

for hand in linker xhand wuji; do
  output="$V2/data/quality_curriculum/${hand}_train.npz"
  if [[ ! -f "$output" ]]; then
    "$PYTHON" "$V2/prepare/prepare_quality_curriculum.py" \
      --hand "$hand" \
      --manifest "$FORMAL/manifests/$hand.json" \
      --audit "$FORMAL/audit/${hand}_stable_audit.json" \
      --unfiltered-data-dir "$V2/data/final/${hand}_unfiltered" \
      --filtered-data-dir "$V2/data/final/$hand" \
      --output "$output"
  fi
done

"$PYTHON" "$V2/prepare/build_quality_curriculum_configs.py"
for hand in linker xhand wuji; do
  summary="$V2/runs/candidates_v1/$hand/geometry_pca_quality_curriculum/training_summary.json"
  if [[ -f "$summary" ]]; then
    echo "[$hand] 已有完整训练摘要，跳过以免覆盖。"
    continue
  fi
  echo "[$hand] 质量课程PCA训练"
  "$PYTHON" -u "$V2/train_quality_curriculum_pca.py" \
    --config "$V2/configs/generated/${hand}_geometry_pca_quality_curriculum.json" \
    --device cuda 2>&1 | sed -u "s/^/[$hand] /" &
done
wait

echo "质量课程训练完成；先审查离线valid，不自动启动PhysX。"
