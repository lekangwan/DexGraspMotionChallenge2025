#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/lekangwan/projects/DexGraspMotionChallenge2025-final-release"
PYTHON="/home/lekangwan/miniconda3/envs/hand-retarget/bin/python"
FORMAL="$ROOT/retarget_research/outputs/reboot_synergy_rank5_formal1000_v1/postconfirmed_rank5_v1"
DATA="$ROOT/retarget_research/advanced_policy_v2/data/final"
RUNS="$ROOT/retarget_research/advanced_policy_v2/runs/candidates_v1"
TRAIN="$ROOT/retarget_research/advanced_policy_v2/train_pca_residual_ppo.py"

run_hand() {
  local hand="$1"
  local base="$2"
  local output="$RUNS/$hand/pca_contact_residual_ppo_smoke"
  if [[ -f "$output/training_log.json" ]]; then
    echo "[$hand] 已有完整training_log.json，跳过，避免重复运行覆盖。"
    return
  fi
  echo "[$hand] 自主PCA + 接触运输残差PPO可行性训练"
  "$PYTHON" -u "$TRAIN" \
    --hand "$hand" \
    --manifest "$FORMAL/manifests/$hand.json" \
    --target-dir "$FORMAL/targets/$hand" \
    --data-dir "$DATA/$hand" \
    --base-checkpoint "$RUNS/$hand/$base/best.pt" \
    --output-dir "$output" \
    --iterations 12 \
    --num-envs 20 \
    --residual-scale 0.12 \
    --device cuda
}

# 三只手顺序运行，避免同时建立三个PhysX场景造成显存或主存竞争。
run_hand linker geometry_pca32
run_hand xhand geometry_pca16
run_hand wuji geometry_pca16
