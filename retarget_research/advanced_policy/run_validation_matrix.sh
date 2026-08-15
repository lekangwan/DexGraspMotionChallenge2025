#!/usr/bin/env bash
# 输入：三手九个正式best checkpoint、最终重定向候选、正式manifest和策略split。
# 输出：每个模型各100条valid闭环报告，以及独立的policy_evaluation_summary.json。
# 内部逻辑：同一GPU上按手和模型顺序执行，`--resume`复用已完成且checkpoint一致的轨迹。
# 作用：只在训练物体的留出轨迹上选择模型，避免提前查看对象级未见test结果。

set -euo pipefail

PROJECT_ROOT="/home/lekangwan/projects/DexGraspMotionChallenge2025-final-release"
PYTHON_BIN="/home/lekangwan/miniconda3/envs/hand-retarget/bin/python"
MANIFEST="$PROJECT_ROOT/retarget_research/manifests/formal_50c_100o_1000t_seed20260808.json"
POLICY_SPLIT="$PROJECT_ROOT/retarget_research/advanced_policy/data/formal_v1/policy_split_seed20260813.json"
RUN_ROOT="$PROJECT_ROOT/retarget_research/advanced_policy/runs/formal_v1"
DATA_ROOT="$PROJECT_ROOT/retarget_research/advanced_policy/data/formal_v1"
EVALUATOR="$PROJECT_ROOT/retarget_research/advanced_policy/evaluate_policy_manifest.py"

cd "$PROJECT_ROOT"
export MPLCONFIGDIR=/tmp/matplotlib-retarget

for retarget_hand in linker wuji xhand; do
  case "$retarget_hand" in
    linker)
      target_dir="$PROJECT_ROOT/retarget_research/outputs/formal_1000/linker_object_centric_3mm_v1"
      ;;
    wuji)
      target_dir="$PROJECT_ROOT/retarget_research/outputs/formal_1000/wuji_v1"
      ;;
    xhand)
      target_dir="$PROJECT_ROOT/retarget_research/outputs/formal_1000/xhand_official"
      ;;
  esac

  for policy_model in bc temporal3 diffusion; do
    experiment="${retarget_hand}_${policy_model}_v1"
    experiment_dir="$RUN_ROOT/$experiment"
    echo "=== VALIDATE $experiment ==="
    "$PYTHON_BIN" "$EVALUATOR" \
      --hand "$retarget_hand" \
      --manifest "$MANIFEST" \
      --policy-split "$POLICY_SPLIT" \
      --split valid \
      --target-dir "$target_dir" \
      --checkpoint "$experiment_dir/best.pt" \
      --data-dir "$DATA_ROOT/$retarget_hand" \
      --output-dir "$experiment_dir/closed_loop_valid" \
      --device cuda \
      --workers 1 \
      --resume
  done
done

echo "VALIDATION_MATRIX=COMPLETE"
