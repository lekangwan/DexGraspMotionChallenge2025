#!/usr/bin/env bash
# 补测自主Residual PPO第1/10轮，和已完成的第20轮一起选择checkpoint。
set -euo pipefail

ROOT="/home/lekangwan/projects/DexGraspMotionChallenge2025-final-release"
PY="/home/lekangwan/miniconda3/envs/hand-retarget/bin/python"
EVAL="$ROOT/retarget_research/advanced_policy/evaluate_policy_manifest.py"
MANIFEST="$ROOT/retarget_research/manifests/formal_50c_100o_1000t_seed20260808.json"
SPLIT="$ROOT/retarget_research/advanced_policy/data/formal_v1/policy_split_seed20260813.json"
DATA="$ROOT/retarget_research/advanced_policy/data/formal_final_30cm"
RUN="$ROOT/retarget_research/advanced_policy/runs/autonomous_residual_ppo_pilot_v1"

cd "$ROOT"

evaluate() {
  local label="$1" hand="$2" target="$3" base="$4" iteration="$5"
  "$PY" "$EVAL" --hand "$hand" --manifest "$MANIFEST" \
    --policy-split "$SPLIT" --split valid --max-tasks-per-category 1 \
    --target-dir "$target" --checkpoint "$base" --data-dir "$DATA/$label" \
    --autonomous-residual-rl-checkpoint \
      "$RUN/$label/train/autonomous_residual_ppo_${iteration}.pt" \
    --output-dir "$RUN/$label/closed_loop_valid50_iter${iteration}" \
    --device cuda --workers 2 --autonomous-only --resume
}

pids=()
for iteration in 0001 0010; do
  evaluate linker linker \
    "$ROOT/retarget_research/outputs/formal_1000/linker_vector_v2alpha_c3g8_v1" \
    "$ROOT/retarget_research/advanced_policy/runs/autonomous_initial_phase_delta_v1/linker_initial_phase_delta_v1/best.pt" \
    "$iteration" 2>&1 | sed -u "s/^/[linker_${iteration}] /" & pids+=("$!")
  evaluate xhand_official xhand \
    "$ROOT/retarget_research/outputs/formal_1000/xhand_official" \
    "$ROOT/retarget_research/advanced_policy/runs/autonomous_initial_phase_delta_v1/xhand_official_initial_phase_delta_v1/best.pt" \
    "$iteration" 2>&1 | sed -u "s/^/[xhand_${iteration}] /" & pids+=("$!")
  evaluate wuji_old wuji \
    "$ROOT/retarget_research/outputs/formal_1000/wuji_v1" \
    "$ROOT/retarget_research/advanced_policy/runs/autonomous_state_aligned_dagger_safe_v1/wuji_old_state_aligned_dagger_safe_v1/best.pt" \
    "$iteration" 2>&1 | sed -u "s/^/[wuji_${iteration}] /" & pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done
echo "AUTONOMOUS_RESIDUAL_PPO_CHECKPOINT_EVAL=COMPLETE"
