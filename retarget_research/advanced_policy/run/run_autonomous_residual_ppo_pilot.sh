#!/usr/bin/env bash
# 三手真正自主Residual PPO短程筛选：每类一个训练环境，共50类。
set -euo pipefail

ROOT="/home/lekangwan/projects/DexGraspMotionChallenge2025-final-release"
PY="/home/lekangwan/miniconda3/envs/hand-retarget/bin/python"
TRAIN="$ROOT/retarget_research/advanced_policy/train_autonomous_residual_ppo.py"
EVAL="$ROOT/retarget_research/advanced_policy/evaluate_policy_manifest.py"
MANIFEST="$ROOT/retarget_research/manifests/formal_50c_100o_1000t_seed20260808.json"
SPLIT="$ROOT/retarget_research/advanced_policy/data/formal_v1/policy_split_seed20260813.json"
DATA="$ROOT/retarget_research/advanced_policy/data/formal_final_30cm"
RUN="$ROOT/retarget_research/advanced_policy/runs/autonomous_residual_ppo_pilot_v1"

cd "$ROOT"

pipeline() {
  local label="$1" hand="$2" target="$3" base="$4"
  local train_dir="$RUN/$label/train"
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 "$PY" "$TRAIN" \
    --hand "$hand" --manifest "$MANIFEST" --policy-split "$SPLIT" \
    --target-dir "$target" --base-checkpoint "$base" \
    --data-dir "$DATA/$label" --checkpoint-dir "$train_dir" \
    --iterations 20 --num-envs 50 --device cuda --residual-scale 0.10
  "$PY" "$EVAL" --hand "$hand" --manifest "$MANIFEST" \
    --policy-split "$SPLIT" --split valid --max-tasks-per-category 1 \
    --target-dir "$target" --checkpoint "$base" --data-dir "$DATA/$label" \
    --autonomous-residual-rl-checkpoint "$train_dir/autonomous_residual_ppo_0020.pt" \
    --output-dir "$RUN/$label/closed_loop_valid50" \
    --device cuda --workers 2 --autonomous-only --resume
}

pids=()
pipeline linker linker \
  "$ROOT/retarget_research/outputs/formal_1000/linker_vector_v2alpha_c3g8_v1" \
  "$ROOT/retarget_research/advanced_policy/runs/autonomous_initial_phase_delta_v1/linker_initial_phase_delta_v1/best.pt" \
  2>&1 | sed -u 's/^/[linker] /' & pids+=("$!")
pipeline xhand_official xhand \
  "$ROOT/retarget_research/outputs/formal_1000/xhand_official" \
  "$ROOT/retarget_research/advanced_policy/runs/autonomous_initial_phase_delta_v1/xhand_official_initial_phase_delta_v1/best.pt" \
  2>&1 | sed -u 's/^/[xhand] /' & pids+=("$!")
pipeline wuji_old wuji \
  "$ROOT/retarget_research/outputs/formal_1000/wuji_v1" \
  "$ROOT/retarget_research/advanced_policy/runs/autonomous_state_aligned_dagger_safe_v1/wuji_old_state_aligned_dagger_safe_v1/best.pt" \
  2>&1 | sed -u 's/^/[wuji] /' & pids+=("$!")

for pid in "${pids[@]}"; do wait "$pid"; done
echo "AUTONOMOUS_RESIDUAL_PPO_PILOT=COMPLETE"
