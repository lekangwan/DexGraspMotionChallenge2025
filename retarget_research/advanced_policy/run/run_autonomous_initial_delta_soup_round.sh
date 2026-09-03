#!/usr/bin/env bash
# 同盆地第二seed微调、0.5参数Soup，并对两种候选各做100条valid。
set -euo pipefail

ROOT="/home/lekangwan/projects/DexGraspMotionChallenge2025-final-release"
PY="/home/lekangwan/miniconda3/envs/hand-retarget/bin/python"
BASE="$ROOT/retarget_research/advanced_policy/runs/autonomous_initial_phase_delta_v1"
RUN="$ROOT/retarget_research/advanced_policy/runs/autonomous_initial_phase_delta_soup_v1"
INDEX="$ROOT/retarget_research/advanced_policy/configs/generated/autonomous_initial_phase_delta_soup_v1/config_index.json"
EVAL="$ROOT/retarget_research/advanced_policy/evaluate_policy_manifest.py"
MANIFEST="$ROOT/retarget_research/manifests/formal_50c_100o_1000t_seed20260808.json"
SPLIT="$ROOT/retarget_research/advanced_policy/data/formal_v1/policy_split_seed20260813.json"
DATA="$ROOT/retarget_research/advanced_policy/data/formal_final_30cm"

cd "$ROOT"
"$PY" retarget_research/advanced_policy/prepare/prepare_initial_delta_soup_round.py --project-root "$ROOT"
pids=()
for label in linker xhand_official wuji_old; do
  "$PY" retarget_research/advanced_policy/run_training_matrix.py --index "$INDEX" \
    --filter "$label" --device cuda 2>&1 | sed -u "s/^/[train-${label}] /" & pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done

for label in linker xhand_official wuji_old; do
  mkdir -p "$RUN/${label}_initial_phase_delta_soup_v1"
  if [[ ! -f "$RUN/${label}_initial_phase_delta_soup_v1/best.pt" ]]; then
    "$PY" retarget_research/advanced_policy/prepare/make_model_soup.py \
      --ingredient "$BASE/${label}_initial_phase_delta_v1/best.pt" \
      --ingredient "$RUN/${label}_initial_phase_delta_ft_v1/best.pt" \
      --weight 0.5 --weight 0.5 \
      --output "$RUN/${label}_initial_phase_delta_soup_v1/best.pt"
  fi
done

run_eval() {
  local label="$1" hand="$2" target="$3" suffix="$4"
  "$PY" "$EVAL" --hand "$hand" --manifest "$MANIFEST" --policy-split "$SPLIT" \
    --split valid --target-dir "$target" --checkpoint "$RUN/${label}_${suffix}/best.pt" \
    --data-dir "$DATA/$label" --output-dir "$RUN/${label}_${suffix}/closed_loop_valid" \
    --device cuda --workers 1 --autonomous-only --resume
}
pids=()
for suffix in initial_phase_delta_ft_v1 initial_phase_delta_soup_v1; do
  run_eval linker linker "$ROOT/retarget_research/outputs/formal_1000/linker_vector_v2alpha_c3g8_v1" "$suffix" 2>&1 | sed -u "s/^/[linker-${suffix}] /" & pids+=("$!")
  run_eval xhand_official xhand "$ROOT/retarget_research/outputs/formal_1000/xhand_official" "$suffix" 2>&1 | sed -u "s/^/[xhand-${suffix}] /" & pids+=("$!")
  run_eval wuji_old wuji "$ROOT/retarget_research/outputs/formal_1000/wuji_v1" "$suffix" 2>&1 | sed -u "s/^/[wuji-${suffix}] /" & pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done

for suffix in initial_phase_delta_ft_v1 initial_phase_delta_soup_v1; do
  "$PY" retarget_research/advanced_policy/summarize_autonomous_initial_phase.py \
    --run-root "$RUN" --split valid --experiment-suffix "$suffix"
done
echo "AUTONOMOUS_INITIAL_DELTA_SOUP_ROUND=COMPLETE"
