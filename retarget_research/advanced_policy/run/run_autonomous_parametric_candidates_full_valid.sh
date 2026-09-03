#!/usr/bin/env bash
# 将目前只有valid50的四个纯参数候选补齐到同一套valid100。
set -euo pipefail

ROOT="/home/lekangwan/projects/DexGraspMotionChallenge2025-final-release"
PY="/home/lekangwan/miniconda3/envs/hand-retarget/bin/python"
EVAL="$ROOT/retarget_research/advanced_policy/evaluate_policy_manifest.py"
MANIFEST="$ROOT/retarget_research/manifests/formal_50c_100o_1000t_seed20260808.json"
SPLIT="$ROOT/retarget_research/advanced_policy/data/formal_v1/policy_split_seed20260813.json"
DATA="$ROOT/retarget_research/advanced_policy/data/formal_final_30cm"

cd "$ROOT"

evaluate() {
  local tag="$1" label="$2" hand="$3" checkpoint="$4" target="$5" screen="$6" output="$7"
  mkdir -p "$output"
  cp -a "$screen/." "$output/"
  "$PY" "$EVAL" --hand "$hand" --manifest "$MANIFEST" \
    --policy-split "$SPLIT" --split valid --target-dir "$target" \
    --checkpoint "$checkpoint" --data-dir "$DATA/$label" \
    --output-dir "$output" --device cuda --workers 1 \
    --autonomous-only --resume 2>&1 | sed -u "s/^/[$tag] /"
}

LROOT="$ROOT/retarget_research/advanced_policy/runs/autonomous_state_aligned_dagger_safe_v1/linker_state_aligned_dagger_safe_v1"
XROOT="$ROOT/retarget_research/advanced_policy/runs/autonomous_xhand_huber_tuning_v1/xhand_huber_beta20_v1"
W005="$ROOT/retarget_research/advanced_policy/runs/autonomous_wuji_state_aligned_feedback_sweep_v1/limit_005"
WFOURIER="$ROOT/retarget_research/advanced_policy/runs/autonomous_initial_fourier_delta_v1/wuji_old_initial_fourier_delta_v1"

pids=()
evaluate linker-temporal linker linker "$LROOT/best.pt" \
  "$ROOT/retarget_research/outputs/formal_1000/linker_vector_v2alpha_c3g8_v1" \
  "$LROOT/closed_loop_valid50" "$LROOT/closed_loop_valid" & pids+=("$!")
evaluate xhand-huber-beta2 xhand_official xhand "$XROOT/best.pt" \
  "$ROOT/retarget_research/outputs/formal_1000/xhand_official" \
  "$XROOT/closed_loop_valid50" "$XROOT/closed_loop_valid" & pids+=("$!")
evaluate wuji-feedback-005 wuji_old wuji "$W005/best.pt" \
  "$ROOT/retarget_research/outputs/formal_1000/wuji_v1" \
  "$W005/closed_loop_valid50" "$W005/closed_loop_valid" & pids+=("$!")
evaluate wuji-fourier wuji_old wuji "$WFOURIER/best.pt" \
  "$ROOT/retarget_research/outputs/formal_1000/wuji_v1" \
  "$WFOURIER/closed_loop_valid50" "$WFOURIER/closed_loop_valid" & pids+=("$!")

status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
if [[ "$status" -ne 0 ]]; then
  echo "至少一个valid100候选失败；修复后重跑本命令即可resume。"
  exit "$status"
fi
echo "AUTONOMOUS_PARAMETRIC_CANDIDATES_FULL_VALID=COMPLETE"
