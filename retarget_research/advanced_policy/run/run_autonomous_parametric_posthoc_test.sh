#!/usr/bin/env bash
# 冻结方案后筛出的四个纯参数候选，在同一500条test上补充评测。
set -euo pipefail

ROOT="/home/lekangwan/projects/DexGraspMotionChallenge2025-final-release"
PY="/home/lekangwan/miniconda3/envs/hand-retarget/bin/python"
EVAL="$ROOT/retarget_research/advanced_policy/evaluate_policy_manifest.py"
MANIFEST="$ROOT/retarget_research/manifests/formal_50c_100o_1000t_seed20260808.json"
SPLIT="$ROOT/retarget_research/advanced_policy/data/formal_v1/policy_split_seed20260813.json"
DATA="$ROOT/retarget_research/advanced_policy/data/formal_final_30cm"
RUN="$ROOT/retarget_research/advanced_policy/runs/autonomous_parametric_posthoc_test_v1"
SELECTION="$ROOT/retarget_research/advanced_policy/configs/autonomous_parametric_posthoc_candidates_v1.json"

cd "$ROOT"

status="$($PY -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["status"])' "$SELECTION")"
if [[ "$status" != "approved_for_test500" ]]; then
  echo "当前候选尚未完成valid100筛选，禁止启动补充test500（status=$status）。"
  exit 2
fi

evaluate() {
  local tag="$1" label="$2" hand="$3" checkpoint="$4" target="$5"
  "$PY" "$EVAL" --hand "$hand" --manifest "$MANIFEST" \
    --policy-split "$SPLIT" --split test --target-dir "$target" \
    --checkpoint "$checkpoint" --data-dir "$DATA/$label" \
    --output-dir "$RUN/$tag" --device cuda --workers 1 \
    --autonomous-only --resume 2>&1 | sed -u "s/^/[$tag] /"
}

run_selected() {
  local pids=()
  evaluate linker_ft linker linker \
    "$ROOT/retarget_research/advanced_policy/runs/autonomous_initial_phase_delta_soup_v1/linker_initial_phase_delta_ft_v1/best.pt" \
    "$ROOT/retarget_research/outputs/formal_1000/linker_vector_v2alpha_c3g8_v1" & pids+=("$!")
  evaluate xhand_huber_beta2 xhand_official xhand \
    "$ROOT/retarget_research/advanced_policy/runs/autonomous_xhand_huber_tuning_v1/xhand_huber_beta20_v1/best.pt" \
    "$ROOT/retarget_research/outputs/formal_1000/xhand_official" & pids+=("$!")
  evaluate wuji_feedback_005 wuji_old wuji \
    "$ROOT/retarget_research/advanced_policy/runs/autonomous_wuji_state_aligned_feedback_sweep_v1/limit_005/best.pt" \
    "$ROOT/retarget_research/outputs/formal_1000/wuji_v1" & pids+=("$!")
  evaluate wuji_blend_fourier_025 wuji_old wuji \
    "$ROOT/retarget_research/advanced_policy/runs/autonomous_wuji_parametric_tuning_v1/blend_fourier_025/best.pt" \
    "$ROOT/retarget_research/outputs/formal_1000/wuji_v1" & pids+=("$!")
  local status=0
  for pid in "${pids[@]}"; do wait "$pid" || status=1; done
  return "$status"
}

run_selected
"$PY" "$ROOT/retarget_research/advanced_policy/run/summarize_autonomous_parametric_posthoc_test.py"
echo "AUTONOMOUS_PARAMETRIC_POSTHOC_TEST=COMPLETE"
