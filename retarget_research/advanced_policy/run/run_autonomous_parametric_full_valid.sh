#!/usr/bin/env bash
# 纯参数自主策略完整valid100：Linker已有结果，只补齐XHand和Wuji。
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
    --output-dir "$output" --device cuda --workers 2 \
    --autonomous-only --resume 2>&1 | sed -u "s/^/[$tag] /"
}

XROOT="$ROOT/retarget_research/advanced_policy/runs/autonomous_initial_phase_huber_v1/xhand_official_initial_phase_huber_v1"
WROOT="$ROOT/retarget_research/advanced_policy/runs/autonomous_state_aligned_dagger_safe_v1/wuji_old_state_aligned_dagger_safe_v1"

pids=()
evaluate xhand-huber xhand_official xhand "$XROOT/best.pt" \
  "$ROOT/retarget_research/outputs/formal_1000/xhand_official" \
  "$XROOT/closed_loop_valid50" "$XROOT/closed_loop_valid" & pids+=("$!")
evaluate wuji-feedback wuji_old wuji "$WROOT/best.pt" \
  "$ROOT/retarget_research/outputs/formal_1000/wuji_v1" \
  "$WROOT/closed_loop_valid50" "$WROOT/closed_loop_valid" & pids+=("$!")

for pid in "${pids[@]}"; do wait "$pid"; done

"$PY" - <<'PY'
import json
from pathlib import Path
paths = {
    "linker_mse": Path("retarget_research/advanced_policy/runs/autonomous_initial_phase_delta_v1/linker_initial_phase_delta_v1/closed_loop_valid/policy_evaluation_summary.json"),
    "xhand_huber": Path("retarget_research/advanced_policy/runs/autonomous_initial_phase_huber_v1/xhand_official_initial_phase_huber_v1/closed_loop_valid/policy_evaluation_summary.json"),
    "wuji_feedback": Path("retarget_research/advanced_policy/runs/autonomous_state_aligned_dagger_safe_v1/wuji_old_state_aligned_dagger_safe_v1/closed_loop_valid/policy_evaluation_summary.json"),
}
for name, path in paths.items():
    data = json.loads(path.read_text(encoding="utf-8"))
    print(f"{name}: {data['success_count']}/{data['trajectory_count']}")
PY
echo "AUTONOMOUS_PARAMETRIC_FULL_VALID=COMPLETE"
