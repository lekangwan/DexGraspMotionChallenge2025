#!/usr/bin/env bash
# Wuji最后一轮定向参数优化：反馈0.075，以及反馈与Fourier的两种纯参数融合。
set -euo pipefail

ROOT="/home/lekangwan/projects/DexGraspMotionChallenge2025-final-release"
PY="/home/lekangwan/miniconda3/envs/hand-retarget/bin/python"
SCALE="$ROOT/retarget_research/advanced_policy/prepare/scale_temporal_feedback.py"
BLEND="$ROOT/retarget_research/advanced_policy/prepare/build_parametric_blend.py"
EVAL="$ROOT/retarget_research/advanced_policy/evaluate_policy_manifest.py"
MANIFEST="$ROOT/retarget_research/manifests/formal_50c_100o_1000t_seed20260808.json"
SPLIT="$ROOT/retarget_research/advanced_policy/data/formal_v1/policy_split_seed20260813.json"
DATA="$ROOT/retarget_research/advanced_policy/data/formal_final_30cm/wuji_old"
TARGET="$ROOT/retarget_research/outputs/formal_1000/wuji_v1"
RUN="$ROOT/retarget_research/advanced_policy/runs/autonomous_wuji_parametric_tuning_v1"
FEEDBACK="$ROOT/retarget_research/advanced_policy/runs/autonomous_state_aligned_dagger_safe_v1/wuji_old_state_aligned_dagger_safe_v1/best.pt"
FOURIER="$ROOT/retarget_research/advanced_policy/runs/autonomous_initial_fourier_delta_v1/wuji_old_initial_fourier_delta_v1/best.pt"

cd "$ROOT"
"$PY" "$SCALE" --input "$FEEDBACK" --output "$RUN/feedback_0075/best.pt" --feedback-limit 0.075
"$PY" "$BLEND" --first "$FEEDBACK" --second "$FOURIER" --alpha 0.25 --output "$RUN/blend_fourier_025/best.pt"
"$PY" "$BLEND" --first "$FEEDBACK" --second "$FOURIER" --alpha 0.50 --output "$RUN/blend_fourier_050/best.pt"

evaluate() {
  local tag="$1"
  "$PY" "$EVAL" --hand wuji --manifest "$MANIFEST" --policy-split "$SPLIT" \
    --split valid --target-dir "$TARGET" --checkpoint "$RUN/$tag/best.pt" \
    --data-dir "$DATA" --output-dir "$RUN/$tag/closed_loop_valid" \
    --device cuda --workers 1 --autonomous-only --resume 2>&1 | sed -u "s/^/[$tag] /"
}

pids=()
for tag in feedback_0075 blend_fourier_025 blend_fourier_050; do
  evaluate "$tag" & pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
exit "$status"
