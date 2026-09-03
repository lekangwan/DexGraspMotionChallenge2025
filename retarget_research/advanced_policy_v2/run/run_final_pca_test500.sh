#!/usr/bin/env bash
# 进阶任务最终收尾：在对象隔离test500上评测valid50选出的自主PCA策略。
set -euo pipefail

ROOT="/home/lekangwan/projects/DexGraspMotionChallenge2025-final-release"
PY="/home/lekangwan/miniconda3/envs/hand-retarget/bin/python"
V2="$ROOT/retarget_research/advanced_policy_v2"
FINAL="$ROOT/retarget_research/outputs/reboot_synergy_rank5_formal1000_v1/postconfirmed_rank5_v1"
SPLIT="$ROOT/retarget_research/advanced_policy/data/formal_v1/policy_split_seed20260813.json"
EVAL="$ROOT/retarget_research/advanced_policy/evaluate_policy_manifest.py"
RUN="$V2/runs/final_pca_test500_v1"

cd "$ROOT"

evaluate() {
  local hand="$1" model="$2"
  "$PY" -u "$EVAL" \
    --hand "$hand" \
    --manifest "$FINAL/manifests/$hand.json" \
    --policy-split "$SPLIT" \
    --target-dir "$FINAL/targets/$hand" \
    --checkpoint "$V2/runs/candidates_v1/$hand/$model/best.pt" \
    --data-dir "$V2/data/final/$hand" \
    --output-dir "$RUN/$hand" \
    --split test \
    --device cuda \
    --workers 2 \
    --lift-threshold 0.15 \
    --autonomous-only \
    --resume 2>&1 | sed -u "s/^/[$hand] /"
}

pids=()
evaluate linker geometry_pca32 & pids+=("$!")
evaluate xhand geometry_pca16 & pids+=("$!")
evaluate wuji geometry_pca16 & pids+=("$!")

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
if [[ "$status" -ne 0 ]]; then
  echo "FINAL_PCA_TEST500=FAILED" >&2
  exit "$status"
fi

"$PY" - "$RUN" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
print("\n自主PCA最终test500：")
for hand in ("linker", "xhand", "wuji"):
    result = json.loads(
        (root / hand / "policy_evaluation_summary.json").read_text(encoding="utf-8")
    )
    print(
        f"{hand:6s}: {result['success_count']}/{result['trajectory_count']} "
        f"({100.0 * result['trajectory_micro_success_rate']:.2f}%)"
    )
PY

echo "FINAL_PCA_TEST500=COMPLETE"
