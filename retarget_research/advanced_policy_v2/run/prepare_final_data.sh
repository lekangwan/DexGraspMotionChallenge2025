#!/usr/bin/env bash
set -euo pipefail

# 在Rank-5正式1000条的重放、trace和v3审计全部完成后运行。
ROOT=/home/lekangwan/projects/DexGraspMotionChallenge2025-final-release
PY=/home/lekangwan/miniconda3/envs/hand-retarget/bin/python
FINAL=$ROOT/retarget_research/outputs/reboot_synergy_rank5_formal1000_v1/postconfirmed_rank5_v1
SPLIT=$ROOT/retarget_research/advanced_policy/data/formal_v1/policy_split_seed20260813.json
SPECS=$ROOT/retarget_research/advanced_policy/configs/hand_data_specs_v5.json
OUT=$ROOT/retarget_research/advanced_policy_v2/data/final

cd "$ROOT"
for hand in linker xhand wuji; do
  base="$OUT/${hand}_unfiltered"
  final="$OUT/$hand"
  "$PY" -u retarget_research/advanced_policy/prepare/prepare_policy_dataset.py \
    --manifest "$FINAL/manifests/${hand}.json" \
    --policy-split "$SPLIT" --hand "$hand" \
    --trace-dir "$FINAL/traces/$hand" \
    --evaluation-summary "$FINAL/evaluation/$hand/manifest_evaluation_summary.json" \
    --output-dir "$base" --hand-specs "$SPECS" --lift-goal 0.15 \
    --include-all-train-valid
  "$PY" -u retarget_research/advanced_policy_v2/prepare/prepare_geometry_data.py \
    --hand "$hand" --manifest "$FINAL/manifests/${hand}.json" \
    --audit "$FINAL/audit/${hand}_stable_audit.json" \
    --base-data-dir "$base" --output-dir "$final" --point-count 128
done

"$PY" retarget_research/advanced_policy_v2/prepare/generate_candidate_configs.py \
  --matrix retarget_research/advanced_policy_v2/configs/candidate_matrix_v1.json \
  --output-dir retarget_research/advanced_policy_v2/configs/generated

"$PY" retarget_research/advanced_policy_v2/prepare/verify_final_data.py \
  --data-root "$OUT" \
  --release-lock "$FINAL/FINAL_RETARGETING_LOCK.json" \
  --output "$OUT/FINAL_DATA_AUDIT.json"

echo "ADVANCED_POLICY_V2_DATA=READY"
