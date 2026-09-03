#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/lekangwan/projects/DexGraspMotionChallenge2025-final-release
PYTHON=/home/lekangwan/miniconda3/envs/hand-retarget/bin/python
RUNNER=$ROOT/retarget_research/retargeting/run/run_phase_cem_manifest.py
SYNERGY=$ROOT/retarget_research/retargeting/run/run_synergy_cem_manifest.py
PREPARE=$ROOT/retarget_research/retargeting/prepare/prepare_physics_cem_independent_eval.py
EVALUATOR=$ROOT/retarget_research/retargeting/evaluate/evaluate_hand_manifest.py
AUDITOR=$ROOT/retarget_research/retargeting/evaluate/audit_stable_success.py
FORMAL=$ROOT/retarget_research/manifests/formal_50c_100o_1000t_seed20260808.json
PROTOCOL=$ROOT/retarget_research/retargeting/configs/stable_success_protocol_v3_selected_methods.json
CAL_BASIS=$ROOT/retarget_research/outputs/reboot_synergy_cem_cal50_v2_rank5/raw
INITIAL_LINKER=$ROOT/retarget_research/outputs/formal_1000/linker_vector_v2alpha_c3g8_v1
INITIAL_XHAND=$ROOT/retarget_research/outputs/formal_1000/xhand_official
INITIAL_WUJI=$ROOT/retarget_research/outputs/formal_1000/wuji_thumb_nullspace_v1
OUT=$ROOT/retarget_research/outputs/reboot_synergy_rank5_formal1000_v1
BASE1=$OUT/base_global1
BASE2=$OUT/base_linker_global2
FINAL=$OUT/final_synergy_rank5

if [[ -f "$FINAL/audit/three_hand_stable_audit_summary.json" ]]; then
  echo "正式1000条已经完整结束，不重复覆盖：$FINAL"
  exit 0
fi
mkdir -p "$OUT/logs"
cd "$ROOT"

prepare_stage() {
  local raw_root=$1
  local stage_root=$2
  local hand=$3
  if [[ -s "$stage_root/manifests/${hand}.json" ]]; then
    echo "复用已整理manifest：$stage_root/manifests/${hand}.json"
    return 0
  fi
  mkdir -p "$stage_root/targets/$hand" "$stage_root/manifests"
  "$PYTHON" "$PREPARE" \
    --screen-summary "$raw_root/$hand/screen_summary.json" \
    --formal-manifest "$FORMAL" --target-dir "$stage_root/targets/$hand" \
    --manifest-output "$stage_root/manifests/${hand}.json"
}

run_global1() {
  local hand=$1
  local target_dir=$2
  mkdir -p "$BASE1/raw/$hand"
  MPLCONFIGDIR=/tmp/matplotlib-retarget OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  "$PYTHON" -u "$RUNNER" --hand "$hand" --manifest "$FORMAL" \
    --target-dir "$target_dir" --output-dir "$BASE1/raw/$hand" \
    --parameterization global --population 8 --elite 2 --iterations 2 \
    --selection-margin 0 --seed 20260827 --device cpu \
    --accumulate-object-trajectories > "$OUT/logs/base1_${hand}.log" 2>&1
}

echo "[1/4] 第一层global CEM：三只手各1000条并行，可按单条续跑。"
run_global1 linker "$INITIAL_LINKER" & p1=$!
run_global1 xhand "$INITIAL_XHAND" & p2=$!
run_global1 wuji "$INITIAL_WUJI" & p3=$!
status=0; wait "$p1" || status=1; wait "$p2" || status=1; wait "$p3" || status=1
[[ "$status" -eq 0 ]] || { tail -n 40 "$OUT"/logs/base1_*.log; exit 1; }
for hand in linker xhand wuji; do prepare_stage "$BASE1/raw" "$BASE1" "$hand"; done

run_final_synergy() {
  local hand=$1
  local base=$2
  mkdir -p "$FINAL/raw/$hand"
  cp "$CAL_BASIS/$hand/synergy_basis.npy" "$FINAL/raw/$hand/synergy_basis.npy"
  MPLCONFIGDIR=/tmp/matplotlib-retarget OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  "$PYTHON" -u "$SYNERGY" --hand "$hand" \
    --manifest "$base/manifests/${hand}.json" --target-dir "$base/targets/$hand" \
    --output-dir "$FINAL/raw/$hand" --rank 5 --population 8 --elite 2 \
    --iterations 2 --selection-margin 1 --seed 20260828 --device cpu \
    --accumulate-object-trajectories > "$OUT/logs/synergy_${hand}.log" 2>&1
}

echo "[2/4] Linker第二层global CEM，同时XHand/Wuji直接进入冻结rank5协同CEM。"
mkdir -p "$BASE2/raw/linker"
MPLCONFIGDIR=/tmp/matplotlib-retarget OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
"$PYTHON" -u "$RUNNER" --hand linker \
  --manifest "$BASE1/manifests/linker.json" --target-dir "$BASE1/targets/linker" \
  --output-dir "$BASE2/raw/linker" --parameterization global \
  --population 8 --elite 2 --iterations 2 --selection-margin 0 \
  --seed 20260828 --device cpu --accumulate-object-trajectories \
  > "$OUT/logs/base2_linker.log" 2>&1 & p1=$!
run_final_synergy xhand "$BASE1" & p2=$!
run_final_synergy wuji "$BASE1" & p3=$!
status=0; wait "$p1" || status=1; wait "$p2" || status=1; wait "$p3" || status=1
[[ "$status" -eq 0 ]] || { tail -n 40 "$OUT"/logs/base2_linker.log "$OUT"/logs/synergy_*.log; exit 1; }
prepare_stage "$BASE2/raw" "$BASE2" linker
prepare_stage "$FINAL/raw" "$FINAL" xhand
prepare_stage "$FINAL/raw" "$FINAL" wuji

echo "[3/4] Linker进入冻结rank5协同CEM。"
run_final_synergy linker "$BASE2"
prepare_stage "$FINAL/raw" "$FINAL" linker

echo "[4/4] 三只手各1000条最终轨迹独立回放并执行稳定运输审计。"
mkdir -p "$FINAL/evaluation" "$FINAL/traces" "$FINAL/audit"
evaluate_hand() {
  local hand=$1
  MPLCONFIGDIR=/tmp/matplotlib-retarget OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  "$PYTHON" -u "$EVALUATOR" --hand "$hand" \
    --manifest "$FINAL/manifests/${hand}.json" --target-dir "$FINAL/targets/$hand" \
    --output-dir "$FINAL/evaluation/$hand" --policy-trace-dir "$FINAL/traces/$hand" \
    --workers 1 --resume --steps-per-frame 3 --hold-steps 30 \
    > "$OUT/logs/evaluate_${hand}.log" 2>&1
}
evaluate_hand linker & p1=$!
evaluate_hand xhand & p2=$!
evaluate_hand wuji & p3=$!
status=0; wait "$p1" || status=1; wait "$p2" || status=1; wait "$p3" || status=1
[[ "$status" -eq 0 ]] || { tail -n 40 "$OUT"/logs/evaluate_*.log; exit 1; }

"$PYTHON" -u "$AUDITOR" --config "$PROTOCOL" \
  --linker-report "$FINAL/evaluation/linker/manifest_evaluation_summary.json" \
  --xhand-report "$FINAL/evaluation/xhand/manifest_evaluation_summary.json" \
  --wuji-report "$FINAL/evaluation/wuji/manifest_evaluation_summary.json" \
  --output-dir "$FINAL/audit" | tee "$OUT/logs/audit.log"
