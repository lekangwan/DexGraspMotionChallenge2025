#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/lekangwan/projects/DexGraspMotionChallenge2025-final-release
PYTHON=/home/lekangwan/miniconda3/envs/hand-retarget/bin/python
GLOBAL=$ROOT/retarget_research/retargeting/run/run_physics_cem_screen.py
PHASE=$ROOT/retarget_research/retargeting/run/run_phase_cem_manifest.py
SYNERGY=$ROOT/retarget_research/retargeting/run/run_synergy_cem_manifest.py
CONFIRM=$ROOT/retarget_research/retargeting/run/confirm_physics_cem_manifest.py
PREPARE=$ROOT/retarget_research/retargeting/prepare/prepare_physics_cem_independent_eval.py
EVALUATOR=$ROOT/retarget_research/retargeting/evaluate/evaluate_hand_manifest.py
AUDITOR=$ROOT/retarget_research/retargeting/evaluate/audit_stable_success.py
FORMAL=$ROOT/retarget_research/manifests/formal_50c_100o_1000t_seed20260808.json
SOURCE_AUDIT=$ROOT/retarget_research/outputs/formal_1000/selected_methods_audit_v3
PROTOCOL=$ROOT/retarget_research/retargeting/configs/stable_success_protocol_v3.json
CAL_SYNERGY=$ROOT/retarget_research/outputs/reboot_synergy_cem_cal50_v2_rank5/raw
OUT=$ROOT/retarget_research/outputs/reboot_finalists_holdout50_v1
BASE1=$OUT/base_global1
BASE2=$OUT/base_linker_global2
PHASE_OUT=$OUT/phase_cem
SYNERGY_OUT=$OUT/synergy_rank5
CRADLE_RAW=$OUT/cradle_raw
CRADLE_OUT=$OUT/cradle_confirmed

if [[ -f "$OUT/final_summary.json" ]]; then
  echo "最终候选holdout50已经完整结束，不重复覆盖：$OUT"
  exit 0
fi
mkdir -p "$OUT/logs"
cd "$ROOT"

prepare_targets() {
  local raw_root=$1
  local output_root=$2
  local hand=$3
  mkdir -p "$output_root/targets/$hand" "$output_root/manifests"
  "$PYTHON" "$PREPARE" \
    --screen-summary "$raw_root/$hand/screen_summary.json" \
    --formal-manifest "$FORMAL" --target-dir "$output_root/targets/$hand" \
    --manifest-output "$output_root/manifests/${hand}.json"
}

echo "[1/6] 从heldout冻结50类50条，并生成第一层统一全局CEM。"
mkdir -p "$BASE1/raw"
for hand in linker xhand wuji; do
  MPLCONFIGDIR=/tmp/matplotlib-retarget OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  "$PYTHON" -u "$GLOBAL" --hand "$hand" \
    --audit "$SOURCE_AUDIT/${hand}_stable_audit.json" --manifest "$FORMAL" \
    --output-dir "$BASE1/raw/$hand" --split heldout --limit 50 \
    --population 8 --elite 2 --iterations 2 --seed 20260830 --device cpu \
    > "$OUT/logs/base1_${hand}.log" 2>&1 &
  eval "base1_${hand}_pid=$!"
done
status=0
wait "$base1_linker_pid" || status=1
wait "$base1_xhand_pid" || status=1
wait "$base1_wuji_pid" || status=1
[[ "$status" -eq 0 ]] || { tail -n 30 "$OUT"/logs/base1_*.log; exit 1; }
for hand in linker xhand wuji; do prepare_targets "$BASE1/raw" "$BASE1" "$hand"; done

echo "[2/6] 复现Linker冻结的第二层全局CEM起点。"
mkdir -p "$BASE2/raw/linker"
MPLCONFIGDIR=/tmp/matplotlib-retarget OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
"$PYTHON" -u "$PHASE" --hand linker \
  --manifest "$BASE1/manifests/linker.json" --target-dir "$BASE1/targets/linker" \
  --output-dir "$BASE2/raw/linker" --parameterization global \
  --population 8 --elite 2 --iterations 2 --selection-margin 0 \
  --seed 20260831 --device cpu > "$OUT/logs/base2_linker.log" 2>&1
prepare_targets "$BASE2/raw" "$BASE2" linker

run_structured() {
  local method=$1
  local hand=$2
  local base=$BASE1
  [[ "$hand" == "linker" ]] && base=$BASE2
  local runner=$PHASE
  local extra=(--parameterization phase)
  if [[ "$method" == "synergy" ]]; then
    runner=$SYNERGY
    extra=(--rank 5)
  fi
  local destination=$PHASE_OUT
  [[ "$method" == "synergy" ]] && destination=$SYNERGY_OUT
  mkdir -p "$destination/raw/$hand"
  if [[ "$method" == "synergy" ]]; then
    cp "$CAL_SYNERGY/$hand/synergy_basis.npy" \
      "$destination/raw/$hand/synergy_basis.npy"
  fi
  MPLCONFIGDIR=/tmp/matplotlib-retarget OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  "$PYTHON" -u "$runner" --hand "$hand" \
    --manifest "$base/manifests/${hand}.json" --target-dir "$base/targets/$hand" \
    --output-dir "$destination/raw/$hand" "${extra[@]}" \
    --population 8 --elite 2 --iterations 2 --selection-margin 1 \
    --seed 20260832 --device cpu > "$OUT/logs/${method}_${hand}.log" 2>&1
}

echo "[3/6] 生成分阶段CEM候选。"
run_structured phase linker & p1=$!
run_structured phase xhand & p2=$!
run_structured phase wuji & p3=$!
status=0; wait "$p1" || status=1; wait "$p2" || status=1; wait "$p3" || status=1
[[ "$status" -eq 0 ]] || { tail -n 30 "$OUT"/logs/phase_*.log; exit 1; }
for hand in linker xhand wuji; do prepare_targets "$PHASE_OUT/raw" "$PHASE_OUT" "$hand"; done

echo "[4/6] 使用cal50冻结的rank5基底生成协同CEM候选。"
run_structured synergy linker & p1=$!
run_structured synergy xhand & p2=$!
run_structured synergy wuji & p3=$!
status=0; wait "$p1" || status=1; wait "$p2" || status=1; wait "$p3" || status=1
[[ "$status" -eq 0 ]] || { tail -n 30 "$OUT"/logs/synergy_*.log; exit 1; }
for hand in linker xhand wuji; do prepare_targets "$SYNERGY_OUT/raw" "$SYNERGY_OUT" "$hand"; done

echo "[5/6] 在分阶段CEM上生成并单环境确认托举候选。"
mkdir -p "$CRADLE_RAW/raw" "$CRADLE_OUT/raw"
for hand in linker xhand wuji; do
  MPLCONFIGDIR=/tmp/matplotlib-retarget OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  "$PYTHON" -u "$PHASE" --hand "$hand" \
    --manifest "$PHASE_OUT/manifests/${hand}.json" \
    --target-dir "$PHASE_OUT/targets/$hand" --output-dir "$CRADLE_RAW/raw/$hand" \
    --parameterization cradle --population 8 --elite 2 --iterations 2 \
    --selection-margin 1 --seed 20260833 --device cpu \
    > "$OUT/logs/cradle_${hand}.log" 2>&1 &
  eval "cradle_${hand}_pid=$!"
done
status=0
wait "$cradle_linker_pid" || status=1
wait "$cradle_xhand_pid" || status=1
wait "$cradle_wuji_pid" || status=1
[[ "$status" -eq 0 ]] || { tail -n 30 "$OUT"/logs/cradle_*.log; exit 1; }
for hand in linker xhand wuji; do
  MPLCONFIGDIR=/tmp/matplotlib-retarget OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  "$PYTHON" -u "$CONFIRM" --hand "$hand" \
    --manifest "$PHASE_OUT/manifests/${hand}.json" \
    --baseline-target-dir "$PHASE_OUT/targets/$hand" \
    --candidate-summary "$CRADLE_RAW/raw/$hand/screen_summary.json" \
    --output-dir "$CRADLE_OUT/raw/$hand" --selection-margin 1 --device cpu \
    > "$OUT/logs/confirm_${hand}.log" 2>&1 &
  eval "confirm_${hand}_pid=$!"
done
status=0
wait "$confirm_linker_pid" || status=1
wait "$confirm_xhand_pid" || status=1
wait "$confirm_wuji_pid" || status=1
[[ "$status" -eq 0 ]] || { tail -n 30 "$OUT"/logs/confirm_*.log; exit 1; }
for hand in linker xhand wuji; do prepare_targets "$CRADLE_OUT/raw" "$CRADLE_OUT" "$hand"; done

evaluate_method() {
  local name=$1
  local method_root=$2
  mkdir -p "$method_root/evaluation" "$method_root/traces" "$method_root/audit"
  for hand in linker xhand wuji; do
    MPLCONFIGDIR=/tmp/matplotlib-retarget OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    "$PYTHON" -u "$EVALUATOR" --hand "$hand" \
      --manifest "$method_root/manifests/${hand}.json" \
      --target-dir "$method_root/targets/$hand" \
      --output-dir "$method_root/evaluation/$hand" \
      --policy-trace-dir "$method_root/traces/$hand" --workers 1 --resume \
      --steps-per-frame 3 --hold-steps 30 \
      > "$OUT/logs/eval_${name}_${hand}.log" 2>&1 &
    eval "eval_${hand}_pid=$!"
  done
  local status=0
  wait "$eval_linker_pid" || status=1
  wait "$eval_xhand_pid" || status=1
  wait "$eval_wuji_pid" || status=1
  [[ "$status" -eq 0 ]] || return "$status"
  "$PYTHON" -u "$AUDITOR" --config "$PROTOCOL" \
    --linker-report "$method_root/evaluation/linker/manifest_evaluation_summary.json" \
    --xhand-report "$method_root/evaluation/xhand/manifest_evaluation_summary.json" \
    --wuji-report "$method_root/evaluation/wuji/manifest_evaluation_summary.json" \
    --output-dir "$method_root/audit" > "$OUT/logs/audit_${name}.log" 2>&1
}

echo "[6/6] 三种冻结候选依次独立回放，每种内部三手并行。"
evaluate_method phase "$PHASE_OUT"
evaluate_method synergy "$SYNERGY_OUT"
evaluate_method cradle "$CRADLE_OUT"

"$PYTHON" - <<PY
import json
from pathlib import Path
root = Path("$OUT")
methods = {
    "phase_cem": root / "phase_cem/audit/three_hand_stable_audit_summary.json",
    "synergy_rank5": root / "synergy_rank5/audit/three_hand_stable_audit_summary.json",
    "cradle_confirmed": root / "cradle_confirmed/audit/three_hand_stable_audit_summary.json",
}
summary = {}
for name, path in methods.items():
    data = json.loads(path.read_text())
    counts = {hand: data[hand]["transport_quality_success_count"]
              for hand in ("linker", "xhand", "wuji")}
    summary[name] = {"transport": counts, "total": sum(counts.values())}
    print(name, counts, "total", summary[name]["total"])
(root / "final_summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
PY
