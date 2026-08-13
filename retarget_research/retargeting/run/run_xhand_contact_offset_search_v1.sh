#!/usr/bin/env bash
# 输入：冻结20轨迹XHand官方基线，以及-1/-5 mm两个新指腹内缩量。
# 输出：两个新细化候选、与当前-3 mm统一比较的PhysX汇总。
# 内部逻辑：两个候选并行生成，复用已有-3 mm结果，再顺序物理评测。
# 作用：确定XHand指腹需要轻触还是适度预压，避免在1000条上盲目选择穿透深度。

set -u -o pipefail

project_root="/home/lekangwan/projects/DexGraspMotionChallenge2025-final-release"
retarget_python="/home/lekangwan/miniconda3/envs/hand-retarget/bin/python"
manifest_path="retarget_research/retargeting/configs/linker_independent_validation_v1.json"
baseline_dir="retarget_research/outputs/xhand_independent_validation_v1"
candidate_root="retarget_research/outputs/xhand_contact_offset_search_v1"

cd "$project_root" || exit 1

run_candidate() {
  local candidate_name="$1"
  local contact_offset="$2"
  echo "[start refinement] $candidate_name"
  MPLCONFIGDIR=/tmp/matplotlib-retarget \
  OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  "$retarget_python" retarget_research/retargeting/run/run_xhand_contact_manifest.py \
    --manifest "$manifest_path" \
    --baseline-dir "$baseline_dir" \
    --output-dir "$candidate_root/$candidate_name" \
    --contact-pad-config retarget_research/retargeting/configs/xhand_contact_pads_v1.json \
    --workers 1 --resume --maxeval 20 \
    --contact-weight 5 --normal-weight 0.05 \
    --penetration-weight 1 --joint-prior-weight 2 \
    --contact-threshold 0.02 --min-contact-tips 2 --lift-delta 0.03 \
    --region-neighbors 32 --contact-offset "$contact_offset" \
    --min-signed-distance -0.006
}

run_candidate contact_offset_m1mm -0.001 & candidate_pid_0=$!
run_candidate contact_offset_m5mm -0.005 & candidate_pid_1=$!
candidate_failed=0
wait "$candidate_pid_0" || candidate_failed=1
wait "$candidate_pid_1" || candidate_failed=1
if [[ "$candidate_failed" -ne 0 ]]; then
  echo "至少一个XHand内缩候选失败；修正后重新执行可续跑。" >&2
  exit 1
fi

MPLCONFIGDIR=/tmp/matplotlib-retarget \
OMP_NUM_THREADS=4 \
MKL_NUM_THREADS=4 \
"$retarget_python" retarget_research/retargeting/evaluate/evaluate_candidate_sweep.py \
  --config retarget_research/retargeting/configs/xhand_contact_offset_search_v1.json \
  --output retarget_research/outputs/xhand_contact_offset_search_v1_evaluation/search_summary.json \
  --workers 1 --resume
