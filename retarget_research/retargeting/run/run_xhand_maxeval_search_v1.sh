#!/usr/bin/env bash
# 输入：冻结20轨迹XHand官方基线，以及10/40两个新SLSQP评估上限。
# 输出：两个新细化候选、与当前20次统一比较的PhysX汇总。
# 内部逻辑：两个候选并行生成，复用maxeval20/先验0候选，再顺序物理评测。
# 作用：在成功率不下降的条件下减少正式1000条优化成本，或验证更多迭代是否必要。

set -u -o pipefail

project_root="/home/lekangwan/projects/DexGraspMotionChallenge2025-final-release"
retarget_python="/home/lekangwan/miniconda3/envs/hand-retarget/bin/python"
manifest_path="retarget_research/retargeting/configs/linker_independent_validation_v1.json"
baseline_dir="retarget_research/outputs/xhand_independent_validation_v1"
candidate_root="retarget_research/outputs/xhand_maxeval_search_v1"

cd "$project_root" || exit 1

run_candidate() {
  local candidate_name="$1"
  local maxeval="$2"
  echo "[start refinement] $candidate_name"
  MPLCONFIGDIR=/tmp/matplotlib-retarget \
  OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  "$retarget_python" retarget_research/retargeting/run/run_xhand_contact_manifest.py \
    --manifest "$manifest_path" \
    --baseline-dir "$baseline_dir" \
    --output-dir "$candidate_root/$candidate_name" \
    --contact-pad-config retarget_research/retargeting/configs/xhand_contact_pads_v1.json \
    --workers 1 --resume --maxeval "$maxeval" \
    --contact-weight 5 --normal-weight 0.05 \
    --penetration-weight 2 --joint-prior-weight 0 \
    --contact-threshold 0.02 --min-contact-tips 2 --lift-delta 0.03 \
    --region-neighbors 32 --contact-offset -0.003 --min-signed-distance -0.006
}

run_candidate maxeval_10 10 & candidate_pid_0=$!
run_candidate maxeval_40 40 & candidate_pid_1=$!
candidate_failed=0
wait "$candidate_pid_0" || candidate_failed=1
wait "$candidate_pid_1" || candidate_failed=1
if [[ "$candidate_failed" -ne 0 ]]; then
  echo "至少一个XHand迭代数候选失败；修正后重新执行可续跑。" >&2
  exit 1
fi

MPLCONFIGDIR=/tmp/matplotlib-retarget \
OMP_NUM_THREADS=4 \
MKL_NUM_THREADS=4 \
"$retarget_python" retarget_research/retargeting/evaluate/evaluate_candidate_sweep.py \
  --config retarget_research/retargeting/configs/xhand_maxeval_search_v1.json \
  --output retarget_research/outputs/xhand_maxeval_search_v1_evaluation/search_summary.json \
  --workers 1 --resume
