#!/usr/bin/env bash
# 输入：冻结20轨迹XHand官方基线和两组新接触权重。
# 输出：两组新细化候选、三组统一PhysX报告及搜索汇总。
# 内部逻辑：两个轻量细化任务并行生成，复用已有权重5结果，再顺序物理评测。
# 作用：在正式1000条前确定XHand接触项是否过强。

set -u -o pipefail

project_root="/home/lekangwan/projects/DexGraspMotionChallenge2025-final-release"
retarget_python="/home/lekangwan/miniconda3/envs/hand-retarget/bin/python"
manifest_path="retarget_research/retargeting/configs/linker_independent_validation_v1.json"
baseline_dir="retarget_research/outputs/xhand_independent_validation_v1"
candidate_root="retarget_research/outputs/xhand_contact_weight_search_v1"

cd "$project_root" || exit 1

run_candidate() {
  local candidate_name="$1"
  local contact_weight="$2"
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
    --contact-weight "$contact_weight" --normal-weight 0.05 \
    --penetration-weight 1 --joint-prior-weight 2 \
    --contact-threshold 0.02 --min-contact-tips 2 --lift-delta 0.03 \
    --region-neighbors 32 --contact-offset -0.003 --min-signed-distance -0.006
}

run_candidate contact_weight_1p25 1.25 & candidate_pid_0=$!
run_candidate contact_weight_2p5 2.5 & candidate_pid_1=$!
candidate_failed=0
wait "$candidate_pid_0" || candidate_failed=1
wait "$candidate_pid_1" || candidate_failed=1
if [[ "$candidate_failed" -ne 0 ]]; then
  echo "至少一个XHand候选失败；修正后重新执行可续跑。" >&2
  exit 1
fi

MPLCONFIGDIR=/tmp/matplotlib-retarget \
OMP_NUM_THREADS=4 \
MKL_NUM_THREADS=4 \
"$retarget_python" retarget_research/retargeting/evaluate/evaluate_candidate_sweep.py \
  --config retarget_research/retargeting/configs/xhand_contact_weight_search_v1.json \
  --output retarget_research/outputs/xhand_contact_weight_search_v1_evaluation/search_summary.json \
  --workers 1 --resume
