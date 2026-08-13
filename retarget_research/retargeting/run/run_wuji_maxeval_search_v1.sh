#!/usr/bin/env bash
# 输入：冻结20轨迹、Wuji v1映射，以及50/150两个新SLSQP评估上限。
# 输出：两个新Wuji候选、与当前100次统一比较的PhysX汇总。
# 内部逻辑：50和150两个候选各用1个worker并行生成，复用100次结果，再顺序物理评测。
# 作用：在正式1000条前决定Wuji优化收敛与运行成本的折中。

set -u -o pipefail

project_root="/home/lekangwan/projects/DexGraspMotionChallenge2025-final-release"
retarget_python="/home/lekangwan/miniconda3/envs/hand-retarget/bin/python"
manifest_path="retarget_research/retargeting/configs/linker_independent_validation_v1.json"
candidate_root="retarget_research/outputs/wuji_maxeval_search_v1"
mapping_config="retarget_research/retargeting/configs/wuji_keypoint_map.json"

cd "$project_root" || exit 1

run_candidate() {
  local candidate_name="$1"
  local maxeval="$2"
  echo "[start retargeting] $candidate_name"
  MPLCONFIGDIR=/tmp/matplotlib-retarget \
  OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  "$retarget_python" retarget_research/retargeting/run/run_wuji_manifest.py \
    --manifest "$manifest_path" \
    --output-dir "$candidate_root/$candidate_name" \
    --workers 1 --resume --maxeval "$maxeval" \
    --translation-bound 2.0 --source-z-offset 0.4 \
    --mapping-config "$mapping_config" \
    --joint-temporal-weight 0 \
    --translation-temporal-weight 0 \
    --rotation-temporal-weight 0
}

run_candidate maxeval_50 50 & candidate_pid_0=$!
run_candidate maxeval_150 150 & candidate_pid_1=$!
candidate_failed=0
wait "$candidate_pid_0" || candidate_failed=1
wait "$candidate_pid_1" || candidate_failed=1
if [[ "$candidate_failed" -ne 0 ]]; then
  echo "至少一个Wuji迭代数候选失败；修正后重新执行可续跑。" >&2
  exit 1
fi

MPLCONFIGDIR=/tmp/matplotlib-retarget \
OMP_NUM_THREADS=4 \
MKL_NUM_THREADS=4 \
"$retarget_python" retarget_research/retargeting/evaluate/evaluate_candidate_sweep.py \
  --config retarget_research/retargeting/configs/wuji_maxeval_search_v1.json \
  --output retarget_research/outputs/wuji_maxeval_search_v1_evaluation/search_summary.json \
  --workers 1 --resume
