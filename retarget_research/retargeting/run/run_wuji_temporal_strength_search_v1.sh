#!/usr/bin/env bash
# 输入：冻结20轨迹、Wuji v1映射、已选maxeval50和三组非零时序强度。
# 输出：三组新候选、与无时序基线统一比较的PhysX报告及搜索汇总JSON。
# 内部逻辑：三组候选各用1个worker并行生成，复用0倍基线，然后顺序物理评测。
# 作用：判断跨帧平滑约束能否提高真实抓取成功，并为正式1000条冻结最后一个Wuji参数。

set -u -o pipefail

project_root="/home/lekangwan/projects/DexGraspMotionChallenge2025-final-release"
retarget_python="/home/lekangwan/miniconda3/envs/hand-retarget/bin/python"
manifest_path="retarget_research/retargeting/configs/linker_independent_validation_v1.json"
candidate_root="retarget_research/outputs/wuji_temporal_strength_search_v1"
mapping_config="retarget_research/retargeting/configs/wuji_keypoint_map.json"

cd "$project_root" || exit 1

# 输入候选名和三类时序权重；输出该候选的10个物体、20条轨迹。
# 每组只开一个worker，三组之间并行，避免嵌套线程争抢CPU。
run_candidate() {
  local candidate_name="$1"
  local joint_weight="$2"
  local translation_weight="$3"
  local rotation_weight="$4"
  echo "[start retargeting] $candidate_name"
  MPLCONFIGDIR=/tmp/matplotlib-retarget \
  OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  "$retarget_python" retarget_research/retargeting/run/run_wuji_manifest.py \
    --manifest "$manifest_path" \
    --output-dir "$candidate_root/$candidate_name" \
    --workers 1 --resume --maxeval 50 \
    --translation-bound 2.0 --source-z-offset 0.4 \
    --mapping-config "$mapping_config" \
    --joint-temporal-weight "$joint_weight" \
    --translation-temporal-weight "$translation_weight" \
    --rotation-temporal-weight "$rotation_weight"
}

run_candidate temporal_0p1 0.1 30 0.1 & candidate_pid_0=$!
run_candidate temporal_0p3 0.3 90 0.3 & candidate_pid_1=$!
run_candidate temporal_1p0 1 300 1 & candidate_pid_2=$!

candidate_failed=0
for candidate_pid in "$candidate_pid_0" "$candidate_pid_1" "$candidate_pid_2"; do
  wait "$candidate_pid" || candidate_failed=1
done
if [[ "$candidate_failed" -ne 0 ]]; then
  echo "至少一个Wuji时序候选失败；修正后重新执行可续跑。" >&2
  exit 1
fi

echo "[evaluate] baseline and three temporal candidates"
MPLCONFIGDIR=/tmp/matplotlib-retarget \
OMP_NUM_THREADS=4 \
MKL_NUM_THREADS=4 \
"$retarget_python" retarget_research/retargeting/evaluate/evaluate_candidate_sweep.py \
  --config retarget_research/retargeting/configs/wuji_temporal_strength_search_v1.json \
  --output retarget_research/outputs/wuji_temporal_strength_search_v1_evaluation/search_summary.json \
  --workers 1 --resume

# 输入0倍摘要与一个非零候选摘要；输出逐轨迹新增成功、回退和净变化。
# 自动做三次配对比较，避免总成功数相同时漏掉“一条新增、一条回退”。
baseline_summary="retarget_research/outputs/wuji_maxeval_search_v1_evaluation/maxeval_50/manifest_evaluation_summary.json"
for candidate_name in temporal_0p1 temporal_0p3 temporal_1p0; do
  "$retarget_python" retarget_research/retargeting/evaluate/compare_method_summaries.py \
    --baseline-summary "$baseline_summary" \
    --improved-summary "retarget_research/outputs/wuji_temporal_strength_search_v1_evaluation/$candidate_name/manifest_evaluation_summary.json" \
    --output "retarget_research/outputs/wuji_temporal_strength_search_v1_evaluation/${candidate_name}_vs_baseline.json" || exit 1
done
