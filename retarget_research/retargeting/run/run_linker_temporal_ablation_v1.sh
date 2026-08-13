#!/usr/bin/env bash
# 输入：冻结20轨迹manifest，以及脚本内预声明的4组时序权重。
# 输出：4组原始重定向、相同夹紧后处理、统一PhysX报告和搜索汇总JSON。
# 内部逻辑：四组几何优化并行运行；全部成功后顺序做快速夹紧，再统一评测。
# 作用：用一条可续跑命令完成Linker时序组成消融，避免人工抄错候选参数。

set -u -o pipefail

project_root="/home/lekangwan/projects/DexGraspMotionChallenge2025-final-release"
retarget_python="/home/lekangwan/miniconda3/envs/hand-retarget/bin/python"
manifest_path="retarget_research/retargeting/configs/linker_independent_validation_v1.json"
raw_root="retarget_research/outputs/linker_temporal_ablation_v1_raw"
candidate_root="retarget_research/outputs/linker_temporal_ablation_v1"

cd "$project_root" || exit 1

# run_geometry的输入是候选名和关节/平移/旋转权重；输出是该候选的10个物体npy。
# 每个候选内部只开1个worker，四个候选彼此并行，避免嵌套并行挤占过多CPU。
run_geometry() {
  local candidate_name="$1"
  local joint_weight="$2"
  local translation_weight="$3"
  local rotation_weight="$4"
  echo "[start geometry] $candidate_name"
  MPLCONFIGDIR=/tmp/matplotlib-retarget \
  OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  "$retarget_python" retarget_research/retargeting/run/run_linker_manifest.py \
    --manifest "$manifest_path" \
    --output-dir "$raw_root/$candidate_name" \
    --workers 1 \
    --resume \
    --maxeval 100 \
    --include-thumb-middle \
    --joint-temporal-weight "$joint_weight" \
    --translation-temporal-weight "$translation_weight" \
    --rotation-temporal-weight "$rotation_weight"
}

candidate_names=(no_temporal joint_only translation_only rotation_only)
run_geometry no_temporal 0 0 0 & geometry_pid_0=$!
run_geometry joint_only 1 0 0 & geometry_pid_1=$!
run_geometry translation_only 0 300 0 & geometry_pid_2=$!
run_geometry rotation_only 0 0 1 & geometry_pid_3=$!

geometry_failed=0
for geometry_pid in "$geometry_pid_0" "$geometry_pid_1" "$geometry_pid_2" "$geometry_pid_3"; do
  wait "$geometry_pid" || geometry_failed=1
done
if [[ "$geometry_failed" -ne 0 ]]; then
  echo "至少一个几何候选失败；修正后原命令可用--resume续跑。" >&2
  exit 1
fi

# 四组都采用已经选定的同一夹紧参数，确保只比较基础重定向的时序权重。
for candidate_name in "${candidate_names[@]}"; do
  echo "[squeeze] $candidate_name"
  MPLCONFIGDIR=/tmp/matplotlib-retarget \
  OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  "$retarget_python" retarget_research/retargeting/run/refine_linker_squeeze.py \
    --manifest "$manifest_path" \
    --baseline-dir "$raw_root/$candidate_name" \
    --output-dir "$candidate_root/$candidate_name" \
    --method-name "linker_o6_temporal_${candidate_name}_squeeze_v1" \
    --thumb-yaw-delta 0.075 \
    --thumb-pitch-delta 0.1875 \
    --finger-delta 0.425 \
    --contact-threshold 0.02 \
    --min-contact-tips 2 \
    --lift-delta 0.03 || exit 1
done

echo "[evaluate] five temporal candidates"
MPLCONFIGDIR=/tmp/matplotlib-retarget \
OMP_NUM_THREADS=4 \
MKL_NUM_THREADS=4 \
"$retarget_python" retarget_research/retargeting/evaluate/evaluate_candidate_sweep.py \
  --config retarget_research/retargeting/configs/linker_temporal_ablation_v1.json \
  --output retarget_research/outputs/linker_temporal_ablation_v1_evaluation/search_summary.json \
  --workers 1 \
  --resume
