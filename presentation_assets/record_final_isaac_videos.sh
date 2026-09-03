#!/usr/bin/env bash
set -euo pipefail

# 这条命令需要本机 GPU/图形环境，预计超过 3 分钟，因此由用户在终端执行。
# 输入：两个项目已冻结的模型、轨迹与案例编号。
# 输出：重定向 9 段视频，以及 ShadowHand 四类各 1 段最终策略成功视频。

ROOT=/home/lekangwan/projects/DexGraspMotionChallenge2025-final-release
DEX_PY=/home/lekangwan/miniconda3/envs/dexgrasp/bin/python
OLD=/home/lekangwan/projects/DexGraspMotionChallenge2025

# 该环境的 PyTorch 来自用户 site，而 ninja 安装在 conda 环境中；显式加入 PATH，
# 让 Isaac Gym 首次加载 gymtorch 时能找到 C++ 扩展构建器。
export PATH=/home/lekangwan/miniconda3/envs/dexgrasp/bin:$PATH

cd "$ROOT"

echo "[1/2] 录制三只目标手的成功、失稳、失败案例（已有合格视频会跳过）"
if [[ "${SKIP_RETARGET:-0}" != "1" ]]; then
  bash retarget_research/reports/run_record_final_retargeting_videos.sh
else
  echo "[skip] retargeting videos"
fi

echo "[2/2] 录制 ShadowHand 最终 Chunk8 策略的四类成功案例"
mkdir -p presentation_assets/videos/shadow

record_shadow() {
  local category=$1
  local object_id=$2
  local env_index=$3
  local capture_dir="$ROOT/presentation_assets/videos/shadow/${category}"
  local result="$ROOT/presentation_assets/videos/shadow/${category}.yaml"
  local env_tag
  printf -v env_tag "%03d" "$env_index"
  if [[ "${FORCE_RECORD:-0}" != "1" && -s "$capture_dir/env${env_tag}.mp4" && -s "$result" ]]; then
    echo "[reuse] $category"
    return
  fi
  rm -f "$result"
  rm -f "$capture_dir/env${env_tag}.mp4" \
        "$capture_dir/env${env_tag}_first.png" \
        "$capture_dir/env${env_tag}_final.png" \
        "$capture_dir/env${env_tag}_success.png"
  "$DEX_PY" -u custom_tools/evaluate_bc_checkpoints_batched.py \
    --checkpoint custom_tools/checkpoints/shadow_chunk8_final.ckpt \
    --bc-config custom_tools/configs/unified_student_temporal_chunk8_demo80_v1.yaml \
    --residual-config custom_tools/configs/residual_ppo_stage1.yaml \
    --trajectory-root "$OLD/dexgrasp/dataset/scaled_category_final_v1_preprocessed" \
    --meshdata-root "$OLD/assets/meshdata" \
    --object-selection custom_tools/configs/scaled_final_holdout_all8.yaml \
    --object-id "$object_id" \
    --output "$result" \
    --seed 2025 \
    --policy-motion-steps 70 \
    --temporal-ensemble-decay 0 \
    --late-lift-z-boost 0.20 \
    --late-lift-start-step 40 \
    --capture-dir "$capture_dir" \
    --capture-env "$env_index" \
    --capture-width 960 \
    --capture-height 720 \
    --min-free-vram-mb 4500
}

record_shadow bottle core-bottle-70172e6afe6aff7847f90c1ac631b97f 11
record_shadow mug core-mug-15bd6225c209a8e3654b0ce7754570c8 7
record_shadow bowl core-bowl-fa23aa60ec51c8e4c40fe5637f0a27e1 1
record_shadow camera core-camera-147183af1ba4e97b8a94168388287ad5 0

echo "ISAAC_RECORDING=COMPLETE"
