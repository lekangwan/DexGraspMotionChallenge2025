# 重定向收尾 A/B 方法选择手册

## 当前已经完成的短步骤

- A组：50类别、50物体、50条calibration轨迹。
- B组：相同50类别、另外50个物体、50条calibration轨迹。
- XHand A组三个动态残差候选已经由正式1000条轨迹秒级生成：`r0/r05/r1`。
- Linker A/B无夹紧基线和夹紧候选、XHand A/B官方候选都已经从完整轨迹切出。
- XHand官方、旧接触和Linker夹紧的物理结果可从正式1000条摘要直接复用。

以下只有Isaac Gym物理重放可能超过3分钟，按约定由用户终端运行。第一阶段五条命令互不依赖；为避免同时启动过多CPU PhysX进程，每条使用一个worker，可开五个终端并行。如果机器明显卡顿，可以只并行三条，完成后再运行剩余两条。

推荐直接使用已经写好的总入口。它先并行三个XHand进程，完成后再并行两个Linker进程，失败后可原命令续跑：

```bash
cd /home/lekangwan/projects/DexGraspMotionChallenge2025-final-release
bash retarget_research/retargeting/run/run_method_selection_stage_a.sh
```

下面保留五条展开命令，便于检查参数或在不同终端中单独恢复。

## 第一阶段：A组三个XHand候选 + Linker A/B基线

### 终端1：XHand残差最终归零

```bash
cd /home/lekangwan/projects/DexGraspMotionChallenge2025-final-release
MPLCONFIGDIR=/tmp/matplotlib-retarget OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/home/lekangwan/miniconda3/envs/hand-retarget/bin/python \
  retarget_research/retargeting/evaluate/evaluate_hand_manifest.py \
  --hand xhand \
  --manifest retarget_research/manifests/formal_method_selection_a_50c_50t_seed20260814.json \
  --target-dir retarget_research/outputs/method_selection_ab/a/xhand_dynamic_r0 \
  --output-dir retarget_research/outputs/method_selection_ab/a/xhand_dynamic_r0_evaluation \
  --workers 1 --resume
```

### 终端2：XHand保留一半残差

```bash
cd /home/lekangwan/projects/DexGraspMotionChallenge2025-final-release
MPLCONFIGDIR=/tmp/matplotlib-retarget OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/home/lekangwan/miniconda3/envs/hand-retarget/bin/python \
  retarget_research/retargeting/evaluate/evaluate_hand_manifest.py \
  --hand xhand \
  --manifest retarget_research/manifests/formal_method_selection_a_50c_50t_seed20260814.json \
  --target-dir retarget_research/outputs/method_selection_ab/a/xhand_dynamic_r05 \
  --output-dir retarget_research/outputs/method_selection_ab/a/xhand_dynamic_r05_evaluation \
  --workers 1 --resume
```

### 终端3：XHand完整保留残差

```bash
cd /home/lekangwan/projects/DexGraspMotionChallenge2025-final-release
MPLCONFIGDIR=/tmp/matplotlib-retarget OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/home/lekangwan/miniconda3/envs/hand-retarget/bin/python \
  retarget_research/retargeting/evaluate/evaluate_hand_manifest.py \
  --hand xhand \
  --manifest retarget_research/manifests/formal_method_selection_a_50c_50t_seed20260814.json \
  --target-dir retarget_research/outputs/method_selection_ab/a/xhand_dynamic_r1 \
  --output-dir retarget_research/outputs/method_selection_ab/a/xhand_dynamic_r1_evaluation \
  --workers 1 --resume
```

### 终端4：Linker A组无夹紧基线

```bash
cd /home/lekangwan/projects/DexGraspMotionChallenge2025-final-release
MPLCONFIGDIR=/tmp/matplotlib-retarget OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/home/lekangwan/miniconda3/envs/hand-retarget/bin/python \
  retarget_research/retargeting/evaluate/evaluate_hand_manifest.py \
  --hand linker \
  --manifest retarget_research/manifests/formal_method_selection_a_50c_50t_seed20260814.json \
  --target-dir retarget_research/outputs/method_selection_ab/a/linker_baseline \
  --output-dir retarget_research/outputs/method_selection_ab/a/linker_baseline_evaluation \
  --workers 1 --resume \
  --linker-finger-stiffness 120 --linker-finger-damping 5 \
  --linker-mimic-stiffness 120 --linker-mimic-damping 5
```

### 终端5：Linker B组无夹紧基线

```bash
cd /home/lekangwan/projects/DexGraspMotionChallenge2025-final-release
MPLCONFIGDIR=/tmp/matplotlib-retarget OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/home/lekangwan/miniconda3/envs/hand-retarget/bin/python \
  retarget_research/retargeting/evaluate/evaluate_hand_manifest.py \
  --hand linker \
  --manifest retarget_research/manifests/formal_method_selection_b_50c_50t_seed20260814.json \
  --target-dir retarget_research/outputs/method_selection_ab/b/linker_baseline \
  --output-dir retarget_research/outputs/method_selection_ab/b/linker_baseline_evaluation \
  --workers 1 --resume \
  --linker-finger-stiffness 120 --linker-finger-damping 5 \
  --linker-mimic-stiffness 120 --linker-mimic-damping 5
```

## 第一阶段完成后的分析规则

XHand先比较严格成功数，再比较相对官方的配对新增/丢失；若成功数仍相同，依次看最终抬升、最大抬升和接触步数。A组只留下一个系数进入B组，不能逐物体混合三种候选。Linker把新跑的无夹紧摘要与正式夹紧摘要过滤到同一A/B键；若夹紧在A和B均无净收益，则最终应回退到几何基线，不能因为正式夹紧已经跑完1000条就强行保留。

B组XHand候选必须等A组第一名确定后再生成；该生成仍是秒级。B组只重放“官方已有结果 vs A组第一名”，不再比较三个系数，也不重新调整参数。

## A组已完成后的冻结决定

- `r=0/0.5/1`为28/50、29/50、27/50，A组唯一入选系数为`r=0.5`。
- Linker夹紧相对无夹紧在A净增5、B净增7，Linker最终保留夹紧。
- B组`r=0.5`候选已经生成。执行下面最后50条确认，不再改系数：

```bash
cd /home/lekangwan/projects/DexGraspMotionChallenge2025-final-release
MPLCONFIGDIR=/tmp/matplotlib-retarget OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/home/lekangwan/miniconda3/envs/hand-retarget/bin/python \
  retarget_research/retargeting/evaluate/evaluate_hand_manifest.py \
  --hand xhand \
  --manifest retarget_research/manifests/formal_method_selection_b_50c_50t_seed20260814.json \
  --target-dir retarget_research/outputs/method_selection_ab/b/xhand_dynamic_r05 \
  --output-dir retarget_research/outputs/method_selection_ab/b/xhand_dynamic_r05_evaluation \
  --workers 1 --resume
```

## 最终结果

B组已完成：`r=0.5`为26/50，官方为28/50；新增0、回退2。因此XHand最终使用官方参考，不再运行动态残差1000条。Linker夹紧在A/B分别净增5/7条，最终保留夹紧；Wuji保持v1。至此本手册的所有运行任务结束。
