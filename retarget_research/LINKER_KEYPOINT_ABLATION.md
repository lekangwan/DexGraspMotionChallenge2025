# Linker O6 10/11/15关键点消融

## 要回答的问题

在保持Linker真实6个主动自由度、时序权重、渐进夹紧和PhysX参数全部不变时，把Shadow→Linker几何点对从10个增加到11或15个，是否能提高严格物理成功率。

三组不变的部分是：同一`linker_independent_validation_v1.json`的20条轨迹、每帧SLSQP `maxeval=100`、时序权重1/300/1、同一渐进夹紧残差和同一严格成功判据。这20条已经用于方法选择，因此本消融不是最终测试。

## 三种点集

- 10点：官方Linker口径，掌心+四指近端/指尖+拇指指尖。
- 11点：10点+拇指中段，即当前主方法的几何基线。
- 15点：11点+食指/中指/无名指/小指中段。由于Linker每根普通指只有近端和远端两个运动link，新点设在远端link从关节原点到已校准指尖向量的37%处。该比例由Shadow“近端+中段”占整指长的比例，与Linker近端/远端长度折算得到，不是额外自由度。

## 决策规则

三组都必须每条轨迹只输出一个候选，不取成功并集。优先比较严格成功数和物体宏平均；若只差1条，继续比较平均最大/最终抬升、成功回退数和接触持续步数，不用关键点MSE单独决定。

当前11点+渐进夹紧为7/20。15点只有在同口径下明确高于7/20，且没有大量丢失原成功轨迹时，才能取代11点进入正式1000条。

## 阶段1：并行生成10点和15点几何基线

11点基线已存在于`outputs/linker_independent_validation_v1_baseline`，不重复生成。下面两条可分别在两个终端运行：

```bash
MPLCONFIGDIR=/tmp/matplotlib-retarget OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
/home/lekangwan/miniconda3/envs/hand-retarget/bin/python \
  retarget_research/retargeting/run/run_linker_manifest.py \
  --manifest retarget_research/retargeting/configs/linker_independent_validation_v1.json \
  --output-dir retarget_research/outputs/linker_keypoint10_independent_v1 \
  --workers 1 --resume --maxeval 100 \
  --joint-temporal-weight 1 \
  --translation-temporal-weight 300 \
  --rotation-temporal-weight 1
```

```bash
MPLCONFIGDIR=/tmp/matplotlib-retarget OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
/home/lekangwan/miniconda3/envs/hand-retarget/bin/python \
  retarget_research/retargeting/run/run_linker_manifest.py \
  --manifest retarget_research/retargeting/configs/linker_independent_validation_v1.json \
  --output-dir retarget_research/outputs/linker_keypoint15_independent_v1 \
  --workers 1 --resume --maxeval 100 --include-finger-middle \
  --joint-temporal-weight 1 \
  --translation-temporal-weight 300 \
  --rotation-temporal-weight 1
```

## 阶段2：应用完全相同的渐进夹紧

阶段1两个摘要均为`all_successful=true`后，并行运行：

```bash
MPLCONFIGDIR=/tmp/matplotlib-retarget OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
/home/lekangwan/miniconda3/envs/hand-retarget/bin/python \
  retarget_research/retargeting/run/refine_linker_squeeze.py \
  --manifest retarget_research/retargeting/configs/linker_independent_validation_v1.json \
  --baseline-dir retarget_research/outputs/linker_keypoint10_independent_v1 \
  --output-dir retarget_research/outputs/linker_keypoint10_squeeze_independent_v1 \
  --method-name linker_o6_keypoint10_squeeze_v1 \
  --thumb-yaw-delta 0.10 --thumb-pitch-delta 0.25 --finger-delta 0.55 \
  --contact-threshold 0.02 --min-contact-tips 2 --lift-delta 0.03
```

```bash
MPLCONFIGDIR=/tmp/matplotlib-retarget OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
/home/lekangwan/miniconda3/envs/hand-retarget/bin/python \
  retarget_research/retargeting/run/refine_linker_squeeze.py \
  --manifest retarget_research/retargeting/configs/linker_independent_validation_v1.json \
  --baseline-dir retarget_research/outputs/linker_keypoint15_independent_v1 \
  --output-dir retarget_research/outputs/linker_keypoint15_squeeze_independent_v1 \
  --method-name linker_o6_keypoint15_squeeze_v1 \
  --thumb-yaw-delta 0.10 --thumb-pitch-delta 0.25 --finger-delta 0.55 \
  --contact-threshold 0.02 --min-contact-tips 2 --lift-delta 0.03
```

## 阶段3：统一PhysX评测与配对比较

阶段2完成后并行评测10点和15点；11点结果已存在`outputs/linker_independent_validation_v1_squeeze_evaluation`。

```bash
/home/lekangwan/miniconda3/envs/hand-retarget/bin/python \
  retarget_research/retargeting/evaluate/evaluate_hand_manifest.py \
  --hand linker \
  --manifest retarget_research/retargeting/configs/linker_independent_validation_v1.json \
  --target-dir retarget_research/outputs/linker_keypoint10_squeeze_independent_v1 \
  --output-dir retarget_research/outputs/linker_keypoint10_squeeze_independent_v1_evaluation \
  --workers 1
```

```bash
/home/lekangwan/miniconda3/envs/hand-retarget/bin/python \
  retarget_research/retargeting/evaluate/evaluate_hand_manifest.py \
  --hand linker \
  --manifest retarget_research/retargeting/configs/linker_independent_validation_v1.json \
  --target-dir retarget_research/outputs/linker_keypoint15_squeeze_independent_v1 \
  --output-dir retarget_research/outputs/linker_keypoint15_squeeze_independent_v1_evaluation \
  --workers 1
```
