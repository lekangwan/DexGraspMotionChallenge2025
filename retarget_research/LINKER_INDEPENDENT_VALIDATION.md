# Linker O6 v2独立验证手册

## 目的与冻结边界

本验证只回答：在不再修改任何v2参数的情况下，开发集5/10能否迁移到从未用于调参的物体。清单固定为`retargeting/configs/linker_independent_validation_v1.json`，种子20260811，10个新物体、每物体2条，共20条轨迹。

清单自动排除了早期Camera/Bottle/Jar和开发集Lime/HotPot/Planter/Loafer/Vase，共8个已暴露物体；新旧物体交集为0。方法固定为`retargeting/configs/linker_o6_adaptive_v2.json`，清单保存其SHA-256。看到验证结果后不得修改残差、阈值或PD再覆盖本次结果。

## 0. 运行前审计（短命令）

```bash
/home/lekangwan/miniconda3/envs/hand-retarget/bin/python \
  retarget_research/retargeting/prepare/verify_validation_manifest.py \
  --manifest retarget_research/retargeting/configs/linker_independent_validation_v1.json
```

必须输出`validation_manifest_ok=true`、`object_count=10`、`trajectory_count=20`和冻结方法名，否则停止。

## 1. 生成O6关键点基线（长命令，由用户运行）

```bash
MPLCONFIGDIR=/tmp/matplotlib-retarget \
OMP_NUM_THREADS=4 \
MKL_NUM_THREADS=4 \
/home/lekangwan/miniconda3/envs/hand-retarget/bin/python \
  retarget_research/retargeting/run/run_linker_manifest.py \
  --manifest retarget_research/retargeting/configs/linker_independent_validation_v1.json \
  --output-dir retarget_research/outputs/linker_independent_validation_v1_baseline \
  --workers 2 \
  --resume \
  --maxeval 100 \
  --include-thumb-middle \
  --joint-temporal-weight 1 \
  --translation-temporal-weight 300 \
  --rotation-temporal-weight 1
```

20条预计超过3分钟。完成后只检查`manifest_run_summary.json`中的`all_successful=true`和`trajectory_count=20`，不能先挑选“看起来容易”的物体。

## 2. 应用冻结渐进夹紧（依赖阶段1）

```bash
MPLCONFIGDIR=/tmp/matplotlib-retarget OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
/home/lekangwan/miniconda3/envs/hand-retarget/bin/python \
  retarget_research/retargeting/run/refine_linker_squeeze.py \
  --manifest retarget_research/retargeting/configs/linker_independent_validation_v1.json \
  --baseline-dir retarget_research/outputs/linker_independent_validation_v1_baseline \
  --output-dir retarget_research/outputs/linker_independent_validation_v1_squeeze \
  --method-name linker_o6_dynamic_squeeze_v1 \
  --thumb-yaw-delta 0.10 \
  --thumb-pitch-delta 0.25 \
  --finger-delta 0.55 \
  --contact-threshold 0.02 \
  --min-contact-tips 2 \
  --lift-delta 0.03
```

## 3. 应用冻结六关节残差（依赖阶段2）

```bash
MPLCONFIGDIR=/tmp/matplotlib-retarget OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
/home/lekangwan/miniconda3/envs/hand-retarget/bin/python \
  retarget_research/retargeting/run/refine_linker_squeeze.py \
  --manifest retarget_research/retargeting/configs/linker_independent_validation_v1.json \
  --baseline-dir retarget_research/outputs/linker_independent_validation_v1_squeeze \
  --output-dir retarget_research/outputs/linker_independent_validation_v1_v2 \
  --method-name linker_o6_force_redistribution_v2 \
  --joint-residuals -0.12530 0.03100 -0.13028 -0.10577 0.10630 0.00391 \
  --contact-threshold 0.02 \
  --min-contact-tips 2 \
  --lift-delta 0.03
```

## 4. 用冻结自适应PD统一重放（依赖阶段3）

```bash
MPLCONFIGDIR=/tmp/matplotlib-retarget OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
/home/lekangwan/miniconda3/envs/hand-retarget/bin/python \
  retarget_research/retargeting/evaluate/evaluate_hand_manifest.py \
  --hand linker \
  --manifest retarget_research/retargeting/configs/linker_independent_validation_v1.json \
  --target-dir retarget_research/outputs/linker_independent_validation_v1_v2 \
  --output-dir retarget_research/outputs/linker_independent_validation_v1_v2_evaluation \
  --workers 1 \
  --linker-adaptive-gains \
  --linker-adaptive-scale-threshold 0.06 \
  --linker-adaptive-joint-std-threshold 0.25 \
  --linker-high-stiffness 400 \
  --linker-high-damping 20
```

## 结果解释

主结果只报告20条的严格成功率和10物体宏平均。本轮不是类别平衡抽样，不能把它表述为考核要求的50类别正式结果。无论成功率高低，都保留完整结果：若接近50%，说明v2有初步迁移证据；若明显下降，说明开发集阈值或固定残差过拟合，不能回头修改本清单上的参数后继续称其为独立验证。

## 实际结果与方法选择

| 阶段 | 严格成功 | 平均最大抬升 | 平均最终抬升 | 平均关键点误差 |
| --- | ---: | ---: | ---: | ---: |
| B0 几何基线，默认PD | 4/20（20%） | 68.3 mm | 28.8 mm | 14.34 mm |
| B1 统一渐进夹紧，默认PD | **7/20（35%）** | **135.4 mm** | **93.5 mm** | 25.88 mm |
| B2 B1+固定六关节残差，默认PD | 6/20（30%） | 108.1 mm | 54.9 mm | 25.21 mm |
| B3 B2+输入条件自适应PD | 6/20（30%） | 107.7 mm | 54.5 mm | 25.21 mm |

B0→B1新增CoughDrops两条、Wingtip11、FruitBasket30和Candle2，但丢失Saucepan两条，净增3条。B1→B2没有新增，反而使FruitBasket30从最大抬升341.9 mm成功降为6.4 mm失败。B2→B3只有CoughDrops30触发高增益；它在默认PD下本来就成功，因此成功集合完全不变。

最终选择`linker_o6_dynamic_squeeze_v1`作为后续正式主方法，使用默认主动/跟随PD 120/5。v2开发集5/10仍保留为开发结果，但独立集只得6/20，固定残差和自适应PD均没有证明泛化。B0/B1/B2/B3的20条Wilson 95%区间分别约为8.1–41.6%、18.1–56.7%、14.5–51.9%和14.5–51.9%，相互重叠，因此只能作为方法选择证据，不能声称统计显著。

由于我们已经用这20条比较并选择了B1，它们从此属于方法选择验证集，不再是最终测试集。正式结论必须来自后续冻结的100物体/1000轨迹，且不能再用这20条调整夹紧参数。
