# Wuji解剖约束初筛运行说明

目标不是先追求更高的旧成功率，而是修复四根普通手指远端反向弯曲。全部候选使用相同的v1关键点映射、`maxeval=50`和无时序项；唯一变量是手型约束。初筛清单只含策略train的20个不同类别、每类1条，不读取任何物理成功结果。

## 第一阶段：两个终端并行生成候选

终端A运行远端硬边界。输入是20条Shadow轨迹；输出是20条Wuji轨迹。优化时保留20自由度，只把四个`joint4`的反向伸展限制为最多5度。

```bash
cd /home/lekangwan/projects/DexGraspMotionChallenge2025-final-release
MPLCONFIGDIR=/tmp/matplotlib-retarget OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
/home/lekangwan/miniconda3/envs/hand-retarget/bin/python \
retarget_research/retargeting/run/run_wuji_manifest.py \
--manifest retarget_research/manifests/wuji_anatomy_train20_seed20260817.json \
--output-dir retarget_research/outputs/wuji_anatomy_ablation_v1/distal_bound \
--workers 1 --resume --maxeval 50 --translation-bound 2.0 \
--source-z-offset 0.4 \
--mapping-config retarget_research/retargeting/configs/wuji_keypoint_map.json \
--joint-temporal-weight 0 --translation-temporal-weight 0 \
--rotation-temporal-weight 0 \
--anatomy-config retarget_research/retargeting/configs/wuji_anatomy_distal_v1.json
```

终端B同时运行硬边界加弱协调。除PIP/DIP最多反向5度外，损失中加入`DIP≈2/3×PIP`的弱先验，权重仅0.02，不覆盖关键点目标。

```bash
cd /home/lekangwan/projects/DexGraspMotionChallenge2025-final-release
MPLCONFIGDIR=/tmp/matplotlib-retarget OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
/home/lekangwan/miniconda3/envs/hand-retarget/bin/python \
retarget_research/retargeting/run/run_wuji_manifest.py \
--manifest retarget_research/manifests/wuji_anatomy_train20_seed20260817.json \
--output-dir retarget_research/outputs/wuji_anatomy_ablation_v1/coupled_flexion \
--workers 1 --resume --maxeval 50 --translation-bound 2.0 \
--source-z-offset 0.4 \
--mapping-config retarget_research/retargeting/configs/wuji_keypoint_map.json \
--joint-temporal-weight 0 --translation-temporal-weight 0 \
--rotation-temporal-weight 0 \
--anatomy-config retarget_research/retargeting/configs/wuji_anatomy_coupled_v1.json
```

两边都出现`all_successful=True`以后再进入第二阶段。命令支持`--resume`，中断后可原样重跑。

## 第一阶段补充：完成2×2因素消融

首轮物理重放表明：只限制DIP时，末帧仍有27.375%的受检关节反向超过5度；同时限制PIP/DIP并加入协调项后，末帧反弯降为0，但稳定运输由旧基线12/20降至11/20。由于首轮两个候选同时改变了“PIP硬边界”和“PIP-DIP协调项”，还不能判断性能下降由哪个因素造成。

因此固定全部已有参数，只补齐下面两个缺失组合。四种受约束候选由两个二值因素组成：是否限制PIP反弯、是否加入弱协调项。这是封闭的2×2消融，不根据测试结果继续扩展搜索空间。

终端A运行PIP+DIP硬边界，但不加入协调损失：

```bash
cd /home/lekangwan/projects/DexGraspMotionChallenge2025-final-release
MPLCONFIGDIR=/tmp/matplotlib-retarget OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
/home/lekangwan/miniconda3/envs/hand-retarget/bin/python \
retarget_research/retargeting/run/run_wuji_manifest.py \
--manifest retarget_research/manifests/wuji_anatomy_train20_seed20260817.json \
--output-dir retarget_research/outputs/wuji_anatomy_ablation_v1/flexion_bounds \
--workers 1 --resume --maxeval 50 --translation-bound 2.0 \
--source-z-offset 0.4 \
--mapping-config retarget_research/retargeting/configs/wuji_keypoint_map.json \
--joint-temporal-weight 0 --translation-temporal-weight 0 \
--rotation-temporal-weight 0 \
--anatomy-config retarget_research/retargeting/configs/wuji_anatomy_flexion_bounds_v1.json
```

终端B只限制DIP反弯，同时加入弱协调损失：

```bash
cd /home/lekangwan/projects/DexGraspMotionChallenge2025-final-release
MPLCONFIGDIR=/tmp/matplotlib-retarget OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
/home/lekangwan/miniconda3/envs/hand-retarget/bin/python \
retarget_research/retargeting/run/run_wuji_manifest.py \
--manifest retarget_research/manifests/wuji_anatomy_train20_seed20260817.json \
--output-dir retarget_research/outputs/wuji_anatomy_ablation_v1/distal_coupled \
--workers 1 --resume --maxeval 50 --translation-bound 2.0 \
--source-z-offset 0.4 \
--mapping-config retarget_research/retargeting/configs/wuji_keypoint_map.json \
--joint-temporal-weight 0 --translation-temporal-weight 0 \
--rotation-temporal-weight 0 \
--anatomy-config retarget_research/retargeting/configs/wuji_anatomy_distal_coupled_v1.json
```

两条生成命令各需约14分钟，适合在两个终端并行运行。生成完成后的物理重放约1分钟，由分析脚本统一执行和比较即可。

## 第二阶段：三个终端并行物理重放

旧基线的20条已经从正式1000条候选中秒级切出，不重新执行重定向。下面三条命令分别评测旧基线、远端硬边界和协调屈曲；默认成功标准已是末段稳定30 cm。每条同时保存掌物位姿trace，之后可继续检查滑移。

终端A：

```bash
cd /home/lekangwan/projects/DexGraspMotionChallenge2025-final-release
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
/home/lekangwan/miniconda3/envs/hand-retarget/bin/python \
retarget_research/retargeting/evaluate/evaluate_hand_manifest.py \
--hand wuji \
--manifest retarget_research/manifests/wuji_anatomy_train20_seed20260817.json \
--target-dir retarget_research/outputs/wuji_anatomy_ablation_v1/legacy_unconstrained \
--output-dir retarget_research/outputs/wuji_anatomy_ablation_v1_evaluation/legacy_unconstrained \
--policy-trace-dir retarget_research/outputs/wuji_anatomy_ablation_v1_traces/legacy_unconstrained \
--workers 1 --resume
```

终端B：

```bash
cd /home/lekangwan/projects/DexGraspMotionChallenge2025-final-release
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
/home/lekangwan/miniconda3/envs/hand-retarget/bin/python \
retarget_research/retargeting/evaluate/evaluate_hand_manifest.py \
--hand wuji \
--manifest retarget_research/manifests/wuji_anatomy_train20_seed20260817.json \
--target-dir retarget_research/outputs/wuji_anatomy_ablation_v1/distal_bound \
--output-dir retarget_research/outputs/wuji_anatomy_ablation_v1_evaluation/distal_bound \
--policy-trace-dir retarget_research/outputs/wuji_anatomy_ablation_v1_traces/distal_bound \
--workers 1 --resume
```

终端C：

```bash
cd /home/lekangwan/projects/DexGraspMotionChallenge2025-final-release
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
/home/lekangwan/miniconda3/envs/hand-retarget/bin/python \
retarget_research/retargeting/evaluate/evaluate_hand_manifest.py \
--hand wuji \
--manifest retarget_research/manifests/wuji_anatomy_train20_seed20260817.json \
--target-dir retarget_research/outputs/wuji_anatomy_ablation_v1/coupled_flexion \
--output-dir retarget_research/outputs/wuji_anatomy_ablation_v1_evaluation/coupled_flexion \
--policy-trace-dir retarget_research/outputs/wuji_anatomy_ablation_v1_traces/coupled_flexion \
--workers 1 --resume
```

## 决策顺序

先检查20条中是否仍有受约束关节低于-5度；越界候选直接淘汰。然后比较稳定30 cm成功数以及相对旧基线的逐轨迹新增/回退，再检查掌物滑移和关键点误差。只有胜出的受约束方法会扩展到train全部50类；不会直接在1000条上同时跑两种方法。

## 已完成结果与晋级决定

五组均在相同train20上完成同步物理解剖限位重放。完整数表和逐轨迹得失见`outputs/wuji_anatomy_ablation_v1_analysis/WUJI_ANATOMY_ABLATION_RESULTS.md`。

- 旧方法稳定13/20、运输12/20，但末段DIP明显反弯比例92.50%，不能作为专家。
- `distal_bound`与`flexion_bounds`均为稳定11/20、运输11/20，平均优化loss均约0.17283；全程DIP明显反弯约4.25%。
- `coupled_flexion`为稳定11/20、运输11/20，loss降至0.16605，全程DIP明显反弯降至2.53%，末段为0；PIP全程明显反弯0.79%。
- `distal_coupled`只有运输10/20，且未硬限制的PIP全程明显反弯达到16.29%，淘汰。
- 硬PD 240/10不提高稳定成功并把运输11/20降到9/20，淘汰。

所有受约束候选的明显DIP越界只出现在前120个接近阶段，闭合、抬升和末段保持均为0。严格低于-5度的统计仍包含PhysX约0.5度以内的数值穿透，所以同时报告严格和低于-5.5度的明显越界，不能把两者混为一谈。

唯一晋级候选冻结为`coupled_flexion`，配置是`wuji_anatomy_coupled_v1.json`、PD仍为120/5。它还不是正式最终方法：下一道门是在train全部50类各生成并重放1条，再对成功案例和初段反弯最严重案例做分层视频复核。
