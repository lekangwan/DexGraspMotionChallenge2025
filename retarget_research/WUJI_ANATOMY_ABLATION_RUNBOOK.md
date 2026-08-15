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
