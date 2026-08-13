# 正式50类/100物体/1000轨迹运行手册

完整数据已在旧项目只读目录中确认，正式manifest使用对象ID内嵌的`core/sem`显式类别标签生成。所有长命令均支持续跑；不要改用参考仓库的41物体示例覆盖正式名单。

## 0. 数据到位后的环境预检

CPU PhysX是基本重定向的合法正式路径；只有准备在GPU上训练策略时，才额外加`--require-cuda`：

```bash
/home/lekangwan/miniconda3/envs/hand-retarget/bin/python \
  retarget_research/scripts/preflight.py \
  --reference-root retarget_research/reference/HandRetargetTask2026 \
  --dataset-root /home/lekangwan/projects/DexGraspMotionChallenge2025/external_data/dataset \
  --asset-root /home/lekangwan/projects/DexGraspMotionChallenge2025/external_data/meshdata \
  --output retarget_research/outputs/formal_1000/preflight.json
```

必须显示`RETARGET_PREFLIGHT=PASS`。预检会扫描全部轨迹文件，不再只抽前若干个。

## 1. 填写并冻结manifest

本协议不猜测`mujoco/ddg`商品名类别，只从`core-类别-ID`和`sem-类别-ID`读取显式类别段，并按大小写合并。类别表已由下面命令生成：

```bash
/home/lekangwan/miniconda3/envs/hand-retarget/bin/python \
  retarget_research/scripts/build_embedded_category_map.py \
  --trajectory-root /home/lekangwan/projects/DexGraspMotionChallenge2025/external_data/dataset \
  --asset-root /home/lekangwan/projects/DexGraspMotionChallenge2025/external_data/meshdata \
  --output retarget_research/manifests/category_map.csv \
  --audit-output retarget_research/manifests/category_map_audit.json \
  --minimum-trajectories 10 --minimum-objects-per-category 2
```

随后自动匹配轨迹和同名资产、读取真实轨迹数，生成五列inventory：

```bash
/home/lekangwan/miniconda3/envs/hand-retarget/bin/python \
  retarget_research/scripts/build_inventory.py \
  --category-map retarget_research/manifests/category_map.csv \
  --trajectory-root /home/lekangwan/projects/DexGraspMotionChallenge2025/external_data/dataset \
  --asset-root /home/lekangwan/projects/DexGraspMotionChallenge2025/external_data/meshdata \
  --output retarget_research/manifests/inventory.csv \
  --audit-output retarget_research/manifests/inventory_audit.json
```

生成的`manifests/inventory.csv`每行包含：

```text
object_id,category,trajectory_file,asset_dir,trajectory_count
```

其中`category`必须来自数据集标签，`trajectory_file`和`asset_dir`建议填写绝对路径。随后运行：

```bash
/home/lekangwan/miniconda3/envs/hand-retarget/bin/python \
  retarget_research/scripts/build_manifest.py \
  --inventory retarget_research/manifests/inventory.csv \
  --output retarget_research/manifests/formal_50c_100o_1000t_seed20260808.json \
  --seed 20260808 \
  --categories 50 \
  --objects-per-category 2 \
  --trajectories-per-object 10 \
  --calibration-per-object 2
```

脚本会检查每个npy严格为`(N,70,28)`、轨迹级物体字段齐全、资产存在，并冻结文件哈希。输出必须显示`categories=50 objects=100 trajectories=1000`。

随后立即冻结本次实验合同。这个命令会重新计算100个源文件哈希，并记录manifest、三手方法配置和当前实现代码的哈希；正式结果出现后若代码变化，验收会直接失败：

```bash
/home/lekangwan/miniconda3/envs/hand-retarget/bin/python \
  retarget_research/scripts/freeze_formal_experiment.py \
  --experiment-config retarget_research/configs/formal_experiment_v1.json \
  --output retarget_research/manifests/formal_50c_100o_1000t_seed20260808.lock.json
```

先只验收输入门：

```bash
/home/lekangwan/miniconda3/envs/hand-retarget/bin/python \
  retarget_research/scripts/verify_formal_bundle.py \
  --experiment-config retarget_research/configs/formal_experiment_v1.json \
  --lock retarget_research/manifests/formal_50c_100o_1000t_seed20260808.lock.json \
  --stage inputs \
  --output retarget_research/outputs/formal_1000/input_audit.json
```

## 2. 生成三只手候选（长命令）

下面三条互不依赖，可以放在不同终端运行；如果CPU或内存压力过大则顺序运行。开发机实测10条约需5–7分钟生成，因此1000条预计为数小时，必须由用户终端执行。

### Linker O6固定基线

```bash
MPLCONFIGDIR=/tmp/matplotlib-retarget OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
/home/lekangwan/miniconda3/envs/hand-retarget/bin/python \
  retarget_research/retargeting/run/run_linker_manifest.py \
  --manifest retarget_research/manifests/formal_50c_100o_1000t_seed20260808.json \
  --output-dir retarget_research/outputs/formal_1000/linker_no_temporal_baseline_v2 \
  --workers 2 --resume --maxeval 100 --include-thumb-middle \
  --joint-temporal-weight 0 \
  --translation-temporal-weight 0 \
  --rotation-temporal-weight 0
```

方法选择消融中，无时序为9/20，优于关节/平移/旋转单项的7/8/7和完整组合的8/20，因此正式运行显式固定三项为0，而不是依赖默认值。

### Linker O6冻结渐进夹紧（依赖O6基线）

该步骤不重新做SLSQP，每条轨迹只根据Shadow接触阶段对原6维主动关节增加同一组闭合残差，通常远快于基线生成：

```bash
MPLCONFIGDIR=/tmp/matplotlib-retarget OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
/home/lekangwan/miniconda3/envs/hand-retarget/bin/python \
  retarget_research/retargeting/run/refine_linker_squeeze.py \
  --manifest retarget_research/manifests/formal_50c_100o_1000t_seed20260808.json \
  --baseline-dir retarget_research/outputs/formal_1000/linker_no_temporal_baseline_v2 \
  --output-dir retarget_research/outputs/formal_1000/linker_o6_optimized_v2 \
  --method-name linker_o6_no_temporal_dynamic_squeeze_v2 \
  --thumb-yaw-delta 0.075 \
  --thumb-pitch-delta 0.1875 \
  --finger-delta 0.425 \
  --contact-threshold 0.02 \
  --min-contact-tips 2 \
  --lift-delta 0.03
```

早期旧参数下的受力再分配残差和自适应PD均未通过20条泛化检查，而且与当前v2输入轨迹不兼容，因此不进入正式1000条手册；复现信息仍保留在`WORK_LOG.md`。

### Wuji v1固定单流程

```bash
MPLCONFIGDIR=/tmp/matplotlib-retarget OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
/home/lekangwan/miniconda3/envs/hand-retarget/bin/python \
  retarget_research/retargeting/run/run_wuji_manifest.py \
  --manifest retarget_research/manifests/formal_50c_100o_1000t_seed20260808.json \
  --output-dir retarget_research/outputs/formal_1000/wuji_v1 \
  --workers 2 --resume --maxeval 50 \
  --mapping-config retarget_research/retargeting/configs/wuji_keypoint_map.json \
  --joint-temporal-weight 0 \
  --translation-temporal-weight 0 \
  --rotation-temporal-weight 0
```

### XHand官方几何基线

```bash
MPLCONFIGDIR=/tmp/matplotlib-retarget OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
/home/lekangwan/miniconda3/envs/hand-retarget/bin/python \
  retarget_research/retargeting/run/run_xhand_manifest.py \
  --manifest retarget_research/manifests/formal_50c_100o_1000t_seed20260808.json \
  --output-dir retarget_research/outputs/formal_1000/xhand_official \
  --workers 2 --resume --iter-num 100 --sample-frame-num 5 \
  --trans-lr 0.005 --ang-lr 0.01 --trans-bound 2 --device cpu
```

## 3. XHand真实指腹细化（依赖XHand官方基线）

```bash
MPLCONFIGDIR=/tmp/matplotlib-retarget OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
/home/lekangwan/miniconda3/envs/hand-retarget/bin/python \
  retarget_research/retargeting/run/run_xhand_contact_manifest.py \
  --manifest retarget_research/manifests/formal_50c_100o_1000t_seed20260808.json \
  --baseline-dir retarget_research/outputs/formal_1000/xhand_official \
  --output-dir retarget_research/outputs/formal_1000/xhand_phase_contact_v2 \
  --contact-pad-config retarget_research/retargeting/configs/xhand_contact_pads_v1.json \
  --workers 2 --resume --maxeval 20 \
  --contact-weight 5 --normal-weight 0.05 --penetration-weight 2 \
  --joint-prior-weight 0 --contact-threshold 0.02 \
  --min-contact-tips 2 --lift-delta 0.03 --region-neighbors 32 \
  --contact-offset -0.003 --min-signed-distance -0.006
```

## 4. 统一物理评估（长命令）

候选全部完成后，以下四条互不依赖。每条都会输出轨迹微平均、物体宏平均、类别宏平均，以及每物体2条calibration/8条heldout的分开结果。XHand必须同时评估官方几何基线和我们的指腹细化，不能只报改进方法。

```bash
/home/lekangwan/miniconda3/envs/hand-retarget/bin/python \
  retarget_research/retargeting/evaluate/evaluate_hand_manifest.py \
  --hand linker \
  --manifest retarget_research/manifests/formal_50c_100o_1000t_seed20260808.json \
  --target-dir retarget_research/outputs/formal_1000/linker_o6_optimized_v2 \
  --output-dir retarget_research/outputs/formal_1000/linker_o6_optimized_v2_evaluation \
  --policy-trace-dir retarget_research/advanced_policy/traces/formal_v1/linker \
  --workers 2 \
  --linker-finger-stiffness 120 --linker-finger-damping 5 \
  --linker-mimic-stiffness 120 --linker-mimic-damping 5

/home/lekangwan/miniconda3/envs/hand-retarget/bin/python \
  retarget_research/retargeting/evaluate/evaluate_hand_manifest.py \
  --hand wuji \
  --manifest retarget_research/manifests/formal_50c_100o_1000t_seed20260808.json \
  --target-dir retarget_research/outputs/formal_1000/wuji_v1 \
  --output-dir retarget_research/outputs/formal_1000/wuji_v1_evaluation \
  --policy-trace-dir retarget_research/advanced_policy/traces/formal_v1/wuji \
  --workers 2

/home/lekangwan/miniconda3/envs/hand-retarget/bin/python \
  retarget_research/retargeting/evaluate/evaluate_hand_manifest.py \
  --hand xhand \
  --manifest retarget_research/manifests/formal_50c_100o_1000t_seed20260808.json \
  --target-dir retarget_research/outputs/formal_1000/xhand_official \
  --output-dir retarget_research/outputs/formal_1000/xhand_official_evaluation \
  --policy-trace-dir retarget_research/advanced_policy/traces/formal_v1/xhand_official \
  --workers 2

/home/lekangwan/miniconda3/envs/hand-retarget/bin/python \
  retarget_research/retargeting/evaluate/evaluate_hand_manifest.py \
  --hand xhand \
  --manifest retarget_research/manifests/formal_50c_100o_1000t_seed20260808.json \
  --target-dir retarget_research/outputs/formal_1000/xhand_phase_contact_v2 \
  --output-dir retarget_research/outputs/formal_1000/xhand_phase_contact_v2_evaluation \
  --policy-trace-dir retarget_research/advanced_policy/traces/formal_v1/xhand \
  --workers 2
```

三手当前主方法为：Linker无时序渐进夹紧v2+固定PD 120/5，XHand指腹细化v2，Wuji单次v1映射、maxeval50和无时序约束。Wuji 50/100/150均为16/20且成功集合相同，所以选择成本最低的50；时序0/0.1/0.3/1.0倍也均为16/20且没有任何配对新增或回退，非零强度只降低跳变并轻微降低最终抬升，因此正式保持0倍。XHand官方几何方法是必须保留的对照组，它不参与“主方法三选三”，但必须在同一manifest、同一仿真参数和同一成功判据下单独重放。报告时优先给出8条/物体的heldout统计，并把2条/物体的calibration统计分开列出。

XHand报告表中至少并排给出官方基线和指腹细化的成功轨迹数、轨迹微平均、物体宏平均、类别宏平均及配对成功变化（新增成功/丢失成功）。

两个XHand评测都完成后，用下面的短命令按“物体名+源轨迹索引”严格配对比较：

```bash
/home/lekangwan/miniconda3/envs/hand-retarget/bin/python \
  retarget_research/retargeting/evaluate/compare_method_summaries.py \
  --baseline-summary retarget_research/outputs/formal_1000/xhand_official_evaluation/manifest_evaluation_summary.json \
  --improved-summary retarget_research/outputs/formal_1000/xhand_phase_contact_v2_evaluation/manifest_evaluation_summary.json \
  --output retarget_research/outputs/formal_1000/xhand_official_vs_phase_contact_v2.json
```

每阶段先检查对应目录中的`manifest_run_summary.json`或`manifest_evaluation_summary.json`。候选生成摘要必须为`all_successful=true`且数量为100物体/1000轨迹；评测摘要必须为`trajectory_count=1000`，才能进入报告统计和视频选择。

## 5. 自动验收候选、评测和策略trace

四套候选完成后检查`candidates`；四套物理评测完成后同时检查`evaluations`和四套`traces`。三套主方法trace用于策略数据，XHand官方trace只用于软件渲染公平对照。以下命令只读取产物，但会完整打开候选和trace，预计可能超过3分钟，仍由用户终端运行：

```bash
/home/lekangwan/miniconda3/envs/hand-retarget/bin/python \
  retarget_research/scripts/verify_formal_bundle.py \
  --experiment-config retarget_research/configs/formal_experiment_v1.json \
  --lock retarget_research/manifests/formal_50c_100o_1000t_seed20260808.lock.json \
  --stage candidates \
  --output retarget_research/outputs/formal_1000/candidate_audit.json

/home/lekangwan/miniconda3/envs/hand-retarget/bin/python \
  retarget_research/scripts/verify_formal_bundle.py \
  --experiment-config retarget_research/configs/formal_experiment_v1.json \
  --lock retarget_research/manifests/formal_50c_100o_1000t_seed20260808.lock.json \
  --stage evaluations --stage traces \
  --output retarget_research/outputs/formal_1000/evaluation_trace_audit.json
```

trace固定为每条240个60 Hz物理步：70个源帧×每帧3步，再保持30步。每个监督对是“执行新命令前的状态→即将执行的命令”，不是动作后的状态；数据准备脚本会拒绝没有`pre_action_state_to_command_v1`标签的旧文件。

## 6. 冻结进阶策略的对象级划分

每类两个物体中，一个只用于train/valid，另一个只用于同类别未见物体test。角色由固定哈希决定，与三只手的物理成败无关，防止用结果挑容易测试物体：

```bash
/home/lekangwan/miniconda3/envs/hand-retarget/bin/python \
  retarget_research/advanced_policy/prepare/build_policy_split.py \
  --manifest retarget_research/manifests/formal_50c_100o_1000t_seed20260808.json \
  --output retarget_research/advanced_policy/data/formal_v1/policy_split_seed20260813.json \
  --seed 20260813
```

预期为50个训练物体、50个测试物体；每类训练物体原8条heldout作为策略train，原2条calibration作为valid，测试物体10条全部只用于最终test。因此未过滤前数量为400/100/500条。

## 7. 物化三只手的策略数据

三条命令互不依赖，可以并行。train/valid默认只保留严格物理成功的重定向轨迹，test保持500条完整不筛选；归一化均值和方差只由train步骤计算：

```bash
/home/lekangwan/miniconda3/envs/hand-retarget/bin/python \
  retarget_research/advanced_policy/prepare/prepare_policy_dataset.py \
  --manifest retarget_research/manifests/formal_50c_100o_1000t_seed20260808.json \
  --policy-split retarget_research/advanced_policy/data/formal_v1/policy_split_seed20260813.json \
  --hand linker \
  --trace-dir retarget_research/advanced_policy/traces/formal_v1/linker \
  --evaluation-summary retarget_research/outputs/formal_1000/linker_o6_optimized_v2_evaluation/manifest_evaluation_summary.json \
  --output-dir retarget_research/advanced_policy/data/formal_v1/linker

/home/lekangwan/miniconda3/envs/hand-retarget/bin/python \
  retarget_research/advanced_policy/prepare/prepare_policy_dataset.py \
  --manifest retarget_research/manifests/formal_50c_100o_1000t_seed20260808.json \
  --policy-split retarget_research/advanced_policy/data/formal_v1/policy_split_seed20260813.json \
  --hand xhand \
  --trace-dir retarget_research/advanced_policy/traces/formal_v1/xhand \
  --evaluation-summary retarget_research/outputs/formal_1000/xhand_phase_contact_v2_evaluation/manifest_evaluation_summary.json \
  --output-dir retarget_research/advanced_policy/data/formal_v1/xhand

/home/lekangwan/miniconda3/envs/hand-retarget/bin/python \
  retarget_research/advanced_policy/prepare/prepare_policy_dataset.py \
  --manifest retarget_research/manifests/formal_50c_100o_1000t_seed20260808.json \
  --policy-split retarget_research/advanced_policy/data/formal_v1/policy_split_seed20260813.json \
  --hand wuji \
  --trace-dir retarget_research/advanced_policy/traces/formal_v1/wuji \
  --evaluation-summary retarget_research/outputs/formal_1000/wuji_v1_evaluation/manifest_evaluation_summary.json \
  --output-dir retarget_research/advanced_policy/data/formal_v1/wuji
```

先阅读每只手的`dataset_summary.json`。若状态是`ready_with_gaps`，表示成功轨迹过滤后某些类别在train中为0；不要隐瞒，也不要根据test结果重选对象。可以将“成功轨迹不足”作为重定向质量对进阶策略的限制写入报告。随后运行策略数据门：

```bash
/home/lekangwan/miniconda3/envs/hand-retarget/bin/python \
  retarget_research/scripts/verify_formal_bundle.py \
  --experiment-config retarget_research/configs/formal_experiment_v1.json \
  --stage policy_data \
  --output retarget_research/advanced_policy/data/formal_v1/policy_data_audit.json
```

## 8. 先做1 epoch CPU冒烟，再开始正式训练

先生成并运行小网络配置，确认三手×三模型共9条训练链都能读数据、反向传播、写checkpoint和画曲线。该冒烟不是实验结果：

```bash
/home/lekangwan/miniconda3/envs/hand-retarget/bin/python \
  retarget_research/advanced_policy/prepare/generate_training_configs.py \
  --matrix retarget_research/advanced_policy/configs/training_matrix_smoke_v1.json \
  --output-dir retarget_research/advanced_policy/configs/generated/smoke_v1

/home/lekangwan/miniconda3/envs/hand-retarget/bin/python \
  retarget_research/advanced_policy/run_training_matrix.py \
  --index retarget_research/advanced_policy/configs/generated/smoke_v1/config_index.json \
  --device cpu
```

冒烟全部成功后生成正式9个配置。正式训练预计超过3分钟，必须由用户终端运行；同一命令中断后重跑会自动读取每个实验的`last.pt`续训：

```bash
/home/lekangwan/miniconda3/envs/hand-retarget/bin/python \
  retarget_research/advanced_policy/prepare/generate_training_configs.py \
  --matrix retarget_research/advanced_policy/configs/training_matrix_v1.json \
  --output-dir retarget_research/advanced_policy/configs/generated/formal_v1

/home/lekangwan/miniconda3/envs/hand-retarget/bin/python \
  retarget_research/advanced_policy/run_training_matrix.py \
  --index retarget_research/advanced_policy/configs/generated/formal_v1/config_index.json
```

正式矩阵是三只手各自的单帧BC、Temporal3和条件动作Diffusion，共9个实验。它们使用同一对象split、同一成功专家过滤规则和同一150 epoch上限；以valid loss选`best.pt`，早停耐心20，保留`last.pt`做续训。Diffusion使用3帧状态条件生成8步动作，默认闭环每执行2步重新规划。

## 9. 先离线诊断，再做未见物体闭环评测

离线评估示例（把手和模型名替换为其余8组）：

```bash
/home/lekangwan/miniconda3/envs/hand-retarget/bin/python \
  retarget_research/advanced_policy/evaluate_offline.py \
  --checkpoint retarget_research/advanced_policy/runs/formal_v1/xhand_bc_v1/best.pt \
  --data-dir retarget_research/advanced_policy/data/formal_v1/xhand \
  --output retarget_research/advanced_policy/runs/formal_v1/xhand_bc_v1/offline_test.json \
  --device cuda
```

离线MAE/RMSE只说明动作拟合，不是抓取成功率。最终闭环示例：

```bash
/home/lekangwan/miniconda3/envs/hand-retarget/bin/python \
  retarget_research/advanced_policy/evaluate_policy_manifest.py \
  --hand xhand \
  --manifest retarget_research/manifests/formal_50c_100o_1000t_seed20260808.json \
  --policy-split retarget_research/advanced_policy/data/formal_v1/policy_split_seed20260813.json \
  --target-dir retarget_research/outputs/formal_1000/xhand_phase_contact_v2 \
  --checkpoint retarget_research/advanced_policy/runs/formal_v1/xhand_bc_v1/best.pt \
  --data-dir retarget_research/advanced_policy/data/formal_v1/xhand \
  --output-dir retarget_research/advanced_policy/runs/formal_v1/xhand_bc_v1/closed_loop_test \
  --device cuda --workers 1 --resume
```

闭环只运行对象级test中的50个未见物体×10条，共500条；训练物体绝不进入最终成功率。每条只从专家候选首帧取得抓取方向对应的手腕初态，并把手指张开；之后240个物理步完全由policy闭环输出。报告必须给轨迹微平均、物体宏平均和类别宏平均，并同时展示训练loss曲线、成功/失败视频。其余手和模型使用完全同构命令，不能只挑最好模型运行测试。

## 10. 自动选择案例并按需渲染视频

不要人工只挑最漂亮的成功。选择器会优先保持类别多样，并分别选严格成功、接近阈值、抬起后滑落和低抬升失败；同一轨迹不会重复进入多个组。专家重放示例：

```bash
/home/lekangwan/miniconda3/envs/hand-retarget/bin/python \
  retarget_research/scripts/select_report_cases.py \
  --summary retarget_research/outputs/formal_1000/xhand_phase_contact_v2_evaluation/manifest_evaluation_summary.json \
  --output retarget_research/reports/cases/xhand_expert_cases.json \
  --count-per-group 2

/home/lekangwan/miniconda3/envs/hand-retarget/bin/python \
  retarget_research/scripts/render_selected_cases.py \
  --selection retarget_research/reports/cases/xhand_expert_cases.json \
  --manifest retarget_research/manifests/formal_50c_100o_1000t_seed20260808.json \
  --output-dir retarget_research/reports/videos/xhand_expert
```

第二条默认只打印并保存渲染计划；核对命令后原样加`--execute`才会顺序生成MP4。默认`--renderer auto`：只要专家trace或新策略状态齐全，就优先用完全CPU的URDF骨架+真实物体网格后备，不依赖图形驱动；显式`--renderer isaac`才重跑实体手网格画面。策略案例把`--summary`换成对应模型的`closed_loop_test/policy_evaluation_summary.json`，Isaac渲染时再加与正式评测相同的`--device cuda`。

Isaac录像模式才会启用graphics device，普通1000条评测继续使用无图形CPU PhysX。默认视频为640×480、20 fps、80帧/4秒；已在本机用真实XHand成功轨迹生成Isaac H.264 MP4并读取首/中/末关键帧验证。当前无CUDA的软件Vulkan偶尔会段错误，因此报告至少先用默认CPU后备稳定生成：它读取同一240步trace，以NumPy做三手URDF前向运动学并叠加真实COACD物体位姿；本机同一轨迹约8秒生成80帧视频。之后如有可用NVIDIA驱动，再用`--renderer isaac`补更美观的实体网格版。

## 11. 导出报告结果表

基本任务四套摘要完成后用一条命令生成可直接粘贴的Markdown主表、完整CSV和逐类别长表：

```bash
/home/lekangwan/miniconda3/envs/hand-retarget/bin/python \
  retarget_research/scripts/export_result_tables.py \
  --summary linker=retarget_research/outputs/formal_1000/linker_o6_optimized_v2_evaluation/manifest_evaluation_summary.json \
  --summary wuji=retarget_research/outputs/formal_1000/wuji_v1_evaluation/manifest_evaluation_summary.json \
  --summary xhand_official=retarget_research/outputs/formal_1000/xhand_official_evaluation/manifest_evaluation_summary.json \
  --summary xhand_ours=retarget_research/outputs/formal_1000/xhand_phase_contact_v2_evaluation/manifest_evaluation_summary.json \
  --output-markdown retarget_research/reports/basic_result_table.md \
  --output-csv retarget_research/reports/basic_result_table.csv \
  --per-category-csv retarget_research/reports/basic_per_category.csv
```

进阶任务同样把9个`policy_evaluation_summary.json`分别作为`--summary 标签=路径`传入。导出器不会把三种平均混成一个数：成功数/轨迹微平均、物体宏平均和类别宏平均分别成列；报告正文应说明主口径，并把表中分母保留下来。
