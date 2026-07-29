# 最终主线代码阅读指南

这份文档只描述最终采用的串行路线。完整实验动机、失败探索和数值结论见
`FINAL_SUBMISSION/PIPELINE_LEARNING_NOTES.md`，但阅读代码时不需要先看那些探索脚本。

## 1. 一张图看懂整个流程

```text
GraspM3原始轨迹 + mesh
        │
        ▼
[A 数据准备] 预处理DexRep观测，冻结80/12/8划分
        │
        ▼
[B BC起点] 单帧BC + 观测噪声，多seed权重平均为BC Soup
        │
        ▼
[C 类别教师] Bottle/Mug/Bowl/Camera各训练一个单帧教师
        │
        ▼
[D 离线学生] 一个共享网络 + 4维Task ID，模仿四个教师
        │
        ▼
[E Online-R1] 学生闭环访问状态，教师在这些状态重新标注
        │
        ▼
[F Temporal3] 从Online-R1初始化，加入前两步本体与动作历史
        │
        ▼
[G 统一评测] Official BC / Soup / Offline / Online-R1 / Temporal3
```

最终推理只加载一个 Temporal3 checkpoint。前面的模型是训练血缘，不是推理时串联的五个网络。

## 2. 推荐阅读顺序

只想理解最终网络时，按以下顺序阅读：

1. `configs/unified_student_taskid_temporal3_v1.yaml`：最终超参数。
2. `task_conditioning.py`：Task ID和三帧历史怎样接入网络。
3. `graspm3_dexrep_dataset.py`：离线和在线样本怎样组成训练输入。
4. `train_bc.py`：动作监督损失和训练循环。
5. `run_taskid_temporal3_stage.py`：初始化、训练和闭环选模。

想理解完整训练血缘时，再依次阅读：

1. `run_scaled_category_expert_training.py`
2. `generate_routed_teacher_labels.py`
3. `run_taskid_offline_stage.py`
4. `collect_taskid_online_scaled20_isolated.py`
5. `run_taskid_online_r1_stage.py`
6. `collect_taskid_online_r2_scaled20_isolated.py`
7. `merge_taskid_online_rounds.py`
8. `run_taskid_temporal3_stage.py`

## 3. 每一阶段的输入和输出

### A. 数据准备

推荐入口：

- `preprocess_graspm3_isolated.py`
- `stage_final_preprocessed_split.py`
- `prepare_bc_dataset.py`

作用：把28维成功抓取轨迹放入Isaac Gym，提取每帧本体状态和DexRep特征，并按整条轨迹划分训练/内部验证。分块进程只为解决8 GB显存上的PhysX累积问题，不改变样本定义。

主要输出：

```text
dexgrasp/dataset/scaled_category_final_v1_preprocessed/
dexgrasp/dataset/scaled_bc20_train_v1/
dexgrasp/dataset/scaled_bc20_valid_v1/
```

冻结清单是 `configs/scaled_category_split_final_v1.json`。

### B. BC Soup

核心训练入口是 `train_bc.py`，配置是
`configs/multicategory_bc_noise005.yaml`；`make_bc_model_soup.py`对两个独立训练
checkpoint做参数加权平均。

这一阶段仍是单帧28维动作回归。观测噪声用于提高学生偏离示范状态后的局部鲁棒性；Soup降低单次随机初始化带来的参数波动。

### C. 四个类别教师

入口：`run_scaled_category_expert_training.py`。

四个教师网络结构相同，但各自只学习一个类别。它们从BC Soup初始化，使用每类20个物体的数据。开发集负责选择每类checkpoint；固定选择结果写入后续阶段脚本和最终哈希清单。

### D. 离线统一学生

入口：`run_taskid_offline_stage.py`。

1. `generate_routed_teacher_labels.py`根据物体类别查询相应教师；
2. `task_conditioning.py`在共享特征后拼接四维one-hot Task ID；
3. `train_bc.py`让一个学生预测教师动作。

最终离线节点使用100%教师动作目标。配置文件仍能表达70/30对照，但最终checkpoint身份以
`configs/comprehensive_five_model_evaluation_v1.yaml`为准。

### E. Online-R1

采集入口：`collect_taskid_online_scaled20_isolated.py`。
训练入口：`run_taskid_online_r1_stage.py`。

学生动作真正推进仿真；在学生访问到的状态上，类别教师给出监督动作。训练批次把在线状态重采样到25%，其余75%来自离线教师数据。这是DAgger式数据聚合，不是第二个并行网络。

### F. Temporal3

第二批在线状态由 `collect_taskid_online_r2_scaled20_isolated.py` 在不同训练轨迹上采集，再由 `merge_taskid_online_rounds.py` 合并。

入口：`run_taskid_temporal3_stage.py`。

最终输入关系：

```text
当前DexRep/本体观测 → 官方DexRep编码器 → 384维当前特征
当前Task ID                                  4维
前两步本体状态                         2 × 100维
前两步实际动作                          2 × 28维
------------------------------------------------
拼接后动作网络输入                          644维
共享MLP                           1024-1024-512-512
输出                                         28维
```

Temporal3从Online-R1初始化。历史是固定顺序拼接，不是GRU或Transformer；推理时历史动作来自网络自己前两步实际执行的动作。

### G. 统一闭环评测

入口：`run_comprehensive_five_model_evaluation.py`。

它固定五个checkpoint、三种数据划分和三个PhysX seed，最终成功指标仍是官方任务的峰值成功标志。底层调用关系为：

```text
run_comprehensive_five_model_evaluation.py
  → evaluate_bc_checkpoints_isolated.py  每个物体独立CUDA进程
  → evaluate_bc_checkpoints_batched.py   创建仿真并执行零残差BC
  → evaluate_bc.py / shadow_hand_grasp_dexrep_custom.py
  → 官方dexgrasp Isaac Gym任务
```

评测代码复用 `residual_env.py` 和 `train_residual_ppo.py` 中的环境创建函数，但动作残差固定为零。它们在这里是仿真支持代码，不表示最终方法使用PPO。

## 4. 代码分层

### 必须先读的算法层

```text
train_bc.py
graspm3_dexrep_dataset.py
task_conditioning.py
run_taskid_*_stage.py
```

### 只在运行时调用的仿真支持层

```text
evaluate_bc.py
evaluate_bc_checkpoints_*.py
evaluation_loop.py
residual_env.py
train_residual_ppo.py
shadow_hand_grasp_dexrep_custom.py
```

支持层较长，主要处理Isaac Gym初始化、局部重置、显存隔离和指标统计。第一次学习主线时可以先不逐行阅读。

### 回归测试

```text
test_task_conditioning.py
test_notask_temporal.py
test_preprocess_chunking.py
test_residual_ppo.py
```

测试用于防止Task ID、历史缓存、分块索引和评测包装在精简后发生行为变化。

## 5. 选模规则

- 训练数据只更新参数；
- Development12可重复用于checkpoint和超参数选择；
- Final8只在模型锁定后报告；
- 主要指标是逐物体成功率再宏平均；
- 相同候选使用2025、2026、2027三个仿真seed；
- 离线动作loss、教师学生MAE和PPO reward都不能替代闭环官方成功率。

最终模型路径和SHA256见
`configs/comprehensive_five_model_evaluation_v1.yaml` 与
`FINAL_SUBMISSION/manifests/checkpoints.sha256`。
