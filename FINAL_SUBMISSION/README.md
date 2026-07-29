# DexGrasp 最终整理版

本目录是项目最终主线的精简索引，集中保存报告、核心代码、锁定配置、选模证据和统一评测结果。它不会替代仓库根目录，也没有复制数据集和 checkpoint；实际运行仍应回到仓库根目录执行。

## 1. 最终方法

训练链为：

```text
官方单帧 BC
  → 噪声 BC 权重平均（BC Soup）
  → 四个类别教师
  → 离线统一 Task-ID 学生
  → Online-R1 在线模仿
  → Temporal3
```

最终推理时只运行一个 Temporal3 网络，不会串行调用前面的 Soup、教师或学生。

Temporal3 的当前观测包含 DexRep、本体状态和四维类别 Task ID，并额外拼接前两步的100维本体状态和28维实际动作。三帧信息直接进入共享 MLP，输出28维动作；没有使用 GRU、Transformer 或 Diffusion Policy。

## 2. 数据划分

| 划分 | 规模 | 用途 |
|---|---:|---|
| 优化训练集 | 80个物体、1726条轨迹 | 更新网络参数 |
| Seen80独立轨迹 | 80个见过的物体、434条轨迹 | 检查闭环复现，不更新参数 |
| Development12 | 12个未见实例、313条轨迹 | checkpoint和超参数选择，结果有选择偏差 |
| Final8 | 8个未见实例、216条轨迹 | 最终报告，不再用于选模 |

四个类别均为 Bottle、Mug、Bowl 和 Camera。Final8 是训练未使用过的同类别新物体，不是四个全新类别。

## 3. 最重要的统一评测

下表为三个仿真 seed 的物体宏平均官方峰值成功率：

| 模型 | Seen80 | Development12 | Final8 |
|---|---:|---:|---:|
| 同数据官方 BC | 9.47% | 16.19% | 8.77% |
| BC Soup | 14.90% | 17.01% | 18.68% |
| 离线 Task-ID 学生 | 21.05% | 22.06% | 14.87% |
| Online-R1 | 22.36% | 32.80% | 17.46% |
| **Temporal3** | **26.17%** | **37.63%** | **26.15%** |

Temporal3 相比主要公平基线“同数据官方 BC”，Final8 宏成功率从8.77%提高到26.15%，绝对提升17.38个百分点，约为2.98倍。官方发布的单物体 checkpoint 直接迁移到 Final8 为3.30%，只作为额外参考。

成功率仍然不高。Seen80和Final8都约26%，说明主要瓶颈不是新物体几何泛化，而是单帧示范附近学到的动作难以在闭环执行中稳定复现，偏离后也缺少可靠恢复监督。

## 4. Task ID 结论

在 Development12 上：

| 节点 | 无 Task ID | 有 Task ID |
|---|---:|---:|
| 离线学生 | 22.07% | 22.06% |
| Online-R1 | 25.87% | 32.80% |
| Temporal3 | 25.89% | 37.63% |

Task ID 在离线蒸馏阶段几乎无影响，但在在线状态和时序历史加入后明显有效。合理解释是：离线成功轨迹中的几何和状态已能暗示类别；学生偏离示范后，显式任务条件能减少四类纠错动作之间的冲突。Task ID 是统一网络的输入条件，不是硬切换四个网络。

## 5. 目录说明

- `EXPERIMENT_REPORT.pdf`：提交用1–2页实验报告。
- `EXPERIMENT_REPORT.md`：报告可编辑源文件。
- `OFFICIAL_BC_COMPARISON.md`：官方方法与本文方法的公平对比。
- `PIPELINE_LEARNING_NOTES.md`：完整结构、数据、实验探索和结论。
- `custom_tools/configs/`：最终主线、无Task-ID对照、数据划分和评测锁定配置。
- `custom_tools/*.py`：主线训练、在线数据聚合和统一评测的核心代码快照。
- `custom_tools/results/comprehensive_five_model_evaluation_v1/`：五模型80/12/8统一评测。
- `custom_tools/results/taskid_ablation_report_v1/`：Task ID消融。
- `custom_tools/results/taskid_final_report_assets_v1/`：报告图表与汇总表。
- `renders/`：四个类别各一段成功和失败视频，以及自动筛选记录。
- `manifests/checkpoints.sha256`：最终保留权重的路径和哈希。
- `REPRODUCE.md`：检查和复现实验的命令。

渲染案例不是仅凭肉眼挑选：成功案例要求锁定 Temporal3 的官方成功标志为真，并检查抬升过程；失败案例要求官方成功标志为假。`renders/selection_summary.yaml` 保存物体、轨迹索引、最大抬升和最终抬升等选择依据。

## 6. 使用边界

本目录中的代码是便于审阅的快照，不是一份脱离原仓库即可独立运行的副本。Isaac Gym、官方 `dexgrasp` 环境、mesh、预处理数据、在线模仿数据和 checkpoint 仍位于原仓库对应路径。这样避免复制数十GB数据，也避免两份代码继续分叉。
