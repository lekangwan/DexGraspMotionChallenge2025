# 进阶任务：目标手闭环策略

本目录记录新考核特有的策略配置、目标手数据适配器和实验结果。可以从旧项目正式的 `custom_tools` 主实现中复用已经验证过的BC、Temporal、数据集和评测组件，但不复用仅用于阅读学习的 `custom_tools/minimal_impl`，也不把基本重定向代码混入 `custom_tools`。

暂定比较顺序：

1. 单帧MLP BC；
2. 短时历史策略；
3. state-conditioned Diffusion Policy动作片段；
4. 仅在时间允许且基线可信时考虑在线数据聚合或残差强化学习。

进阶阶段的数据输入必须来自本项目重定向后经过物理筛选的目标手轨迹，不能直接把未经验证的候选轨迹当专家数据。

## 代码阅读顺序

1. `observations.py`：先看一帧策略到底能看到什么，离线和闭环共用同一个拼接函数。
2. `prepare/build_policy_split.py`：理解为什么按物体而不是按帧随机划分，如何得到同类别未见物体测试集。
3. `prepare/prepare_policy_dataset.py`：看物理trace如何经过成功过滤、映射和仅train归一化变为NPZ。
4. `dataset.py`：看单帧、Temporal3和Diffusion动作片段怎样取窗口，以及如何阻止跨轨迹读取历史。
5. `models.py`：依次看MLP BC、Temporal3和条件动作Diffusion的输入输出。
6. `train.py`：看监督loss、优化器、验证、best/last checkpoint和loss曲线。
7. `runtime.py`：看训练模型如何维护历史、反归一化并在每个物理步输出动作。
8. `evaluate_offline.py`：只检查动作拟合误差，不能当作抓取成功率。
9. `evaluate_policy_isaac.py`：一条轨迹的真实闭环；除首帧张开手腕初态外不再读取专家未来动作。
10. `evaluate_policy_manifest.py`：先用`--split valid`在训练物体的留出轨迹上选择模型，再用`--split test`只在对象级未见物体上汇总最终微平均、物体宏平均和类别宏平均。

## 三种策略的数据流

单帧BC读取当前特权状态，直接回归下一条目标手绝对关节位置。Temporal3额外读取当前和前两步状态、前两步实际发出的动作，用显式短历史判断接近、闭合和抬升阶段。Diffusion读取三步状态，先从高斯噪声生成未来8步动作片段，闭环时只执行前2步就根据新状态重新规划。

训练trace每条为240步：70个20 Hz源轨迹帧在60 Hz物理中各插值3步，再保持末端30步。监督对严格是“新动作执行前的状态→即将执行的动作”。`prepare_policy_dataset.py`会拒绝旧的动作后状态trace，避免模型通过未来信息得到虚假的低loss。

闭环执行支持可选的train分布动作限速：只统计同一train专家轨迹内相邻60 Hz命令的逐维和L2变化，不读取valid/test。倍率0是原始策略；倍率1表示使用train的99.5%分位边界。该层限制的是相邻绝对位置目标跳变，不修改网络权重，也不把专家未来动作喂给策略。

观测包含手的实际DOF位置/速度、物体位置/四元数/线角速度、相对初始位移、距离10 cm抬升目标的剩余量、手物接触数，以及由COACD网格和当前scale计算的14维实例形状描述。形状描述由轴向尺寸3、顶点协方差6、表面积、体积和径向分位数3组成，计算确定且成本很低，使同类别未见实例不再只有类别ID可区分。它仍依赖仿真真值，也没有逐手指到物体表面的DexRep距离场，因此必须明确它是轻量形状增强的特权状态基线，不是相机感知或完整DexRep。

完整命令不在这里重复，统一见`../FORMAL_1000_RUNBOOK.md`第6至9节。

`training_matrix_smoke_v1.json`是真正的链路冒烟：每个模型只执行4个训练batch和
2个验证batch，检查读取、前向、反向、checkpoint和曲线写出。正式矩阵不设置
batch上限，仍会完整遍历每个epoch；冒烟loss不能作为实验结果。
