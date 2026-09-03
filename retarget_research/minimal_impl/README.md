# 灵巧手重定向：最终主线的独立最小实现

这个目录只保留基础重定向任务，进阶策略、PPO和已淘汰方法都不在此处。
它不导入完整实验目录的Python模块，但正常复用题目数据、URDF/mesh、Isaac Gym和
参考工程的底层可求导运动学解析器。

最终流程：

```text
Shadow 70帧轨迹
  ├─ XHand/Wuji：15个语义点的逐帧SLSQP
  └─ Linker O6：15条功能向量 + 接触锚点的逐帧SLSQP
           ↓
    三手统一Global CEM
           ↓
    Linker额外第二轮Global CEM
           ↓
    三手统一Rank-5协同CEM
           ↓
    只对真正修改的候选，基线/候选各单环境重放2次
           ↓
    候选每次都稳定运输且平均分领先>1才接受，否则恢复基线
```

## 文件与阅读顺序

| 顺序 | 文件 | 职责 |
|---:|---|---|
| 1 | `config.py` | 三手维度、关节名、关键点和Linker向量 |
| 2 | `data.py` | 读写NPY/manifest，建立单条`Case` |
| 3 | `kinematics.py` | URDF/MJCF正向运动学：关节角→世界点 |
| 4 | `retarget.py` | 几何损失、SLSQP、Linker接触锚和70帧初值 |
| 5 | `simulate.py` | 动作映射、PhysX重放、成功与抓取质量 |
| 6 | `prepare.py` | 从校准manifest拟合并冻结Rank-5基底 |
| 7 | `cem.py` | Global/Rank-5参数化、CEM搜索和重复确认 |
| 8 | `test_minimal.py` | 无长时仿真的数学和接口回归测试 |

详细方法、选择原因、数据流、公式、消融、结果及面试问答见
[`RETARGETING_LEARNING_GUIDE.md`](../RETARGETING_LEARNING_GUIDE.md)。

## 最小运行方式

首先生成一条运动学初值（Linker要额外传物体目录）：

```bash
conda activate hand-retarget
PYTHONPATH=. python -m retarget_research.minimal_impl.retarget \
  --hand xhand --source /path/to/source.npy --indices 0 \
  --maxeval 50 --output /tmp/xhand_initial.npy
```

然后在`dexgrasp`环境中做Global CEM。该命令启动多次PhysX，通常超过3分钟：

```bash
conda activate dexgrasp
PYTHONPATH=. python -m retarget_research.minimal_impl.cem \
  --hand xhand --stage global --source /path/to/source.npy \
  --target /tmp/xhand_initial.npy --source-index 0 --object-dir /path/to/object \
  --output /tmp/xhand_global.npy --report /tmp/xhand_global.json
```

Rank-5阶段必须使用校准集上预先拟合并冻结的基底，不能从当前测试物体现场拟合：

```bash
PYTHONPATH=. python -m retarget_research.minimal_impl.prepare \
  --manifest /path/to/calibration_manifest.json \
  --target-dir /path/to/global_cem_targets \
  --output /tmp/xhand_rank5_basis.npy

PYTHONPATH=. python -m retarget_research.minimal_impl.cem \
  --hand xhand --stage synergy --source /path/to/source.npy \
  --target /tmp/xhand_global.npy --source-index 0 --object-dir /path/to/object \
  --synergy-basis /tmp/xhand_rank5_basis.npy \
  --output /tmp/xhand_rank5.npy --report /tmp/xhand_rank5.json
```

单条独立重放：

```bash
PYTHONPATH=. python -m retarget_research.minimal_impl.simulate \
  --hand xhand --source /path/to/source.npy --target /tmp/xhand_rank5.npy \
  --source-index 0 --object-dir /path/to/object --output /tmp/replay.json
```

快速测试：

```bash
conda activate hand-retarget
PYTHONPATH=. python retarget_research/minimal_impl/test_minimal.py
```

正式1000条调度、断点续跑和视频录制没有复制到学习版；冻结产物由
`retarget_research/retargeting/configs/final_retargeting_release_v1.json`唯一索引。
