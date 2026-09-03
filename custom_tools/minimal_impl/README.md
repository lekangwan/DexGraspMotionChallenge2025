# 独立最小实现

这个目录实现当前最终主线，不导入目录外的旧 `custom_tools` 模块。它继续复用官方
`dexgrasp/` 环境、物体mesh和轨迹数据，因为这些是任务本身，不是本项目新增的实验代码。
这里的“独立”是指不调用完整版 `custom_tools`，不是把Isaac Gym、官方任务和数据集
重新复制一份。

当前方法：

```text
2460维策略观测（100维本体 + 2360维DexRep）+ 4维类别 + 两步本体/动作历史
→ 1024-1024-512-512 MLP
→ 未来8步28维动作
→ 重叠等权时间集成
→ 第40到69步腕部z补偿从0线性增至0.20
```

文件分工：

| 文件 | 作用 |
|---|---|
| `model.py` | DexRep编码、Temporal3、Chunk8、时间集成、损失、checkpoint |
| `data.py` | 离线/在线轨迹、历史、未来动作块、均衡采样 |
| `project_data.py` | 读取真实NPY和在线NPZ |
| `train.py` | 普通PyTorch监督训练 |
| `prepare.py` | BC Soup和类别教师标签 |
| `simulate.py` | 官方Isaac Gym评测与在线采集 |
| `test_minimal.py` | 无GPU快速回归测试 |

完整讲解见上一级 [`LEARNING_GUIDE.md`](../LEARNING_GUIDE.md)。

快速测试：

```bash
PYTHONPATH=. python3 custom_tools/minimal_impl/test_minimal.py
```

训练入口：

```bash
PYTHONPATH=. python3 -m custom_tools.minimal_impl.train \
  --offline-dir /path/to/scaled_bc20_train_v1 \
  --teacher-actions /path/to/routed_teacher.npz \
  --online-actions /path/to/online_r1_r2.npz \
  --init-checkpoint custom_tools/checkpoints/ablation/temporal3_demo80.ckpt \
  --output /tmp/shadow_chunk8.ckpt
```

查看Isaac Gym评测参数：

```bash
PYTHONPATH=. python3 -m custom_tools.minimal_impl.simulate --help
```

最小实现从“已经完成DexRep预处理的训练轨迹”开始。原始轨迹转DexRep需要创建官方
Isaac Gym环境，属于数据前处理而不是策略本身；为避免复制上千行环境封装，本目录不重新
实现官方环境。

评测多个物体时，当前仓库自带的 `assets/meshdata` 可能不完整，需要显式传入完整目录：

```bash
PYTHONPATH=. python3 -m custom_tools.minimal_impl.simulate evaluate \
  --student custom_tools/checkpoints/shadow_chunk8_final.ckpt \
  --trajectory-root /path/to/trajectory_npy \
  --meshdata-root /path/to/meshdata \
  --object-id core-bottle-... \
  --output /tmp/evaluation.yaml
```

`train.py` 展示主线的核心监督训练，不复制Lightning日志、断点续训和批量实验调度。
正式实验训练4轮后通过Isaac Gym逐轮选择，最后采用第2轮；精简训练入口只保存最后一轮，
因此用于理解和小规模复现。冻结最终权重已保存在
`custom_tools/checkpoints/shadow_chunk8_final.ckpt`。
