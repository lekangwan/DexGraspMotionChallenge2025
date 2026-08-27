# ShadowHand 当前主线

本目录只面向已经冻结的 ShadowHand 抓取策略：

```text
DexRep + 4维类别ID + 前两步本体/动作历史
→ MLP预测未来8步动作
→ 重叠动作块等权平均
→ 第40步后增加腕部z抬升量
```

最终 checkpoint：`checkpoints/shadow_chunk8_final.ckpt`

唯一配置：`configs/shadow_chunk8_mainline.yaml`

最终结果：`results/final_ablation/summary.yaml`

独立教学实现：`minimal_impl/`

零基础阅读文档：[`LEARNING_GUIDE.md`](LEARNING_GUIDE.md)

建议阅读顺序：

1. `LEARNING_GUIDE.md`：先理解任务、数据流、训练和评测；
2. `minimal_impl/model.py`：理解DexRep、Temporal3、Chunk8和时间集成；
3. `minimal_impl/data.py`：理解训练样本怎样构造；
4. `minimal_impl/train.py`：理解监督学习；
5. `minimal_impl/simulate.py`：理解动作怎样真正进入Isaac Gym；
6. `results/final_ablation/`：核对结果和消融。

CPU快速检查：

```bash
PYTHONPATH=. python3 custom_tools/minimal_impl/test_minimal.py
```

工作过程和失败候选不再拆成大量文档，统一保存在仓库根目录的 `PROJECT.md`。
