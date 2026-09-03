# 自主抓取策略第二版

这条管线只解决进阶任务：从episode初始手—物状态出发，由策略自主生成后续动作。测试时不会读取未来重定向轨迹、专家手腕或参考轨迹。

文件职责：

- `geometry.py`：把真实物体mesh采样到初始手腕坐标系；
- `prepare/prepare_geometry_data.py`：依据v3稳定运输审计筛专家，并生成几何sidecar；
- `dataset.py`：构造初始任务、三帧历史和未来Chunk8；
- `models.py`：PointNet与三种候选策略；
- `train.py`：统一Huber监督训练；
- `runtime.py`：Isaac闭环推理和重叠动作块时间集成；
- `evaluate/run_candidate_valid50.py`：固定valid50闭环首筛。

## 当前冻结输入（2026-09-01）

Rank-5正式1000条已经完成并经过重复确认。该目录现在只读取：

```text
retarget_research/outputs/reboot_synergy_rank5_formal1000_v1/
└── postconfirmed_rank5_v1/
```

最终监督轨迹数量为Linker 183条（train/valid=152/31）、XHand 339条（276/63）、
Wuji 356条（285/71）；每手test保持500条未见物体任务，不按专家成功预筛。数据审计位于
`data/final/FINAL_DATA_AUDIT.json`，三手测试物体与训练/验证物体交集均为0。

## 当前训练顺序

1. 同时训练`geometry_phase`和`geometry_chunk`，三只手使用同一算法与超参数规则；
2. 在每类1条的固定valid50上做Isaac闭环首筛；首轮Phase为4/6/2，闭环Chunk8三手均为0；
3. 不再训练更依赖偏离状态的Temporal，改用只由初始几何和phase生成Chunk8的`geometry_plan_chunk`修复teacher-forcing停滞；
4. 完整valid冻结方法后，500条对象隔离test只运行一次。

所有候选均为纯参数策略：测试时不读取未来专家动作、不检索训练轨迹、不使用专家手腕或类别ID。正式顺序和停止条件见`ADVANCED_POLICY_V2_PLAN.md`。

`verify_autonomous_contract.py`会检查checkpoint只包含配置、维度和模型参数张量，并在闭环报告生成后逐条确认`teacher=None`、`expert_wrist=false`、两类Residual RL均为空。当前六个首轮checkpoint均已通过，证据写入`AUTONOMY_AUDIT.json`。
