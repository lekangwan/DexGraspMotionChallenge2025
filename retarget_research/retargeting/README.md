# 基本任务：轨迹重定向

本目录只负责把官方Shadow Hand轨迹转换成三种目标手的候选轨迹，并完成物理重放评测。它不依赖、也不修改旧项目的 `custom_tools`。

```text
retargeting/
├── configs/          三只手的关节和关键点语义配置
├── prepare/          数据、手模型、关键点定义与可视化校准
├── run/              基线和改进方法的运行入口
├── evaluate/         几何指标、Isaac Gym重放与成功率统计
└── test/             运动学、维度、梯度和小样本回归测试
```

参考仓库 `reference/HandRetargetTask2026` 保持只读。参考实现中经过核对的逻辑会在这里重新组织成统一接口，而不是直接在参考脚本上继续堆改动。

代码阅读前先看 `MODULE_GUIDE.md`；模块和函数说明格式见 `CODING_STANDARD.md`。

## 最终冻结方法与结果（2026-09-01）

三只手现在使用同一条物理优化主线：

```text
运动学初值 → Global CEM → Rank-5关节协同CEM → 两次独立确认 → 正式1000条复验
```

Linker因只有6个主动自由度，在Rank-5之前多做一次Global2。最终参考Isaac成功率为Linker
43.9%、XHand 69.5%、Wuji 71.8%；稳定运输/可训练比例为35.0%、63.0%、66.2%。

冻结入口：

- `configs/final_retargeting_release_v1.json`：方法、结果和所有关键文件SHA-256；
- `../outputs/reboot_synergy_rank5_formal1000_v1/postconfirmed_rank5_v1/`：最终目标轨迹、确认结果和审计；
- `../minimal_impl/cem.py`：适合学习的Global/Rank-5 CEM最小实现。

只验证最终产物是否仍与冻结版本一致（不重新跑1000条仿真）：

```bash
/home/lekangwan/miniconda3/envs/hand-retarget/bin/python \
  retarget_research/retargeting/evaluate/freeze_final_retargeting.py
```

脚本会重新检查50类/100物体/1000轨迹、确认计数、审计计数、300个目标NPY及关键脚本哈希，并打印release SHA。若任何正式文件被改动，生成的SHA会变化，可与Git中冻结JSON比较。

下面“第一阶段顺序”仅说明最初搭建过程，不再代表当前待办。

## 第一阶段顺序

1. 冻结XHand关键点语义配置，并验证参考基线输出；
2. 用相同语义规范校准Linker O6关键点；
3. 接入Wuji 20自由度模型；
4. 让同一个优化器接受三种手配置；
5. 分别接入Isaac Gym重放与批量评测。
