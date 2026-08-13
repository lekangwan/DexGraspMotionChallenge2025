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

## 第一阶段顺序

1. 冻结XHand关键点语义配置，并验证参考基线输出；
2. 用相同语义规范校准Linker O6关键点；
3. 接入Wuji 20自由度模型；
4. 让同一个优化器接受三种手配置；
5. 分别接入Isaac Gym重放与批量评测。
