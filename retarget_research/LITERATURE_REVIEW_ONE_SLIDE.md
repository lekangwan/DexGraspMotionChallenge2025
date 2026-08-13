# 一页PPT：灵巧手动作重定向方法调研与路线选择

## 标题

**从 Shadow Hand 到不同结构灵巧手：重定向方法调研与本文路线**

## 中间主体：四类方法（建议横向排成四栏）

| 方法 | 最直观的理解 | 优点 | 主要问题 | 代表工作 |
| --- | --- | --- | --- | --- |
| 关节直接映射 | 把源手关节角按对应关系复制或缩放 | 最快、实现简单 | 两只手结构、手指长度和关节数不同时，指尖位置容易错误 | 工程基线 |
| 关键点/骨架匹配 | 调整目标手关节，使掌心和指尖接近源手 | 不需要目标手专家标签；只需URDF；结果可解释 | 姿势相似不代表真正接触和抓稳物体 | DexPilot；AnyTeleop / dex-retargeting |
| 物体与接触感知 | 不只模仿手形，还保留手指与物体的接触位置和方向 | 更符合抓取任务本质，跨手型更可靠 | 需要物体网格、距离/碰撞计算，速度更慢 | Kinematic Motion Retargeting for Contact-Rich Manipulations |
| 学习或物理优化 | 用网络快速预测，或在仿真中用轨迹优化/RL修正 | 速度快或物理成功率潜力高 | 学习法需要成对标签；物理法计算量和调参成本高 | DexMV：先重定向，再模仿学习 |

## 底部结论：我们的选择

**选择“关键点初始化 + 物体接触与时序联合优化 + Isaac Gym筛选”的两阶段方案。**

```text
官方Shadow成功轨迹
        ↓
参考基线：逐帧匹配掌心/手指关键点
        ↓
我们的改进：保持接触位置、避免穿模、让70帧连续平滑
        ↓
Isaac Gym重放，以真实抬升成功率筛选轨迹
```

选择原因：题目没有提供目标手动作标签，所以不能一开始直接监督训练MLP；纯关键点法已有参考实现、可快速建立基线，但忽略物体接触和整段动作连续性。我们的方案保留其可解释、无需标签的优点，同时针对“看起来像却抓不住”的核心缺陷进行改进，适合17天周期，并能为后续策略学习生成目标手专家数据。

## 30秒讲稿

“现有灵巧手重定向大致可分为直接关节映射、关键点匹配、接触感知优化和学习/物理方法。直接映射无法适应不同手型；关键点方法只需源轨迹和目标手URDF，是目前最成熟且可解释的起点，但只保证姿势相似，不保证抓住物体。接触感知方法说明，抓取任务还需要保存手指与物体的关系。由于本任务没有目标手监督标签、时间只有17天，我们采用两阶段路线：先复现逐帧关键点基线，再加入物体接触、穿模和时间平滑约束，最终用Isaac Gym抬升成功率验证。成功轨迹再用于后续模仿学习。”

## 参考资料

1. Handa et al., *DexPilot: Vision Based Teleoperation of Dexterous Robotic Hand-Arm System*, ICRA 2020: <https://research.nvidia.com/publication/2020-05_dexpilot-vision-based-teleoperation-dexterous-robotic-hand-arm-system>
2. Qin et al., *AnyTeleop: A General Vision-Based Dexterous Robot Arm-Hand Teleoperation System*, RSS 2023: <https://yzqin.github.io/anyteleop/>
3. DexSuite, *dex-retargeting* 开源实现（位置、向量与DexPilot优化器）: <https://github.com/dexsuite/dex-retargeting>
4. Lakshmipathy et al., *Kinematic Motion Retargeting for Contact-Rich Anthropomorphic Manipulations*, ACM TOG 2025: <https://arxiv.org/abs/2402.04820>
5. Qin et al., *DexMV: Imitation Learning for Dexterous Manipulation from Human Videos*, 2021: <https://yzqin.github.io/dexmv/>

