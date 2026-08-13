# Linker自由度增广实验说明

## 1. 为什么做

真实LinkerHand O6只有6个主动手指控制量：拇指2个，食指、中指、无名指和小指各1个。它的URDF虽然有11个转动关节，但拇指末端和四指末端共5个关节分别被固定为：

```text
thumb_ip = 1.86 × thumb_pitch
finger_dip = 0.89 × finger_mcp
```

因此原优化器不能分别调整一根手指的近端和远端弯曲。当前冻结10轨迹中，O6只有1/10成功，而同一物理回放器中的关节跟踪误差只有0.0207 rad，说明主要问题更可能是可表达抓形不足，而不是控制器没有跟上。

本实验保持Linker的掌形、指长、mesh、关节轴、物体和成功判据不变，只把5个从动关节变为独立优化变量。它回答的问题是：如果同样的Linker外形有11个可控手指关节，成功率是否会提高？

## 2. 两种模式的输入输出

### `coupled6`：真实O6基线

- 优化输入：Shadow每帧28维轨迹和语义关键点。
- 优化变量：6个手指主动角 + 3维手腕平移 + 3维手腕欧拉角，共12维。
- 保存输出：`[手腕6, 主动关节6]`，形状为`(N,70,12)`。
- 回放逻辑：在Isaac中按1.86/0.89补出5个从动角。

### `independent11`：相同外形的解耦增强模型

- 优化输入：与O6完全相同，额外自动启用拇指中段关键点。
- 优化变量：11个手指角 + 手腕6维，共17维。
- 保存输出：`[手腕6, 完整关节11]`，形状为`(N,70,17)`。
- 回放逻辑：11个手指角逐项发给Isaac，不再应用固定倍率。

两种模式都使用原URDF的关节上下限。新增的自由度不是扩大关节活动范围，而是允许原来绑定在一起的两个角分别取值。

## 3. 内部数据流

```text
Shadow 28维轨迹
    ↓ 正向运动学
每帧Shadow语义关键点
    ↓ SLSQP，上一帧结果作为下一帧初值
Linker手腕6维 + 手指6/11维
    ↓ 保存候选npy
独立几何复算 + Isaac CPU PhysX重放
    ↓
关键点误差、耦合偏差、抬升高度、持续时间、成功率
```

11轴几何报告还会计算5个“耦合偏差”：独立关节角减去原O6倍率预测角。如果偏差始终接近0，即使成功率上升也不能说明新增自由度被真正使用；如果偏差明显非零且成功率提高，则支持“原机械耦合限制抓形”的解释。

## 4. 结论边界

`independent11`不是市售LinkerHand O6的真实硬件模型，因为O6没有5个新增电机。它目前是结构消融和算法诊断模型，不能拿它的成绩替代O6成绩。只有在考核允许修改目标手URDF或把目标手定义为6–20自由度的自定义Linker增强版时，才能把它作为正式目标手；否则报告中必须同时给出O6基线，并明确说明11轴结果是假设性上限。

实验按以下顺序决定后续工作：

1. 先在冻结10轨迹比较O6与解耦11轴的几何误差和物理成功率。
2. 若11轴明显提高，继续研究“有限残差解耦”，用较弱约束让DIP靠近原倍率但允许为接触小幅偏离。
3. 若11轴仍然失败，说明瓶颈不只是耦合，应转向物体感知抓形合成，而不是继续增加关节。

## 5. 冻结10轨迹运行命令

候选生成预计超过3分钟，由用户终端执行：

```bash
cd /home/lekangwan/projects/DexGraspMotionChallenge2025-final-release

MPLCONFIGDIR=/tmp/matplotlib-retarget \
OMP_NUM_THREADS=4 \
MKL_NUM_THREADS=4 \
/home/lekangwan/miniconda3/envs/hand-retarget/bin/python \
  retarget_research/retargeting/run/run_linker_manifest.py \
  --manifest retarget_research/retargeting/configs/linker_pilot_holdout_v1.json \
  --output-dir retarget_research/outputs/linker11_pilot_v1 \
  --workers 1 \
  --resume \
  --joint-mode independent11 \
  --maxeval 100 \
  --source-z-offset 0.4 \
  --joint-temporal-weight 1 \
  --translation-temporal-weight 300 \
  --rotation-temporal-weight 1
```

生成全部成功后再执行统一几何与物理评估：

```bash
MPLCONFIGDIR=/tmp/matplotlib-retarget \
/home/lekangwan/miniconda3/envs/hand-retarget/bin/python \
  retarget_research/retargeting/evaluate/evaluate_hand_manifest.py \
  --hand linker11 \
  --manifest retarget_research/retargeting/configs/linker_pilot_holdout_v1.json \
  --target-dir retarget_research/outputs/linker11_pilot_v1 \
  --output-dir retarget_research/outputs/linker11_pilot_v1_evaluation \
  --workers 1
```

第一条完成时应看到`all_successful=True`；第二条最终会打印`success=x/10`。判断时还应读取汇总中的平均关键点误差、平均最大抬升和每条失败轨迹，不能只看成功数。

## 6. 冻结10轨迹实测结论

完全解耦11轴已按上述命令完成：成功率仍为`1/10`。相对真实O6基线，它把平均关键点误差从15.028 mm降到14.190 mm，但平均最大抬升从36.1 mm降到21.9 mm。新增关节相对原mimic关系的平均绝对偏差为0.346 rad，最大1.333 rad，说明优化器确实使用了新增自由度，而不是11轴退化回原6轴。

随后完成两种物体接触细化：

- 闭合后冻结11轴抓形：`0/10`。唯一Planter成功轨迹的持续抬升由32步降到28步，低于30步判据。
- 用11轴成功重放重新标定指腹，闭合和抬升期保持手腕基线、逐帧动态优化手指：恢复`1/10`；Planter最大抬升由175.0 mm提高到326.1 mm，但没有新增成功轨迹。

因此当前证据否定的是“只解除O6现有屈曲联动就能提高成功率”。11轴仍没有新增横向张开/侧摆关节，无法改变普通手指的接触方向。

项目最终决定不以Linker L20替换O6。三只正式目标手继续保持Linker O6、XHand 12和Wuji 20，形成6/12/20自由度的低—中—高完整覆盖。L20虽然可能更容易重定向，但会与Wuji同处20自由度上限，削弱“不同控制复杂度”这一实验设计。11轴只保留为结构消融，不作为正式目标手。

O6下一步保留6个主动命令，研究两条真正属于低自由度手的改进：第一，把5个从动关节从刚硬位置跟踪改成接触可偏离的柔顺mimic；第二，根据O6可达抓形选择拇指与2–3根普通指的可行对向接触，而不是要求它复制Shadow全部五指姿态。
