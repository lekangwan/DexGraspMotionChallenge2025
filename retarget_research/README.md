# 灵巧手重定向研究工作区

本目录用于浙江大学夏令营“灵巧手重定向研究”考核，与此前 `custom_tools` 项目隔离。旧项目代码和数据不在这里修改。

正式工作从2026-08-09开始。项目目标、验收标准和工作原则见 `PROJECT_CHARTER.md`；每日进展、命令、结果和阻塞统一追加到 `WORK_LOG.md`。

## 已锁定的三种目标手

| 目标手 | 主动自由度 | 手指数 | 当前基础 | 选择理由 |
| --- | ---: | ---: | --- | --- |
| LinkerHand O6 | 6 | 5 | 有优化与Isaac Gym骨架，关键点未对齐 | 低自由度、强欠驱动代表 |
| XHand | 12 | 5 | 参考仓库唯一完整基线 | 先跑通全链路的中等难度基准 |
| WujiHand | 20 | 5 | 有完整右手URDF和mesh，无评测器 | 高自由度、接近Shadow形态 |

Allegro只有四指，不满足题目“五指灵巧手”的字面条件；Revo3有21个主动关节，超过“6到20自由度”的上限，因此均不作为正式三手。

## 目录

```text
retarget_research/
├── PLAN_17_DAYS.md              17天逐日计划和验收门
├── PROJECT_CHARTER.md            项目目标、研究问题、交付物和工作原则
├── WORK_LOG.md                   从正式开工日起持续追加的工作记录
├── HAND_READINESS.md             三只手的准备度、缺口和补齐顺序
├── RESEARCH_DESIGN.md           方法、消融、评测和进阶策略设计
├── LINKER_DOF_EXPANSION.md      Linker 6轴耦合与11轴增强实验的易读说明
├── LINKER_INDEPENDENT_VALIDATION.md Linker v2独立20轨迹验证手册
├── THREE_HAND_VALIDATION_RESULTS.md 三手同一20轨迹结果与方法选择
├── FORMAL_1000_RUNBOOK.md       完整数据到位后的正式长命令顺序
├── FORMAL_READINESS.md          大规模实验前五道硬门、已完成项和外部阻塞
├── retargeting/                  与custom_tools隔离的基本重定向实现
├── advanced_policy/              进阶策略适配，可复用旧项目组件
├── ENVIRONMENT.md               环境状态与复现命令
├── BLOCKERS_AND_QUESTIONS.md    当前阻塞与发给报告联系人问题
├── configs/project.yaml         冻结的项目级配置
├── manifests/inventory.csv      待填的完整数据清单格式
├── reports/                     两份1–2页报告模板
├── scripts/preflight.py         环境、资产、数据预检
├── scripts/build_embedded_category_map.py 从core/sem对象ID显式标签构建类别表
├── scripts/build_inventory.py   官方类别表与数据根目录自动匹配成inventory
├── scripts/build_manifest.py    固定seed抽取50类/100物体/1000轨迹
├── scripts/freeze_formal_experiment.py 冻结数据、方法和代码哈希
├── scripts/verify_formal_bundle.py 分阶段验收1000条正式产物
├── scripts/select_report_cases.py 自动选择成功/近失/滑落/失败案例
├── scripts/render_selected_cases.py 按需重跑少量案例并录制MP4
├── scripts/render_software_replay.py 无GPU时由DOF/URDF/物体网格生成诊断MP4
├── scripts/export_result_tables.py 多份摘要自动导出报告Markdown/CSV
└── reference/HandRetargetTask2026
    └── 考核参考仓库，只读使用，不提交到本仓库
```

## 正式开工顺序

1. 向联系人确认 `BLOCKERS_AND_QUESTIONS.md` 中的口径，同时取得完整数据与物体资产。
2. 运行 `scripts/preflight.py`，确认CPU PhysX与三手资产仍可用。
3. 生成完整对象清单 `manifests/inventory.csv`，再用固定seed冻结抽样manifest。
4. 在 `retargeting/` 中先复现XHand小样本和全量baseline，再进入LinkerHand和WujiHand。
5. 所有调参只使用calibration split；最终1000轨迹结果一次性冻结并记录。

正式长命令及依赖关系见`FORMAL_1000_RUNBOOK.md`；这些命令预计超过3分钟，应由用户在终端运行。

## 当前状态（2026-08-13）

- 已下载参考仓库，commit为 `76fc48d80c02ae17cf5f8667fd286ed9c6c5cf46`。
- 已创建独立Conda环境 `hand-retarget`，没有修改旧 `dexgrasp` 环境。
- 三只手的数学重定向、CPU PhysX重放、冻结manifest批处理和统一评估均已跑通。
- 5物体×2轨迹开发集结果：XHand真实指腹细化7/10、Wuji v2单流程7/10、Linker O6受力再分配+自适应PD为5/10（原几何基线1/10、统一渐进夹紧3/10）；只用于开发，不是正式结果。
- Linker每条轨迹仍只有一个12维候选（腕部6维+主动手指6维）；最终使用11点无时序关键点基线上的统一渐进夹紧和固定PD 120/5，不按物理成功结果取候选并集。冻结参数见`retargeting/configs/linker_o6_method_selection_v2.json`。
- Linker独立验证已完成：10个全新物体×2条上，几何基线4/20、统一夹紧7/20、固定残差6/20、自适应PD仍6/20。正式主方法据此选择更简单的统一夹紧v1和默认PD；v2的开发集5/10保留为过拟合边界案例。
- 正式1000条为Linker夹紧231/1000、Wuji v1 638/1000、XHand官方574/1000；XHand指腹细化为541/1000，后续动态残差在A/B仍未超过官方。三手最终专家冻结为Linker统一夹紧、Wuji v1和XHand官方，详见`configs/final_method_decision_v1.json`。
- Linker同外形11轴解耦消融已完成冻结10轨迹评估，仍为1/10，证明单纯解除屈曲联动不足；它不等同真实O6硬件，也不进入正式三手方案。
- 已在旧项目只读目录找到5048个原始轨迹文件和5751套资产，完整预检全部通过。只使用`core/sem`对象ID中的显式类别段，排除类别含糊的`mujoco/ddg`，形成108类、3723物体的候选池。
- seed 20260808的50类、100物体、1000轨迹正式重定向与评测已经完成；A/B各50类的收尾确认也已结束。重定向不再调参，结果目录仍不进入Git，下一阶段使用`advanced_policy/configs/hand_data_specs_v2.json`进入策略学习。
- 大规模运行前的代码链路已补齐：正式实验lock、候选/评测/trace五阶段验收、执行前状态到下一动作的策略trace、对象级无泄漏split、三手BC/Temporal3/Diffusion训练、离线误差和Isaac闭环测试。
- 当前短测试为原重定向71项与进阶/正式工具13项全部通过；另完成XHand单轨迹物理trace、零策略闭环、Isaac相机和纯CPU软件渲染MP4冒烟。
