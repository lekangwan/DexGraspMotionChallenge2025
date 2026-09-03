# 灵巧手重定向研究工作区

本目录用于浙江大学夏令营“灵巧手重定向研究”考核，与此前 `custom_tools` 项目隔离。旧项目代码和数据不在这里修改。

> 2026-09-03范围冻结：PPT和后续整理只保留基础重定向；进阶自主策略暂停。
> `advanced_policy/`与`advanced_policy_v2/`仅作历史实验归档，不代表当前主线。

正式工作从2026-08-09开始。项目目标、验收标准和工作原则见 `PROJECT_CHARTER.md`；每日进展、命令、结果和阻塞统一追加到 `WORK_LOG.md`。

## 已锁定的三种目标手

| 目标手 | 主动自由度 | 手指数 | 当前基础 | 选择理由 |
| --- | ---: | ---: | --- | --- |
| LinkerHand O6 | 6 | 5 | 最终方法已冻结 | 低自由度、强欠驱动代表 |
| XHand | 12 | 5 | 最终方法已冻结 | 参考仓库基线与中等自由度代表 |
| WujiHand | 20 | 5 | 最终方法已冻结 | 高自由度、接近Shadow形态 |

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

## 当前状态（2026-09-03）

- 正式清单固定为50类、每类2个物体、每物体10条轨迹，共100物体和1000条轨迹；选择种子为`20260808`。
- 三只目标手为LinkerHand O6、XHand和WujiHand，主动手指自由度分别为6、12、20。统一使用CPU PhysX 60 Hz；70个专家帧各插值3步，再保持30步，共240步。
- 基础任务已经冻结。统一主线为“运动学初值 → Global CEM（Linker额外Global2）→ Rank-5关节协同CEM → 候选与基线各两次独立确认 → 正式1000条重放”。参考Isaac成功率为Linker 43.9%、XHand 69.5%、Wuji 71.8%；稳定运输/可训练比例为35.0%、63.0%、66.2%。
- 最终轨迹、确认报告与审计位于`outputs/reboot_synergy_rank5_formal1000_v1/postconfirmed_rank5_v1/`；可复核哈希锁位于`retargeting/configs/final_retargeting_release_v1.json`。
- 进阶自主策略因尚未取得足够进展已暂停，不纳入推免PPT展示。相关目录保留仅为实验历史。
- 当前学习入口为`RETARGETING_LEARNING_GUIDE.md`，代码入口为`minimal_impl/`。
- 参考仓库版本为`76fc48d80c02ae17cf5f8667fd286ed9c6c5cf46`。外部数据、URDF、checkpoint和视频默认不进入Git，提交时需按`reports/REPORT_ASSET_CHECKLIST.md`另行附带或提供链接。
