# 自定义主线代码

本目录只保留最终串行主线、必要仿真支持、冻结配置和少量回归测试。官方仓库文件仍在 `ActionDiffusion/`、`dexgrasp/` 和 `assets/`，没有直接改写成自定义实现。

建议从 [`PIPELINE.md`](PIPELINE.md) 开始阅读，不要按文件名字母顺序阅读。

## 六个推荐入口

| 阶段 | 入口 |
|---|---|
| 数据准备 | `preprocess_graspm3_isolated.py`、`prepare_bc_dataset.py` |
| 类别教师 | `run_scaled_category_expert_training.py` |
| 离线统一学生 | `run_taskid_offline_stage.py` |
| 在线模仿 | `collect_taskid_online_scaled20_isolated.py`、`run_taskid_online_r1_stage.py` |
| 三帧学生 | `run_taskid_temporal3_stage.py` |
| 统一评测 | `run_comprehensive_five_model_evaluation.py` |

## 目录约定

- `configs/`：只保留最终主线、对照和评测所需配置；
- `results/`：只保留报告引用的轻量汇总和图片；
- `runs/`：训练权重，受 `.gitignore` 排除；
- `data/distillation/`：教师标签和在线聚合数据，受 `.gitignore` 排除；
- 其他Python文件：上述入口的算法或仿真依赖。

最终结果和案例视频见 `FINAL_SUBMISSION/`。完整开发历史位于 `lekang_baseline` 分支。
