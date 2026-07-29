# 检查与复现

所有命令都从原仓库根目录运行，而不是进入 `FINAL_SUBMISSION` 后运行。

```bash
conda activate dexgrasp
cd /home/lekangwan/projects/DexGraspMotionChallenge2025
```

## 1. 快速完整性检查

检查环境、数据、mesh和官方 checkpoint：

```bash
python custom_tools/preflight_check.py
```

检查最终保留权重是否与锁定模型一致：

```bash
sha256sum -c FINAL_SUBMISSION/manifests/checkpoints.sha256
```

## 2. 只打印主线命令

以下 `--dry-run` 不启动正式训练或长时间评测，用于确认输入路径和将要执行的步骤：

```bash
python custom_tools/run_scaled_category_expert_training.py \
  --scales 20 \
  --seed 2025 \
  --min-free-vram-mb 4500 \
  --dry-run

python custom_tools/run_taskid_offline_stage.py \
  --min-free-vram-mb 4500 \
  --dry-run

python custom_tools/run_taskid_online_r1_stage.py \
  --min-free-vram-mb 4500 \
  --dry-run

python custom_tools/run_taskid_temporal3_stage.py \
  --min-free-vram-mb 4500 \
  --dry-run

python custom_tools/run_comprehensive_five_model_evaluation.py \
  --min-free-vram-mb 4500 \
  --dry-run
```

注意：离线学生配置文件同时描述过70%教师/30%示范候选；最终统一评测使用的 `t100` checkpoint 是阶段脚本将教师权重覆盖为1.0后训练和选择的结果。最终模型身份应以 `custom_tools/configs/comprehensive_five_model_evaluation_v1.yaml` 中的路径与SHA256为准。

## 3. 完整统一评测

该命令按相同协议评测官方BC、BC Soup、离线学生、Online-R1和Temporal3，并覆盖 Seen80、Development12 和 Final8，使用2025、2026、2027三个仿真 seed。耗时较长，已有结果无需重复运行。

```bash
python -u custom_tools/run_comprehensive_five_model_evaluation.py \
  --min-free-vram-mb 4500 \
  --max-attempts 5 \
  2>&1 | tee custom_tools/results/comprehensive_five_model_evaluation_v1.log
```

结果入口：

```text
custom_tools/results/comprehensive_five_model_evaluation_v1/metrics.csv
custom_tools/results/comprehensive_five_model_evaluation_v1/summary.yaml
custom_tools/results/comprehensive_five_model_evaluation_v1/five_model_split_curves.png
```

## 4. Task ID 对照

Task ID 对照的汇总结果已保存为：

```text
custom_tools/results/notask_full_pipeline_ablation_v1/summary.yaml
custom_tools/results/taskid_ablation_report_v1/metrics.csv
custom_tools/results/taskid_ablation_report_v1/taskid_ablation_curve.png
```

如需从已有结果重新导出图：

```bash
python custom_tools/export_taskid_ablation_figure.py
```

## 5. 未打包的大文件

以下内容故意不复制到 `FINAL_SUBMISSION`：

- `dexgrasp/dataset/` 下的原始和预处理轨迹；
- `assets/meshdata/` 下的物体mesh；
- `custom_tools/data/distillation/` 下的教师标签与在线聚合数据；
- `custom_tools/runs/bc/` 和官方目录中的 checkpoint；
- 大量中间搜索结果、日志和失败候选权重。

这些内容仍保留在原项目中。最终目录通过配置、结果摘要和哈希指向它们，不额外占用磁盘。
