# 选模证据

这些文件是从完整实验结果中提取的轻量摘要，不包含checkpoint或逐步仿真日志。

| 文件 | 说明 |
|---|---|
| `training_dataset_summary.json` | 80个训练物体的轨迹数量和内部划分 |
| `category_expert_development_summary.yaml` | 4/10/20物体类别教师与不同epoch的开发集比较 |
| `category_expert_candidates.csv` | 各类别教师候选排序 |
| `offline_student_repeat_summary.yaml` | 离线Task-ID学生的多seed复核 |
| `online_r1_repeat_summary.yaml` | Online-R1锁定checkpoint的多seed复核 |
| `temporal3_repeat_summary.yaml` | Temporal3锁定checkpoint的多seed复核 |
| `seen80_selection.yaml` | 80个见过物体的独立验证轨迹清单 |
| `serial_nodes_seen16_summary.yaml` | 五个串行节点在固定16物体上的三seed诊断 |

Development12参与过模型选择，因此这些结果用于说明“为什么选这个checkpoint”，不作为无偏最终泛化证据。Final8结果只在模型锁定后用于报告。
