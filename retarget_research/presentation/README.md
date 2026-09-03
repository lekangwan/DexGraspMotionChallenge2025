# 五分钟汇报素材

本目录只保存真实实验数据生成的图表、真实仿真视频的截图/GIF，以及不涉及机械手外观臆造的算法示意图。

- `build_assets.py`：生成 SVG/PNG 统计图、算法框图、GIF和关键帧拼图。
- `record_cases.py`：录制两个真实对照：Linker姿态方案与功能方案、XHand监督策略与Residual PPO。
- `assets/figures/`：可编辑SVG和高分辨率PNG。
- `assets/frames/`：真实Isaac Gym视频帧拼图。
- `assets/animations/`：PPT自动播放GIF。
- `assets/videos/`：为方法对照重新录制的MP4。

训练日志没有记录 episode return，因此汇报中不绘制虚构的 return 曲线；主训练图使用训练成功率和平均最终抬升，loss另存为备用图。

注意：`08_*`至`11_*`是早期参考轨迹条件Residual PPO的历史素材，不是最终自主策略。正式汇报的进阶结果应使用`../reports/figures/advanced_policy_results.png`、`advanced_training_curves.png`以及`../advanced_policy/videos/autonomous_parametric_final_test_v1/`。
