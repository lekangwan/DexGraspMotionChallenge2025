# 五分钟汇报素材索引

所有机械手画面均来自本项目真实 Isaac Gym 回放；算法框图、关键点图和统计图由真实配置或日志生成。SVG保留矢量文字和线条，适合在PPT中继续编辑。

## 第1页：封面

- `assets/frames/01_cover_four_hands.png`：Shadow Hand与三种目标手的真实仿真画面。
- `assets/figures/01_task_overview.svg`：从源轨迹、重定向、物理验证到闭环修正的完整任务概览。

封面四手图用于说明整体任务，Shadow画面是既有Shadow抓取示例；第2页三种目标手才是严格来自同一条WineBottle[10]轨迹的对比。

## 第2页：为什么需要重定向

- `assets/frames/02_same_winebottle_three_target_hands.png`：同一物体、同一源轨迹在Linker、XHand和Wuji上的真实重放画面。
- `assets/figures/02_four_hand_size_comparison.svg`：Shadow Hand与三种目标手的原始URDF尺寸对比；同时提供高清PNG。
- 建议在右侧保留6/12/20 DoF表格，不再叠加装饰图标。

## 第3页：不同构型采用不同思路

- `assets/figures/03_keypoint_vs_vector_flow.svg`：关键点法与功能向量法的输入、误差、优化变量和输出对照框图；同时提供高清PNG。
- `assets/figures/03_semantic_keypoint_retargeting.svg`：15个语义关键点对应关系。
- `assets/figures/04_linker_pose_vs_function_principle.svg`：6 DoF下“姿态相似”和“抓取功能相似”的原理区别。
- `assets/frames/04_linker_real_comparison.png`：真实WineBottle[10]案例。旧姿态方案最高抬升28.4 cm但末段落回，最终约0.0 cm；功能向量方案最大/最终抬升均为38.6 cm。
- `assets/animations/04_linker_real_comparison_left.gif`、`..._right.gif`：上述两种方法的完整真实过程。
- 原始MP4：`assets/videos/linker_pose_baseline.mp4`、`linker_function_vector.mp4`。

## 第4页：基础任务结果

- `assets/figures/05_formal_1000_retarget_results.svg`：三只手各1000条正式重放的末段稳定成功率，细横线补充运输合格率。
- `assets/animations/06_xhand_stable_success.gif`：XHand稳定成功案例。
- `assets/animations/06_xhand_lift_then_slip.gif`：XHand抬升后滑落案例。
- 原始正式视频保存在 `advanced_policy/videos/final_basic_isaac_state_v1/`。

## 第5–6页：当前最终自主策略（正式汇报优先使用）

- `../reports/figures/advanced_training_curves.png`：三只手最终候选的真实训练loss曲线。
- `../reports/figures/advanced_policy_results.png`：valid100上的方法/参数比较，以及预选test500与补充测试结果。
- `../advanced_policy/videos/autonomous_parametric_final_test_v1/`：三只手各一段稳定成功和一段抬升后滑落，共6段最终自主策略Isaac Gym视频。
- 正式口径为Linker 35/500、XHand 42/500、Wuji 30/500。策略只使用重定向首帧初始化，之后不读取未来专家动作或检索参考轨迹。

## 历史方法演进素材（不要作为最终自主结果）

- `assets/frames/07_contact_error_accumulation.png`：同一失败轨迹的接近、接触、偏移和滑落四帧。
- `assets/figures/08_residual_ppo_closed_loop.svg`：Residual PPO闭环结构，明确腕部参考与手指残差的分工。
- `assets/figures/09_ppo_training_success_and_lift.svg`：真实训练成功率与平均最终抬升；浅色为原始批次，深色为15轮滑动平均。
- `assets/figures/09b_ppo_training_losses.svg`：真实policy/value loss备用图。

训练程序未记录episode return，因此没有绘制虚构的return曲线。

- `assets/figures/10_supervised_vs_residual_ppo.svg`：专家可行子集上的监督/DAgger与Residual PPO对比，同时标注500条全量结果。
- `assets/frames/11_xhand_supervised_vs_ppo.png`：真实的同物体、同索引XHand–Battery[12]对照；监督策略失败，Residual PPO成功。
- `assets/animations/11_xhand_supervised_vs_ppo_left.gif`、`..._right.gif`：两种策略的完整真实过程。
- 原始MP4：`assets/videos/xhand_supervised_battery12.mp4`、`xhand_residual_ppo_battery12.mp4`。

上述Residual PPO在每一步读取测试参考轨迹，只能用于说明早期探索和任务口径修正，不能写成最终完全自主方法，也不能用它的成功率代替35/42/30。

## 统一视觉规范

- 白底，Noto Sans CJK字体；靛蓝/青绿/柔和玫红分别对应Linker、XHand、Wuji。
- 线框采用0.8–1.8 pt，箭头与边框保持同一粗细量级。
- 流程图使用论文中常见的分阶段面板，将真实仿真帧嵌入算法结构，而不是使用装饰性图标。
- 柱状图直接标注数值，不保留背景网格；黑色短横线表示第二项判据，避免增加一组抢眼颜色。
- 不使用渐变、阴影、发光、立体SmartArt、背景纹理或无意义图标。
- 正文PPT优先放SVG；GIF只用于现场播放，提交版PDF使用对应PNG关键帧。

视觉规范参考：Nature Research Figure Guide（清晰坐标、无障碍配色、矢量输出和去装饰化要求）、Paul Tol的色盲友好科研配色，以及机器人论文常见的“输入—方法模块—真实输出”方法总览结构。
