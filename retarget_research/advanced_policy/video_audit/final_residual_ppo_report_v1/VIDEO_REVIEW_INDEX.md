# Residual PPO 最终报告视频索引

案例来自按物体隔离的500条正式测试结果。成功样本按末段高度波动最小选择；失败样本只在曾越过30 cm的轨迹中选择，并优先展示峰值后明显滑落。视频由评测JSON保存的真实关节状态与物体位姿生成，因此与正式成功率统计完全对应，不重新采样策略。

| 目标手 | 类型 | 物体与轨迹 | 最大/最终抬升 | 视频 |
| --- | --- | --- | ---: | --- |
| LinkerHand O6 | 稳定成功 | Headphones [13] | 33.0/33.0 cm | [打开](../../videos/final_residual_ppo_isaac_state_v1/stable_success_0_sem-Headphones-aa87f1cb1632a08f2764cfba57a5de73_source13.mp4) |
| XHand | 稳定成功 | Piano [35] | 33.9/33.9 cm | [打开](../../videos/final_residual_ppo_isaac_state_v1/stable_success_1_sem-Piano-ce4945cde785aecb478fa0ab37c461c6_source35.mp4) |
| WujiHand | 稳定成功 | Blender [7] | 32.9/32.9 cm | [打开](../../videos/final_residual_ppo_isaac_state_v1/stable_success_2_sem-Blender-dd06add5426f69ddc9c603cb0476780f_source7.mp4) |
| LinkerHand O6 | 抬升后滑落 | CerealBox [30] | 37.0/−2.8 cm | [打开](../../videos/final_residual_ppo_isaac_state_v1/lift_then_slip_failure_0_sem-CerealBox-2ee85d45fe615a734322eb6f7ad3b3a2_source30.mp4) |
| XHand | 抬升后滑落 | Fruit [16] | 38.3/−1.9 cm | [打开](../../videos/final_residual_ppo_isaac_state_v1/lift_then_slip_failure_1_sem-Fruit-473758ca6cb0506ee7697d561711bd2b_source16.mp4) |
| WujiHand | 抬升后滑落 | Clock [13] | 39.4/1.9 cm | [打开](../../videos/final_residual_ppo_isaac_state_v1/lift_then_slip_failure_2_sem-Clock-cca5dddf86affe9a23522985f649a9ae_source13.mp4) |

每条视频为20 FPS、80帧、4秒。画面由 Isaac Gym 相机直接渲染目标手 URDF、物体网格和仿真地面；每帧使用正式评测报告保存的真实关节状态与物体位姿，因此不会因录像时重新执行闭环策略而改变原实验结果。
