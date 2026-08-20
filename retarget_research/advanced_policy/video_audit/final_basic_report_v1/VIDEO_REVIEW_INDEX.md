# 基础重定向最终报告视频索引

六段视频均在 Isaac Gym 中加载目标手 URDF、物体网格和仿真地面，并逐帧回放正式物理评测保存的状态。每段20 FPS、80帧、4秒。

| 目标手 | 类型 | 物体与轨迹 | 最大/最终抬升 | 视频 |
|---|---|---|---:|---|
| LinkerHand O6 | 稳定成功 | Dog [33] | 39.5/39.5 cm | [打开](../../videos/final_basic_isaac_state_v1/stable_success_0_sem-Dog-35f73ca2716aefcfbeccafa1b3b5f850_source33.mp4) |
| XHand | 稳定成功 | Cookie [19] | 35.0/35.0 cm | [打开](../../videos/final_basic_isaac_state_v1/stable_success_1_sem-Cookie-ccfa74e5574678325cde8c99e4b182f9_source19.mp4) |
| WujiHand | 稳定成功 | Thumbtack [1] | 35.3/35.2 cm | [打开](../../videos/final_basic_isaac_state_v1/stable_success_2_sem-Thumbtack-17aa537d3f70e0998c5d696f6e844329_source1.mp4) |
| LinkerHand O6 | 抬升后滑落 | Camera [16] | 33.6/6.1 cm | [打开](../../videos/final_basic_isaac_state_v1/strict_failure_0_sem-Camera-5838f13cd253038ae94020dcb05ab335_source16.mp4) |
| XHand | 严格规则失败 | Camera [34] | 17.1/−2.5 cm | [打开](../../videos/final_basic_isaac_state_v1/strict_failure_1_sem-Camera-5838f13cd253038ae94020dcb05ab335_source34.mp4) |
| WujiHand | 严格规则失败 | Vase [37] | 21.1/0.0 cm | [打开](../../videos/final_basic_isaac_state_v1/strict_failure_2_sem-Vase-1a8f9295b44b48895e8c5748ca5ef3ea_source37.mp4) |

后三条都没有满足最终采用的30 cm末段稳定要求，正好说明“中途抬起”不能代替稳定抓取。Linker的成功和失败案例均来自最终功能向量、动态权重、接触锚与抓握偏置方法。
