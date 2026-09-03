# 进阶自主策略最终收尾

## 最终方法

最终保留 `Geometry-PCA` 整轨迹生成策略。它根据 episode 初始手—物状态和物体点云预测低维
PCA 系数，再将系数还原为 240 个物理步的完整手腕与手指位置命令。测试时只使用重定向轨迹
第一帧完成初始化，不读取未来专家动作、专家手腕、参考轨迹检索、类别 ID 或 Residual RL 基线。

固定 `valid50` 上，Linker/XHand/Wuji 分别为 9/50、10/50、8/50。后续多候选PCA、
动态几何残差、接触反馈、Grasp-FSM、Keypose、潜空间Diffusion、Surface-IK以及无PCA直接
Temporal3均未超过该结果，因此不再继续选择或调参。PCA秩按valid阶段冻结为Linker 32，
XHand和Wuji 16。

## 对象隔离 test500

统一评测条件：每手500条、未见测试物体、15 cm抬升阈值、末段稳定保持，并以掌—物相对
运输稳定作为最终严格成功条件。

| 手 | valid50 | test稳定抬升 | test严格稳定运输 |
|---|---:|---:|---:|
| Linker | 9/50 | 29/500（5.8%） | 22/500（4.4%） |
| XHand | 10/50 | 70/500（14.0%） | 67/500（13.4%） |
| Wuji | 8/50 | 65/500（13.0%） | 57/500（11.4%） |

三个test汇总分别位于：

- `runs/final_pca_test500_v1/linker/policy_evaluation_summary.json`
- `runs/final_pca_test500_v1/xhand/policy_evaluation_summary.json`
- `runs/final_pca_test500_v1/wuji/policy_evaluation_summary.json`

## 冻结权重

- Linker：`runs/candidates_v1/linker/geometry_pca32/best.pt`  
  SHA256 `6a9a52c2a9cfe0641e1b74ff33b42214c72fe2f2c1a1a3b0a8569939105504b2`
- XHand：`runs/candidates_v1/xhand/geometry_pca16/best.pt`  
  SHA256 `0a8c8e1b6b9ae1735ace6e814a8932a68d42d59969304b4299dd9e53bc7a2db8`
- Wuji：`runs/candidates_v1/wuji/geometry_pca16/best.pt`  
  SHA256 `e62126478c32f5858fdcaae6c018745b1c52b4c42dbb3abc4300327081ff0255`

## 结论边界

该结果证明PCA对整段动作的一致性约束明显优于本项目测试过的单步和动作块自主策略，但测试集
成功率仍然较低，尤其Linker在未见物体上出现明显泛化下降。因此该分支仅作为一次完整、无测试
专家信息泄漏的进阶尝试保存，不纳入当前推免PPT的主要成果，也不再继续开发。
