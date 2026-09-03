# 两个项目的 PPT 展示资产

这里只收纳最终 ShadowHand 抓取和基础重定向两条主线，不包含已暂停的进阶策略。

## 目录

- `figures/`：Python 从冻结结果直接生成的 300 dpi PNG 和可编辑 SVG。
- `videos/shadow/`：最终 Chunk8 策略的真实 Isaac Gym 录像。
- `videos/retarget/`：三只目标手最终重定向方法的真实 Isaac Gym 录像。
- `flowcharts/`：Figma 原生绘制后导出的流程图 PNG、SVG，以及布局规格。
- `build_scientific_figures.py`：重新生成全部统计图。

## 推荐在 PPT 中使用

1. ShadowHand：`shadow_ablation.png`、`shadow_category_breakdown.png`、一段成功抓取 MP4。
2. 重定向：`retarget_baseline_vs_final.png`、`retarget_quality_funnel.png`、三只手各一段成功 MP4。
3. PNG 用于直接展示；SVG 用于 PowerPoint/Figma 中继续编辑。

## 矢量流程图

- `flowcharts/shadowhand_pipeline.svg`：ShadowHand 自主抓取完整主线。
- `flowcharts/retargeting_pipeline.svg`：三种灵巧手重定向完整主线。
- 同名 PNG 是 3840×2160 的高分辨率备用版本。
- Figma 源文件：https://www.figma.com/design/StBnVDcGcz8wE92AKBFBPL/Untitled?node-id=0-1

### 推荐的动态案例

- ShadowHand：`animations/shadow_bowl_env001.gif`，动作和物体轮廓最容易看清。
- Linker：文件名包含 `retarget_linker_stable_transport_success` 的 GIF/MP4。
- XHand：文件名包含 `retarget_xhand_stable_transport_success` 的 GIF/MP4。
- Wuji：文件名包含 `retarget_wuji_stable_transport_success` 的 GIF/MP4。

### 推荐的静态案例

- `storyboards/shadow_bowl_env001.jpg`
- `storyboards/retarget_linker_stable_transport_success_*.jpg`
- `storyboards/retarget_xhand_stable_transport_success_*.jpg`
- `storyboards/retarget_wuji_stable_transport_success_*.jpg`

ShadowHand 四段录像均重新执行并通过稳定成功核验；录像为 960×720、122 帧、20 Hz。原始 MP4 不裁切，GIF 和故事板只裁掉上下空白，不修改仿真内容。

制图采用白底、低饱和色盲友好配色、轻网格、无 3D/渐变/厚边框的统一风格。所有数字来自冻结的 YAML/JSON，而非人工重新填写。
