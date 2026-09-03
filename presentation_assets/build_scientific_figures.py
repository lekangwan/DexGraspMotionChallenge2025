"""从冻结实验结果生成 PPT 用统计图（PNG）和可编辑矢量图（SVG）。"""

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "figures"
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"

# Tableau 10 / Matplotlib tab10 中的色盲友好蓝、橙、绿、灰。
BLUE = "#4E79A7"
ORANGE = "#F28E2B"
GREEN = "#59A14F"
GRAY = "#A7A9AC"
INK = "#243447"
GRID = "#D9DEE5"


def configure_style():
    """输入：无；输出：统一 Matplotlib 风格；作用：保证所有图视觉一致。"""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Noto Sans CJK SC", "Noto Sans CJK JP", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "axes.titleweight": "bold",
        "axes.titlesize": 16,
        "axes.labelsize": 12,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })


def finish(fig, name):
    """输入：图对象和文件名；输出：300 dpi PNG、SVG；作用：同时服务展示和后期编辑。"""
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def clean_axis(ax, grid=True):
    """输入：坐标轴；输出：简洁坐标轴；作用：去掉论文图中不必要的装饰。"""
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    if grid:
        ax.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.7)
        ax.set_axisbelow(True)


def load_shadow_summary():
    """输入：冻结 ShadowHand YAML；输出：各方法三种子统计；作用：图表不手抄结果。"""
    path = ROOT / "custom_tools/results/final_ablation/summary.yaml"
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)["methods"]


def plot_shadow_ablation(summary):
    """输入：Shadow 消融；输出：方法演进误差棒图；作用：展示主线组件贡献。"""
    keys = ["released_official_bc", "matched_official_bc", "temporal3",
            "chunk8_no_ensemble", "chunk8_equal", "chunk8_equal_lift"]
    labels = ["仓库 BC", "匹配训练设置的 BC", "Temporal3", "Chunk8\n单次预测",
              "Chunk8\n时间集成", "最终：Chunk8\n集成 + 抬升校正"]
    means = np.array([summary[k]["official_macro_rate"]["mean"] for k in keys]) * 100
    stds = np.array([summary[k]["official_macro_rate"]["std"] for k in keys]) * 100
    x = np.arange(len(keys))
    colors = [GRAY, GRAY, BLUE, BLUE, ORANGE, GREEN]
    fig, ax = plt.subplots(figsize=(11.2, 5.5))
    ax.plot(x, means, color="#B8C0C8", linewidth=1.8, zorder=1)
    ax.errorbar(x, means, yerr=stds, fmt="none", ecolor=INK,
                elinewidth=1.2, capsize=4, zorder=2)
    ax.scatter(x, means, s=95, c=colors, edgecolor="white", linewidth=1.2, zorder=3)
    for i, value in enumerate(means):
        ax.text(i, value + 2.0, f"{value:.1f}%", ha="center", color=INK,
                fontsize=11, fontweight="bold")
    ax.set_xticks(x, labels)
    ax.set_ylabel("物体宏平均成功率（%）")
    ax.set_ylim(0, 46)
    ax.set_title("ShadowHand：从逐帧模仿到动作块时间集成")
    ax.text(0.99, 0.02, "误差棒：3 个随机种子的标准差；测试集：8 个未见实例",
            transform=ax.transAxes, ha="right", va="bottom", color="#66717E", fontsize=9.5)
    clean_axis(ax)
    finish(fig, "shadow_ablation")


def plot_shadow_categories(summary):
    """输入：Shadow 分类结果；输出：类别对比图；作用：展示提升是否只来自单一类别。"""
    categories = ["bottle", "mug", "bowl", "camera"]
    zh = ["瓶子", "杯子", "碗", "相机"]
    methods = ["matched_official_bc", "temporal3", "chunk8_equal_lift"]
    labels = ["BC 基线", "Temporal3", "最终 Chunk8"]
    colors = [GRAY, BLUE, GREEN]
    x = np.arange(4)
    width = 0.23
    fig, ax = plt.subplots(figsize=(10.2, 5.4))
    for index, (method, label, color) in enumerate(zip(methods, labels, colors)):
        values = np.array([summary[method]["category_official_rate_mean"][c]
                           for c in categories]) * 100
        ax.bar(x + (index - 1) * width, values, width, label=label,
               color=color, edgecolor="white", linewidth=0.7)
    ax.set_xticks(x, zh)
    ax.set_ylabel("类别成功率（%）")
    ax.set_ylim(0, 55)
    ax.set_title("ShadowHand：四类未见物体上的成功率")
    ax.legend(ncol=3, loc="upper center")
    clean_axis(ax)
    finish(fig, "shadow_category_breakdown")


def load_retarget_results():
    """输入：冻结正式 1000 条 JSON；输出：三手结果；作用：读取最终统一口径。"""
    path = ROOT / "retarget_research/retargeting/configs/final_retargeting_release_v1.json"
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)["hands"]


def plot_retarget_improvement(hands):
    """输入：正式结果；输出：基线到最终方法哑铃图；作用：突出净提升。"""
    labels = ["LinkerHand O6", "XHand", "WujiHand"]
    keys = ["linker", "xhand", "wuji"]
    baseline = np.array([22.2, 56.5, 47.6])
    final = np.array([hands[k]["metrics"]["reference_isaac_success_count"] / 10 for k in keys])
    y = np.arange(3)
    fig, ax = plt.subplots(figsize=(9.6, 5.2))
    for i in range(3):
        ax.plot([baseline[i], final[i]], [i, i], color="#C5CBD1", linewidth=5,
                solid_capstyle="round", zorder=1)
    ax.scatter(baseline, y, s=105, color=GRAY, label="运动学基线", zorder=2)
    ax.scatter(final, y, s=120, color=GREEN, label="最终物理反馈方法", zorder=3)
    for i in range(3):
        ax.text(baseline[i] - 1.5, i - 0.13, f"{baseline[i]:.1f}%", ha="right", color="#66717E")
        ax.text(final[i] + 1.5, i - 0.13, f"{final[i]:.1f}%", ha="left", color=GREEN,
                fontweight="bold")
        ax.text((baseline[i] + final[i]) / 2, i + 0.22,
                f"+{final[i] - baseline[i]:.1f} pp", ha="center", color=INK, fontsize=10)
    ax.set_yticks(y, labels)
    ax.set_ylim(2.42, -0.42)
    ax.set_xlim(0, 82)
    ax.set_xlabel("参考 Isaac Gym 成功率（%）")
    ax.set_title("重定向：物理反馈优化带来的成功率提升", pad=16)
    ax.legend(ncol=2, loc="upper right")
    clean_axis(ax, grid=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8, alpha=0.7)
    finish(fig, "retarget_baseline_vs_final")


def plot_retarget_quality_funnel(hands):
    """输入：正式结果；输出：四层质量筛选图；作用：区分到达目标和稳定运输。"""
    stages = ["曾到达目标区", "末段仍在目标区", "稳定保持", "稳定运输"]
    fields = ["reference_isaac_success_count", "reference_isaac_terminal_success_count",
              "stable_physics_success_count", "transport_quality_success_count"]
    labels = {"linker": "LinkerHand O6", "xhand": "XHand", "wuji": "WujiHand"}
    colors = {"linker": BLUE, "xhand": ORANGE, "wuji": GREEN}
    x = np.arange(4)
    fig, ax = plt.subplots(figsize=(10.2, 5.4))
    for key in ("linker", "xhand", "wuji"):
        values = [hands[key]["metrics"][field] / 10 for field in fields]
        ax.plot(x, values, marker="o", markersize=7.5, linewidth=2.4,
                color=colors[key], label=labels[key])
        ax.text(3.08, values[-1], f"{values[-1]:.1f}%", va="center",
                color=colors[key], fontweight="bold")
    ax.set_xticks(x, stages)
    ax.set_ylabel("通过比例（%）")
    ax.set_ylim(0, 80)
    ax.set_xlim(-0.12, 3.45)
    ax.set_title("重定向：成功不是单帧抬起，而是逐层通过质量筛选")
    ax.legend(ncol=3, loc="lower center")
    clean_axis(ax)
    finish(fig, "retarget_quality_funnel")


def main():
    """输入：两项目冻结结果；输出：四组 PPT 图；作用：一键重建全部统计资产。"""
    configure_style()
    shadow = load_shadow_summary()
    retarget = load_retarget_results()
    plot_shadow_ablation(shadow)
    plot_shadow_categories(shadow)
    plot_retarget_improvement(retarget)
    plot_retarget_quality_funnel(retarget)
    print(f"Figures saved to {OUT}")


if __name__ == "__main__":
    main()
