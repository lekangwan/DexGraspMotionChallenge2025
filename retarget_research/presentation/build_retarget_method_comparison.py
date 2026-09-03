"""生成关键点法与向量法的科研风格对照框图。"""

from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


OUT = Path(__file__).resolve().parent / "assets" / "figures"
INK = "#20252B"
MUTED = "#66717D"
LINE = "#CBD3DB"
PURPLE = "#332288"
TEAL = "#228C7B"


def rounded_box(ax, x, y, w, h, title, body, edge, face):
    """输入位置和文字，输出带标题的圆角方法节点。"""
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.07",
        facecolor=face, edgecolor=edge, linewidth=1.25))
    ax.text(x + 0.18, y + h - 0.21, title, fontsize=10.5,
            fontweight="bold", color=edge, va="top")
    ax.text(x + 0.18, y + h - 0.54, body, fontsize=8.4,
            color=INK, va="top", linespacing=1.25)


def arrow(ax, start, end, color=MUTED):
    """输入两个坐标，输出统一样式的流程箭头。"""
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=12,
        linewidth=1.15, color=color, shrinkA=2, shrinkB=2))


def build_figure():
    """绘制两种重定向方法的输入、误差、优化变量和输出。"""
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Noto Sans CJK SC", "Noto Sans CJK JP", "DejaVu Sans"],
        "svg.fonttype": "none",
        "axes.unicode_minus": False,
    })
    fig, ax = plt.subplots(figsize=(13.33, 7.5))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set_xlim(0, 13.33)
    ax.set_ylim(0, 7.5)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(0.62, 7.02, "两种轨迹重定向方法的计算逻辑",
            fontsize=22, fontweight="bold", color=INK, va="center")
    ax.text(0.64, 6.62, "共同目标：逐帧求出目标手的手腕位姿和关节角",
            fontsize=11, color=MUTED, va="center")

    rounded_box(ax, 4.42, 5.55, 4.50, 0.78,
                "共同输入：Shadow Hand当前帧",
                "28维轨迹帧 → 正向运动学（FK） → 源手空间几何",
                "#4E5965", "#F1F3F5")

    ax.add_patch(Rectangle((0.45, 0.56), 6.05, 4.58,
                           facecolor="#FBFAFE", edgecolor="#CDC8E4", linewidth=1.0))
    ax.add_patch(Rectangle((6.82, 0.56), 6.05, 4.58,
                           facecolor="#F8FCFB", edgecolor="#BFD9D4", linewidth=1.0))
    ax.text(0.74, 4.82, "a  关键点法", fontsize=15, fontweight="bold", color=PURPLE)
    ax.text(2.62, 4.82, "XHand / WujiHand", fontsize=9.5, color=MUTED, va="center")
    ax.text(7.11, 4.82, "b  功能向量法", fontsize=15, fontweight="bold", color=TEAL)
    ax.text(9.44, 4.82, "LinkerHand O6", fontsize=9.5, color=MUTED, va="center")

    arrow(ax, (5.44, 5.55), (3.41, 5.12), PURPLE)
    arrow(ax, (7.90, 5.55), (9.91, 5.12), TEAL)

    rounded_box(ax, 0.76, 3.70, 2.42, 0.82,
                "提取目标数据", "掌心、指节、指尖\n共15个语义点", PURPLE, "#EEEAF8")
    rounded_box(ax, 3.52, 3.70, 2.63, 0.82,
                "计算目标手当前值", "候选手腕+关节角 → FK\n得到对应的15个点", PURPLE, "#EEEAF8")
    arrow(ax, (3.18, 4.11), (3.52, 4.11), PURPLE)

    rounded_box(ax, 1.35, 2.45, 4.22, 0.82,
                "计算误差", "15对点的平均平方距离 × 1000\n+ 相邻帧关节/手腕变化惩罚", PURPLE, "white")
    arrow(ax, (2.00, 3.70), (2.68, 3.27), PURPLE)
    arrow(ax, (4.84, 3.70), (4.22, 3.27), PURPLE)

    rounded_box(ax, 1.35, 1.16, 4.22, 0.82,
                "SLSQP反复调整", "改变目标手的手腕位姿和关节角\n直到对应点尽量重合", PURPLE, "white")
    arrow(ax, (3.46, 2.45), (3.46, 1.98), PURPLE)
    ax.text(3.46, 0.83, "输出：目标手当前帧动作",
            fontsize=10.5, fontweight="bold", color=PURPLE, ha="center")
    arrow(ax, (3.46, 1.16), (3.46, 0.98), PURPLE)

    rounded_box(ax, 7.13, 3.70, 2.56, 0.82,
                "构造Shadow目标", "10条位置向量\n5条单位方向向量", TEAL, "#E6F3F0")
    rounded_box(ax, 9.98, 3.70, 2.56, 0.82,
                "适配Linker尺寸", "用零姿态长度比例\n缩放位置向量", TEAL, "#E6F3F0")
    arrow(ax, (9.69, 4.11), (9.98, 4.11), TEAL)

    rounded_box(ax, 7.40, 2.45, 4.83, 0.82,
                "计算Linker当前值和误差",
                "FK得到当前向量；Huber比较长度与方向\n+ 掌心、接触锚、抓紧和相邻帧约束", TEAL, "white")
    arrow(ax, (8.40, 3.70), (9.10, 3.27), TEAL)
    arrow(ax, (11.26, 3.70), (10.66, 3.27), TEAL)

    rounded_box(ax, 7.40, 1.16, 4.83, 0.82,
                "SLSQP反复调整", "改变Linker手腕和6个主动量\n优先恢复指尖对置与包覆", TEAL, "white")
    arrow(ax, (9.82, 2.45), (9.82, 1.98), TEAL)
    ax.text(9.82, 0.83, "输出：Linker当前帧动作",
            fontsize=10.5, fontweight="bold", color=TEAL, ha="center")
    arrow(ax, (9.82, 1.16), (9.82, 0.98), TEAL)

    ax.text(6.66, 0.22,
            "关键区别：关键点法追求“对应位置接近”；向量法追求“尺寸适配后的方向、对置和抓取功能接近”。",
            fontsize=10.2, color=INK, ha="center")

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "03_keypoint_vs_vector_flow.svg",
                facecolor="white", bbox_inches="tight", pad_inches=0.08)
    fig.savefig(OUT / "03_keypoint_vs_vector_flow.png", dpi=260,
                facecolor="white", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


if __name__ == "__main__":
    build_figure()
