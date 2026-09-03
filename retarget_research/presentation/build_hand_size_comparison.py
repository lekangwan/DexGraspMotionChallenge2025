"""生成四种灵巧手几何尺寸对比图，用于PPT。"""

from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle


OUT = Path(__file__).resolve().parent / "assets" / "figures"

HANDS = ["Shadow Hand", "LinkerHand O6", "XHand", "WujiHand"]
COLORS = ["#5B6573", "#332288", "#44AA99", "#CC6677"]
VALUES = {
    "手掌纵向总长度": [210.0, 177.0, 208.0, 206.0],
    "食指根—小指根宽度": [67.0, 52.6, 62.5, 63.4],
    "手腕—中指根": [99.0, 103.0, 108.0, 90.0],
    "中指根—指尖": [100.0, 74.0, 100.0, 104.0],
}


def build_figure():
    """输入四手尺寸数据，输出适合16:9幻灯片的SVG和PNG。"""
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Noto Sans CJK SC", "Noto Sans CJK JP", "DejaVu Sans"],
        "svg.fonttype": "none",
        "axes.unicode_minus": False,
    })

    fig, ax = plt.subplots(figsize=(13.33, 7.5))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set_xlim(0, 13.33)
    ax.set_ylim(0, 7.5)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ink = "#20252B"
    muted = "#68717C"
    line = "#D8DDE3"
    cell = "#F5F7F9"

    ax.text(0.65, 6.92, "四种灵巧手的实际几何尺寸", fontsize=23,
            fontweight="bold", color=ink, va="center")
    ax.text(0.67, 6.47, "同一测量口径 · 单位：mm", fontsize=11.5,
            color=muted, va="center")

    left = 0.65
    label_w = 2.35
    metric_w = 2.43
    gap = 0.12
    top = 5.88
    header_h = 0.72
    row_h = 0.92

    ax.add_patch(Rectangle((left, top), label_w, header_h,
                           facecolor="#ECEFF3", edgecolor="none"))
    ax.text(left + 0.24, top + header_h / 2, "机械手",
            fontsize=11.5, fontweight="bold", color=ink, va="center")

    maxima = [220.0, 72.0, 115.0, 110.0]
    for col, (metric, _) in enumerate(VALUES.items()):
        x = left + label_w + gap + col * (metric_w + gap)
        ax.add_patch(Rectangle((x, top), metric_w, header_h,
                               facecolor="#ECEFF3", edgecolor="none"))
        title = metric.replace("—", "\n—") if metric == "食指根—小指根宽度" else metric
        ax.text(x + metric_w / 2, top + header_h / 2, title,
                fontsize=10.2, fontweight="bold", color=ink,
                ha="center", va="center", linespacing=1.05)

    for row, (hand, color) in enumerate(zip(HANDS, COLORS)):
        y = top - (row + 1) * row_h
        ax.add_patch(Rectangle((left, y), label_w, row_h - 0.07,
                               facecolor="white", edgecolor=line, linewidth=0.7))
        ax.add_patch(Rectangle((left, y), 0.09, row_h - 0.07,
                               facecolor=color, edgecolor="none"))
        ax.text(left + 0.25, y + (row_h - 0.07) / 2, hand,
                fontsize=11.3, fontweight="bold", color=ink, va="center")
        if row == 0:
            ax.text(left + label_w - 0.20, y + (row_h - 0.07) / 2,
                    "源手", fontsize=8.8, color=muted, ha="right", va="center")

        for col, (_, values) in enumerate(VALUES.items()):
            x = left + label_w + gap + col * (metric_w + gap)
            ax.add_patch(Rectangle((x, y), metric_w, row_h - 0.07,
                                   facecolor=cell, edgecolor=line, linewidth=0.7))
            bar_x = x + 0.17
            bar_y = y + 0.23
            bar_w = 1.45
            bar_h = 0.31
            ax.add_patch(FancyBboxPatch(
                (bar_x, bar_y), bar_w, bar_h,
                boxstyle="round,pad=0,rounding_size=0.07",
                facecolor="#E1E5E9", edgecolor="none"))
            actual_w = bar_w * values[row] / maxima[col]
            ax.add_patch(FancyBboxPatch(
                (bar_x, bar_y), actual_w, bar_h,
                boxstyle="round,pad=0,rounding_size=0.07",
                facecolor=color, edgecolor="none"))
            value_text = f"{values[row]:.0f}" if values[row].is_integer() else f"{values[row]:.1f}"
            ax.text(x + metric_w - 0.17, y + (row_h - 0.07) / 2,
                    value_text, fontsize=12.2, fontweight="bold",
                    color=ink, ha="right", va="center")

    note_y = 0.77
    ax.add_patch(FancyBboxPatch(
        (0.65, note_y), 12.03, 0.82,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        facecolor="#F2F5F8", edgecolor="#CBD3DB", linewidth=0.8))
    ax.text(0.94, note_y + 0.53,
            "尺寸关系：XHand、WujiHand与Shadow Hand接近；LinkerHand更短、更窄",
            fontsize=12.0, fontweight="bold", color=ink, va="center")
    ax.text(0.94, note_y + 0.23,
            "LinkerHand相对Shadow Hand：总长度约 −16%，掌宽约 −22%，中指长度约 −26%",
            fontsize=10.2, color=muted, va="center")
    ax.text(0.67, 0.35,
            "测量口径：URDF原始可视网格，关节角为0，hand_scale = 1.0；数值取约数。",
            fontsize=9.3, color=muted, va="center")

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "02_four_hand_size_comparison.svg",
                facecolor="white", bbox_inches="tight", pad_inches=0.08)
    fig.savefig(OUT / "02_four_hand_size_comparison.png", dpi=260,
                facecolor="white", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


if __name__ == "__main__":
    build_figure()
