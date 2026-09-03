from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "assets"
REGULAR = font_manager.FontProperties(
    fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc"
)
BOLD = font_manager.FontProperties(
    fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
)


def box(ax, x, y, w, h, title, lines, face, edge):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.8, edgecolor=edge, facecolor=face,
    ))
    ax.text(x + w / 2, y + h * 0.70, title, ha="center", va="center",
            fontsize=22, color="#123B5D", fontproperties=BOLD)
    for i, line in enumerate(lines):
        ax.text(x + w / 2, y + h * (0.42 - i * 0.20), line,
                ha="center", va="center", fontsize=16.5,
                color="#29485F", fontproperties=REGULAR)


def arrow(ax, x1, y1, x2, y2, color="#405464", dashed=False):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=22,
        linewidth=2.4, color=color, linestyle="--" if dashed else "-",
    ))


def main():
    plt.rcParams["svg.fonttype"] = "path"
    fig, ax = plt.subplots(figsize=(19.2, 10.8), dpi=150)
    fig.patch.set_facecolor("#FAFCFD")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.035, 0.95, "进阶策略：由重定向示范学习自主抓取",
            fontsize=34, color="#123B5D", fontproperties=BOLD, va="center")
    ax.text(0.035, 0.905,
            "核心：训练时学习专家动作；测试时只使用初始状态、执行相位，以及 Wuji 的短时真实反馈",
            fontsize=18, color="#496274", fontproperties=REGULAR, va="center")

    ax.add_patch(FancyBboxPatch((0.025, 0.57), 0.95, 0.29,
                 boxstyle="round,pad=0.01,rounding_size=0.018",
                 facecolor="#F2F7FA", edgecolor="#D8E4EA", linewidth=1.5))
    ax.text(0.045, 0.82, "训练阶段", fontsize=24, color="#123B5D",
            fontproperties=BOLD, va="center")
    box(ax, 0.055, 0.625, 0.19, 0.135, "成功重定向轨迹",
        ["手—物状态 + 专家动作", "只保留稳定成功样本"], "#E8F1F7", "#B9CEDB")
    box(ax, 0.295, 0.625, 0.19, 0.135, "构造监督样本",
        ["输入：初始观测 o0 + 相位 pt", "标签：a*t − a0"], "#EAF6F3", "#A9D8CF")
    box(ax, 0.535, 0.625, 0.18, 0.135, "MLP 策略网络",
        ["512 → 512 → 384", "监督损失：MSE / Huber"], "#FFF4E8", "#E9C79F")
    box(ax, 0.765, 0.625, 0.18, 0.135, "学得自主策略",
        ["phase 控制动作进程", "不记忆测试轨迹"], "#EDEAF8", "#C7BEE4")
    arrow(ax, 0.245, 0.692, 0.292, 0.692)
    arrow(ax, 0.485, 0.692, 0.532, 0.692)
    arrow(ax, 0.715, 0.692, 0.762, 0.692)

    ax.add_patch(FancyBboxPatch((0.025, 0.105), 0.95, 0.415,
                 boxstyle="round,pad=0.01,rounding_size=0.018",
                 facecolor="#FFFFFF", edgecolor="#D8E4EA", linewidth=1.5))
    ax.text(0.045, 0.48, "自主测试阶段（每个物理步 60 Hz）",
            fontsize=24, color="#123B5D", fontproperties=BOLD, va="center")
    box(ax, 0.055, 0.285, 0.20, 0.14, "Episode 开始",
        ["保存初始手—物观测 o0", "张手初始命令 a0"], "#E8F1F7", "#B9CEDB")
    box(ax, 0.305, 0.285, 0.16, 0.14, "策略输入",
        ["初始观测 o0", "相位 pt：0 → 1"], "#EAF6F3", "#A9D8CF")
    box(ax, 0.515, 0.285, 0.18, 0.14, "MLP 生成动作",
        ["预测动作增量 Δat", "at = a0 + Δat"], "#FFF4E8", "#E9C79F")
    box(ax, 0.745, 0.285, 0.20, 0.14, "Isaac Gym 执行",
        ["手腕 + 手指位置命令", "PD控制、接触与抬升"], "#EDEAF8", "#C7BEE4")
    arrow(ax, 0.255, 0.355, 0.302, 0.355)
    arrow(ax, 0.465, 0.355, 0.512, 0.355)
    arrow(ax, 0.695, 0.355, 0.742, 0.355)

    ax.add_patch(FancyBboxPatch((0.305, 0.145), 0.64, 0.085,
                 boxstyle="round,pad=0.01,rounding_size=0.014",
                 facecolor="#E7F7F5", edgecolor="#17A398",
                 linewidth=1.8, linestyle="--"))
    ax.text(0.325, 0.198, "Wuji 额外反馈：", fontsize=18.5, color="#087E77",
            fontproperties=BOLD, va="center")
    ax.text(0.445, 0.198,
            "最近三帧真实状态 → 反馈网络 → 有界修正 0.1·tanh(ct) → 加到名义动作",
            fontsize=17, color="#29485F", fontproperties=REGULAR, va="center")
    arrow(ax, 0.845, 0.285, 0.845, 0.232, color="#17A398", dashed=True)
    arrow(ax, 0.63, 0.145, 0.61, 0.282, color="#17A398", dashed=True)

    ax.add_patch(FancyBboxPatch((0.025, 0.035), 0.95, 0.045,
                 boxstyle="round,pad=0.006,rounding_size=0.01",
                 facecolor="#123B5D", edgecolor="#123B5D"))
    ax.text(0.5, 0.057,
            "测试约束：不读取未来专家动作 · 不检索参考轨迹 · 不使用专家手腕 · 仅第一帧用于任务初始化",
            ha="center", va="center", fontsize=17, color="white", fontproperties=BOLD)

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "advanced_policy_pipeline.png", dpi=200,
                facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.12)
    fig.savefig(OUT / "advanced_policy_pipeline_embedded.svg",
                facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


if __name__ == "__main__":
    main()
