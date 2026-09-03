from __future__ import annotations

import json
from pathlib import Path

import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "retarget_research/presentation/assets"
FIG = OUT / "figures"
ANIM = OUT / "animations"
FRAME = OUT / "frames"
VIDEO = OUT / "videos"
RUNS = ROOT / "retarget_research/advanced_policy/runs"

BLUE = "#332288"
LIGHT_BLUE = "#E8E7F2"
ORANGE = "#44AA99"
LIGHT_ORANGE = "#E0F1ED"
GREEN = "#CC6677"
LIGHT_GREEN = "#F4E6EA"
GOLD = "#DDAA33"
RED = "#BB5566"
INK = "#252525"
MID = "#666666"
LIGHT = "#E5E5E5"


def setup_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Noto Sans CJK SC", "Noto Sans CJK JP", "DejaVu Sans"],
        "font.size": 10,
        "axes.edgecolor": INK,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "svg.fonttype": "none",
    })


def save_figure(fig: plt.Figure, stem: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{stem}.svg", bbox_inches="tight", facecolor="white")
    fig.savefig(FIG / f"{stem}.png", dpi=260, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def box(ax, xy, width, height, text, face="white", edge=INK, fontsize=10, weight="normal"):
    patch = Rectangle(xy, width, height, facecolor=face, edgecolor=edge, linewidth=1.2)
    ax.add_patch(patch)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text,
            ha="center", va="center", fontsize=fontsize, color=INK, weight=weight)
    return patch


def arrow(ax, start, end, color=MID, connection="arc3"):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12,
                                 linewidth=1.15, color=color,
                                 connectionstyle=connection, shrinkA=2, shrinkB=2))


def task_overview() -> None:
    shadow_video = ROOT / "FINAL_SUBMISSION/renders/bottle_success.mp4"
    shared = ROOT / "retarget_research/advanced_policy/videos/expert_success_v1"
    target_paths = [
        shared / "linker/shared_success_5_sem-WineBottle-f331ad8d0e6654ef8f992b1fe7075c8f_source10.mp4",
        shared / "xhand/shared_success_5_sem-WineBottle-f331ad8d0e6654ef8f992b1fe7075c8f_source10.mp4",
        shared / "wuji/shared_success_5_sem-WineBottle-f331ad8d0e6654ef8f992b1fe7075c8f_source10.mp4",
    ]
    outcome_video = VIDEO / "linker_function_vector.mp4"
    fig, ax = plt.subplots(figsize=(14.2, 4.8))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 5)
    ax.axis("off")

    stages = [
        (0.15, 3.05, "a", "源抓取轨迹"),
        (3.35, 10.55, "b", "跨构型轨迹重定向"),
        (10.85, 15.85, "c", "物理验证与闭环执行"),
    ]
    for left, right, panel, title in stages:
        ax.add_patch(Rectangle((left, 0.20), right - left, 4.55,
                               facecolor="#FAFAFA", edgecolor="#C8C8C8", linewidth=0.9))
        ax.add_patch(Rectangle((left, 4.18), right - left, 0.57,
                               facecolor="#F0F0F0", edgecolor="none"))
        ax.text(left + 0.18, 4.46, panel, ha="left", va="center", fontsize=11, weight="bold")
        ax.text(left + 0.52, 4.46, title, ha="left", va="center", fontsize=10.5, weight="bold")

    source = np.asarray(read_frame(shadow_video, 20).crop((155, 0, 505, 350)))
    ax.imshow(source, extent=(0.45, 2.75, 0.95, 3.65), aspect="auto", zorder=2)
    ax.text(1.60, 0.62, "Shadow Hand 专家轨迹", ha="center", fontsize=8.8, color=INK)
    arrow(ax, (2.78, 2.30), (3.62, 2.30), INK)

    palm = np.asarray([4.65, 1.30])
    ax.add_patch(Rectangle((4.25, 0.94), 0.8, 0.68, facecolor="white", edgecolor=BLUE, linewidth=1.2))
    branches = [
        [(-0.64, 0.34), (-0.70, 1.05), (-0.72, 1.63)],
        [(-0.23, 0.40), (-0.25, 1.25), (-0.24, 1.93)],
        [(0.20, 0.37), (0.27, 1.16), (0.34, 1.78)],
        [(0.57, 0.25), (0.70, 0.90), (0.80, 1.42)],
        [(-0.48, -0.08), (-0.95, -0.37)],
    ]
    for chain in branches:
        pts = np.vstack([palm, palm + np.asarray(chain)])
        ax.plot(pts[:, 0], pts[:, 1], color=BLUE, lw=1.65)
        ax.scatter(pts[1:, 0], pts[1:, 1], s=16, facecolor="white", edgecolor=BLUE, linewidth=0.9, zorder=4)
    ax.text(4.66, 3.62, "15个语义关键点", ha="center", fontsize=8.8, weight="bold")
    ax.text(4.66, 3.30, "正向运动学 + 优化 q", ha="center", fontsize=8.2, color=MID)
    arrow(ax, (5.65, 2.28), (6.52, 2.28), INK)

    target_labels = [("Linker O6", "6 DoF", BLUE), ("XHand", "12 DoF", ORANGE), ("WujiHand", "20 DoF", GREEN)]
    ys = [(3.08, 4.02), (1.91, 2.85), (0.74, 1.68)]
    for path, (name, dof, color), (bottom, top) in zip(target_paths, target_labels, ys):
        image = np.asarray(read_frame(path, 55).crop((145, 5, 495, 355)))
        ax.imshow(image, extent=(6.72, 8.52, bottom, top), aspect="auto", zorder=2)
        ax.add_patch(Rectangle((6.72, bottom), 1.80, top - bottom, facecolor="none", edgecolor=color, linewidth=1.3))
        ax.text(8.72, (bottom + top) / 2 + 0.12, name, ha="left", va="center", fontsize=8.8, weight="bold")
        ax.text(8.72, (bottom + top) / 2 - 0.18, dof, ha="left", va="center", fontsize=8.2, color=MID)
    ax.text(9.56, 0.38, "每条源轨迹分别适配三种目标构型", ha="center", fontsize=8.2, color=MID)
    arrow(ax, (10.52, 2.30), (11.08, 2.30), INK)

    if outcome_video.is_file():
        outcome = np.asarray(read_frame(outcome_video, 50))
        ax.imshow(outcome, extent=(11.18, 13.55, 1.28, 3.60), aspect="auto", zorder=2)
        ax.add_patch(Rectangle((11.18, 1.28), 2.37, 2.32, facecolor="none", edgecolor=INK, linewidth=0.9))
    ax.text(12.36, 0.92, "参考目标区域 + 稳定运输", ha="center", fontsize=8.8, weight="bold")
    box(ax, (14.05, 2.55), 1.38, 0.70, "参考轨迹\nq_ref", LIGHT_BLUE, BLUE, 8.4, "bold")
    box(ax, (14.05, 1.38), 1.38, 0.70, "策略残差\nΔq", LIGHT_ORANGE, ORANGE, 8.4, "bold")
    ax.add_patch(Circle((15.60, 2.31), 0.18, facecolor="white", edgecolor=INK, linewidth=1.0))
    ax.text(15.60, 2.31, "+", ha="center", va="center", fontsize=12)
    arrow(ax, (15.43, 2.88), (15.52, 2.46), BLUE)
    arrow(ax, (15.43, 1.72), (15.52, 2.16), ORANGE)
    ax.text(14.74, 3.68, "保留主路线", ha="center", fontsize=8.0, color=MID)
    ax.text(14.74, 0.97, "只学习必要修正", ha="center", fontsize=8.0, color=MID)
    save_figure(fig, "01_task_overview")


def draw_hand(ax, points, color, label):
    palm = points["palm"]
    ax.add_patch(Rectangle((palm[0] - 0.42, palm[1] - 0.33), 0.84, 0.66,
                           facecolor="white", edgecolor=color, linewidth=1.8))
    for finger in ("index", "middle", "ring", "little", "thumb"):
        chain = np.asarray(points[finger])
        chain = np.vstack([palm, chain])
        ax.plot(chain[:, 0], chain[:, 1], color=color, lw=2.1, solid_capstyle="round")
    ordered = [palm]
    for finger in ("index", "middle", "ring", "little"):
        ordered.extend(points[finger])
    ordered.extend(points["thumb"])
    ordered = np.asarray(ordered)
    ax.scatter(ordered[:, 0], ordered[:, 1], s=31, facecolor="white", edgecolor=color,
               linewidth=1.4, zorder=3)
    ax.scatter(ordered[[3, 6, 9, 12, 14], 0], ordered[[3, 6, 9, 12, 14], 1],
               s=42, facecolor=color, edgecolor="white", linewidth=0.6, zorder=4)
    ax.text(palm[0], -0.05, label, ha="center", va="top", fontsize=11, weight="bold", color=color)
    return ordered


def semantic_keypoints() -> None:
    source = {
        "palm": (0.0, 0.8),
        "index": [(-0.72, 1.55), (-0.82, 2.35), (-0.84, 3.05)],
        "middle": [(-0.25, 1.62), (-0.26, 2.55), (-0.24, 3.38)],
        "ring": [(0.25, 1.58), (0.31, 2.45), (0.38, 3.18)],
        "little": [(0.68, 1.45), (0.82, 2.15), (0.94, 2.72)],
        "thumb": [(-0.62, 0.55), (-1.18, 0.18)],
    }
    target = {
        "palm": (5.1, 0.8),
        "index": [(4.48, 1.48), (4.36, 2.15), (4.33, 2.72)],
        "middle": [(4.88, 1.58), (4.84, 2.38), (4.84, 3.02)],
        "ring": [(5.30, 1.55), (5.35, 2.31), (5.40, 2.90)],
        "little": [(5.68, 1.43), (5.79, 2.05), (5.89, 2.52)],
        "thumb": [(4.52, 0.62), (4.02, 0.30)],
    }
    fig, ax = plt.subplots(figsize=(10.2, 5.2))
    left = draw_hand(ax, source, BLUE, "Shadow Hand：15个语义关键点")
    right = draw_hand(ax, target, ORANGE, "目标手：对应位置")
    for i in range(15):
        ax.plot([left[i, 0], right[i, 0]], [left[i, 1], right[i, 1]],
                color="#A8A8A8", lw=0.7, linestyle=(0, (3, 4)), zorder=0)
    ax.text(2.55, 3.58, "同名部位建立对应", ha="center", va="center", color=INK, fontsize=11, weight="bold")
    ax.text(2.55, 3.30, "指根 · 中段 · 指尖 · 掌心", ha="center", color=MID, fontsize=9)
    arrow(ax, (2.05, 0.70), (3.05, 0.70), INK)
    ax.text(2.55, 0.95, "优化关节角 q", ha="center", fontsize=10, color=INK)
    ax.set_xlim(-1.7, 6.5)
    ax.set_ylim(-0.35, 3.9)
    ax.set_aspect("equal")
    ax.axis("off")
    save_figure(fig, "03_semantic_keypoint_retargeting")


def linker_principle() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4))
    for ax, functional in zip(axes, (False, True)):
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_xlim(-2.2, 2.2)
        ax.set_ylim(-1.6, 2.2)
        obj = Circle((0, 0.25), 0.68, facecolor="#E8E8E8", edgecolor=INK, linewidth=1.2)
        ax.add_patch(obj)
        palm_y = 1.45 if not functional else 1.20
        ax.add_patch(Rectangle((-0.55, palm_y), 1.1, 0.38, facecolor="white", edgecolor=BLUE, linewidth=1.8))
        if functional:
            tips = [(-0.65, 0.55), (-0.35, -0.25), (0.35, -0.25), (0.65, 0.55)]
            starts = [(-0.45, palm_y), (-0.15, palm_y), (0.15, palm_y), (0.45, palm_y)]
        else:
            tips = [(-1.12, 0.78), (-0.62, 0.82), (0.62, 0.82), (1.12, 0.78)]
            starts = [(-0.45, palm_y), (-0.15, palm_y), (0.15, palm_y), (0.45, palm_y)]
        for s, t in zip(starts, tips):
            mid = ((s[0] + t[0]) / 2, (s[1] + t[1]) / 2 + (0.10 if functional else 0.0))
            ax.plot([s[0], mid[0], t[0]], [s[1], mid[1], t[1]], color=BLUE, lw=3, solid_capstyle="round")
            ax.scatter(*t, s=34, color=BLUE, zorder=3)
        thumb_tip = (-0.58, 0.02) if functional else (-1.22, 0.25)
        ax.plot([-0.52, -0.92, thumb_tip[0]], [palm_y + 0.05, 0.72, thumb_tip[1]], color=BLUE, lw=3, solid_capstyle="round")
        ax.scatter(*thumb_tip, s=34, color=BLUE)
        if functional:
            contacts = [(-0.58, 0.55), (-0.35, -0.25), (0.35, -0.25), (0.58, 0.55), (-0.58, 0.02)]
            for p in contacts:
                ax.add_patch(Circle(p, 0.07, facecolor=RED, edgecolor="white", linewidth=0.6, zorder=4))
            title = "抓取功能优先"
            subtitle = "姿态不必相同，但形成包覆与对向接触"
            color = ORANGE
        else:
            title = "几何姿态优先"
            subtitle = "外形接近，但6 DoF无法建立足够接触"
            color = RED
        ax.text(0, 2.08, title, ha="center", va="top", fontsize=12, weight="bold", color=color)
        ax.text(0, -1.28, subtitle, ha="center", va="top", fontsize=9.5, color=MID)
    fig.subplots_adjust(wspace=0.10)
    save_figure(fig, "04_linker_pose_vs_function_principle")


def retarget_results() -> None:
    names = ["LinkerHand O6", "XHand", "WujiHand"]
    stable = np.asarray([43.9, 69.5, 71.8])
    transport = np.asarray([35.0, 63.0, 66.2])
    x = np.arange(3)
    fig, ax = plt.subplots(figsize=(7.6, 4.35))
    bars = ax.bar(x, stable, width=0.48, color=[BLUE, ORANGE, GREEN], edgecolor="none")
    ax.bar_label(bars, labels=[f"{v:.1f}%" for v in stable], padding=3, fontsize=11, weight="bold")
    for xi, value in zip(x, transport):
        ax.plot([xi - 0.22, xi + 0.22], [value, value], color=INK, lw=1.7)
        ax.scatter([xi], [value], s=20, facecolor="white", edgecolor=INK, zorder=4)
        ax.text(xi + 0.27, value, f"运输 {value:.1f}%", ha="left", va="center", fontsize=8.2, color=INK)
    ax.set_xticks(x, names)
    ax.set_ylabel("成功率（%）")
    ax.set_ylim(0, 80)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_yticks([0, 20, 40, 60, 80])
    ax.tick_params(direction="out", length=3.5, width=0.8)
    fig.tight_layout()
    save_figure(fig, "05_formal_1000_retarget_results")


def final_retargeting_pipeline() -> None:
    """绘制最终冻结的运动学初始化—两层物理CEM—确认流程。"""
    fig, ax = plt.subplots(figsize=(14.2, 4.8))
    ax.set_xlim(0, 15.7)
    ax.set_ylim(0, 5.1)
    ax.axis("off")

    stages = [
        (0.25, 1.55, 2.45, 2.05, "运动学初值", "X/W：15点匹配\nLinker：功能向量", LIGHT_BLUE, BLUE),
        (3.25, 1.55, 2.45, 2.05, "Global CEM", "腕部位姿 +\n成组闭合残差", LIGHT_ORANGE, ORANGE),
        (6.25, 1.55, 2.45, 2.05, "Rank-5 协同", r"$\Delta q=Bc$" + "\n分阶段平滑施加", "#F5EEDC", GOLD),
        (9.25, 1.55, 2.45, 2.05, "重复确认", "候选与基线\n各独立重放2次", LIGHT_GREEN, GREEN),
        (12.25, 1.55, 2.95, 2.05, "最终轨迹", "参考成功 +\n稳定运输质量审计", "#F1F1F1", INK),
    ]
    for index, (x, y, width, height, title, subtitle, face, edge) in enumerate(stages, 1):
        ax.add_patch(Rectangle((x, y), width, height, facecolor=face, edgecolor=edge, linewidth=1.6))
        ax.text(x + 0.18, y + height - 0.30, f"{index}", ha="left", va="center",
                fontsize=9.5, color=edge, weight="bold")
        ax.text(x + width / 2, y + 1.33, title, ha="center", va="center",
                fontsize=13.5, color=INK, weight="bold")
        ax.text(x + width / 2, y + 0.67, subtitle, ha="center", va="center",
                fontsize=10.3, color=MID, linespacing=1.45)
    for left, right in zip(stages[:-1], stages[1:]):
        arrow(ax, (left[0] + left[2] + 0.10, 2.58), (right[0] - 0.10, 2.58), INK)

    ax.text(4.47, 4.35, "每个候选都完整执行240个PhysX步", ha="center",
            fontsize=10.3, color=ORANGE, weight="bold")
    ax.plot([3.48, 8.43], [4.08, 4.08], color=ORANGE, lw=1.0)
    ax.text(7.48, 0.72,
            "统一设置：population 8 · elite 2 · iteration 2    ｜    Linker在Rank-5前增加一次Global2",
            ha="center", fontsize=10.2, color=INK)
    ax.text(7.48, 0.30,
            "物理分数决定搜索方向；几何误差只负责提供可执行初值",
            ha="center", fontsize=9.5, color=MID)
    save_figure(fig, "05b_final_retargeting_pipeline")


def residual_loop() -> None:
    fig, ax = plt.subplots(figsize=(13.2, 4.7))
    ax.set_xlim(0, 15.2)
    ax.set_ylim(0, 5.0)
    ax.axis("off")

    panels = [
        (0.15, 2.85, "a", "状态输入"),
        (3.10, 10.65, "b", "参考引导的残差策略"),
        (10.90, 15.05, "c", "物理闭环执行"),
    ]
    for left, right, letter, title in panels:
        ax.add_patch(Rectangle((left, 0.35), right - left, 4.35,
                               facecolor="#FAFAFA", edgecolor="#C8C8C8", linewidth=0.85))
        ax.add_patch(Rectangle((left, 4.13), right - left, 0.57,
                               facecolor="#F0F0F0", edgecolor="none"))
        ax.text(left + 0.18, 4.41, letter, fontsize=10.5, weight="bold", va="center")
        ax.text(left + 0.51, 4.41, title, fontsize=10.2, weight="bold", va="center")

    # 输入不画成一个笼统大框，而是明确说明策略在每个时刻看见哪些量。
    box(ax, (0.55, 2.23), 1.90, 0.83, r"当前状态  $s_t$", "white", INK, 10, "bold")
    ax.text(1.50, 1.82, "手部本体状态", ha="center", fontsize=8.4, color=MID)
    ax.text(1.50, 1.49, "物体位姿与相对关系", ha="center", fontsize=8.4, color=MID)
    for y in (1.87, 1.54):
        ax.add_patch(Rectangle((0.77, y - 0.08), 0.18, 0.10,
                               facecolor="#B8B8B8", edgecolor="none"))
    arrow(ax, (2.46, 2.64), (3.52, 2.64), INK)

    # 上支路是固定参考，下支路由 PPO 更新。两者用固定语义色区分。
    ax.add_patch(Rectangle((3.55, 2.82), 5.15, 0.98,
                           facecolor=LIGHT_BLUE, edgecolor="#B8B5CF", linewidth=0.8))
    ax.add_patch(Rectangle((3.55, 1.22), 5.15, 1.18,
                           facecolor="#FAF3DE", edgecolor="#D8C99D", linewidth=0.8))
    ax.text(3.78, 3.52, "固定", fontsize=7.8, color=MID,
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="#B8B5CF", linewidth=0.6))
    box(ax, (4.50, 3.04), 1.72, 0.53, "重定向参考", "white", BLUE, 8.8, "bold")
    ax.text(7.23, 3.31, r"$q_{ref,t}$", ha="center", va="center", fontsize=10.0, weight="bold")
    arrow(ax, (6.22, 3.31), (6.78, 3.31), BLUE)

    ax.text(3.78, 2.10, "更新", fontsize=7.8, color=MID,
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="#D8C99D", linewidth=0.6))
    # 用三层节点简洁表达 MLP，而不是额外堆砌神经网络术语。
    layer_x = (4.65, 5.20, 5.75)
    counts = (3, 4, 2)
    for i in range(len(layer_x) - 1):
        ys0 = np.linspace(1.48, 2.08, counts[i])
        ys1 = np.linspace(1.57, 1.99, counts[i + 1])
        for y0 in ys0:
            for y1 in ys1:
                ax.plot([layer_x[i], layer_x[i + 1]], [y0, y1], color="#C4B889", lw=0.42, zorder=1)
    for x0, count in zip(layer_x, counts):
        ys = np.linspace(1.48 if count != 2 else 1.57, 2.08 if count != 2 else 1.99, count)
        ax.scatter(np.full(count, x0), ys, s=18, facecolor="white", edgecolor=GOLD, linewidth=0.7, zorder=2)
    ax.text(5.20, 1.33, "PPO Actor", ha="center", fontsize=8.3, weight="bold")
    ax.text(7.23, 1.81, r"$\Delta q_t$", ha="center", va="center", fontsize=10.0, weight="bold")
    arrow(ax, (5.92, 1.80), (6.78, 1.80), GOLD)

    ax.add_patch(Circle((9.48, 2.57), 0.25, facecolor="white", edgecolor=INK, linewidth=1.0))
    ax.text(9.48, 2.57, "+", ha="center", va="center", fontsize=14)
    arrow(ax, (7.68, 3.31), (9.23, 2.72), BLUE)
    arrow(ax, (7.68, 1.80), (9.23, 2.42), GOLD)
    ax.text(9.48, 1.17, r"$q_t=q_{ref,t}+\Delta q_t$", ha="center", fontsize=9.0, weight="bold")
    arrow(ax, (9.73, 2.57), (11.18, 2.57), INK)

    success_video = VIDEO / "xhand_residual_ppo_battery12.mp4"
    if success_video.is_file():
        result = np.asarray(read_frame(success_video, 50))
        ax.imshow(result, extent=(11.25, 14.67, 1.25, 3.76), aspect="auto", zorder=2)
        ax.add_patch(Rectangle((11.25, 1.25), 3.42, 2.51,
                               facecolor="none", edgecolor=INK, linewidth=0.8, zorder=3))
    else:
        box(ax, (11.25, 1.25), 3.42, 2.51, "Isaac Gym\n物理环境", "white", INK, 10, "bold")
    ax.text(12.96, 0.88, "执行动作并返回下一时刻状态", ha="center", fontsize=8.4, color=MID)
    arrow(ax, (12.80, 1.20), (1.48, 1.25), MID, "arc3,rad=-0.12")
    ax.text(7.12, 0.43, "物理反馈形成闭环：偏离参考轨迹后，策略在下一时刻继续修正", ha="center", fontsize=8.5, color=MID)
    save_figure(fig, "08_residual_ppo_closed_loop")


def moving_average(values, window=15):
    values = np.asarray(values, dtype=float)
    if len(values) < window:
        return values
    pad = np.pad(values, (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(pad, np.ones(window) / window, mode="valid")


def training_curves() -> None:
    logs = {
        "LinkerHand O6": RUNS / "residual_rl_general/linker/training_log.json",
        "XHand": RUNS / "residual_rl_general/xhand_official/training_log.json",
        "WujiHand": RUNS / "residual_rl_general/wuji_old/training_log.json",
    }
    colors = {"LinkerHand O6": BLUE, "XHand": ORANGE, "WujiHand": GREEN}
    rows = {name: json.loads(path.read_text()) for name, path in logs.items()}
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.0))
    for name, data in rows.items():
        x = np.asarray([r["iteration"] for r in data])
        success = np.asarray([r["success_rate"] for r in data]) * 100
        lift = np.asarray([r["mean_final_lift_m"] for r in data]) * 100
        axes[0].plot(x, success, color=colors[name], alpha=0.16, lw=0.75)
        axes[0].plot(x, moving_average(success), color=colors[name], lw=1.8, label=name)
        axes[1].plot(x, lift, color=colors[name], alpha=0.16, lw=0.75)
        axes[1].plot(x, moving_average(lift), color=colors[name], lw=1.8, label=name)
    axes[0].set_ylabel("训练批次成功率（%）")
    axes[1].set_ylabel("平均最终抬升（cm）")
    for ax in axes:
        ax.set_xlabel("PPO iteration")
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(direction="out", length=3.5, width=0.8)
    axes[0].legend(frameon=False, fontsize=8, ncol=1)
    fig.tight_layout()
    save_figure(fig, "09_ppo_training_success_and_lift")

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.0))
    for name, data in rows.items():
        x = np.asarray([r["iteration"] for r in data])
        policy = np.asarray([r["policy_loss"] for r in data])
        value = np.asarray([r["value_loss"] for r in data])
        axes[0].plot(x, policy, color=colors[name], alpha=0.14, lw=0.7)
        axes[0].plot(x, moving_average(policy), color=colors[name], lw=1.6, label=name)
        axes[1].plot(x, value, color=colors[name], alpha=0.14, lw=0.7)
        axes[1].plot(x, moving_average(value), color=colors[name], lw=1.6, label=name)
    axes[0].set_ylabel("Policy loss")
    axes[1].set_ylabel("Value loss")
    for ax in axes:
        ax.set_xlabel("PPO iteration")
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(direction="out", length=3.5, width=0.8)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save_figure(fig, "09b_ppo_training_losses")


def policy_results() -> None:
    names = ["LinkerHand O6", "XHand", "WujiHand"]
    supervised = np.asarray([12.3, 19.1, 12.9])
    ppo = np.asarray([54.3, 54.1, 45.4])
    full = np.asarray([12.0, 28.8, 30.2])
    x = np.arange(3)
    width = 0.31
    fig, ax = plt.subplots(figsize=(8.2, 4.45))
    a = ax.bar(x - width / 2, supervised, width, label="监督学习 / DAgger", color="#C7C7C7", edgecolor="none")
    b = ax.bar(x + width / 2, ppo, width, label="Residual PPO", color=BLUE, edgecolor="none")
    ax.bar_label(a, labels=[f"{v:.1f}%" for v in supervised], padding=2, fontsize=9)
    ax.bar_label(b, labels=[f"{v:.1f}%" for v in ppo], padding=2, fontsize=9, weight="bold")
    for xi, value in zip(x, full):
        ax.text(xi, 61.2, f"全量500条：{value:.1f}%", ha="center", va="center", fontsize=8.5, color=MID)
    ax.set_xticks(x, names)
    ax.set_ylabel("专家可行子集复现率（%）")
    ax.set_ylim(0, 66)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(direction="out", length=3.5, width=0.8)
    ax.legend(frameon=False, loc="upper left", ncol=2, fontsize=9)
    fig.tight_layout()
    save_figure(fig, "10_supervised_vs_residual_ppo")


def read_frame(path: Path, frame_index: int) -> Image.Image:
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"无法读取 {path} 第 {frame_index} 帧")
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    return image.crop((0, 52, image.width, image.height))


def font(size: int, bold=False):
    path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc" if bold else "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    return ImageFont.truetype(path, size)


def labeled_panel(image: Image.Image, label: str, width=560, height=420) -> Image.Image:
    image = image.copy()
    image.thumbnail((width, height - 48), Image.Resampling.LANCZOS)
    panel = Image.new("RGB", (width, height), "white")
    x = (width - image.width) // 2
    panel.paste(image, (x, 0))
    draw = ImageDraw.Draw(panel)
    bbox = draw.textbbox((0, 0), label, font=font(24, True))
    draw.text(((width - (bbox[2] - bbox[0])) / 2, height - 40), label, fill=INK, font=font(24, True))
    return panel


def three_hand_comparison() -> None:
    base = ROOT / "retarget_research/advanced_policy/videos/expert_success_v1"
    paths = [
        base / "linker/shared_success_5_sem-WineBottle-f331ad8d0e6654ef8f992b1fe7075c8f_source10.mp4",
        base / "xhand/shared_success_5_sem-WineBottle-f331ad8d0e6654ef8f992b1fe7075c8f_source10.mp4",
        base / "wuji/shared_success_5_sem-WineBottle-f331ad8d0e6654ef8f992b1fe7075c8f_source10.mp4",
    ]
    labels = ["LinkerHand O6 · 6 DoF", "XHand · 12 DoF", "WujiHand · 20 DoF"]
    panels = []
    for path, label in zip(paths, labels):
        image = read_frame(path, 55).crop((145, 5, 495, 355))
        panels.append(labeled_panel(image, label))
    canvas = Image.new("RGB", (sum(p.width for p in panels), panels[0].height), "white")
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 0))
        x += panel.width
    FRAME.mkdir(parents=True, exist_ok=True)
    canvas.save(FRAME / "02_same_winebottle_three_target_hands.png")


def cover_composite() -> None:
    shadow = ROOT / "FINAL_SUBMISSION/renders/bottle_success.mp4"
    base = ROOT / "retarget_research/advanced_policy/videos/expert_success_v1"
    target_paths = [
        base / "linker/shared_success_5_sem-WineBottle-f331ad8d0e6654ef8f992b1fe7075c8f_source10.mp4",
        base / "xhand/shared_success_5_sem-WineBottle-f331ad8d0e6654ef8f992b1fe7075c8f_source10.mp4",
        base / "wuji/shared_success_5_sem-WineBottle-f331ad8d0e6654ef8f992b1fe7075c8f_source10.mp4",
    ]
    source_image = read_frame(shadow, 20).crop((155, 0, 505, 350))
    images = [source_image]
    for path in target_paths:
        images.append(read_frame(path, 55).crop((145, 5, 495, 355)))
    labels = ["Shadow Hand\n源轨迹", "LinkerHand O6\n6 DoF", "XHand\n12 DoF", "WujiHand\n20 DoF"]
    panels = [labeled_panel(image, label, 410, 350) for image, label in zip(images, labels)]
    gap = 34
    canvas = Image.new("RGB", (4 * 410 + 3 * gap, 350), "white")
    draw = ImageDraw.Draw(canvas)
    for i, panel in enumerate(panels):
        x = i * (410 + gap)
        canvas.paste(panel, (x, 0))
        if i < 3:
            y = 150
            draw.line((x + 412, y, x + 410 + gap - 7, y), fill=BLUE if i == 0 else MID, width=3)
            draw.polygon([(x + 410 + gap - 7, y), (x + 410 + gap - 17, y - 6),
                          (x + 410 + gap - 17, y + 6)], fill=BLUE if i == 0 else MID)
    canvas.save(FRAME / "01_cover_four_hands.png")


def video_to_gif(source: Path, output: Path, stride=2, width=480) -> None:
    cap = cv2.VideoCapture(str(source))
    frames = []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % stride == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb).crop((0, 52, frame.shape[1], frame.shape[0]))
            height = round(image.height * width / image.width)
            frames.append(image.resize((width, height), Image.Resampling.LANCZOS))
        i += 1
    cap.release()
    output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(output, save_all=True, append_images=frames[1:], duration=1000 // 10,
                   loop=0, optimize=True, disposal=2)


def animations() -> None:
    basic = ROOT / "retarget_research/advanced_policy/videos/final_basic_isaac_state_v1"
    success = basic / "stable_success_1_sem-Cookie-ccfa74e5574678325cde8c99e4b182f9_source19.mp4"
    residual = ROOT / "retarget_research/advanced_policy/videos/final_residual_ppo_isaac_state_v1"
    failure = residual / "lift_then_slip_failure_1_sem-Fruit-473758ca6cb0506ee7697d561711bd2b_source16.mp4"
    video_to_gif(success, ANIM / "06_xhand_stable_success.gif")
    video_to_gif(failure, ANIM / "06_xhand_lift_then_slip.gif")


def accumulation_frames() -> None:
    source = ROOT / "retarget_research/advanced_policy/videos/final_residual_ppo_isaac_state_v1/lift_then_slip_failure_1_sem-Fruit-473758ca6cb0506ee7697d561711bd2b_source16.mp4"
    indices = [12, 34, 54, 76]
    labels = ["t0  接近", "t1  形成接触", "t2  抬升中偏移", "t3  接触丢失"]
    panels = [labeled_panel(read_frame(source, idx), label, 420, 350) for idx, label in zip(indices, labels)]
    gap = 42
    canvas = Image.new("RGB", (4 * 420 + 3 * gap, 350), "white")
    draw = ImageDraw.Draw(canvas)
    for i, panel in enumerate(panels):
        x = i * (420 + gap)
        canvas.paste(panel, (x, 0))
        if i < 3:
            y = 150
            draw.line((x + 424, y, x + 420 + gap - 8, y), fill=MID, width=3)
            draw.polygon([(x + 420 + gap - 8, y), (x + 420 + gap - 20, y - 7),
                          (x + 420 + gap - 20, y + 7)], fill=MID)
    FRAME.mkdir(parents=True, exist_ok=True)
    canvas.save(FRAME / "07_contact_error_accumulation.png")


def optional_real_comparisons() -> None:
    pairs = [
        (VIDEO / "linker_pose_baseline.mp4", VIDEO / "linker_function_vector.mp4",
         "04_linker_real_comparison.png", 50,
         "旧姿态方案｜最终 0.0 cm（后续滑落）", "功能向量方案｜最终 38.6 cm"),
        (VIDEO / "xhand_supervised_battery12.mp4", VIDEO / "xhand_residual_ppo_battery12.mp4",
         "11_xhand_supervised_vs_ppo.png", 40,
         "监督策略｜最终 −6.9 cm", "Residual PPO｜最终 37.4 cm"),
    ]
    FRAME.mkdir(parents=True, exist_ok=True)
    for left, right, name, frame_index, left_label, right_label in pairs:
        if not left.is_file() or not right.is_file():
            continue
        panels = [labeled_panel(read_frame(left, frame_index), left_label, 640, 470),
                  labeled_panel(read_frame(right, frame_index), right_label, 640, 470)]
        canvas = Image.new("RGB", (1280, 470), "white")
        canvas.paste(panels[0], (0, 0))
        canvas.paste(panels[1], (640, 0))
        canvas.save(FRAME / name)
        video_to_gif(left, ANIM / name.replace(".png", "_left.gif"))
        video_to_gif(right, ANIM / name.replace(".png", "_right.gif"))


def main() -> None:
    for directory in (FIG, ANIM, FRAME, VIDEO):
        directory.mkdir(parents=True, exist_ok=True)
    setup_style()
    task_overview()
    semantic_keypoints()
    linker_principle()
    retarget_results()
    final_retargeting_pipeline()
    residual_loop()
    training_curves()
    policy_results()
    cover_composite()
    three_hand_comparison()
    animations()
    accumulation_frames()
    optional_real_comparisons()
    print(OUT)


if __name__ == "__main__":
    main()
