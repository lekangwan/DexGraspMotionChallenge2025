from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"
RUNS = ROOT / "advanced_policy" / "runs"
RUNS_V2 = ROOT / "advanced_policy_v2" / "runs" / "candidates_v1"

COLORS = {
    "Linker O6": "#4C78A8",
    "XHand": "#F58518",
    "WujiHand": "#54A24B",
    "Wuji-repaired": "#B279A2",
}

FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
font_manager.fontManager.addfont(FONT_PATH)
CHINESE_FONT = font_manager.FontProperties(fname=FONT_PATH).get_name()

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": [CHINESE_FONT, "DejaVu Sans"],
    "axes.unicode_minus": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def save_basic_results() -> None:
    hands = ["Linker O6", "XHand", "WujiHand"]
    baseline = np.array([22.2, 56.5, 47.6])
    selected = np.array([43.9, 69.5, 71.8])
    stable = np.array([39.4, 65.5, 68.2])
    transport = np.array([35.0, 63.0, 66.2])
    x = np.arange(len(hands))
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.7))

    width = 0.32
    bars = [
        axes[0].bar(x - width / 2, baseline, width, label="运动学初始方法", color="#8C9AA6"),
        axes[0].bar(x + width / 2, selected, width, label="最终物理在环方法", color="#2878B5"),
    ]
    for item in bars:
        axes[0].bar_label(item, fmt="%.1f", padding=2, fontsize=7)
    axes[0].set_title("(a) 参考Isaac成功率")
    axes[0].set_xticks(x, hands)
    axes[0].set_ylabel("成功率（%）")
    axes[0].set_ylim(0, 80)
    axes[0].grid(axis="y", alpha=0.22)
    axes[0].legend(frameon=False, fontsize=7, loc="upper left")

    width = 0.34
    bars_a = axes[1].bar(x - width / 2, stable, width, label="稳定抬升", color="#2878B5")
    bars_b = axes[1].bar(x + width / 2, transport, width, label="运输合格", color="#3A9D7A")
    axes[1].bar_label(bars_a, fmt="%.1f", padding=2, fontsize=8)
    axes[1].bar_label(bars_b, fmt="%.1f", padding=2, fontsize=8)
    axes[1].set_title("(b) 最终轨迹的两级质量")
    axes[1].set_xticks(x, hands)
    axes[1].set_ylim(0, 80)
    axes[1].grid(axis="y", alpha=0.22)
    axes[1].legend(frameon=False, fontsize=7, loc="upper left")
    fig.subplots_adjust(left=0.065, right=0.99, bottom=0.18, top=0.88, wspace=0.18)
    fig.savefig(FIGURE_DIR / "basic_success_rates.png", dpi=220)
    plt.close(fig)


def save_policy_results() -> None:
    hands = ["Linker O6", "XHand", "WujiHand"]
    absolute = np.array([7.0, 5.0, 2.0])
    relative = np.array([10.0, 9.0, 13.0])
    soup = np.array([8.0, 9.0, 10.0])
    valid = np.array([10.0, 14.0, 19.0])
    test = np.array([7.0, 8.4, 6.0])
    posthoc = np.array([7.0, 8.8, 7.6])
    x = np.arange(len(hands))

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.7))
    width = 0.19
    groups = [
        axes[0].bar(x - 1.5 * width, absolute, width, label="初态绝对动作", color="#A9B4BE"),
        axes[0].bar(x - 0.5 * width, relative, width, label="初态相对动作", color="#72A4C2"),
        axes[0].bar(x + 0.5 * width, soup, width, label="参数Soup", color="#E5A36A"),
        axes[0].bar(x + 1.5 * width, valid, width, label="逐手最终方法", color="#2878B5"),
    ]
    for item in groups:
        axes[0].bar_label(item, fmt="%.0f", padding=2, fontsize=6.5)
    axes[0].set_title("(a) valid100方法消融")
    axes[0].set_xticks(x, hands)
    axes[0].tick_params(axis="x", labelsize=8)
    axes[0].set_ylabel("自主成功率（%）")
    axes[0].set_ylim(0, 23)
    axes[0].grid(axis="y", alpha=0.22)
    axes[0].legend(frameon=False, fontsize=6.8, ncol=2, loc="upper left")

    width = 0.24
    b1 = axes[1].bar(x - width, valid, width, label="冻结前valid100", color="#8DB3D3")
    b2 = axes[1].bar(x, test, width, label="预选test500", color="#315B7D")
    b3 = axes[1].bar(
        x + width, posthoc, width, label="补充测试最佳观察",
        color="white", edgecolor="#D97941", hatch="///", linewidth=1.2,
    )
    for item in (b1, b2, b3):
        axes[1].bar_label(item, fmt="%.1f", padding=2, fontsize=7)
    axes[1].set_title("(b) 未见物体泛化与补充结果")
    axes[1].set_xticks(x, hands)
    axes[1].tick_params(axis="x", labelsize=8)
    axes[1].set_ylim(0, 23)
    axes[1].grid(axis="y", alpha=0.22)
    axes[1].legend(frameon=False, fontsize=6.8, loc="upper left")
    fig.subplots_adjust(left=0.065, right=0.99, bottom=0.18, top=0.88, wspace=0.28)
    fig.savefig(FIGURE_DIR / "advanced_policy_results.png", dpi=220)
    plt.close(fig)


def read_csv(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {
        key: np.asarray([float(row[key]) for row in rows])
        for key in ("epoch", "train_loss", "valid_loss")
    }


def moving_average(values: np.ndarray, window: int = 15) -> np.ndarray:
    if len(values) < window:
        return values
    left = window // 2
    right = window - 1 - left
    padded = np.pad(values, (left, right), mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def save_training_curves() -> None:
    supervised = {
        "Linker O6": RUNS / "autonomous_initial_phase_delta_v1/linker_initial_phase_delta_v1/metrics.csv",
        "XHand": RUNS / "autonomous_initial_phase_huber_v1/xhand_official_initial_phase_huber_v1/metrics.csv",
        "WujiHand": RUNS / "autonomous_state_aligned_dagger_v1/wuji_old_state_aligned_dagger_v1/metrics.csv",
    }
    titles = {
        "Linker O6": "Linker：相对动作MSE",
        "XHand": "XHand：相对动作Huber",
        "WujiHand": "Wuji：状态对齐反馈",
    }

    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.35))
    for axis, (hand, path) in zip(axes, supervised.items()):
        data = read_csv(path)
        color = COLORS[hand]
        axis.plot(data["epoch"], data["train_loss"], color=color, label="训练集")
        axis.plot(data["epoch"], data["valid_loss"], color="#555555", linestyle="--", label="验证集")
        axis.set_yscale("log")
        axis.set_title(titles[hand])
        axis.set_xlabel("训练轮次")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("归一化监督损失（对数坐标）")
    axes[2].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "advanced_training_curves.png", dpi=220)
    plt.close(fig)


def save_v2_training_curves() -> None:
    """绘制当前进阶主线两种候选的真实训练/验证Huber曲线。"""
    hands = [("linker", "Linker O6"), ("xhand", "XHand"), ("wuji", "WujiHand")]
    model_style = {
        "geometry_phase": ("逐步Phase", "-"),
        "geometry_chunk": ("Chunk8", "--"),
    }
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.35), sharey=True)
    for axis, (hand_key, hand_name) in zip(axes, hands):
        for model_type, (label, linestyle) in model_style.items():
            path = RUNS_V2 / hand_key / model_type / "metrics.csv"
            if not path.is_file():
                continue
            data = read_csv(path)
            axis.plot(
                data["epoch"], data["train_loss"],
                color="#9AA6B2" if model_type == "geometry_phase" else "#A8C7BC",
                linestyle=linestyle, linewidth=1.1, alpha=0.8,
            )
            axis.plot(
                data["epoch"], data["valid_loss"],
                color="#2878B5" if model_type == "geometry_phase" else "#2A9D8F",
                linestyle=linestyle, linewidth=1.8, label=f"{label}验证",
            )
        axis.set_yscale("log")
        axis.set_title(hand_name)
        axis.set_xlabel("训练轮次")
        axis.grid(alpha=0.22)
    axes[0].set_ylabel("归一化Huber损失（对数坐标）")
    axes[2].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "advanced_v2_training_curves.png", dpi=220)
    plt.close(fig)


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    save_basic_results()
    save_policy_results()
    save_training_curves()
    save_v2_training_curves()


if __name__ == "__main__":
    main()
