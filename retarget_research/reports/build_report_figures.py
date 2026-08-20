from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"
RUNS = ROOT / "advanced_policy" / "runs"

COLORS = {
    "Linker O6": "#4C78A8",
    "XHand": "#F58518",
    "WujiHand": "#54A24B",
    "Wuji-repaired": "#B279A2",
}


def save_basic_results() -> None:
    hands = ["Linker O6", "XHand", "WujiHand"]
    stable = np.array([19.3, 51.6, 58.1])
    transport = np.array([15.3, 50.1, 55.7])
    x = np.arange(len(hands))
    width = 0.34

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    bars_a = ax.bar(x - width / 2, stable, width, label="Stable lift >= 30 cm", color="#4C78A8")
    bars_b = ax.bar(x + width / 2, transport, width, label="Transport-quality", color="#72B7B2")
    ax.bar_label(bars_a, fmt="%.1f%%", padding=2, fontsize=9)
    ax.bar_label(bars_b, fmt="%.1f%%", padding=2, fontsize=9)
    ax.set_xticks(x, hands)
    ax.set_ylabel("Success rate (%)")
    ax.set_ylim(0, 68)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "basic_success_rates.png", dpi=220)
    plt.close(fig)


def save_policy_results() -> None:
    hands = ["Linker O6", "XHand", "WujiHand"]
    phase = np.array([12.3, 19.1, 12.9])
    residual = np.array([54.3, 54.1, 45.4])
    full_test = np.array([12.0, 28.8, 30.2])
    x = np.arange(len(hands))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9.2, 3.9))
    b1 = ax.bar(x - width, phase, width, label="PhaseResidual / best DAgger", color="#9ECAE9")
    b2 = ax.bar(x, residual, width, label="Residual PPO (expert-success subset)", color="#F58518")
    b3 = ax.bar(x + width, full_test, width, label="Residual PPO (all 500 tests)", color="#54A24B")
    for bars in (b1, b2, b3):
        ax.bar_label(bars, fmt="%.1f%%", padding=2, fontsize=8)
    ax.set_xticks(x, hands)
    ax.set_ylabel("Closed-loop success rate (%)")
    ax.set_ylim(0, 64)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
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
        "Linker O6": RUNS / "formal_final_online3/linker_phase_residual_v1/metrics.csv",
        "XHand": RUNS / "formal_final_online3/xhand_official_phase_residual_v1/metrics.csv",
        "WujiHand": RUNS / "formal_final_online3/wuji_old_phase_residual_v1/metrics.csv",
    }
    ppo = {
        "Linker O6": RUNS / "residual_rl_general/linker/training_log.json",
        "XHand": RUNS / "residual_rl_general/xhand_official/training_log.json",
        "WujiHand": RUNS / "residual_rl_general/wuji_old/training_log.json",
    }

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.5))
    for hand, path in supervised.items():
        data = read_csv(path)
        color = COLORS[hand]
        axes[0].plot(data["epoch"], data["train_loss"], color=color, label=hand)
        axes[0].plot(data["epoch"], data["valid_loss"], color=color, linestyle="--", alpha=0.8)
    axes[0].set_yscale("log")
    axes[0].set_title("Supervised residual loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE (log scale)")
    axes[0].text(0.98, 0.96, "solid: train\ndashed: valid", transform=axes[0].transAxes,
                 ha="right", va="top", fontsize=8)

    for hand, path in ppo.items():
        rows = json.loads(path.read_text())
        iteration = np.asarray([row["iteration"] for row in rows])
        success = np.asarray([row["success_rate"] for row in rows]) * 100
        value_loss = np.asarray([row["value_loss"] for row in rows])
        axes[1].plot(iteration, moving_average(success), color=COLORS[hand], label=hand)
        axes[2].plot(iteration, moving_average(value_loss), color=COLORS[hand], label=hand)
    axes[1].set_title("PPO training success (MA-15)")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Success rate (%)")
    axes[2].set_title("PPO value loss (MA-15)")
    axes[2].set_xlabel("Iteration")
    axes[2].set_ylabel("Value loss")

    for ax in axes:
        ax.grid(alpha=0.25)
    axes[2].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "advanced_training_curves.png", dpi=220)
    plt.close(fig)


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    save_basic_results()
    save_policy_results()
    save_training_curves()


if __name__ == "__main__":
    main()
