#!/usr/bin/env python3
"""分解PCA多候选策略的候选多样性、路由误差和离线Oracle上限。"""

import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models import build_pca_mixture, build_pca_model  # noqa: E402


def load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def gather(values, indices):
    return values[np.arange(len(indices)), indices]


def analyze(hand, runs_root, data_root, device):
    data = load_npz(data_root / hand / "mixture_valid.npz")
    checkpoint = torch.load(
        runs_root / hand / "geometry_mixture_gate/best.pt", map_location=device
    )
    config = checkpoint["config"]
    model = build_pca_mixture(
        config,
        checkpoint["dimensions"]["task_observation_dim"],
        checkpoint["dimensions"]["action_dim"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    tensors = [
        torch.from_numpy(data[name]).to(device)
        for name in ("task_observation", "initial_command", "object_points")
    ]
    with torch.no_grad():
        condition, logits, candidates = model.generate(*tensors)
        quality = model.score(condition, candidates)
    candidates = candidates.cpu().numpy()
    logits = logits.cpu().numpy()
    quality = quality.cpu().numpy()
    target = data["pca_coefficient"]
    scale = np.asarray(checkpoint["coefficient_std"], np.float32)
    scale = scale / np.sqrt(np.mean(scale ** 2))
    errors = np.mean(((candidates - target[:, None]) * scale) ** 2, axis=-1)
    oracle = np.argmin(errors, axis=1)
    gate = np.argmax(logits, axis=1)
    prior = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
    critic = np.argmax(quality + float(config.get("critic_prior_weight", 0.15)) * prior, axis=1)

    centers = np.asarray(checkpoint["cluster_centers"], np.float32)
    target_mode = np.argmin(np.mean((target[:, None] - centers[None]) ** 2, axis=-1), axis=1)
    positive = data["success"].astype(bool)
    pairwise = []
    for first in range(candidates.shape[1]):
        for second in range(first + 1, candidates.shape[1]):
            pairwise.append(np.sqrt(np.mean((candidates[:, first] - candidates[:, second]) ** 2, axis=-1)))

    pca_checkpoint = torch.load(config["pca_checkpoint"], map_location=device)
    pca_model = build_pca_model(
        pca_checkpoint["config"],
        pca_checkpoint["dimensions"]["task_observation_dim"],
        pca_checkpoint["dimensions"]["action_dim"],
    ).to(device)
    pca_model.load_state_dict(pca_checkpoint["model_state"])
    pca_model.eval()
    with torch.no_grad():
        single = pca_model(*tensors).cpu().numpy()
    single_error = np.mean(((single - target) * scale) ** 2, axis=-1)

    def distribution(indices):
        return np.bincount(indices, minlength=candidates.shape[1]).tolist()

    return {
        "hand": hand,
        "sample_count": int(len(target)),
        "positive_count": int(positive.sum()),
        "candidate_pairwise_rms_mean": float(np.mean(pairwise)),
        "single_pca_mse_all": float(single_error.mean()),
        "mixture_oracle_mse_all": float(errors.min(axis=1).mean()),
        "mixture_gate_mse_all": float(gather(errors, gate).mean()),
        "mixture_critic_mse_all": float(gather(errors, critic).mean()),
        "single_pca_mse_positive": float(single_error[positive].mean()),
        "mixture_oracle_mse_positive": float(errors[positive].min(axis=1).mean()),
        "mixture_gate_mse_positive": float(gather(errors, gate)[positive].mean()),
        "mixture_critic_mse_positive": float(gather(errors, critic)[positive].mean()),
        "gate_equals_oracle": float(np.mean(gate == oracle)),
        "critic_equals_oracle": float(np.mean(critic == oracle)),
        "gate_target_mode_accuracy_positive": float(np.mean(gate[positive] == target_mode[positive])),
        "gate_distribution": distribution(gate),
        "critic_distribution": distribution(critic),
        "oracle_distribution": distribution(oracle),
        "target_mode_distribution_positive": distribution(target_mode[positive]),
    }


def main():
    research = Path(__file__).resolve().parents[2]
    runs = research / "advanced_policy_v2/runs/candidates_v1"
    data = research / "advanced_policy_v2/data/final"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = [analyze(hand, runs, data, device) for hand in ("linker", "xhand", "wuji")]
    output = runs / "pca_mixture_diagnostic.json"
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for result in results:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"DIAGNOSTIC={output}")


if __name__ == "__main__":
    main()
