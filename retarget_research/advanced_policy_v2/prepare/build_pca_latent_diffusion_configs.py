#!/usr/bin/env python3
"""生成三只手统一结构的PCA潜空间Diffusion训练配置。"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    for hand, rank in (("linker", 32), ("xhand", 16), ("wuji", 16)):
        name = "geometry_pca_latent_diffusion"
        config = {
            "seed": 20260902, "device": "cuda", "hand": hand,
            "model_type": name, "pca_rank": rank,
            "hidden_dim": 256, "point_feature_dim": 64,
            "time_frequencies": 8, "diffusion_steps": 50,
            "candidate_count": 8, "sample_seed": 20260902,
            "selection_margin": 0.1,
            "batch_size": 64, "epochs": 400, "evaluation_interval": 5,
            "early_stopping_patience": 45,
            "learning_rate": 2e-4, "weight_decay": 1e-3,
            "point_jitter_std": 0.02,
            "regression_weight": 0.25,
            "compatibility_weight": 0.1,
            "data_dir": str(ROOT / f"data/final/{hand}"),
            "base_checkpoint": str(
                ROOT / f"runs/candidates_v1/{hand}/geometry_pca{rank}/best.pt"
            ),
            "output_dir": str(
                ROOT / f"runs/candidates_v1/{hand}/{name}"
            ),
            "category_id_used": False,
            "checkpoint_contains_training_trajectories": False,
        }
        path = ROOT / f"configs/generated/{hand}_{name}.json"
        path.write_text(
            json.dumps(config, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
