#!/usr/bin/env python3
"""生成三只手共用结构的PCA多候选训练配置。"""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main():
    for hand in ("linker", "xhand", "wuji"):
        config = {
            "seed": 20260901, "device": "cuda", "hand": hand,
            "model_type": "geometry_pca_mixture", "pca_rank": 32,
            "mode_count": 4, "hidden_dim": 192, "point_feature_dim": 64,
            "batch_size": 64, "epochs": 250, "early_stopping_patience": 30,
            "learning_rate": 3e-4, "weight_decay": 1e-3,
            "critic_prior_weight": 0.15,
            "data_dir": str(ROOT / f"data/final/{hand}"),
            "output_dir": str(ROOT / f"runs/candidates_v1/{hand}/geometry_pca_mixture"),
            "pca_checkpoint": str(ROOT / f"runs/candidates_v1/{hand}/geometry_pca32/best.pt"),
            "category_id_used": False, "trajectory_retrieval_used": False,
        }
        path = ROOT / f"configs/generated/{hand}_geometry_pca_mixture.json"
        path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()
