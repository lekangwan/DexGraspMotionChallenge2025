#!/usr/bin/env python3
"""生成三只手共享结构的初始手物交互PCA配置。"""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RANK = {"linker": 32, "xhand": 16, "wuji": 16}


def main():
    output = ROOT / "configs/generated"
    output.mkdir(parents=True, exist_ok=True)
    for hand, rank in RANK.items():
        config = {
            "seed": 20260901,
            "device": "cuda",
            "hand": hand,
            "model_type": "geometry_pca_initial_interaction",
            "pca_rank": rank,
            "hidden_dim": 192,
            "point_feature_dim": 64,
            "interaction_dim": 75,
            "interaction_feature_dim": 64,
            "batch_size": 64,
            "epochs": 300,
            "early_stopping_patience": 35,
            "learning_rate": 0.0003,
            "weight_decay": 0.001,
            "point_jitter_std": 0.02,
            "data_dir": str(ROOT / f"data/final/{hand}"),
            "output_dir": str(
                ROOT / f"runs/candidates_v1/{hand}/geometry_pca_initial_interaction"
            ),
            "category_id_used": False,
            "checkpoint_contains_training_trajectories": False,
        }
        path = output / f"{hand}_geometry_pca_initial_interaction.json"
        path.write_text(
            json.dumps(config, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(output)


if __name__ == "__main__":
    main()
