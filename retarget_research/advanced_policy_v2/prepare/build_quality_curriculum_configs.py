#!/usr/bin/env python3
"""生成三手质量课程PCA的统一训练配置。"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    for hand, rank in (("linker", 32), ("xhand", 16), ("wuji", 16)):
        config = {
            "seed": 20260902, "hand": hand,
            "base_checkpoint": str(
                ROOT / f"runs/candidates_v1/{hand}/geometry_pca{rank}/best.pt"
            ),
            "data_dir": str(ROOT / f"data/final/{hand}"),
            "curriculum_data": str(
                ROOT / f"data/quality_curriculum/{hand}_train.npz"
            ),
            "output_dir": str(
                ROOT / f"runs/candidates_v1/{hand}/geometry_pca_quality_curriculum"
            ),
            "reference_epochs": 60, "strict_epochs": 80,
            "reference_learning_rate": 1e-4,
            "strict_learning_rate": 5e-5,
            "batch_size": 64, "weight_decay": 1e-3,
            "point_jitter_std": 0.02, "early_stopping_patience": 30,
            "category_id_used": False,
            "checkpoint_contains_training_trajectories": False,
        }
        path = ROOT / f"configs/generated/{hand}_geometry_pca_quality_curriculum.json"
        path.write_text(
            json.dumps(config, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
