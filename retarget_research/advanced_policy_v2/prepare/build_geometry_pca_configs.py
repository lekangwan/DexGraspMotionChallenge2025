#!/usr/bin/env python3
"""生成两档整轨迹PCA策略配置。"""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs/generated"
RUNS = ROOT / "runs/candidates_v1"


def main():
    for hand in ("linker", "xhand", "wuji"):
        for rank in (16, 32):
            name = f"geometry_pca{rank}"
            config = {
                "seed": 20260901, "device": "cuda", "hand": hand,
                "model_type": name, "pca_rank": rank,
                "hidden_dim": 192, "point_feature_dim": 64,
                "batch_size": 64, "epochs": 300,
                "early_stopping_patience": 35,
                "learning_rate": 3e-4, "weight_decay": 1e-3,
                "point_jitter_std": 0.02,
                "data_dir": str(ROOT / f"data/final/{hand}"),
                "output_dir": str(RUNS / hand / name),
                "category_id_used": False,
                "checkpoint_contains_training_trajectories": False,
            }
            (CONFIGS / f"{hand}_{name}.json").write_text(
                json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
    print(CONFIGS)


if __name__ == "__main__":
    main()
