#!/usr/bin/env python3
"""生成三手统一结构的动态交互残差训练配置。"""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RANK = {"linker": 32, "xhand": 16, "wuji": 16}
ACTION_DIM = {"linker": 12, "xhand": 18, "wuji": 26}


def main():
    output = ROOT / "configs/generated"
    for hand in ("linker", "xhand", "wuji"):
        limit = [0.05] * 3 + [0.25] * 3 + [0.50] * (ACTION_DIM[hand] - 6)
        config = {
            "seed": 20260901, "device": "cuda", "hand": hand,
            "model_type": "geometry_pca_interaction_residual",
            "hidden_dim": 256, "phase_frequencies": 4,
            "batch_size": 512, "epochs": 120, "early_stopping_patience": 20,
            "learning_rate": 0.0003, "weight_decay": 0.001,
            "data_dir": str(ROOT / f"data/final/{hand}"),
            "output_dir": str(ROOT / f"runs/candidates_v1/{hand}/geometry_pca_interaction"),
            "base_checkpoint": str(
                ROOT / f"runs/candidates_v1/{hand}/geometry_pca{RANK[hand]}/best.pt"
            ),
            "residual_limit": limit,
            "category_id_used": False,
            "trajectory_retrieval_used": False,
        }
        (output / f"{hand}_geometry_pca_interaction.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(output)


if __name__ == "__main__":
    main()
