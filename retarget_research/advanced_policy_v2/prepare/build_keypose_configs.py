#!/usr/bin/env python3
"""生成三只手统一的关键状态策略训练配置。"""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main():
    output = ROOT / "configs/generated"
    for hand in ("linker", "xhand", "wuji"):
        config = {
            "seed": 20260902,
            "device": "cuda",
            "hand": hand,
            "model_type": "geometry_keypose_lift",
            "hidden_dim": 256,
            "point_feature_dim": 64,
            "interaction_dim": 75,
            "interaction_feature_dim": 64,
            "batch_size": 64,
            "epochs": 250,
            "early_stopping_patience": 35,
            "learning_rate": 0.0003,
            "weight_decay": 0.001,
            "data_dir": str(ROOT / f"data/final/{hand}"),
            "output_dir": str(
                ROOT / f"runs/candidates_v1/{hand}/geometry_keypose_lift"
            ),
            "category_id_used": False,
            "trajectory_retrieval_used": False,
        }
        (output / f"{hand}_geometry_keypose_lift.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(output)


if __name__ == "__main__":
    main()
