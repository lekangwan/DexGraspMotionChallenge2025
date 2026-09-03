#!/usr/bin/env python3
"""生成三只手统一的无PCA Temporal3 直接策略配置。"""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main():
    output = ROOT / "configs/generated"
    output.mkdir(parents=True, exist_ok=True)
    for hand in ("linker", "xhand", "wuji"):
        config = {
            "seed": 20260902,
            "hand": hand,
            "model_type": "direct_interaction_temporal3",
            "history": 3,
            "action_horizon": 1,
            "motion_steps": 210,
            "hidden_dim": 256,
            "state_feature_dim": 192,
            "point_feature_dim": 96,
            "recurrent_layers": 2,
            "phase_frequencies": 4,
            "batch_size": 512,
            "num_workers": 4,
            "epochs": 120,
            "early_stopping_patience": 18,
            "learning_rate": 3e-4,
            "weight_decay": 1e-4,
            "gradient_clip": 1.0,
            "huber_beta": 0.5,
            "state_noise_std": 0.01,
            "data_dir": str(ROOT / f"data/final/{hand}"),
            "output_dir": str(
                ROOT / f"runs/candidates_v1/{hand}/direct_interaction_temporal3"
            ),
            "category_id_used": False,
            "pca_used": False,
        }
        (output / f"{hand}_direct_interaction_temporal3.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
