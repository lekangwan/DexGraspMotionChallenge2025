#!/usr/bin/env python3
"""生成两档用于验证过拟合假设的紧凑Phase训练配置。"""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs/generated"
RUNS = ROOT / "runs/candidates_v1"


def main():
    variants = {
        "phase_compact192": {
            "hidden_dim": 192, "point_feature_dim": 64,
            "phase_frequencies": 2, "weight_decay": 1e-3,
            "point_jitter_std": 0.02,
        },
        "phase_compact96": {
            "hidden_dim": 96, "point_feature_dim": 32,
            "phase_frequencies": 1, "weight_decay": 3e-3,
            "point_jitter_std": 0.03,
        },
    }
    for hand in ("linker", "xhand", "wuji"):
        base = json.loads((CONFIGS / f"{hand}_geometry_phase.json").read_text(encoding="utf-8"))
        for name, changes in variants.items():
            config = dict(base)
            config.update(changes)
            config["model_type"] = "geometry_phase"
            config["output_dir"] = str(RUNS / hand / name)
            config["early_stopping_patience"] = 18
            (CONFIGS / f"{hand}_{name}.json").write_text(
                json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
    print(CONFIGS)


if __name__ == "__main__":
    main()
