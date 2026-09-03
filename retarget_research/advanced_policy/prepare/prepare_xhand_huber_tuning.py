#!/usr/bin/env python3
"""生成XHand Huber beta与warm-start对照配置。"""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "advanced_policy/configs/generated/autonomous_initial_phase_huber_v1/xhand_official_initial_phase_huber_v1.json"
OUTPUT = ROOT / "advanced_policy/configs/generated/autonomous_xhand_huber_tuning_v1"
RUN = ROOT / "advanced_policy/runs/autonomous_xhand_huber_tuning_v1"
BASE = ROOT / "advanced_policy/runs/autonomous_initial_phase_delta_v1/xhand_official_initial_phase_delta_v1/best.pt"


def main():
    base = json.loads(SOURCE.read_text(encoding="utf-8"))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    variants = {
        "xhand_huber_beta05_v1": {"huber_beta": 0.5},
        "xhand_huber_beta20_v1": {"huber_beta": 2.0},
        "xhand_huber_warm_v1": {
            "huber_beta": 1.0,
            "init_checkpoint": str(BASE.resolve()),
            "learning_rate": 0.00008,
            "epochs": 80,
            "early_stopping_patience": 15,
            "seed": 20260825,
        },
    }
    index = {}
    for name, changes in variants.items():
        config = dict(base)
        config.update(changes)
        config["experiment_name"] = name
        config["output_dir"] = str((RUN / name).resolve())
        path = OUTPUT / f"{name}.json"
        path.write_text(
            json.dumps(config, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        index[name] = str(path.resolve())
    (OUTPUT / "config_index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"XHAND_HUBER_TUNING_CONFIGS={len(index)}")


if __name__ == "__main__":
    main()
