#!/usr/bin/env python3
"""由当前最强InitialPhase配置生成Huber损失对照配置。"""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "advanced_policy/configs/generated/autonomous_initial_phase_delta_v1"
OUTPUT = ROOT / "advanced_policy/configs/generated/autonomous_initial_phase_huber_v1"
RUN = ROOT / "advanced_policy/runs/autonomous_initial_phase_huber_v1"


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    index = {}
    for label in ("linker", "xhand_official", "wuji_old"):
        source = SOURCE / f"{label}_initial_phase_delta_v1.json"
        config = json.loads(source.read_text(encoding="utf-8"))
        name = f"{label}_initial_phase_huber_v1"
        config.update({
            "experiment_name": name,
            "seed": 20260824,
            "loss_type": "huber",
            "huber_beta": 1.0,
            "output_dir": str((RUN / name).resolve()),
        })
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
    print(f"INITIAL_HUBER_CONFIGS={len(index)}")


if __name__ == "__main__":
    main()
