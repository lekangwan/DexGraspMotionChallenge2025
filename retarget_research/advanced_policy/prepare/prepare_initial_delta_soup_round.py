#!/usr/bin/env python3
"""从首轮模型生成同盆地第二seed微调配置，并在微调后构造参数Soup。"""

import argparse
import json
from pathlib import Path


LABELS = ("linker", "xhand_official", "wuji_old")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--make-soup", action="store_true")
    args = parser.parse_args()
    root = args.project_root.resolve()
    base_root = root / "retarget_research/advanced_policy/runs/autonomous_initial_phase_delta_v1"
    run_root = root / "retarget_research/advanced_policy/runs/autonomous_initial_phase_delta_soup_v1"
    config_root = root / "retarget_research/advanced_policy/configs/generated/autonomous_initial_phase_delta_soup_v1"
    config_root.mkdir(parents=True, exist_ok=True)
    index = {}
    for label in LABELS:
        base_run = base_root / f"{label}_initial_phase_delta_v1"
        config = json.loads((base_run / "config.json").read_text(encoding="utf-8"))
        output = run_root / f"{label}_initial_phase_delta_ft_v1"
        config.update({
            "experiment_name": output.name,
            "seed": 20260823,
            "epochs": 80,
            "learning_rate": 0.00008,
            "early_stopping_patience": 15,
            "init_checkpoint": str((base_run / "best.pt").resolve()),
            "output_dir": str(output.resolve()),
        })
        path = config_root / f"{output.name}.json"
        path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        index[output.name] = str(path.resolve())
    (config_root / "config_index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"SOUP_FINETUNE_CONFIGS={len(index)}")


if __name__ == "__main__":
    main()
