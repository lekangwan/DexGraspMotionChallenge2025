#!/usr/bin/env python3
"""生成三手Initial-Delta + DAgger反馈策略配置。"""

import argparse
import json
from pathlib import Path


LABELS = ("linker", "xhand_official", "wuji_old")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    base_root = root / "retarget_research/advanced_policy/runs/autonomous_initial_phase_delta_v1"
    run_root = root / "retarget_research/advanced_policy/runs/autonomous_initial_feedback_dagger_v1"
    config_root = root / "retarget_research/advanced_policy/configs/generated/autonomous_initial_feedback_dagger_v1"
    config_root.mkdir(parents=True, exist_ok=True)
    index = {}
    for label in LABELS:
        base = base_root / f"{label}_initial_phase_delta_v1"
        config = json.loads((base / "config.json").read_text(encoding="utf-8"))
        output = run_root / f"{label}_initial_phase_feedback_v1"
        config.update({
            "experiment_name": output.name,
            "model_type": "initial_phase_feedback",
            "seed": 20260824,
            "epochs": 120,
            "learning_rate": 0.0001,
            "early_stopping_patience": 18,
            "feedback_limit": 0.75,
            "online_ratio": 0.35,
            "online_data_path": str((run_root / "online_data" / f"{label}_r1.npz").resolve()),
            "init_checkpoint": str((base / "best.pt").resolve()),
            "output_dir": str(output.resolve()),
        })
        path = config_root / f"{output.name}.json"
        path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        index[output.name] = str(path.resolve())
    (config_root / "config_index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"INITIAL_FEEDBACK_CONFIGS={len(index)}")


if __name__ == "__main__":
    main()
