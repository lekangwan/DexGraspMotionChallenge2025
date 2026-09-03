#!/usr/bin/env python3
"""按单GPU顺序训练指定候选，避免多个训练进程争抢显存。"""

import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=["geometry_phase", "geometry_chunk"])
    parser.add_argument("--hands", nargs="+", default=["linker", "xhand", "wuji"])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    trainer = Path(__file__).resolve().parents[1] / "train.py"
    for hand in args.hands:
        for model_type in args.models:
            config = args.config_dir / f"{hand}_{model_type}.json"
            print(f"TRAIN hand={hand} model={model_type}", flush=True)
            subprocess.run(
                [sys.executable, "-u", str(trainer), "--config", str(config), "--device", args.device],
                check=True,
            )


if __name__ == "__main__":
    main()

