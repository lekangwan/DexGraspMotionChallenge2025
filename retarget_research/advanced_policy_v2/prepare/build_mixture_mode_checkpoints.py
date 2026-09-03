#!/usr/bin/env python3
"""复制多候选checkpoint并固定模式，用于测量候选集合的物理Oracle上限。"""

import copy
from pathlib import Path

import torch


def main():
    runs = Path(__file__).resolve().parents[1] / "runs/candidates_v1"
    for hand in ("linker", "xhand", "wuji"):
        source = torch.load(runs / hand / "geometry_mixture_gate/best.pt", map_location="cpu")
        for mode in range(int(source["config"].get("mode_count", 4))):
            payload = copy.deepcopy(source)
            payload["config"] = dict(
                source["config"], selection="fixed", fixed_mode=mode,
                model_type=f"geometry_mixture_mode{mode}",
            )
            directory = runs / hand / f"geometry_mixture_mode{mode}"
            directory.mkdir(parents=True, exist_ok=True)
            torch.save(payload, directory / "best.pt")
    print(f"MODE_CHECKPOINTS={runs}")


if __name__ == "__main__":
    main()
