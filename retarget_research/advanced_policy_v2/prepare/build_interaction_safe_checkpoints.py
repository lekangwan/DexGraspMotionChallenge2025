#!/usr/bin/env python3
"""从已训练残差构造冻结手腕的两档安全增益候选。"""

import copy
from pathlib import Path

import torch


def main():
    runs = Path(__file__).resolve().parents[1] / "runs/candidates_v1"
    for hand in ("linker", "xhand", "wuji"):
        source = torch.load(
            runs / hand / "geometry_pca_interaction/best.pt", map_location="cpu"
        )
        for label, gain in (("025", 0.25), ("050", 0.50)):
            payload = copy.deepcopy(source)
            model_type = f"geometry_pca_interaction_finger{label}"
            payload["config"] = dict(
                source["config"], model_type=model_type,
                finger_residual_only=True, residual_gain=gain,
            )
            directory = runs / hand / model_type
            directory.mkdir(parents=True, exist_ok=True)
            torch.save(payload, directory / "best.pt")
    print(runs)


if __name__ == "__main__":
    main()
