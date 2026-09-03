#!/usr/bin/env python3
"""复制Diffusion checkpoint并写入各手离线标定后的选择门槛。"""

from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
MARGINS = {"linker": 1.20, "xhand": 1.05, "wuji": 2.25}


def main():
    for hand, margin in MARGINS.items():
        source = ROOT / f"runs/candidates_v1/{hand}/geometry_pca_latent_diffusion/best.pt"
        target_dir = ROOT / f"runs/candidates_v1/{hand}/geometry_pca_latent_diffusion_calibrated"
        payload = torch.load(source, map_location="cpu")
        payload["config"] = dict(
            payload["config"], selection_margin=margin,
            calibration="offline_valid_coefficient_error_v1",
        )
        target_dir.mkdir(parents=True, exist_ok=True)
        torch.save(payload, target_dir / "best.pt")
        print(hand, margin, target_dir / "best.pt")


if __name__ == "__main__":
    main()
