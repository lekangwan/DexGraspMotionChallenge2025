#!/usr/bin/env python3
"""把三手最强PCA包装成统一参数的可微表面IK策略。"""

from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
BASE = {"linker": "geometry_pca32", "xhand": "geometry_pca16", "wuji": "geometry_pca16"}


def main():
    for hand, base_name in BASE.items():
        base_path = ROOT / f"runs/candidates_v1/{hand}/{base_name}/best.pt"
        base = torch.load(base_path, map_location="cpu")
        with np.load(ROOT / f"data/final/{hand}/train.npz", allow_pickle=False) as archive:
            fingers = archive["actions"][:, 6:].astype(np.float32)
        payload = {
            "schema": "geometry_pca_surface_ik_policy_v1",
            "config": {
                "hand": hand, "model_type": "geometry_pca_surface_ik",
                "surface_offset_m": 0.002, "joint_delta_bound": 0.60,
                "wrist_translation_bound_m": 0.03,
                "ik_steps": 120, "ik_learning_rate": 0.03,
                "anchor_weight": 0.005, "translation_anchor_weight": 0.02,
                "category_id_used": False,
            },
            "dimensions": base["dimensions"],
            "base_checkpoint": str(base_path.resolve()),
            "finger_lower": np.quantile(fingers, 0.001, axis=0).astype(np.float32),
            "finger_upper": np.quantile(fingers, 0.999, axis=0).astype(np.float32),
        }
        output = ROOT / f"runs/candidates_v1/{hand}/geometry_pca_surface_ik/best.pt"
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, output)
        print(output)


if __name__ == "__main__":
    main()
