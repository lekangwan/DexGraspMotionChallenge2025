#!/usr/bin/env python3
"""把三只手当前最强PCA策略包装成同参数的在线接触反馈策略。"""

from pathlib import Path

import torch


def main():
    runs = Path(__file__).resolve().parents[1] / "runs/candidates_v1"
    bases = {"linker": "geometry_pca32", "xhand": "geometry_pca16", "wuji": "geometry_pca16"}
    for hand, base_name in bases.items():
        base_path = (runs / hand / base_name / "best.pt").resolve()
        base = torch.load(base_path, map_location="cpu")
        payload = {
            "schema": "geometry_pca_contact_feedback_v1",
            "config": {
                "model_type": "geometry_pca_contact_feedback",
                "hand": hand,
                "contact_threshold": 0.02,
                "contact_stable_steps": 2,
                "release_steps": 2,
                "grip_step": 0.003,
                "max_grip": 0.15,
            },
            "dimensions": base["dimensions"],
            "base_checkpoint": str(base_path),
        }
        output = runs / hand / "geometry_pca_contact_feedback" / "best.pt"
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, output)
        print(output)

        grasp_payload = {
            **payload,
            "config": {
                **payload["config"],
                "model_type": "geometry_pca_grasp_fsm",
                "pause_for_grasp": True,
                "grip_step": 0.006,
                "max_grip": 0.12,
                "max_grasp_hold_steps": 20,
            },
        }
        grasp_output = runs / hand / "geometry_pca_grasp_fsm" / "best.pt"
        grasp_output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(grasp_payload, grasp_output)
        print(grasp_output)


if __name__ == "__main__":
    main()
