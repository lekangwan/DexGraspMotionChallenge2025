#!/usr/bin/env python3
"""生成闭合时机与反馈手指的自主复合策略checkpoint。"""

import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs/candidates_v1"


def write_candidate(hand, name, composite_type, secondary, **parameters):
    """把两个已训练网络及组合规则封装成可审计checkpoint。"""
    primary = (RUNS / hand / "geometry_phase/best.pt").resolve()
    secondary_path = (RUNS / hand / secondary / "best.pt").resolve()
    output = RUNS / hand / name
    output.mkdir(parents=True, exist_ok=True)
    config = {
        "model_type": name,
        "composite_type": composite_type,
        "autonomous_inputs": [
            "initial_observation", "initial_command", "object_point_cloud",
            "phase", "current_observation_if_feedback",
        ],
        "forbidden_runtime_inputs": [
            "future_expert_action", "reference_trajectory", "category_id",
        ],
        **parameters,
    }
    torch.save(
        {
            "schema": "geometry_composite_policy_v1",
            "config": config,
            "primary_checkpoint": str(primary),
            "secondary_checkpoint": str(secondary_path),
        },
        output / "best.pt",
    )
    (output / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main():
    for hand in ("linker", "xhand", "wuji"):
        write_candidate(
            hand, "phase_lead05", "phase_lead", "geometry_phase",
            finger_phase_lead=0.05, finger_scale=1.0,
        )
        write_candidate(
            hand, "phase_lead10", "phase_lead", "geometry_phase",
            finger_phase_lead=0.10, finger_scale=1.0,
        )
        write_candidate(
            hand, "phase_feedback_fingers", "hybrid", "geometry_chunk",
            finger_phase_lead=0.0, finger_scale=1.0,
        )
    print(RUNS)


if __name__ == "__main__":
    main()
