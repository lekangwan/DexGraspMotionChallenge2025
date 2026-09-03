#!/usr/bin/env python3
"""把SO(3)检索轨迹与初态MLP按固定比例融合。"""

import argparse
from pathlib import Path

import torch


def load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--learned", type=Path, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    retrieval, learned = load(args.retrieval), load(args.learned)
    payload = {
        "config": {
            "model_type": "trajectory_se3_blend",
            "retrieval_k": retrieval["config"]["retrieval_k"],
            "translation_frame": retrieval["config"]["translation_frame"],
            "blend_alpha": args.alpha,
            "motion_steps": int(learned["config"].get("motion_steps", 210)),
        },
        "dimensions": retrieval["dimensions"],
        "retrieval_initial_observations": retrieval["retrieval_initial_observations"],
        "retrieval_local_translation": retrieval["retrieval_local_translation"],
        "retrieval_relative_rotvec": retrieval["retrieval_relative_rotvec"],
        "retrieval_finger_actions": retrieval["retrieval_finger_actions"],
        "base_model_config": learned["config"],
        "model_state": learned["model_state"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(f"TRAJECTORY_SE3_BLEND={args.output} alpha={args.alpha}")


if __name__ == "__main__":
    main()
