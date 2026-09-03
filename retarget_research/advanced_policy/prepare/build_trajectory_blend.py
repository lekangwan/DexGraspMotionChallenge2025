#!/usr/bin/env python3
"""将5NN完整轨迹与相对初态MLP按固定比例融合。"""

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
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--wrist-alpha", type=float)
    parser.add_argument("--finger-alpha", type=float)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    retrieval, learned = load(args.retrieval), load(args.learned)
    if retrieval["dimensions"] != learned["dimensions"]:
        raise ValueError("检索与MLP维度不一致")
    if learned["config"]["model_type"] != "initial_phase_delta":
        raise ValueError("融合基模型必须是initial_phase_delta")
    if args.alpha is not None:
        wrist_alpha = finger_alpha = float(args.alpha)
    else:
        if args.wrist_alpha is None or args.finger_alpha is None:
            raise ValueError("必须提供alpha，或同时提供wrist-alpha与finger-alpha")
        wrist_alpha, finger_alpha = float(args.wrist_alpha), float(args.finger_alpha)
    payload = {
        "config": {
            "model_type": "trajectory_blend",
            "retrieval_k": int(retrieval["config"]["retrieval_k"]),
            "wrist_blend_alpha": wrist_alpha,
            "finger_blend_alpha": finger_alpha,
            "motion_steps": int(learned["config"].get("motion_steps", 210)),
        },
        "dimensions": retrieval["dimensions"],
        "retrieval_initial_observations": retrieval["retrieval_initial_observations"],
        "retrieval_action_deltas": retrieval["retrieval_action_deltas"],
        "retrieval_feature_indices": retrieval["retrieval_feature_indices"],
        "base_model_config": learned["config"],
        "model_state": learned["model_state"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(f"TRAJECTORY_BLEND={args.output} wrist={wrist_alpha} finger={finger_alpha}")


if __name__ == "__main__":
    main()
