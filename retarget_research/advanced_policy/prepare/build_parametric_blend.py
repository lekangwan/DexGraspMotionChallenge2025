import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    first = torch.load(args.first, map_location="cpu", weights_only=False)
    second = torch.load(args.second, map_location="cpu", weights_only=False)
    if first["dimensions"] != second["dimensions"]:
        raise ValueError("两个策略的输入输出维度不同")
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("alpha必须位于[0,1]")
    payload = {
        "config": {
            "model_type": "parametric_blend",
            "blend_alpha": args.alpha,
            "first_checkpoint": str(args.first.resolve()),
            "second_checkpoint": str(args.second.resolve()),
            "motion_steps": first["config"].get("motion_steps", 210),
        },
        "dimensions": first["dimensions"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(f"OUTPUT={args.output}")


if __name__ == "__main__":
    main()
