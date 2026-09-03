#!/usr/bin/env python3
"""复制纯参数Temporal checkpoint并缩小反馈门。"""

import argparse
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--feedback-limit", type=float, required=True)
    args = parser.parse_args()
    try:
        payload = torch.load(args.input, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(args.input, map_location="cpu")
    payload["config"] = dict(payload["config"])
    payload["config"]["feedback_limit"] = args.feedback_limit
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(f"SCALED_TEMPORAL_FEEDBACK={args.output} limit={args.feedback_limit}")


if __name__ == "__main__":
    main()
