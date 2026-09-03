#!/usr/bin/env python3
"""汇总三只手合法自主评测结果。"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--split", choices=("valid", "test"), default="valid")
    parser.add_argument("--experiment-suffix", default="initial_phase_v1")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summary = {}
    for hand in ("linker", "xhand_official", "wuji_old"):
        path = (
            args.run_root / f"{hand}_{args.experiment_suffix}" / f"closed_loop_{args.split}"
            / "policy_evaluation_summary.json"
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not payload.get("autonomous_only"):
            raise ValueError(f"结果没有自主评测标志: {path}")
        if payload.get("expert_wrist") or payload.get("residual_rl_checkpoint"):
            raise ValueError(f"结果混入测试专家轨迹: {path}")
        summary[hand] = {
            "success_count": int(payload["success_count"]),
            "trajectory_count": int(payload["trajectory_count"]),
            "success_rate": float(payload["trajectory_micro_success_rate"]),
            "mean_max_lift_m": float(payload["mean_max_lift_m"]),
            "mean_final_lift_m": float(payload["mean_final_lift_m"]),
            "mean_contact_steps": float(payload["mean_hand_object_contact_steps"]),
        }
        print(hand, json.dumps(summary[hand], ensure_ascii=False))
    output = args.output or (
        args.run_root / f"autonomous_{args.experiment_suffix}_{args.split}_summary.json"
    )
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"AUTONOMOUS_SUMMARY={output.resolve()}")


if __name__ == "__main__":
    main()
