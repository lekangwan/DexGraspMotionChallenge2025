#!/usr/bin/env python3
"""按同一v3字段比较旧基线与新正式1000条独立审计。"""

import argparse
import json
from pathlib import Path


FIELDS = (
    "source_10cm_success_count",
    "stable_physics_success_count",
    "transport_quality_success_count",
)


def compact(audit):
    """提取数量和成功率，避免把大型逐轨迹结果复制进对比文件。"""
    total = int(audit["trajectory_count"])
    return {
        "trajectory_count": total,
        **{
            field: {
                "count": int(audit[field]),
                "rate": float(audit[field]) / total,
            }
            for field in FIELDS
        },
    }


def compare(baseline, candidate, minimum_gain):
    """输出逐指标差值，并以运输成功率是否达到预设增益作最终判断。"""
    if baseline["trajectory_count"] != candidate["trajectory_count"]:
        raise ValueError("两个审计的轨迹数量不同，不能直接比较")
    old = compact(baseline)
    new = compact(candidate)
    deltas = {
        field: {
            "count": new[field]["count"] - old[field]["count"],
            "rate": new[field]["rate"] - old[field]["rate"],
        }
        for field in FIELDS
    }
    gain = deltas["transport_quality_success_count"]["rate"]
    return {
        "baseline": old,
        "candidate": new,
        "delta": deltas,
        "minimum_transport_gain": minimum_gain,
        "candidate_is_meaningfully_better": bool(gain >= minimum_gain),
        "decision": (
            "keep_candidate" if gain >= minimum_gain
            else "open_success_only_recovery"
        ),
    }


def main():
    """读取两份独立审计并保存机器可读的冻结决策。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--minimum-transport-gain", type=float, default=0.05)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    result = compare(baseline, candidate, args.minimum_transport_gain)
    result["baseline_path"] = str(args.baseline.resolve())
    result["candidate_path"] = str(args.candidate.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

