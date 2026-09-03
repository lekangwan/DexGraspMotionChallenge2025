#!/usr/bin/env python3
"""在固定valid前50条上依次闭环评估已训练的候选。"""

import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["geometry_phase", "geometry_chunk"])
    parser.add_argument("--hands", nargs="+", default=["linker", "xhand", "wuji"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--parallel-hands", action="store_true")
    args = parser.parse_args()
    research_root = Path(__file__).resolve().parents[2]
    project_root = research_root.parent
    final = research_root / "outputs/reboot_synergy_rank5_formal1000_v1/postconfirmed_rank5_v1"
    split = research_root / "advanced_policy/data/formal_v1/policy_split_seed20260813.json"
    evaluator = research_root / "advanced_policy/evaluate_policy_manifest.py"
    runs = research_root / "advanced_policy_v2/runs/candidates_v1"
    jobs = []
    for hand in args.hands:
        for model_type in args.models:
            checkpoint = runs / hand / model_type / "best.pt"
            output = runs / hand / model_type / "closed_loop_valid50"
            command = [
                sys.executable, "-u", str(evaluator),
                "--hand", hand,
                "--manifest", str(final / f"manifests/{hand}.json"),
                "--policy-split", str(split),
                "--target-dir", str(final / f"targets/{hand}"),
                "--checkpoint", str(checkpoint),
                "--data-dir", str(research_root / f"advanced_policy_v2/data/final/{hand}"),
                "--output-dir", str(output),
                "--split", "valid", "--max-tasks-per-category", "1",
                "--device", args.device, "--workers", str(args.workers),
                "--lift-threshold", "0.15", "--autonomous-only", "--resume",
            ]
            jobs.append((hand, model_type, output, command))
    if args.parallel_hands:
        processes = [
            (job, subprocess.Popen(job[3], cwd=str(project_root)))
            for job in jobs
        ]
        for (hand, model_type, _, _), process in processes:
            if process.wait() != 0:
                raise RuntimeError(f"{hand}/{model_type}评测失败")
    else:
        for _, _, _, command in jobs:
            subprocess.run(command, cwd=str(project_root), check=True)
    summaries = {hand: {} for hand in args.hands}
    for hand, model_type, output, _ in jobs:
        summary = json.loads((output / "policy_evaluation_summary.json").read_text(encoding="utf-8"))
        summaries[hand][model_type] = {
            "success_count": summary["success_count"],
            "trajectory_count": summary["trajectory_count"],
        }
    output = runs / "valid50_comparison.json"
    output.write_text(json.dumps(summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
