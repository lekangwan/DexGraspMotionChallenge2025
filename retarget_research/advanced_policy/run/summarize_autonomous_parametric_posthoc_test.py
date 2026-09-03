import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "retarget_research/advanced_policy/runs/autonomous_parametric_posthoc_test_v1"
FROZEN = ROOT / "retarget_research/advanced_policy/runs/autonomous_parametric_final_test_v1"


def read_result(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "status": data["status"],
        "trajectory_count": data["trajectory_count"],
        "success_count": data["success_count"],
        "success_rate": data["trajectory_micro_success_rate"],
        "mean_max_lift_m": data["mean_max_lift_m"],
        "mean_final_lift_m": data["mean_final_lift_m"],
        "autonomous_only": data["autonomous_only"],
        "teacher_checkpoint": data["teacher_checkpoint"],
        "residual_rl_checkpoint": data["residual_rl_checkpoint"],
    }


def main() -> None:
    methods = {
        path.parent.name: read_result(path)
        for path in sorted(RUN.glob("*/policy_evaluation_summary.json"))
    }
    frozen = {
        path.parent.name: read_result(path)
        for path in sorted(FROZEN.glob("*/policy_evaluation_summary.json"))
    }
    output = {
        "status": "complete" if len(methods) == 4 else "partial",
        "note": "post-hoc test comparison after the original valid-set selection was frozen",
        "candidate_count": len(methods),
        "candidates": methods,
        "original_frozen_methods": frozen,
    }
    destination = RUN / "comparison_summary.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for name, result in sorted(methods.items(), key=lambda item: item[1]["success_rate"], reverse=True):
        print(f"{name}: {result['success_count']}/{result['trajectory_count']} = {result['success_rate']:.1%}")
    print(f"SUMMARY={destination}")


if __name__ == "__main__":
    main()
