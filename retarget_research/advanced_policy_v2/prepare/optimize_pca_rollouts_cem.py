#!/usr/bin/env python3
"""在valid近失案例上检查：物理CEM能否把自主PCA轨迹修成稳定抓取。"""

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np


RESEARCH = Path(__file__).resolve().parents[2]
for path in (
    RESEARCH / "advanced_policy",
    RESEARCH / "retargeting/evaluate",
    RESEARCH / "retargeting/run",
):
    sys.path.insert(0, str(path))

from train_residual_ppo_general import GeneralResidualEnv, rollout  # noqa: E402
from physics_cem_refine import (  # noqa: E402
    PhysicsCase, apply_trajectory_parameters, physics_score,
)


BASE_MODELS = {
    "linker": "geometry_pca32",
    "xhand": "geometry_pca16",
    "wuji": "geometry_pca16",
}


def environment_args(device):
    """PCA已经是240个物理步，因此不再做70帧三倍插值。"""
    return SimpleNamespace(
        device=device, dt=1.0 / 60.0, substeps=2,
        finger_stiffness=120.0, finger_damping=5.0,
        mimic_stiffness=120.0, mimic_damping=5.0,
        clearance=0.005, object_friction=1.0, settle_steps=30,
        steps_per_frame=1, hold_steps=0, lift_threshold=0.15,
    )


def parameter_distribution(hand):
    """返回与基础任务全局CEM一致的低维修部、手指和抬升搜索范围。"""
    if hand == "linker":
        finger_std = [0.08, 0.12] + [0.15] * 4
        finger_bound = [0.20, 0.30] + [0.35] * 4
    else:
        finger_std = [0.12] * 5
        finger_bound = [0.30] * 5
    std = np.asarray([0.010] * 3 + [0.10] * 3 + finger_std + [0.020], np.float32)
    bounds = np.asarray([0.025] * 3 + [0.25] * 3 + finger_bound + [0.050], np.float32)
    return std, bounds


def autonomous_success(metric):
    """进阶任务统一采用稳定抬升且掌物运输稳定的成功定义。"""
    return bool(metric["success"] and metric.get("transport_stability_success", False))


def score_metric(metric):
    """先把success改成进阶统一口径，再调用已有物理抓取质量分数。"""
    value = dict(metric)
    value["stable_lift_success"] = bool(value["success"])
    value["success"] = autonomous_success(value)
    return value, float(physics_score(value, 0.15, grasp_quality=True))


def make_case(hand, report, frames):
    """由自主闭环报告构造只含PCA预测轨迹的物理优化案例。"""
    source = np.load(report["source"], allow_pickle=True).item()
    source_index = int(report["source_trajectory_index"])
    return PhysicsCase(
        hand=hand,
        category=report["category"],
        object_name=report["object_name"],
        source_index=source_index,
        target_frames=np.asarray(frames, dtype=np.float32),
        mesh_path=Path(report["object_dir"]) / "coacd/decomposed.obj",
        scale=float(np.asarray(source["obj_scale"])[source_index]),
        rotation=np.asarray(source["obj_rotmat"])[source_index],
    )


def evaluate(hand, report, target_data, base_frames, population, device):
    """在并行PhysX环境中评估一批低维轨迹修正。"""
    cases = [
        make_case(hand, report, apply_trajectory_parameters(base_frames, parameters))
        for parameters in population
    ]
    env = GeneralResidualEnv(cases, target_data, environment_args(device))
    try:
        raw_metrics = rollout(env)
    finally:
        env.close()
    converted = [score_metric(item) for item in raw_metrics]
    return [item[0] for item in converted], np.asarray([item[1] for item in converted])


def optimize_case(hand, summary_item, args):
    """对一条PCA失败轨迹运行CEM，并单环境复验最佳候选。"""
    report = json.loads(Path(summary_item["report"]).read_text(encoding="utf-8"))
    base_frames = np.asarray(report["predicted_policy_actions"], dtype=np.float32)
    target_data = np.load(report["target"], allow_pickle=True).item()
    rng = np.random.default_rng(args.seed + int(report["source_trajectory_index"]))
    std, bounds = parameter_distribution(hand)
    mean = np.zeros_like(std)
    best_parameters = mean.copy()
    best_score = -np.inf
    best_metric = None
    baseline_metric = None
    history = []
    for iteration in range(1, args.iterations + 1):
        population = rng.normal(mean, std, size=(args.population, len(mean))).astype(np.float32)
        population = np.clip(population, -bounds, bounds)
        population[0] = 0.0
        metrics, scores = evaluate(
            hand, report, target_data, base_frames, population, args.device
        )
        if baseline_metric is None:
            baseline_metric = metrics[0]
            best_metric = metrics[0]
            best_score = float(scores[0])
        ranking = np.argsort(scores)[::-1]
        elite = population[ranking[:args.elite]]
        mean = elite.mean(axis=0)
        std = np.maximum(elite.std(axis=0), bounds * 0.05)
        winner = int(ranking[0])
        if scores[winner] > best_score:
            best_score = float(scores[winner])
            best_parameters = population[winner].copy()
            best_metric = metrics[winner]
        item = {
            "iteration": iteration,
            "best_score": float(scores[winner]),
            "best_success": bool(metrics[winner]["success"]),
            "best_max_lift_m": float(metrics[winner]["max_lift_m"]),
        }
        history.append(item)
        print(json.dumps({"hand": hand, "object": report["object_name"], **item}, ensure_ascii=False), flush=True)

    confirm_metrics, confirm_scores = evaluate(
        hand, report, target_data, base_frames,
        np.stack([np.zeros_like(best_parameters), best_parameters]), args.device,
    )
    accepted = bool(
        confirm_metrics[1]["success"]
        and confirm_scores[1] > confirm_scores[0]
    )
    return {
        "hand": hand,
        "category": report["category"],
        "object_name": report["object_name"],
        "source_trajectory_index": int(report["source_trajectory_index"]),
        "original_report": str(Path(summary_item["report"]).resolve()),
        "original_success": bool(summary_item["success"]),
        "original_max_lift_m": float(summary_item["max_lift_m"]),
        "cem_baseline_metric": baseline_metric,
        "search_best_metric": best_metric,
        "confirmation_baseline_metric": confirm_metrics[0],
        "confirmation_candidate_metric": confirm_metrics[1],
        "parameters": best_parameters.tolist(),
        "accepted": accepted,
        "history": history,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hand", choices=tuple(BASE_MODELS), required=True)
    parser.add_argument("--cases", type=int, default=2)
    parser.add_argument("--population", type=int, default=8)
    parser.add_argument("--elite", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.elite <= args.population:
        parser.error("elite必须位于1到population之间")
    runs = RESEARCH / "advanced_policy_v2/runs/candidates_v1"
    summary_path = runs / args.hand / BASE_MODELS[args.hand] / "closed_loop_valid50/policy_evaluation_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    failures = [item for item in summary["results"] if not item["success"]]
    failures.sort(
        key=lambda item: (item["max_lift_m"], item["hand_object_contact_steps"]),
        reverse=True,
    )
    selected = failures[:args.cases]
    if args.dry_run:
        print(json.dumps([
            {k: item[k] for k in ("category", "object_name", "source_trajectory_index", "max_lift_m")}
            for item in selected
        ], ensure_ascii=False, indent=2))
        return
    results = [optimize_case(args.hand, item, args) for item in selected]
    output = {
        "schema_version": 1,
        "purpose": "valid-only feasibility diagnostic; never used as training data",
        "hand": args.hand,
        "population": args.population,
        "iterations": args.iterations,
        "result_count": len(results),
        "accepted_count": sum(item["accepted"] for item in results),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
