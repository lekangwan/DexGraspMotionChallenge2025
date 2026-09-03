#!/usr/bin/env python3
"""审计进阶v2 checkpoint和闭环报告是否违反自主策略边界。"""

import argparse
import hashlib
import json
from pathlib import Path

import torch


HANDS = ("linker", "xhand", "wuji")
MODELS = (
    "geometry_phase", "geometry_chunk", "geometry_plan_chunk",
    "phase_lead05", "phase_lead10", "phase_feedback_fingers",
)
FORBIDDEN_CHECKPOINT_KEYS = {
    "expert_actions", "expert_trajectory", "reference_trajectory",
    "retrieval_features", "retrieval_actions", "train_trajectories", "category_id",
}


def sha256(path):
    """输入文件路径，输出SHA-256；作用是锁定被审计checkpoint。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_checkpoint(path):
    """输入单个checkpoint，输出结构与禁用字段审计结果。"""
    payload = torch.load(path, map_location="cpu")
    top_keys = set(payload)
    config_keys = set(payload.get("config", {}))
    schema = payload.get("schema")
    state = payload.get("model_state", {})
    forbidden = sorted((top_keys | config_keys | set(state)) & FORBIDDEN_CHECKPOINT_KEYS)
    if schema not in {"geometry_action_chunk_policy_v1", "geometry_composite_policy_v1"}:
        raise ValueError(f"checkpoint规格错误: {path}")
    if forbidden:
        raise ValueError(f"checkpoint包含禁用字段: {path}: {forbidden}")
    if schema == "geometry_composite_policy_v1":
        components = {}
        for name in ("primary_checkpoint", "secondary_checkpoint"):
            component = Path(payload[name])
            child = torch.load(component, map_location="cpu")
            if child.get("schema") != "geometry_action_chunk_policy_v1":
                raise ValueError(f"复合策略子模型规格错误: {component}")
            child_keys = set(child) | set(child.get("config", {})) | set(child.get("model_state", {}))
            child_forbidden = sorted(child_keys & FORBIDDEN_CHECKPOINT_KEYS)
            if child_forbidden:
                raise ValueError(f"复合策略子模型包含禁用字段: {component}: {child_forbidden}")
            components[name] = {"path": str(component.resolve()), "sha256": sha256(component)}
        return {
            "path": str(path.resolve()), "sha256": sha256(path), "schema": schema,
            "model_type": payload["config"]["model_type"],
            "top_level_keys": sorted(top_keys), "config_keys": sorted(config_keys),
            "components": components, "forbidden_keys": forbidden,
        }
    if not state or not all(torch.is_tensor(value) for value in state.values()):
        raise ValueError(f"model_state不是纯参数张量: {path}")
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "schema": schema,
        "model_type": payload["config"]["model_type"],
        "epoch": int(payload["epoch"]),
        "best_valid_loss": float(payload["best_valid_loss"]),
        "top_level_keys": sorted(top_keys),
        "config_keys": sorted(config_keys),
        "parameter_tensor_count": len(state),
        "forbidden_keys": forbidden,
    }


def audit_rollout_summary(path):
    """输入valid/test汇总，检查每条报告未启用教师、专家腕部或残差参考轨迹。"""
    summary = json.loads(path.read_text(encoding="utf-8"))
    failures = []
    for item in summary["results"]:
        report = json.loads(Path(item["report"]).read_text(encoding="utf-8"))
        if (
            report.get("teacher_checkpoint") is not None
            or report.get("residual_rl_checkpoint") is not None
            or report.get("autonomous_residual_rl_checkpoint") is not None
            or bool(report.get("expert_wrist", False))
            or "no_future_expert_actions" not in report.get("initialization_rule", "")
        ):
            failures.append([item["object_name"], int(item["source_trajectory_index"])])
    if failures:
        raise ValueError(f"闭环报告违反自主边界: {path}: {failures[:5]}")
    return {
        "path": str(path.resolve()),
        "trajectory_count": int(summary["trajectory_count"]),
        "success_count": int(summary["success_count"]),
        "violations": failures,
    }


def main():
    """审计六个候选checkpoint；若valid结果已存在则同时审计闭环报告。"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs-root", type=Path,
        default=Path("retarget_research/advanced_policy_v2/runs/candidates_v1"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("retarget_research/advanced_policy_v2/AUTONOMY_AUDIT.json"),
    )
    args = parser.parse_args()
    checkpoints = {}
    rollouts = {}
    for hand in HANDS:
        checkpoints[hand] = {}
        rollouts[hand] = {}
        for model in MODELS:
            directory = args.runs_root / hand / model
            checkpoints[hand][model] = audit_checkpoint(directory / "best.pt")
            summary = directory / "closed_loop_valid50/policy_evaluation_summary.json"
            if summary.is_file():
                rollouts[hand][model] = audit_rollout_summary(summary)
    output = {
        "schema_version": 1,
        "contract": {
            "checkpoint_is_parametric_only": True,
            "category_id_used": False,
            "future_expert_actions_allowed": False,
            "expert_wrist_allowed": False,
            "trajectory_retrieval_allowed": False,
        },
        "checkpoints": checkpoints,
        "closed_loop_reports": rollouts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"AUTONOMY_AUDIT={args.output.resolve()}")


if __name__ == "__main__":
    main()
