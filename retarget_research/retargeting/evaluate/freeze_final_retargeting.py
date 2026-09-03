#!/usr/bin/env python3
"""校验最终三手1000条结果，并生成可复核的冻结清单。"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil


PROJECT_ROOT = Path(__file__).resolve().parents[3]
HANDS = ("linker", "xhand", "wuji")


def sha256(path):
    """流式计算文件SHA-256，避免一次读入大型目标轨迹。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path):
    """把绝对路径转换为仓库内可移植相对路径。"""
    return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))


def artifact(path):
    """记录单个复现文件的相对路径、大小和内容哈希。"""
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": relative(path),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def copy_manifests(source_root, final_root):
    """将冻结输入manifest复制到最终目录，使下游不再跨阶段取文件。"""
    output = final_root / "manifests"
    output.mkdir(parents=True, exist_ok=True)
    for hand in HANDS:
        source = source_root / f"{hand}.json"
        target = output / f"{hand}.json"
        if target.exists() and sha256(target) != sha256(source):
            raise ValueError(f"最终目录已有不同manifest: {target}")
        if not target.exists():
            shutil.copy2(source, target)


def validate_hand(hand, final_root):
    """检查一只手的样本规模、确认计数、评测计数和可训练键。"""
    manifest_path = final_root / "manifests" / f"{hand}.json"
    confirmation_path = final_root / "targets" / hand / "confirmation_summary.json"
    evaluation_path = final_root / "evaluation" / hand / "manifest_evaluation_summary.json"
    audit_path = final_root / "audit" / f"{hand}_stable_audit.json"
    eligible_path = final_root / "audit" / f"{hand}_training_eligible_keys.json"
    manifest = load_json(manifest_path)
    confirmation = load_json(confirmation_path)
    evaluation = load_json(evaluation_path)
    audit = load_json(audit_path)
    eligible = load_json(eligible_path)

    entries = manifest["entries"]
    trajectory_count = sum(len(item["trajectory_indices"]) for item in entries)
    if (len(entries), len({item["category"] for item in entries}), trajectory_count) != (100, 50, 1000):
        raise ValueError(f"{hand} manifest不是50类/100物体/1000轨迹")
    if int(evaluation["trajectory_count"]) != 1000 or int(audit["trajectory_count"]) != 1000:
        raise ValueError(f"{hand}独立评测或审计不完整")
    if len(eligible) != int(audit["training_eligible_count"]):
        raise ValueError(f"{hand}可训练键数量与审计不一致")
    changed = int(confirmation["changed_trajectory_count"])
    accepted = int(confirmation["accepted_change_count"])
    restored = int(confirmation["restored_baseline_count"])
    if accepted + restored != changed:
        raise ValueError(f"{hand}重复确认计数不闭合")

    target_paths = sorted((final_root / "targets" / hand).glob("*.npy"))
    if len(target_paths) != 100:
        raise ValueError(f"{hand}最终目标文件应为100个，实际{len(target_paths)}")
    return {
        "objects": 100,
        "categories": 50,
        "trajectories": 1000,
        "confirmation": {
            "changed": changed,
            "accepted": accepted,
            "restored": restored,
            "repeats_per_baseline_and_candidate": int(confirmation["confirmation_repeats"]),
            "selection_margin": float(confirmation["selection_margin"]),
        },
        "metrics": {
            "reference_isaac_success_count": int(audit["reference_isaac_success_count"]),
            "reference_isaac_terminal_success_count": int(audit["reference_isaac_terminal_success_count"]),
            "stable_physics_success_count": int(audit["stable_physics_success_count"]),
            "transport_quality_success_count": int(audit["transport_quality_success_count"]),
            "training_eligible_count": int(audit["training_eligible_count"]),
            "per_policy_split": audit["per_policy_split"],
        },
        "artifacts": {
            "manifest": artifact(manifest_path),
            "confirmation_summary": artifact(confirmation_path),
            "evaluation_summary": artifact(evaluation_path),
            "stable_audit": artifact(audit_path),
            "training_eligible_keys": artifact(eligible_path),
            "targets": [artifact(path) for path in target_paths],
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--formal-root", type=Path,
        default=PROJECT_ROOT / "retarget_research/outputs/reboot_synergy_rank5_formal1000_v1",
    )
    parser.add_argument(
        "--tracked-lock", type=Path,
        default=PROJECT_ROOT / "retarget_research/retargeting/configs/final_retargeting_release_v1.json",
    )
    args = parser.parse_args()
    formal_root = args.formal_root.resolve()
    preconfirm = formal_root / "final_synergy_rank5"
    final_root = formal_root / "postconfirmed_rank5_v1"
    copy_manifests(preconfirm / "manifests", final_root)

    method_files = [
        PROJECT_ROOT / "retarget_research/manifests/formal_50c_100o_1000t_seed20260808.json",
        PROJECT_ROOT / "retarget_research/retargeting/configs/stable_success_protocol_v3_selected_methods.json",
        PROJECT_ROOT / "retarget_research/retargeting/run/run_reboot_synergy_rank5_formal1000_v1.sh",
        PROJECT_ROOT / "retarget_research/retargeting/run/run_postconfirm_rank5_formal1000_v1.sh",
        PROJECT_ROOT / "retarget_research/retargeting/success_only/confirm_existing_candidates_manifest.py",
        PROJECT_ROOT / "retarget_research/retargeting/success_only/confirm_existing_candidates_isolated.py",
    ]
    for hand in HANDS:
        method_files.extend([
            preconfirm / "raw" / hand / "synergy_basis.npy",
            preconfirm / "raw" / hand / "screen_summary.json",
        ])
    release = {
        "schema_version": 1,
        "release_id": "rank5_cem_postconfirmed_formal1000_v1",
        "frozen_date": "2026-09-01",
        "status": "final",
        "dataset_contract": {"categories": 50, "objects": 100, "trajectories_per_hand": 1000},
        "method": {
            "shared_pipeline": [
                "kinematic_retargeting_initialization",
                "global_physics_cem",
                "rank5_joint_synergy_phase_cem",
                "two_repeat_isolated_candidate_confirmation",
                "independent_1000_trajectory_replay",
                "reference_and_stable_transport_audit",
            ],
            "linker_extra_stage": "second_global_physics_cem_before_rank5",
            "rank": 5,
            "cem_population": 8,
            "cem_elite": 2,
            "cem_iterations": 2,
            "primary_report_metric": "reference_isaac_success",
            "expert_training_gate": "stable_15cm_terminal_and_palm_relative_transport",
        },
        "hands": {hand: validate_hand(hand, final_root) for hand in HANDS},
        "method_artifacts": [artifact(path) for path in method_files],
    }
    canonical = json.dumps(release, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    release["release_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    content = json.dumps(release, ensure_ascii=False, indent=2) + "\n"
    output_lock = final_root / "FINAL_RETARGETING_LOCK.json"
    output_lock.write_text(content, encoding="utf-8")
    args.tracked_lock.parent.mkdir(parents=True, exist_ok=True)
    args.tracked_lock.write_text(content, encoding="utf-8")
    print(f"FINAL_RETARGETING_LOCK={output_lock}")
    print(f"TRACKED_RETARGETING_LOCK={args.tracked_lock}")
    print(f"RELEASE_SHA256={release['release_sha256']}")


if __name__ == "__main__":
    main()
