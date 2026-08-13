#!/usr/bin/env python3
"""分阶段验收正式1000轨迹实验产物，阻止缺失结果进入统计或训练。

输入：正式实验配置、可选冻结lock、要检查的inputs/candidates/evaluations/traces/policy_data阶段。
输出：逐阶段JSON报告和`FORMAL_BUNDLE=PASS/FAIL`退出码。
内部逻辑：按manifest的1000个唯一键核对候选、评测和trace，再检查策略split及三手NPZ。
作用：用户完成每个长命令后只需运行一次短审计，不必人工翻查数千个文件。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

try:
    from .freeze_formal_experiment import PROJECT_ROOT, resolve_project_path, verify_manifest
except ImportError:
    from freeze_formal_experiment import PROJECT_ROOT, resolve_project_path, verify_manifest


STAGES = ("inputs", "candidates", "evaluations", "traces", "policy_data")


def sha256(path):
    """分块计算任意文件SHA-256。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_keys(manifest):
    """把manifest展开成1000个`(物体,源轨迹索引)`唯一键。"""
    keys = {
        (entry["object_name"], int(index))
        for entry in manifest["entries"]
        for index in entry["trajectory_indices"]
    }
    expected = sum(len(entry["trajectory_indices"]) for entry in manifest["entries"])
    if len(keys) != expected:
        raise ValueError("manifest展开后存在重复轨迹键")
    return keys


def verify_lock(lock_path, experiment_path, manifest_path):
    """检查lock中的实验、manifest、输入凭据和实现文件指纹。

    输入：lock、正式实验配置和manifest路径。
    输出：通过状态及输入/实现指纹数量；不匹配时抛出异常。
    内部逻辑：先核对两个主文件，再分别遍历输入凭据与实现指纹重新计算SHA-256。
    作用：同时防止类别/inventory来源漂移和重定向实现漂移。
    """
    if not lock_path.is_file():
        raise FileNotFoundError(lock_path)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if sha256(experiment_path) != lock["experiment_config_sha256"]:
        raise ValueError("正式实验配置在freeze后发生变化")
    if sha256(manifest_path) != lock["manifest_sha256"]:
        raise ValueError("正式manifest在freeze后发生变化")
    input_mismatches = []
    for relative, expected in lock.get("input_fingerprints", {}).items():
        path = PROJECT_ROOT / relative
        actual = sha256(path) if path.is_file() else None
        if actual != expected:
            input_mismatches.append(relative)
    if input_mismatches:
        raise ValueError(f"freeze后输入凭据变化: {input_mismatches[:10]}")
    implementation_mismatches = []
    for relative, expected in lock["implementation_fingerprints"].items():
        path = PROJECT_ROOT / relative
        actual = sha256(path) if path.is_file() else None
        if actual != expected:
            implementation_mismatches.append(relative)
    if implementation_mismatches:
        raise ValueError(f"freeze后实现文件变化: {implementation_mismatches[:10]}")
    return {
        "status": "PASS",
        "input_fingerprint_count": len(lock.get("input_fingerprints", {})),
        "implementation_fingerprint_count": len(lock["implementation_fingerprints"]),
    }


def verify_candidate_set(manifest, spec):
    """核对一个候选目录中每物体文件的源索引、帧数和动作维度。"""
    directory = resolve_project_path(spec["target_dir"])
    observed = set()
    for entry in manifest["entries"]:
        path = directory / f"{entry['object_name']}.npy"
        if not path.is_file():
            raise FileNotFoundError(path)
        data = np.load(path, allow_pickle=True).item()
        indices = np.asarray(data["source_trajectory_indices"], dtype=np.int64)
        expected = np.asarray(entry["trajectory_indices"], dtype=np.int64)
        if not np.array_equal(indices, expected):
            raise ValueError(f"{path}源索引与manifest不一致")
        frames = np.asarray(data["grasp_seqs"])
        shape = (len(expected), 70, int(spec["action_dimension"]))
        if frames.shape != shape or not np.isfinite(frames).all():
            raise ValueError(f"{path}候选形状/有限性错误: {frames.shape} vs {shape}")
        observed.update((entry["object_name"], int(index)) for index in indices)
    if observed != expected_keys(manifest):
        raise ValueError(f"{directory}候选键集合不完整")
    return {"status": "PASS", "object_count": len(manifest["entries"]), "trajectory_count": len(observed)}


def verify_evaluation_set(manifest, manifest_path, spec):
    """核对统一评测摘要恰好包含manifest中的全部1000个结果。"""
    path = resolve_project_path(spec["evaluation_dir"]) / "manifest_evaluation_summary.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    summary = json.loads(path.read_text(encoding="utf-8"))
    if Path(summary["manifest"]).resolve() != manifest_path.resolve():
        raise ValueError(f"{path}引用了其他manifest")
    results = summary.get("results", [])
    observed = {
        (item["object_name"], int(item["source_trajectory_index"]))
        for item in results
    }
    if len(results) != len(observed) or observed != expected_keys(manifest):
        raise ValueError(f"{path}存在缺失或重复轨迹结果")
    counted = sum(bool(item["success"]) for item in results)
    if counted != int(summary["success_count"]):
        raise ValueError(f"{path}成功计数与逐轨迹结果不一致")
    return {
        "status": "PASS",
        "trajectory_count": len(results),
        "success_count": counted,
        "success_rate": counted / len(results),
    }


def verify_trace_set(manifest, spec):
    """核对一个策略来源的1000条trace字段、长度、维度和执行前对齐标签。"""
    directory = resolve_project_path(spec["trace_dir"])
    required = {
        "hand_dof_position", "hand_dof_velocity", "policy_action",
        "object_position", "object_quaternion_xyzw", "object_linear_velocity",
        "object_angular_velocity", "hand_object_contact_count",
        "source_frame_index", "is_hold", "metadata_json",
    }
    count = 0
    total_steps = 0
    for entry in manifest["entries"]:
        for source_index in entry["trajectory_indices"]:
            path = directory / entry["object_name"] / f"source_{source_index}_trace.npz"
            if not path.is_file():
                raise FileNotFoundError(path)
            with np.load(path, allow_pickle=False) as archive:
                missing = required - set(archive.files)
                if missing:
                    raise ValueError(f"{path}缺字段: {sorted(missing)}")
                metadata = json.loads(str(archive["metadata_json"].item()))
                lengths = {len(archive[name]) for name in archive.files if name != "metadata_json"}
                if len(lengths) != 1 or next(iter(lengths)) <= 0:
                    raise ValueError(f"{path}字段长度不一致")
                if archive["policy_action"].shape[1] != int(spec["action_dimension"]):
                    raise ValueError(f"{path}动作维度错误")
                if metadata.get("trace_alignment") != "pre_action_state_to_command_v1":
                    raise ValueError(f"{path}不是执行前状态到下一动作的对齐数据")
                if (
                    metadata.get("hand") != spec["hand"]
                    or metadata.get("object_name") != entry["object_name"]
                    or int(metadata.get("source_trajectory_index", -1)) != int(source_index)
                ):
                    raise ValueError(f"{path}元数据错配")
                total_steps += next(iter(lengths))
            count += 1
    return {"status": "PASS", "trajectory_count": count, "step_count": total_steps}


def verify_policy_data(experiment, manifest):
    """核对对象级split无泄漏及三只手准备完成的数据文件。"""
    split_path = resolve_project_path(experiment["policy"]["split"])
    if not split_path.is_file():
        raise FileNotFoundError(split_path)
    split = json.loads(split_path.read_text(encoding="utf-8"))
    if split.get("leakage_check") != "PASS":
        raise ValueError("策略split没有通过泄漏检查")
    train_objects = {item["object_name"] for item in split["records"] if item["split"] != "test"}
    test_objects = {item["object_name"] for item in split["records"] if item["split"] == "test"}
    if train_objects & test_objects:
        raise ValueError("策略训练与测试物体重叠")
    if train_objects | test_objects != {item["object_name"] for item in manifest["entries"]}:
        raise ValueError("策略split没有覆盖正式manifest全部物体")
    data_root = resolve_project_path(experiment["policy"]["data_root"])
    hands = {}
    for hand in ("linker", "xhand", "wuji"):
        directory = data_root / hand
        required = [
            directory / "train.npz", directory / "valid.npz", directory / "test.npz",
            directory / "normalization.npz", directory / "mappings.json",
            directory / "dataset_summary.json",
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"{hand}策略数据不完整: {missing}")
        summary = json.loads(required[-1].read_text(encoding="utf-8"))
        if summary.get("hand") != hand or summary.get("status") not in {"ready", "ready_with_gaps"}:
            raise ValueError(f"{hand}数据摘要状态异常")
        hands[hand] = {
            "status": summary["status"],
            "split_summaries": summary["split_summaries"],
            "missing_successful_train_categories": summary["missing_successful_train_categories"],
        }
    return {"status": "PASS", "split": str(split_path), "hands": hands}


def main():
    """按指定阶段执行验收、写报告并用退出码表达结果。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--stage", action="append", choices=STAGES)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    stages = args.stage or list(STAGES)
    experiment_path = args.experiment_config.resolve()
    report = {"experiment_config": str(experiment_path), "stages": {}}
    try:
        experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
        report["experiment"] = experiment["experiment_name"]
        manifest_path = resolve_project_path(experiment["manifest"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if "inputs" in stages:
            sources = verify_manifest(manifest, experiment["protocol"], True)
            input_report = {"status": "PASS", "verified_source_count": len(sources)}
            if args.lock is not None:
                input_report["lock"] = verify_lock(
                    args.lock.resolve(), experiment_path, manifest_path
                )
            report["stages"]["inputs"] = input_report
        if "candidates" in stages:
            report["stages"]["candidates"] = {
                name: verify_candidate_set(manifest, spec)
                for name, spec in experiment["candidate_sets"].items()
            }
        if "evaluations" in stages:
            report["stages"]["evaluations"] = {
                name: verify_evaluation_set(manifest, manifest_path, spec)
                for name, spec in experiment["candidate_sets"].items()
            }
        if "traces" in stages:
            report["stages"]["traces"] = {
                name: verify_trace_set(manifest, spec)
                for name, spec in experiment["candidate_sets"].items()
                if spec.get("trace_dir")
            }
        if "policy_data" in stages:
            report["stages"]["policy_data"] = verify_policy_data(experiment, manifest)
        report["status"] = "PASS"
    except Exception as exc:
        report["status"] = "FAIL"
        report["error_type"] = type(exc).__name__
        report["error"] = str(exc)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(f"FORMAL_BUNDLE={report['status']}")
    raise SystemExit(0 if report["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
