#!/usr/bin/env python3
"""冻结正式1000轨迹实验的数据来源、方法配置和代码指纹。

输入：正式实验配置、已生成manifest、类别/inventory审计文件和输出lock路径。
输出：包含数据来源、100个源文件、方法文件及代码SHA-256的只读审计JSON。
内部逻辑：先核对50×2×10、2/8划分、源文件和资产，再分别哈希输入凭据与代码。
作用：保证正式结果出现后不能悄悄换类别规则、数据、参数或实现；lock只是实验合同。
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_project_path(value):
    """把项目相对路径或绝对路径统一解析为绝对Path。"""
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def sha256(path):
    """分块计算文件SHA-256，避免大轨迹文件整体进入内存。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest(manifest, protocol, verify_source_hashes=True):
    """核对正式manifest的数量、划分、路径、资产与源文件哈希。

    输入：manifest字典、协议数量字典和是否重新计算源哈希。
    输出：每个源文件的验证记录。
    内部逻辑：逐类别/物体检查索引唯一性和calibration/heldout严格分割。
    作用：在数小时批处理前一次性拦截抽样数量或数据版本错误。
    """
    entries = manifest.get("entries", [])
    expected = {
        "category_count": len({item.get("category") for item in entries}),
        "object_count": len(entries),
        "trajectory_count": sum(len(item.get("trajectory_indices", [])) for item in entries),
        "objects_per_category": None,
        "trajectories_per_object": None,
        "calibration_per_object": None,
        "heldout_per_object": None,
    }
    counts = Counter(item.get("category") for item in entries)
    expected["objects_per_category"] = set(counts.values())
    expected["trajectories_per_object"] = {
        len(item.get("trajectory_indices", [])) for item in entries
    }
    expected["calibration_per_object"] = {
        len(item.get("calibration_indices", [])) for item in entries
    }
    expected["heldout_per_object"] = {
        len(item.get("heldout_indices", [])) for item in entries
    }
    for name in ("category_count", "object_count", "trajectory_count"):
        if expected[name] != int(protocol[name]):
            raise ValueError(f"manifest的{name}={expected[name]}，预期{protocol[name]}")
    for name in (
        "objects_per_category",
        "trajectories_per_object",
        "calibration_per_object",
        "heldout_per_object",
    ):
        if expected[name] != {int(protocol[name])}:
            raise ValueError(f"manifest的{name}分布错误: {sorted(expected[name])}")
    if int(manifest.get("selection_seed", -1)) != int(protocol["selection_seed"]):
        raise ValueError("manifest抽样seed与正式协议不一致")

    seen_objects = set()
    seen_pairs = set()
    source_records = []
    for entry in entries:
        name = entry["object_name"]
        if name in seen_objects:
            raise ValueError(f"物体ID重复: {name}")
        seen_objects.add(name)
        selected = set(map(int, entry["trajectory_indices"]))
        calibration = set(map(int, entry["calibration_indices"]))
        heldout = set(map(int, entry["heldout_indices"]))
        if calibration & heldout or calibration | heldout != selected:
            raise ValueError(f"{name}的calibration/heldout不是严格分割")
        if any(index < 0 or index >= int(entry["available_trajectory_count"]) for index in selected):
            raise ValueError(f"{name}含越界轨迹索引")
        for index in selected:
            if (name, index) in seen_pairs:
                raise ValueError(f"轨迹键重复: {name}:{index}")
            seen_pairs.add((name, index))
        source = Path(entry["source_path"]).resolve()
        asset = Path(entry["object_asset_path"]).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        required_assets = [
            asset / "coacd" / "coacd_1.urdf",
            asset / "coacd" / "decomposed.obj",
        ]
        missing = [str(path) for path in required_assets if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"{name}缺少资产: {missing}")
        actual_hash = sha256(source) if verify_source_hashes else entry["source_sha256"]
        if actual_hash != entry["source_sha256"]:
            raise ValueError(f"{name}源轨迹在manifest冻结后发生变化")
        source_records.append(
            {
                "object_name": name,
                "source_path": str(source),
                "source_sha256": actual_hash,
                "object_asset_path": str(asset),
            }
        )
    return source_records


def collect_code_files(roots):
    """收集代码根目录中的Python/JSON/YAML/Markdown文件并排除缓存和输出。"""
    allowed = {".py", ".json", ".yaml", ".yml", ".md"}
    files = set()
    for root in roots:
        if not root.is_dir():
            raise FileNotFoundError(root)
        for path in root.rglob("*"):
            if path.is_file() and path.suffix in allowed and "__pycache__" not in path.parts:
                files.add(path.resolve())
    return sorted(files)


def build_lock(experiment_path, verify_source_hashes=True):
    """读取实验合同并生成完整lock字典。

    输入：实验配置路径和是否重新计算100个源轨迹哈希。
    输出：含数据凭据、manifest、源轨迹、方法和实现指纹的字典。
    内部逻辑：验证manifest后，把inventory及`input_files`与实现代码分组哈希。
    作用：让类别来源变化和代码变化都能被后续验收分别定位。
    """
    experiment_path = experiment_path.resolve()
    experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
    manifest_path = resolve_project_path(experiment["manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sources = verify_manifest(manifest, experiment["protocol"], verify_source_hashes)
    input_paths = [resolve_project_path(experiment["inventory"])] + [
        resolve_project_path(value) for value in experiment.get("input_files", [])
    ]
    missing_inputs = [str(path) for path in input_paths if not path.is_file()]
    if missing_inputs:
        raise FileNotFoundError(f"缺少冻结输入凭据: {missing_inputs}")
    method_paths = [resolve_project_path(value) for value in experiment["method_files"]]
    missing = [str(path) for path in method_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"缺少冻结方法文件: {missing}")
    code_paths = collect_code_files(
        [resolve_project_path(value) for value in experiment["code_roots"]]
    )
    fingerprints = {
        str(path.relative_to(PROJECT_ROOT)): sha256(path)
        for path in sorted(set(method_paths + code_paths + [experiment_path]))
    }
    input_fingerprints = {
        str(path.relative_to(PROJECT_ROOT)): sha256(path)
        for path in sorted(set(input_paths))
    }
    return {
        "schema_version": 2,
        "status": "FROZEN",
        "experiment_name": experiment["experiment_name"],
        "experiment_config": str(experiment_path),
        "experiment_config_sha256": sha256(experiment_path),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "protocol": experiment["protocol"],
        "verified_source_count": len(sources),
        "verified_sources": sources,
        "input_fingerprints": input_fingerprints,
        "implementation_fingerprints": fingerprints,
        "rule": "Any hash mismatch invalidates direct comparison with this formal experiment.",
    }


def main():
    """解析参数、拒绝意外覆盖并写出正式lock。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-source-rehash", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.force:
        raise FileExistsError(f"lock已存在；如确认重新冻结请加--force: {args.output}")
    lock = build_lock(args.experiment_config, not args.skip_source_rehash)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(lock, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"verified_sources={lock['verified_source_count']}")
    print(f"fingerprinted_files={len(lock['implementation_fingerprints'])}")
    print(f"FORMAL_EXPERIMENT_FROZEN={args.output.resolve()}")


if __name__ == "__main__":
    main()
