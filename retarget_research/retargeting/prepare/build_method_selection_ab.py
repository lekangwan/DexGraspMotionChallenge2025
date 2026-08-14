#!/usr/bin/env python3
"""从正式manifest的calibration部分冻结类别均衡的A/B方法选择集。

输入：每类恰有两个物体的正式manifest、固定随机种子和两个输出JSON路径。
输出：A/B各50类、50物体、50轨迹的manifest；同类的A/B来自不同物体。
内部逻辑：只用稳定SHA-256哈希决定物体分组和calibration轨迹，不读取物理成败。
作用：用较低重放成本进行方法筛选，同时避免类别数量和物体实例分布失衡。
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path


def stable_rank(seed: int, *parts: object) -> str:
    """为任意抽样键生成跨Python版本稳定的排序值。

    输入：整数种子，以及类别、物体或轨迹索引等可转为字符串的字段。
    输出：64字符SHA-256十六进制摘要，可直接作为确定性随机排序键。
    内部逻辑：用不可混淆的分隔符连接字段后计算哈希，不依赖`hash()`随机盐。
    作用：保证不同机器和重复运行产生完全相同的A/B清单。
    """
    payload = "\x1f".join([str(seed), *(str(part) for part in parts)])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def source_manifest_sha256(path: Path) -> str:
    """计算源manifest字节内容的SHA-256。

    输入：正式manifest路径。
    输出：64字符摘要。
    内部逻辑：直接对JSON原始字节哈希，手工改动空格也会改变摘要。
    作用：让A/B清单明确绑定到生成它们的正式清单版本。
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_calibration_index(entry: dict, seed: int) -> int:
    """从一个物体预先冻结的calibration索引中选出一条。

    输入：正式manifest物体条目和A/B种子。
    输出：被选中的源轨迹整数索引。
    内部逻辑：按`种子+类别+物体+索引`哈希排序，取最小者。
    作用：不接触成功率数据即可把每物体两条calibration缩减为一条。
    """
    calibration = [int(value) for value in entry.get("calibration_indices", [])]
    if len(calibration) < 1:
        raise ValueError(f"{entry.get('object_name')}没有calibration轨迹")
    return min(
        calibration,
        key=lambda index: stable_rank(
            seed, entry["category"], entry["object_name"], index
        ),
    )


def subset_entry(entry: dict, source_index: int) -> dict:
    """把正式物体条目缩减成单轨迹方法选择条目。

    输入：原条目和已选源轨迹索引。
    输出：保留路径/哈希/资产元数据、但只含一个索引的新字典。
    内部逻辑：复制原条目并同步改写三个轨迹划分字段。
    作用：让现有run/evaluate入口无需特殊分支即可读取小样本manifest。
    """
    result = dict(entry)
    result["trajectory_indices"] = [int(source_index)]
    result["calibration_indices"] = [int(source_index)]
    result["heldout_indices"] = []
    return result


def build_ab_manifests(source: dict, seed: int, source_path: Path) -> tuple[dict, dict]:
    """构造物体实例互斥、类别完全一致的A/B manifest。

    输入：已解析正式manifest、固定种子和源manifest路径。
    输出：`(A, B)`两个manifest字典。
    内部逻辑：逐类别要求恰有两个物体；哈希决定哪个进A，再各选一条calibration。
    作用：A可用于少量候选筛选，B可验证入选候选是否跨同类另一实例成立。
    """
    grouped: dict[str, list[dict]] = defaultdict(list)
    for entry in source.get("entries", []):
        grouped[str(entry["category"])].append(entry)
    declared_categories = [str(value) for value in source.get("categories", [])]
    categories = declared_categories or sorted(grouped)
    if len(categories) != 50 or set(categories) != set(grouped):
        raise ValueError(
            f"源manifest必须恰含50个完整类别，实际声明{len(categories)}、条目{len(grouped)}"
        )

    split_entries = {"A": [], "B": []}
    for category in sorted(categories):
        entries = grouped[category]
        if len(entries) != 2:
            raise ValueError(f"类别{category}应恰有2个物体，实际{len(entries)}")
        ordered = sorted(
            entries,
            key=lambda item: stable_rank(seed, category, item["object_name"]),
        )
        for split, entry in zip(("A", "B"), ordered):
            selected = select_calibration_index(entry, seed)
            split_entries[split].append(subset_entry(entry, selected))

    source_hash = source_manifest_sha256(source_path)
    outputs = []
    for split in ("A", "B"):
        entries = sorted(
            split_entries[split], key=lambda item: (item["category"], item["object_name"])
        )
        outputs.append(
            {
                "schema_version": 2,
                "purpose": f"balanced_method_selection_{split.lower()}_50c_50o_50t",
                "selection_seed": int(seed),
                "selection_split": split,
                "selection_source": "formal_manifest_calibration_only",
                "source_manifest": str(source_path.resolve()),
                "source_manifest_sha256": source_hash,
                "category_count": 50,
                "object_count": 50,
                "trajectory_count": 50,
                "objects_per_category": 1,
                "trajectories_per_object": 1,
                "calibration_per_object": 1,
                "categories": sorted(categories),
                "entries": entries,
            }
        )
    return outputs[0], outputs[1]


def write_manifest(path: Path, manifest: dict) -> None:
    """以可读JSON格式写出一个A/B manifest。

    输入：目标路径和manifest字典。
    输出：无返回值；创建父目录并写入UTF-8 JSON。
    内部逻辑：启用中文直写、两空格缩进和末尾换行。
    作用：使抽样结果便于人工审查和Git外本机留档。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    """解析参数并一次生成A/B两份冻结清单。

    输入：`--source-manifest/--output-a/--output-b/--seed`。
    输出：两个JSON文件及终端数量、路径摘要。
    内部逻辑：读取正式清单、调用纯构造函数，再检查A/B物体完全不重叠。
    作用：作为两天重定向收尾实验的唯一小样本抽样入口。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-a", type=Path, required=True)
    parser.add_argument("--output-b", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260814)
    args = parser.parse_args()

    source = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    manifest_a, manifest_b = build_ab_manifests(
        source, args.seed, args.source_manifest
    )
    objects_a = {item["object_name"] for item in manifest_a["entries"]}
    objects_b = {item["object_name"] for item in manifest_b["entries"]}
    if objects_a & objects_b:
        raise AssertionError("A/B物体集合意外重叠")
    write_manifest(args.output_a, manifest_a)
    write_manifest(args.output_b, manifest_b)
    print(f"A={manifest_a['trajectory_count']} -> {args.output_a.resolve()}")
    print(f"B={manifest_b['trajectory_count']} -> {args.output_b.resolve()}")
    print("categories=50, objects_disjoint=true, source=calibration_only")


if __name__ == "__main__":
    main()
