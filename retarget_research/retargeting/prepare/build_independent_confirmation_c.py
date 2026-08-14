#!/usr/bin/env python3
"""为正式结果之后的新方法冻结50类别独立确认C组。

输入：正式manifest、完整inventory CSV、固定种子和输出JSON。
输出：每类1条的C组；优先选正式清单外新物体，无第三实例时选从未使用的轨迹。
内部逻辑：所有物体和轨迹均按稳定SHA-256排序，不读取任何重放成功率。
作用：避免在已经看过A/B和正式1000条后，继续用同一批结果证明新方法泛化。
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

try:
    from .build_method_selection_ab import stable_rank
except ImportError:
    from build_method_selection_ab import stable_rank


def file_sha256(path: Path) -> str:
    """分块计算源轨迹文件SHA-256。

    输入：本地npy路径。
    输出：64字符十六进制摘要。
    内部逻辑：按1 MiB分块读取，避免一次加载完整文件到内存。
    作用：让新物体条目和正式manifest一样可检测源数据替换。
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_inventory(path: Path) -> list[dict]:
    """读取C组所需的inventory字段并检查路径存在。

    输入：正式数据inventory CSV。
    输出：规范化物体字典列表。
    内部逻辑：解析ID、类别、轨迹数及绝对源/资产路径，拒绝缺失项。
    作用：为正式清单之外的第三实例提供显式类别标签和真实资产来源。
    """
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            source = Path(row["trajectory_file"]).expanduser().resolve()
            asset = Path(row["asset_dir"]).expanduser().resolve()
            if not source.is_file() or not asset.is_dir():
                raise FileNotFoundError(f"inventory路径缺失: {source}, {asset}")
            rows.append(
                {
                    "object_name": row["object_id"].strip(),
                    "category": row["category"].strip(),
                    "source_path": str(source),
                    "object_asset_path": str(asset),
                    "available_trajectory_count": int(row["trajectory_count"]),
                }
            )
    return rows


def new_object_entry(row: dict, source_index: int) -> dict:
    """把inventory新物体转换为批处理兼容的单轨迹条目。

    输入：inventory行和已选源索引。
    输出：含路径、哈希、形状元数据和确认标志的manifest条目。
    内部逻辑：对源文件计算哈希，轨迹划分只保留一条confirmation索引。
    作用：让现有三手run/evaluate入口无需修改即可处理C组。
    """
    source = Path(row["source_path"])
    return {
        "object_name": row["object_name"],
        "category": row["category"],
        "source_path": row["source_path"],
        "source_sha256": file_sha256(source),
        "object_asset_path": row["object_asset_path"],
        "available_trajectory_count": int(row["available_trajectory_count"]),
        "frame_count": 70,
        "action_dimension": 28,
        "trajectory_indices": [int(source_index)],
        "calibration_indices": [],
        "heldout_indices": [int(source_index)],
        "confirmation_indices": [int(source_index)],
        "new_object_instance": True,
        "confirmation_source": "object_not_present_in_formal_manifest",
    }


def unused_trajectory_entry(entry: dict, source_index: int) -> dict:
    """为没有第三实例的类别建立已知物体新轨迹条目。

    输入：正式manifest物体条目和未使用源索引。
    输出：复制路径/哈希但只含新索引、并明确标为非新实例的条目。
    内部逻辑：不修改正式条目，重建三个轨迹划分字段和确认来源标志。
    作用：保持50类别全覆盖，同时诚实区分实例泛化与同实例新方向确认。
    """
    result = dict(entry)
    result.update(
        {
            "trajectory_indices": [int(source_index)],
            "calibration_indices": [],
            "heldout_indices": [int(source_index)],
            "confirmation_indices": [int(source_index)],
            "new_object_instance": False,
            "confirmation_source": "unused_trajectory_from_formal_object_no_third_instance",
        }
    )
    return result


def build_confirmation_manifest(
    formal: dict,
    inventory_rows: list[dict],
    seed: int,
    formal_path: Path,
    excluded_manifests: list[dict] | None = None,
    split_label: str = "C",
    excluded_manifest_paths: list[Path] | None = None,
) -> dict:
    """构造每类一条且尽量使用新实例的C组manifest。

    输入：正式manifest、inventory行、种子、正式清单路径，
    可选已使用manifest列表和新分组名。
    输出：批处理兼容的确认manifest字典。
    内部逻辑：每类先哈希选择所有已用清单外的新物体；若没有，
    再从旧物体的全部未使用索引中哈希取一条。任何类别无候选都会立即报错。
    作用：可依次冻结C/D等独立确认集，阻止事后换样本或重复轨迹。
    """
    excluded_manifests = excluded_manifests or []
    excluded_manifest_paths = excluded_manifest_paths or []
    split_label = str(split_label).upper()
    if not split_label.isalpha():
        raise ValueError(f"分组名必须为字母: {split_label}")
    categories = sorted(str(value) for value in formal.get("categories", []))
    if len(categories) != 50:
        raise ValueError(f"正式manifest应含50类，实际{len(categories)}")
    formal_by_category: dict[str, list[dict]] = defaultdict(list)
    formal_objects = set()
    used_objects = set()
    used_keys = set()
    for entry in formal.get("entries", []):
        formal_by_category[str(entry["category"])].append(entry)
        formal_objects.add(str(entry["object_name"]))
        used_objects.add(str(entry["object_name"]))
        used_keys.update(
            (str(entry["object_name"]), int(index))
            for index in entry["trajectory_indices"]
        )
    for manifest in excluded_manifests:
        for entry in manifest.get("entries", []):
            object_name = str(entry["object_name"])
            used_objects.add(object_name)
            used_keys.update(
                (object_name, int(index))
                for index in entry["trajectory_indices"]
            )
    inventory_by_category: dict[str, list[dict]] = defaultdict(list)
    for row in inventory_rows:
        if row["category"] in categories and row["object_name"] not in used_objects:
            inventory_by_category[row["category"]].append(row)

    entries = []
    for category in categories:
        unseen = inventory_by_category.get(category, [])
        if unseen:
            row = min(
                unseen,
                key=lambda item: stable_rank(
                    seed, split_label, category, "object", item["object_name"]
                ),
            )
            candidates = range(int(row["available_trajectory_count"]))
            source_index = min(
                candidates,
                key=lambda index: stable_rank(
                    seed, split_label, category, row["object_name"], index
                ),
            )
            entries.append(new_object_entry(row, source_index))
            continue

        unused = []
        for formal_entry in formal_by_category.get(category, []):
            selected = {int(value) for value in formal_entry["trajectory_indices"]}
            for source_index in range(int(formal_entry["available_trajectory_count"])):
                if (
                    source_index not in selected
                    and (str(formal_entry["object_name"]), source_index)
                    not in used_keys
                ):
                    unused.append((formal_entry, source_index))
        if not unused:
            raise ValueError(f"类别{category}既无第三实例，也无未使用轨迹")
        formal_entry, source_index = min(
            unused,
            key=lambda item: stable_rank(
                seed, split_label, category, item[0]["object_name"], item[1]
            ),
        )
        entries.append(unused_trajectory_entry(formal_entry, source_index))

    entries.sort(key=lambda item: item["category"])
    new_count = sum(bool(item["new_object_instance"]) for item in entries)
    exclusion_records = []
    for index, path in enumerate(excluded_manifest_paths):
        exclusion_records.append(
            {
                "path": str(path.resolve()),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "purpose": excluded_manifests[index].get("purpose"),
            }
        )
    split_lower = split_label.lower()
    return {
        "schema_version": 2,
        "purpose": f"post_formal_independent_confirmation_{split_lower}_50c_50t",
        "selection_seed": int(seed),
        "source_formal_manifest": str(formal_path.resolve()),
        "source_formal_manifest_sha256": hashlib.sha256(
            formal_path.read_bytes()
        ).hexdigest(),
        "excluded_prior_manifests": exclusion_records,
        "category_count": 50,
        "object_count": 50,
        "trajectory_count": 50,
        "objects_per_category": 1,
        "trajectories_per_object": 1,
        "new_object_instance_count": new_count,
        "known_object_new_trajectory_count": 50 - new_count,
        "categories": categories,
        "entries": entries,
    }


def main() -> None:
    """解析正式数据路径并写出冻结C组。

    输入：`--formal-manifest/--inventory/--output/--seed`，可重复指定
    `--exclude-manifest`并用`--split-label`命名新分组。
    输出：50条确认JSON及新实例/新轨迹数量摘要。
    内部逻辑：读取两个输入，调用纯构造函数并以UTF-8缩进JSON写盘。
    作用：作为阶段重定时或以后新方法唯一允许的最终小样本确认入口。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-manifest", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--exclude-manifest", type=Path, action="append", default=[])
    parser.add_argument("--split-label", default="C")
    args = parser.parse_args()
    formal = json.loads(args.formal_manifest.read_text(encoding="utf-8"))
    inventory = load_inventory(args.inventory)
    excluded = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in args.exclude_manifest
    ]
    manifest = build_confirmation_manifest(
        formal,
        inventory,
        args.seed,
        args.formal_manifest,
        excluded,
        args.split_label,
        args.exclude_manifest,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"trajectories={manifest['trajectory_count']}")
    print(f"new_object_instances={manifest['new_object_instance_count']}")
    print(
        "known_object_new_trajectories="
        f"{manifest['known_object_new_trajectory_count']}"
    )
    print(f"output={args.output.resolve()}")


if __name__ == "__main__":
    main()
