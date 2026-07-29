"""Select a policy-blind, nested object split for category-expert scaling.

The existing four training objects per category are kept as the smallest
training set.  New objects are selected using raw trajectory integrity and
geometry statistics only.  Objects that appeared in any earlier local split,
evaluation result, or preprocessed dataset are excluded from the new test set.
"""

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np

from custom_tools.select_object_split import (
    FEATURE_KEYS,
    collect_category,
    normalized_features,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
OBJECT_PATTERN = re.compile(
    r"core-(?:bottle|mug|bowl|camera)-[0-9a-f]+")
TEXT_SUFFIXES = {".csv", ".json", ".md", ".txt", ".yaml", ".yml"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Select nested 4/10/20-object category training sets.")
    parser.add_argument(
        "--base-manifest",
        default=str(REPO_ROOT / "custom_tools" / "configs" /
                    "object_split_final.json"))
    parser.add_argument(
        "--dataset-root",
        default=str(REPO_ROOT / "external_data" / "dataset"))
    parser.add_argument(
        "--mesh-root",
        default=str(REPO_ROOT / "external_data" / "meshdata"))
    parser.add_argument("--train-sizes", nargs="+", type=int, default=[4, 10, 20])
    parser.add_argument("--test-per-category", type=int, default=5)
    parser.add_argument("--backups-per-category", type=int, default=10)
    parser.add_argument("--min-trajectories", type=int, default=20)
    parser.add_argument("--min-rotation-std", type=float, default=0.01)
    parser.add_argument("--min-action-dispersion", type=float, default=0.01)
    parser.add_argument(
        "--output-json",
        default=str(REPO_ROOT / "custom_tools" / "configs" /
                    "scaled_category_split_candidates_v1.json"))
    parser.add_argument(
        "--output-csv",
        default=str(REPO_ROOT / "custom_tools" / "configs" /
                    "scaled_category_split_candidates_v1.csv"))
    return parser.parse_args()


def historical_object_ids():
    """Return every target-category object ID already exposed locally."""
    object_ids = set()
    text_roots = (
        REPO_ROOT / "custom_tools" / "configs",
        REPO_ROOT / "custom_tools" / "results",
    )
    scanned_files = 0
    for root in text_roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            # Result folders can contain media or accidental large logs.  The
            # split audit only needs compact manifests and text summaries.
            if path.stat().st_size > 20 * 1024 * 1024:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            object_ids.update(OBJECT_PATTERN.findall(text))
            scanned_files += 1

    dataset_root = REPO_ROOT / "dexgrasp" / "dataset"
    if dataset_root.exists():
        for path in dataset_root.rglob("*.npy"):
            match = OBJECT_PATTERN.fullmatch(path.stem)
            if match:
                object_ids.add(match.group(0))
    return object_ids, scanned_files


def pairwise_distance_to(features, index, chosen):
    return min(float(np.linalg.norm(features[index] - features[j])) for j in chosen)


def select_test_indices(objects, features, available, count):
    """Choose representative but diverse held-out objects without policy data."""
    if len(available) < count:
        raise RuntimeError("not enough historically unseen objects for test split")
    center = np.median(features, axis=0)
    center_distance = np.linalg.norm(features - center, axis=1)
    cutoff = float(np.percentile(center_distance[available], 95))
    non_extreme = [i for i in available if center_distance[i] <= cutoff]
    first = min(non_extreme, key=lambda i: (
        center_distance[i], objects[i]["object_id"]))
    chosen = [first]
    while len(chosen) < count:
        remaining = [i for i in non_extreme if i not in chosen]
        next_index = max(remaining, key=lambda i: (
            pairwise_distance_to(features, i, chosen),
            -center_distance[i],
            objects[i]["object_id"],
        ))
        chosen.append(next_index)
    return chosen


def extend_training_indices(objects, features, base_indices, available, target_count):
    """Greedily cover geometry space while preserving the historical base set."""
    chosen = list(base_indices)
    trajectory_counts = np.asarray(
        [item["trajectory_count"] for item in objects], dtype=np.float64)
    while len(chosen) < target_count:
        remaining = [i for i in available if i not in chosen]
        if not remaining:
            raise RuntimeError("not enough eligible objects to extend training split")
        next_index = max(remaining, key=lambda i: (
            pairwise_distance_to(features, i, chosen),
            trajectory_counts[i],
            objects[i]["object_id"],
        ))
        chosen.append(next_index)
    return chosen


def select_backup_indices(objects, features, available, chosen, count):
    backups = []
    anchor = list(chosen)
    while len(backups) < count:
        remaining = [i for i in available if i not in anchor]
        if not remaining:
            raise RuntimeError("not enough eligible objects for backups")
        next_index = max(remaining, key=lambda i: (
            pairwise_distance_to(features, i, anchor),
            objects[i]["trajectory_count"],
            objects[i]["object_id"],
        ))
        backups.append(next_index)
        anchor.append(next_index)
    return backups


def write_csv(path, rows):
    fieldnames = [
        "category", "split", "first_train_size", "rank", "object_id",
        "trajectory_count", "geometry_distance_to_category_center",
        "geometry_proxy", "physical_longest_extent", "physical_bbox_volume",
        "bbox_aspect_ratio", "convex_piece_count", "vertex_count",
        "rotation_angle_std", "final_action_dispersion",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fieldnames} for row in rows)


def main():
    args = parse_args()
    train_sizes = sorted(set(args.train_sizes))
    if not train_sizes or train_sizes[0] < 1:
        raise ValueError("--train-sizes must contain positive integers")
    if min(args.test_per_category, args.backups_per_category) < 1:
        raise ValueError("test and backup counts must be positive")

    output_json = Path(args.output_json).expanduser().resolve()
    output_csv = Path(args.output_csv).expanduser().resolve()
    if output_json.exists() or output_csv.exists():
        raise FileExistsError(
            "Refusing to replace an existing split; choose new output paths")

    base_path = Path(args.base_manifest).expanduser().resolve()
    with base_path.open(encoding="utf-8") as handle:
        base = json.load(handle)
    categories = list(base["criteria"]["categories"])
    base_count = len(base["categories"][categories[0]]["train"])
    if train_sizes[0] != base_count:
        raise ValueError(
            "smallest train size must equal base split size ({})".format(base_count))

    historical_ids, scanned_files = historical_object_ids()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    mesh_root = Path(args.mesh_root).expanduser().resolve()
    manifest = {
        "selection_status": "policy_blind_candidates_not_yet_replayed",
        "warning": (
            "Do not evaluate learned policies on the test objects until the "
            "preprocessing check has frozen this split."),
        "selection_rule": (
            "Keep the original 4 training objects per category; choose 5 fresh "
            "test objects and extend training to 10 then 20 objects using only "
            "raw-trajectory integrity and geometry diversity. No policy score "
            "is used for selection."),
        "base_manifest": str(base_path),
        "criteria": {
            "categories": categories,
            "train_sizes": train_sizes,
            "train_per_category": train_sizes[-1],
            "test_per_category": args.test_per_category,
            "backups_per_category": args.backups_per_category,
            "min_trajectories": args.min_trajectories,
            "min_rotation_std": args.min_rotation_std,
            "min_action_dispersion": args.min_action_dispersion,
            "feature_keys": list(FEATURE_KEYS),
        },
        "history_audit": {
            "scanned_text_file_count": scanned_files,
            "excluded_object_count": len(historical_ids),
            "excluded_object_ids": sorted(historical_ids),
        },
        "categories": {},
        "objects": {},
        "rejected_counts": {},
    }
    rows = []

    for category in categories:
        # The historical 4-object baseline used a 20-trajectory minimum.  Keep
        # those exact objects for comparability, while requiring every newly
        # selected object to satisfy the stricter threshold below.
        collection_minimum = min(
            int(base["criteria"].get("min_trajectories", 20)),
            args.min_trajectories)
        objects, rejected = collect_category(
            category, dataset_root, mesh_root, collection_minimum,
            args.min_rotation_std, args.min_action_dispersion)
        by_id = {item["object_id"]: index for index, item in enumerate(objects)}
        base_ids = list(base["categories"][category]["train"])
        missing = [object_id for object_id in base_ids if object_id not in by_id]
        if missing:
            raise RuntimeError(
                "base objects are not eligible in {}: {}".format(category, missing))

        features = normalized_features(objects)
        center = np.median(features, axis=0)
        center_distances = np.linalg.norm(features - center, axis=1)
        low, high = np.percentile(center_distances, [33, 66])
        base_indices = [by_id[object_id] for object_id in base_ids]
        fresh = [
            index for index, item in enumerate(objects)
            if (item["object_id"] not in historical_ids
                and item["trajectory_count"] >= args.min_trajectories)
        ]
        test_indices = select_test_indices(
            objects, features, fresh, args.test_per_category)
        test_set = set(test_indices)
        training_pool = [i for i in fresh if i not in test_set]
        train_indices = extend_training_indices(
            objects, features, base_indices, training_pool, train_sizes[-1])
        train_set = set(train_indices)
        backup_pool = [i for i in fresh if i not in test_set and i not in train_set]
        backup_indices = select_backup_indices(
            objects, features, backup_pool, train_indices + test_indices,
            args.backups_per_category)

        nested = {
            str(size): [objects[i]["object_id"] for i in train_indices[:size]]
            for size in train_sizes
        }
        train_ids = nested[str(train_sizes[-1])]
        test_ids = [objects[i]["object_id"] for i in test_indices]
        backup_ids = [objects[i]["object_id"] for i in backup_indices]
        if not set(nested[str(train_sizes[0])]) == set(base_ids):
            raise RuntimeError("base training set changed for {}".format(category))
        for small, large in zip(train_sizes, train_sizes[1:]):
            if not set(nested[str(small)]).issubset(nested[str(large)]):
                raise RuntimeError("training sets are not nested")
        if (set(train_ids) & set(test_ids) or set(train_ids) & set(backup_ids)
                or set(test_ids) & set(backup_ids)):
            raise RuntimeError("split overlap in {}".format(category))

        manifest["categories"][category] = {
            "eligible_count": len(objects),
            "fresh_eligible_count": len(fresh),
            "train_nested": nested,
            # Compatibility with staging/preprocessing tools.
            "train": train_ids,
            "test": test_ids,
            "backups": backup_ids,
        }
        manifest["rejected_counts"][category] = len(rejected)

        index_groups = (
            ("train", train_indices),
            ("test", test_indices),
            ("backups", backup_indices),
        )
        for split_name, indices in index_groups:
            for rank, index in enumerate(indices, 1):
                item = dict(objects[index])
                item["geometry_distance_to_category_center"] = float(
                    center_distances[index])
                item["geometry_proxy"] = (
                    "typical" if center_distances[index] <= low else
                    "medium" if center_distances[index] <= high else "unusual")
                item["split"] = split_name
                item["rank"] = rank
                if split_name == "train":
                    item["first_train_size"] = next(
                        size for size in train_sizes if rank <= size)
                else:
                    item["first_train_size"] = ""
                manifest["objects"][item["object_id"]] = item
                rows.append(item)

        print(
            "{}: eligible={}, fresh={}, train={}, test={}, backups={}".format(
                category, len(objects), len(fresh), len(train_ids),
                len(test_ids), len(backup_ids)))

    all_train = {
        object_id for split in manifest["categories"].values()
        for object_id in split["train"]}
    all_test = {
        object_id for split in manifest["categories"].values()
        for object_id in split["test"]}
    if all_train & all_test:
        raise RuntimeError("global train/test overlap")
    if all_test & historical_ids:
        raise RuntimeError("historically exposed object leaked into fresh test")

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    write_csv(output_csv, rows)
    print("Wrote {}".format(output_json))
    print("Wrote {}".format(output_csv))
    print("SCALED_SPLIT_RESULT=CANDIDATES_READY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
