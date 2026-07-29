"""Select geometry-nearest, policy-blind replacements for failed preflight objects."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from custom_tools.select_object_split import collect_category, normalized_features


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate-manifest", default=str(
            REPO_ROOT / "custom_tools/configs/scaled_category_split_candidates_v1.json"))
    parser.add_argument(
        "--preprocess-summary", default=str(
            REPO_ROOT / "custom_tools/results/scaled_category_preprocess_summary_v1.json"))
    parser.add_argument(
        "--dataset-root", default=str(REPO_ROOT / "external_data/dataset"))
    parser.add_argument(
        "--mesh-root", default=str(REPO_ROOT / "external_data/meshdata"))
    parser.add_argument("--extra-per-category", type=int, default=4)
    parser.add_argument(
        "--output-json", default=str(
            REPO_ROOT / "custom_tools/configs/scaled_replacement_candidates_v1.json"))
    parser.add_argument(
        "--output-csv", default=str(
            REPO_ROOT / "custom_tools/configs/scaled_replacement_candidates_v1.csv"))
    return parser.parse_args()


def main():
    args = parse_args()
    if args.extra_per_category < 1:
        raise ValueError("--extra-per-category must be positive")
    output_json = Path(args.output_json).expanduser().resolve()
    output_csv = Path(args.output_csv).expanduser().resolve()
    if output_json.exists() or output_csv.exists():
        raise FileExistsError("refusing to overwrite a replacement candidate manifest")

    with Path(args.candidate_manifest).open(encoding="utf-8") as handle:
        source = json.load(handle)
    with Path(args.preprocess_summary).open(encoding="utf-8") as handle:
        summary = json.load(handle)
    if summary["present_object_count"] != summary["expected_object_count"]:
        raise RuntimeError("source preprocessing is incomplete")
    failures = summary["failed_objects"]
    if not failures:
        raise RuntimeError("there are no failed objects to replace")

    historical = set(source["history_audit"]["excluded_object_ids"])
    current_train_test = {
        object_id for split in source["categories"].values()
        for name in ("train", "test") for object_id in split[name]}
    failure_by_category = {}
    for failure in failures:
        failure_by_category.setdefault(failure["category"], []).append(failure)

    manifest = {
        "selection_status": "policy_blind_replacement_candidates_not_yet_replayed",
        "selection_rule": (
            "For each object with fewer than 12 official-final trajectories, "
            "cycle through failed objects and choose the nearest unused object "
            "in normalized raw-trajectory/geometry feature space. Select four "
            "extra candidates per affected category. No learned-policy result "
            "is used."),
        "source_candidate_manifest": str(Path(args.candidate_manifest).resolve()),
        "source_preprocess_summary": str(Path(args.preprocess_summary).resolve()),
        "criteria": {
            "categories": list(failure_by_category),
            "extra_per_category": args.extra_per_category,
            "min_trajectories": source["criteria"]["min_trajectories"],
        },
        "categories": {},
        "objects": {},
        "failed_objects": failures,
    }
    rows = []
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    mesh_root = Path(args.mesh_root).expanduser().resolve()

    for category, category_failures in failure_by_category.items():
        objects, _ = collect_category(
            category, dataset_root, mesh_root,
            source["criteria"]["min_trajectories"],
            source["criteria"]["min_rotation_std"],
            source["criteria"]["min_action_dispersion"])
        by_id = {item["object_id"]: index for index, item in enumerate(objects)}
        features = normalized_features(objects)
        failed_indices = []
        for failure in category_failures:
            if failure["object_id"] not in by_id:
                raise RuntimeError("failed object is absent from eligible pool")
            failed_indices.append(by_id[failure["object_id"]])
        available = [
            index for index, item in enumerate(objects)
            if item["object_id"] not in historical
            and item["object_id"] not in current_train_test]
        required = len(category_failures) + args.extra_per_category
        if len(available) < required:
            raise RuntimeError("not enough replacement candidates for {}".format(category))

        selected = []
        for selection_round in range(required):
            target_position = selection_round % len(category_failures)
            target_index = failed_indices[target_position]
            remaining = [index for index in available if index not in selected]
            candidate_index = min(remaining, key=lambda index: (
                float(np.linalg.norm(features[index] - features[target_index])),
                -objects[index]["trajectory_count"],
                objects[index]["object_id"],
            ))
            selected.append(candidate_index)
            item = dict(objects[candidate_index])
            distances = {
                failure["object_id"]: float(np.linalg.norm(
                    features[candidate_index] - features[failed_index]))
                for failure, failed_index in zip(category_failures, failed_indices)
            }
            item.update({
                "split": "backups",
                "selection_rank": selection_round + 1,
                "target_failed_object_id": category_failures[target_position]["object_id"],
                "target_geometry_distance": distances[
                    category_failures[target_position]["object_id"]],
                "geometry_distances_to_failed_objects": distances,
            })
            manifest["objects"][item["object_id"]] = item
            rows.append(item)

        selected_ids = [objects[index]["object_id"] for index in selected]
        manifest["categories"][category] = {
            "train": [], "test": [], "backups": selected_ids,
            "failure_count": len(category_failures),
            "candidate_count": len(selected_ids),
        }
        print("{}: failures={}, replacement_candidates={}".format(
            category, len(category_failures), len(selected_ids)))

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    fields = (
        "category", "selection_rank", "object_id", "target_failed_object_id",
        "target_geometry_distance", "trajectory_count", "physical_longest_extent",
        "physical_bbox_volume", "bbox_aspect_ratio", "convex_piece_count",
        "vertex_count", "rotation_angle_std", "final_action_dispersion")
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in rows)
    print("Wrote {}".format(output_json))
    print("Wrote {}".format(output_csv))
    print("REPLACEMENT_SELECTION_RESULT=READY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
