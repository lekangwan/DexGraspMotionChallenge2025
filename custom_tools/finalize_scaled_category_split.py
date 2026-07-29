"""Freeze the nested 4/10/20 split after policy-blind replay preflight."""

import argparse
import copy
import csv
import json
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate-manifest", default=str(
            REPO_ROOT / "custom_tools/configs/scaled_category_split_candidates_v1.json"))
    parser.add_argument(
        "--candidate-summary", default=str(
            REPO_ROOT / "custom_tools/results/scaled_category_preprocess_summary_v1.json"))
    parser.add_argument(
        "--replacement-manifest", default=str(
            REPO_ROOT / "custom_tools/configs/scaled_replacement_candidates_v1.json"))
    parser.add_argument(
        "--replacement-summary", default=str(
            REPO_ROOT / "custom_tools/results/scaled_replacement_preprocess_summary_v1.json"))
    parser.add_argument(
        "--base-manifest", default=str(
            REPO_ROOT / "custom_tools/configs/object_split_final.json"))
    parser.add_argument(
        "--base-preprocessed-root", default=str(
            REPO_ROOT / "dexgrasp/dataset/object_split_candidates_preprocessed"))
    parser.add_argument(
        "--expanded-preprocessed-root", default=str(
            REPO_ROOT / "dexgrasp/dataset/scaled_category_candidates_v1_preprocessed"))
    parser.add_argument(
        "--output-json", default=str(
            REPO_ROOT / "custom_tools/configs/scaled_category_split_final_v1.json"))
    parser.add_argument(
        "--output-csv", default=str(
            REPO_ROOT / "custom_tools/configs/scaled_category_split_final_v1.csv"))
    return parser.parse_args()


def load(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def preprocessed_metadata(path):
    data = np.load(str(path), allow_pickle=True).item()
    if data.get("selection_metric") != "official_final":
        raise RuntimeError("non-official preprocessing source: {}".format(path))
    return {
        "raw_count": int(len(data["maximum_lift"])),
        "retained_count": int(len(data["grasp_seqs"])),
        "retention_rate": (
            len(data["grasp_seqs"]) / len(data["maximum_lift"])
            if len(data["maximum_lift"]) else 0.0),
    }


def main():
    args = parse_args()
    output_json = Path(args.output_json).expanduser().resolve()
    output_csv = Path(args.output_csv).expanduser().resolve()
    if output_json.exists() or output_csv.exists():
        raise FileExistsError("refusing to overwrite the frozen split")

    source = load(args.candidate_manifest)
    source_summary = load(args.candidate_summary)
    replacements = load(args.replacement_manifest)
    replacement_summary = load(args.replacement_summary)
    base = load(args.base_manifest)
    if source_summary["present_object_count"] != source_summary["expected_object_count"]:
        raise RuntimeError("expanded candidate preprocessing is incomplete")
    if replacement_summary["status"] != "COMPLETE":
        raise RuntimeError("replacement preprocessing is incomplete")
    minimum = int(source_summary["min_retained_for_bc"])
    passing_replacements = {
        row["object_id"] for row in replacement_summary["objects"]
        if row["status"] == "PASS"}
    base_ids = {
        object_id for split in base["categories"].values()
        for object_id in split["train"]}
    base_root = Path(args.base_preprocessed_root).expanduser().resolve()
    expanded_root = Path(args.expanded_preprocessed_root).expanduser().resolve()

    final_categories = copy.deepcopy(source["categories"])
    used_replacements = set()
    replacement_records = []
    for failed in source_summary["failed_objects"]:
        failed_id = failed["object_id"]
        category = failed["category"]
        candidates = [
            object_id for object_id in replacements["categories"][category]["backups"]
            if object_id in passing_replacements and object_id not in used_replacements]
        if not candidates:
            raise RuntimeError("no passing replacement remains for {}".format(failed_id))
        replacement_id = min(candidates, key=lambda object_id: (
            replacements["objects"][object_id][
                "geometry_distances_to_failed_objects"][failed_id],
            object_id,
        ))
        used_replacements.add(replacement_id)
        category_split = final_categories[category]
        replaced_locations = []
        for split_name in ("train", "test"):
            if failed_id in category_split[split_name]:
                index = category_split[split_name].index(failed_id)
                category_split[split_name][index] = replacement_id
                replaced_locations.append(split_name)
        for size, object_ids in category_split["train_nested"].items():
            if failed_id in object_ids:
                object_ids[object_ids.index(failed_id)] = replacement_id
                replaced_locations.append("train_nested_{}".format(size))
        if not replaced_locations:
            raise RuntimeError("failed object was not found in final split")
        replacement_records.append({
            "category": category,
            "split": failed["split"],
            "failed_object_id": failed_id,
            "failed_retained_count": failed["retained_count"],
            "replacement_object_id": replacement_id,
            "geometry_distance": replacements["objects"][replacement_id][
                "geometry_distances_to_failed_objects"][failed_id],
            "replaced_locations": replaced_locations,
        })

    final_ids = {
        object_id for split in final_categories.values()
        for name in ("train", "test") for object_id in split[name]}
    if len(final_ids) != 100:
        raise RuntimeError("final split does not contain 100 unique objects")
    final_objects = {}
    rows = []
    nested_totals = {
        str(size): {"by_category": {}, "total": 0}
        for size in source["criteria"]["train_sizes"]}
    for category, split in final_categories.items():
        if not (set(split["train_nested"]["4"])
                <= set(split["train_nested"]["10"])
                <= set(split["train_nested"]["20"])):
            raise RuntimeError("nested split was broken for {}".format(category))
        if set(split["train"]) & set(split["test"]):
            raise RuntimeError("train/test overlap for {}".format(category))
        for split_name in ("train", "test"):
            for rank, object_id in enumerate(split[split_name], 1):
                if object_id in source["objects"]:
                    metadata = copy.deepcopy(source["objects"][object_id])
                else:
                    metadata = copy.deepcopy(replacements["objects"][object_id])
                data_path = (
                    base_root / (object_id + ".npy") if object_id in base_ids
                    else expanded_root / (object_id + ".npy"))
                if not data_path.is_file():
                    raise FileNotFoundError(data_path)
                data_metrics = preprocessed_metadata(data_path)
                if data_metrics["retained_count"] < minimum:
                    raise RuntimeError("final object is below minimum: {}".format(object_id))
                metadata.update(data_metrics)
                metadata.update({
                    "split": split_name,
                    "rank": rank,
                    "preprocessed_source": str(data_path),
                    "uses_frozen_base_preprocessing": object_id in base_ids,
                })
                final_objects[object_id] = metadata
                rows.append({
                    "category": category, "split": split_name, "rank": rank,
                    "object_id": object_id,
                    "retained_count": data_metrics["retained_count"],
                    "retention_rate": data_metrics["retention_rate"],
                    "uses_frozen_base_preprocessing": object_id in base_ids,
                })
        for size in source["criteria"]["train_sizes"]:
            total = sum(
                final_objects[object_id]["retained_count"]
                for object_id in split["train_nested"][str(size)])
            nested_totals[str(size)]["by_category"][category] = total
            nested_totals[str(size)]["total"] += total

    final = {
        "status": "frozen_preflight_passed",
        "selection_rule": (
            "Keep the original nested ordering. Replace each object below 12 "
            "official-final trajectories with the nearest same-category candidate "
            "that passes 12, using geometry only. Original 4-object/category data "
            "reuse the previously frozen preprocessing for exact comparability."),
        "min_retained_for_bc": minimum,
        "criteria": source["criteria"],
        "categories": final_categories,
        "objects": final_objects,
        "replacements": replacement_records,
        "counts": {"train": 80, "test": 20, "replacements": len(replacement_records)},
        "nested_retained_trajectory_totals": nested_totals,
        "source_files": {
            "candidate_manifest": str(Path(args.candidate_manifest).resolve()),
            "candidate_summary": str(Path(args.candidate_summary).resolve()),
            "replacement_manifest": str(Path(args.replacement_manifest).resolve()),
            "replacement_summary": str(Path(args.replacement_summary).resolve()),
            "base_manifest": str(Path(args.base_manifest).resolve()),
        },
        "leakage_note": (
            "Replacement selection used official demonstration replay retention "
            "and geometry only; no learned-policy score was read."),
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(final, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for record in replacement_records:
        print("[REPLACED] {failed_object_id} -> {replacement_object_id} "
              "(geometry_distance={geometry_distance:.4f})".format(**record))
    for size, values in nested_totals.items():
        print("train{} retained: total={}, by_category={}".format(
            size, values["total"], values["by_category"]))
    print("FINAL_SCALED_SPLIT_RESULT=FROZEN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
