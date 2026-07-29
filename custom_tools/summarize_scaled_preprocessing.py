"""Audit preprocessing for the nested category-expert object split."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default=str(REPO_ROOT / "custom_tools" / "configs" /
                    "scaled_category_split_candidates_v1.json"))
    parser.add_argument(
        "--preprocessed-root",
        default=str(REPO_ROOT / "dexgrasp" / "dataset" /
                    "scaled_category_candidates_v1_preprocessed"))
    parser.add_argument(
        "--required-split", action="append", choices=("train", "test", "backups"),
        default=[])
    parser.add_argument("--min-retained", type=int, default=12)
    parser.add_argument(
        "--output-json",
        default=str(REPO_ROOT / "custom_tools" / "results" /
                    "scaled_category_preprocess_summary_v1.json"))
    parser.add_argument(
        "--output-csv",
        default=str(REPO_ROOT / "custom_tools" / "results" /
                    "scaled_category_preprocess_summary_v1.csv"))
    return parser.parse_args()


def main():
    args = parse_args()
    if args.min_retained < 2:
        raise ValueError("--min-retained must be at least 2")
    manifest_path = Path(args.manifest).expanduser().resolve()
    data_root = Path(args.preprocessed_root).expanduser().resolve()
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("selection_status") != "policy_blind_candidates_not_yet_replayed":
        raise RuntimeError("unexpected candidate manifest status")

    required_splits = args.required_split or ["train", "test"]
    rows = []
    missing = []
    by_id = {}
    for category in manifest["criteria"]["categories"]:
        category_split = manifest["categories"][category]
        for split_name in required_splits:
            for object_id in category_split[split_name]:
                path = data_root / (object_id + ".npy")
                metadata = manifest["objects"][object_id]
                if not path.is_file():
                    missing.append(object_id)
                    row = {
                        "category": category,
                        "split": split_name,
                        "first_train_size": metadata.get("first_train_size", ""),
                        "geometry_proxy": metadata["geometry_proxy"],
                        "object_id": object_id,
                        "raw_count": metadata["trajectory_count"],
                        "retained_count": 0,
                        "retention_rate": 0.0,
                        "official_final_status": "MISSING",
                    }
                    rows.append(row)
                    by_id[object_id] = row
                    continue

                data = np.load(str(path), allow_pickle=True).item()
                if data.get("selection_metric") != "official_final":
                    raise RuntimeError(
                        "{} was not preprocessed with official_final".format(object_id))
                raw_count = int(len(data["maximum_lift"]))
                retained_count = int(len(data["grasp_seqs"]))
                row = {
                    "category": category,
                    "split": split_name,
                    "first_train_size": metadata.get("first_train_size", ""),
                    "geometry_proxy": metadata["geometry_proxy"],
                    "object_id": object_id,
                    "raw_count": raw_count,
                    "retained_count": retained_count,
                    "retention_rate": retained_count / raw_count if raw_count else 0.0,
                    "official_final_status": (
                        "PASS" if retained_count >= args.min_retained else "REPLACE"),
                }
                rows.append(row)
                by_id[object_id] = row

    failures = [row for row in rows if row["official_final_status"] != "PASS"]
    nested_totals = {}
    for size in manifest["criteria"]["train_sizes"]:
        category_totals = {}
        for category in manifest["criteria"]["categories"]:
            object_ids = manifest["categories"][category]["train_nested"][str(size)]
            category_totals[category] = sum(
                by_id[object_id]["retained_count"] for object_id in object_ids)
        nested_totals[str(size)] = {
            "by_category": category_totals,
            "total": sum(category_totals.values()),
        }

    summary = {
        "status": "PASS" if not failures else "NEEDS_REPLACEMENTS",
        "manifest": str(manifest_path),
        "preprocessed_root": str(data_root),
        "required_splits": required_splits,
        "min_retained_for_bc": args.min_retained,
        "expected_object_count": len(rows),
        "present_object_count": len(rows) - len(missing),
        "pass_count": len(rows) - len(failures),
        "failure_count": len(failures),
        "missing_object_ids": missing,
        "failed_objects": failures,
        "nested_retained_trajectory_totals": nested_totals,
        "objects": rows,
        "metric_note": (
            "Retention uses the unmodified official final-frame success flag; "
            "no learned policy result participates in this audit."),
    }
    output_json = Path(args.output_json).expanduser().resolve()
    output_csv = Path(args.output_csv).expanduser().resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("[{}] present {}/{} objects; passing minimum {}: {}".format(
        summary["status"], summary["present_object_count"],
        summary["expected_object_count"], args.min_retained,
        summary["pass_count"]))
    for size, values in nested_totals.items():
        print("train{} retained: total={}, by_category={}".format(
            size, values["total"], values["by_category"]))
    for row in failures:
        print("[{}] {}/{} {}: retained={}".format(
            row["official_final_status"], row["category"], row["split"],
            row["object_id"], row["retained_count"]))
    print("Wrote {}".format(output_json))
    print("Wrote {}".format(output_csv))
    print("SCALED_PREPROCESS_RESULT={}".format(summary["status"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
