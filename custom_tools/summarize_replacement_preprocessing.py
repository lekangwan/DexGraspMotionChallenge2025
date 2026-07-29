"""Summarize official-final replay retention for replacement candidates."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", default=str(
            REPO_ROOT / "custom_tools/configs/scaled_replacement_candidates_v1.json"))
    parser.add_argument(
        "--preprocessed-root", default=str(
            REPO_ROOT / "dexgrasp/dataset/scaled_category_candidates_v1_preprocessed"))
    parser.add_argument("--min-retained", type=int, default=12)
    parser.add_argument(
        "--output-json", default=str(
            REPO_ROOT / "custom_tools/results/scaled_replacement_preprocess_summary_v1.json"))
    parser.add_argument(
        "--output-csv", default=str(
            REPO_ROOT / "custom_tools/results/scaled_replacement_preprocess_summary_v1.csv"))
    return parser.parse_args()


def main():
    args = parse_args()
    with Path(args.manifest).open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    root = Path(args.preprocessed_root).expanduser().resolve()
    rows = []
    for object_id, metadata in manifest["objects"].items():
        path = root / (object_id + ".npy")
        if not path.is_file():
            row = {
                "category": metadata["category"], "object_id": object_id,
                "target_failed_object_id": metadata["target_failed_object_id"],
                "target_geometry_distance": metadata["target_geometry_distance"],
                "raw_count": metadata["trajectory_count"], "retained_count": 0,
                "retention_rate": 0.0, "status": "MISSING"}
        else:
            data = np.load(str(path), allow_pickle=True).item()
            if data.get("selection_metric") != "official_final":
                raise RuntimeError("non-official selection for {}".format(object_id))
            raw_count = int(len(data["maximum_lift"]))
            retained_count = int(len(data["grasp_seqs"]))
            row = {
                "category": metadata["category"], "object_id": object_id,
                "target_failed_object_id": metadata["target_failed_object_id"],
                "target_geometry_distance": metadata["target_geometry_distance"],
                "raw_count": raw_count, "retained_count": retained_count,
                "retention_rate": retained_count / raw_count if raw_count else 0.0,
                "status": "PASS" if retained_count >= args.min_retained else "FAIL"}
        rows.append(row)

    missing = [row for row in rows if row["status"] == "MISSING"]
    passing = [row for row in rows if row["status"] == "PASS"]
    summary = {
        "status": "COMPLETE" if not missing else "INCOMPLETE",
        "manifest": str(Path(args.manifest).resolve()),
        "preprocessed_root": str(root),
        "min_retained": args.min_retained,
        "expected_count": len(rows), "present_count": len(rows) - len(missing),
        "pass_count": len(passing), "fail_count": len(rows) - len(passing) - len(missing),
        "objects": rows,
    }
    output_json = Path(args.output_json).expanduser().resolve()
    output_csv = Path(args.output_csv).expanduser().resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("Replacement candidates: present={}/{}, pass={}, fail={}".format(
        summary["present_count"], summary["expected_count"],
        summary["pass_count"], summary["fail_count"]))
    print("REPLACEMENT_PREPROCESS_RESULT={}".format(summary["status"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
