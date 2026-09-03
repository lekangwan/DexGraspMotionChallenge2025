#!/usr/bin/env python3
"""核对最终策略数据规模、成功标签、几何sidecar和测试物体隔离。"""

import argparse
import json
from pathlib import Path

import numpy as np


HANDS = ("linker", "xhand", "wuji")
SPLITS = ("train", "valid", "test")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--release-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    release = json.loads(args.release_lock.read_text(encoding="utf-8"))
    result = {
        "schema_version": 1,
        "retargeting_release_id": release["release_id"],
        "retargeting_release_sha256": release["release_sha256"],
        "quality_rule": "train_valid_filtered_by_training_eligible; test_unfiltered",
        "hands": {},
    }
    for hand in HANDS:
        directory = args.data_root / hand
        mappings = json.loads((directory / "mappings.json").read_text(encoding="utf-8"))
        object_name = {int(value): name for name, value in mappings["object_to_id"].items()}
        expected = release["hands"][hand]["metrics"]["per_policy_split"]
        objects = {}
        hand_result = {"splits": {}}
        for split in SPLITS:
            with np.load(directory / f"{split}.npz", allow_pickle=False) as data:
                trajectory_ids = np.unique(data["trajectory_id"])
                if len(data["actions"]) != 240 * len(trajectory_ids):
                    raise ValueError(f"{hand}/{split}不是每轨迹240步")
                labels = data["expert_replay_success"].reshape(-1, 240)
                if not np.all(labels == labels[:, :1]):
                    raise ValueError(f"{hand}/{split}同一轨迹标签不一致")
                positive = int(labels[:, 0].sum())
                objects[split] = {object_name[int(value)] for value in np.unique(data["object_id"])}
                categories = int(len(np.unique(data["category_id"])))
            with np.load(directory / f"geometry_{split}.npz", allow_pickle=False) as geometry:
                if len(geometry["trajectory_id"]) != len(trajectory_ids):
                    raise ValueError(f"{hand}/{split}几何sidecar数量不一致")
                if geometry["object_points"].shape[1:] != (128, 3):
                    raise ValueError(f"{hand}/{split}物体点云不是128x3")
            expected_count = (
                int(expected[split]["training_eligible_count"])
                if split in {"train", "valid"}
                else int(expected[split]["trajectory_count"])
            )
            expected_positive = int(expected[split]["training_eligible_count"])
            if len(trajectory_ids) != expected_count or positive != expected_positive:
                raise ValueError(f"{hand}/{split}数量与最终审计不一致")
            hand_result["splits"][split] = {
                "trajectories": len(trajectory_ids),
                "positive_labels": positive,
                "objects": len(objects[split]),
                "categories": categories,
            }
        leakage = (objects["train"] | objects["valid"]) & objects["test"]
        if leakage:
            raise ValueError(f"{hand}测试物体泄漏: {sorted(leakage)}")
        hand_result["test_object_leakage_count"] = 0
        result["hands"][hand] = hand_result
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"FINAL_DATA_AUDIT={args.output}")


if __name__ == "__main__":
    main()
