"""Validate and merge Task-ID online-imitation rounds 1 and 2."""

from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "custom_tools/data/distillation"
ROUND_PATHS = (
    DATA_ROOT / "online_taskid_scaled20_r1_train4.npz",
    DATA_ROOT / "online_taskid_scaled20_r2_train4_offset4.npz",
)
OUTPUT = DATA_ROOT / "online_taskid_scaled20_r1_r2_aggregated.npz"
ARRAY_KEYS = (
    "observations",
    "teacher_actions",
    "student_actions",
    "category_indices",
    "object_indices",
    "trajectory_indices",
    "frame_indices",
)


def load_round(path):
    data = np.load(path, allow_pickle=False)
    expected = set(ARRAY_KEYS) | {"object_ids"}
    if set(data.files) != expected:
        raise RuntimeError("Unexpected keys in {}".format(path))
    count = len(data["observations"])
    if data["observations"].shape != (count, 2460):
        raise RuntimeError("Invalid observations in {}".format(path))
    if data["teacher_actions"].shape != (count, 28):
        raise RuntimeError("Invalid teacher actions in {}".format(path))
    if data["student_actions"].shape != (count, 28):
        raise RuntimeError("Invalid student actions in {}".format(path))
    if not all(
        np.isfinite(data[key]).all()
        for key in ("observations", "teacher_actions", "student_actions")
    ):
        raise RuntimeError("Non-finite values in {}".format(path))
    return data


def main():
    if OUTPUT.exists() and OUTPUT.with_suffix(".yaml").exists():
        print("[REUSE] {}".format(OUTPUT))
        return
    if OUTPUT.exists() != OUTPUT.with_suffix(".yaml").exists():
        raise RuntimeError("Aggregated output is incomplete; inspect it first")
    missing = [str(path) for path in ROUND_PATHS if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing online rounds: {}".format(missing))

    rounds = [load_round(path) for path in ROUND_PATHS]
    try:
        object_ids = rounds[0]["object_ids"]
        if not np.array_equal(object_ids, rounds[1]["object_ids"]):
            raise RuntimeError("Online rounds use different object ordering")
        trajectory_pairs = []
        for data in rounds:
            pairs = set(
                zip(
                    data["object_indices"].tolist(),
                    data["trajectory_indices"].tolist(),
                )
            )
            trajectory_pairs.append(pairs)
        overlap = trajectory_pairs[0] & trajectory_pairs[1]
        if overlap:
            raise RuntimeError(
                "Online rounds overlap on {} object/trajectory pairs".format(
                    len(overlap)
                )
            )

        merged = {
            key: np.concatenate([data[key] for data in rounds], axis=0)
            for key in ARRAY_KEYS
        }
        merged["object_ids"] = object_ids.copy()
        np.savez_compressed(OUTPUT, **merged)

        category_counts = np.bincount(
            merged["category_indices"].astype(np.int64), minlength=4
        )
        summary = {
            "method": "DAgger dataset aggregation",
            "training_split_only": True,
            "formal_final_holdout_used": False,
            "source_rounds": [str(path) for path in ROUND_PATHS],
            "round_sample_counts": [
                int(len(data["observations"])) for data in rounds
            ],
            "sample_count": int(len(merged["observations"])),
            "object_count": int(len(object_ids)),
            "trajectory_pair_counts": [
                len(pairs) for pairs in trajectory_pairs
            ],
            "trajectory_pair_overlap": 0,
            "category_sample_counts": {
                category: int(category_counts[index])
                for index, category in enumerate(
                    ("bottle", "mug", "bowl", "camera")
                )
            },
        }
        with OUTPUT.with_suffix(".yaml").open(
            "w", encoding="utf-8"
        ) as handle:
            yaml.safe_dump(
                summary, handle, allow_unicode=True, sort_keys=False
            )
        print(
            "TASKID_ONLINE_ROUNDS_MERGED samples={} overlap=0".format(
                len(merged["observations"])
            )
        )
    finally:
        for data in rounds:
            data.close()


if __name__ == "__main__":
    main()
