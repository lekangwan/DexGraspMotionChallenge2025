"""Create category-aligned demonstration labels for Temporal3 experts."""

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
DEXGRASP_ROOT = ROOT / "dexgrasp"
for path in (str(ROOT), str(DEXGRASP_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from custom_tools.graspm3_dexrep_dataset import (  # noqa: E402
    GraspM3DexRepDataset,
)


CATEGORIES = ("bottle", "mug", "bowl", "camera")
CONFIG = (
    ROOT / "custom_tools/configs/"
    / "category_temporal3_demo_scaled20_v1.yaml")
MANIFEST = (
    ROOT / "custom_tools/configs/scaled_category_split_final_v1.json")
SUMMARY = (
    ROOT / "custom_tools/results/scaled_bc20_dataset_summary_v1.json")
OUTPUT_ROOT = ROOT / "custom_tools/data/distillation/category_temporal_demo"


def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--category", choices=CATEGORIES, action="append", default=[])
    return parser.parse_args()


def load_json(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def expected_counts(summary, category):
    rows = [
        row for row in summary["objects"]
        if row["category"] == category]
    return (
        sum(int(row["train_count"]) for row in rows),
        sum(int(row["valid_count"]) for row in rows),
    )


def output_path(category):
    return OUTPUT_ROOT / "{}_demo_train.npz".format(category)


def verify(path, category, object_ids, trajectory_count):
    data = np.load(path, allow_pickle=False)
    expected = trajectory_count * 70
    if data["teacher_actions"].shape != (expected, 28):
        raise RuntimeError(
            "{} has unexpected shape {}".format(
                path, data["teacher_actions"].shape))
    if data["category"].tolist() != [category]:
        raise RuntimeError("{} category metadata mismatch".format(path))
    if data["object_ids"].tolist() != object_ids:
        raise RuntimeError("{} object order mismatch".format(path))


def prepare(category, manifest, summary):
    object_ids = manifest["categories"][category]["train_nested"]["20"]
    train_count, _ = expected_counts(summary, category)
    path = output_path(category)
    if path.is_file():
        verify(path, category, object_ids, train_count)
        print("[REUSE] {}".format(path), flush=True)
        return
    args = OmegaConf.load(str(CONFIG))
    args.add_noise = False
    args.seq_num = train_count
    args.train_obj_code_list = object_ids
    args.distillation.enabled = False
    original_cwd = Path.cwd()
    try:
        os.chdir(str(DEXGRASP_ROOT))
        dataset = GraspM3DexRepDataset(args, ds_name="train")
    finally:
        os.chdir(str(original_cwd))
    labels = dataset.data["vis_unscale_actions"].astype(
        np.float32, copy=True)
    if labels.shape != (train_count * dataset.num_frame, 28):
        raise RuntimeError(
            "{} label shape mismatch: {}".format(category, labels.shape))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        teacher_actions=labels,
        sample_count=np.asarray([len(labels)], dtype=np.int64),
        category=np.asarray([category]),
        object_ids=np.asarray(object_ids),
        source=np.asarray(["successful_demonstration_actions"]),
    )
    verify(path, category, object_ids, train_count)
    print(
        "CATEGORY_TEMPORAL_DEMO_LABELS=COMPLETE category={} samples={}"
        .format(category, len(labels)), flush=True)


def main():
    cli = parse_cli()
    categories = tuple(cli.category) if cli.category else CATEGORIES
    manifest = load_json(MANIFEST)
    summary = load_json(SUMMARY)
    for category in categories:
        prepare(category, manifest, summary)


if __name__ == "__main__":
    main()
