"""Create one symlink-only source directory for the frozen scaled split."""

import argparse
import json
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", default=str(
            REPO_ROOT / "custom_tools/configs/scaled_category_split_final_v1.json"))
    parser.add_argument(
        "--output-root", default=str(
            REPO_ROOT / "dexgrasp/dataset/scaled_category_final_v1_preprocessed"))
    return parser.parse_args()


def main():
    args = parse_args()
    manifest_path = Path(args.manifest).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("status") != "frozen_preflight_passed":
        raise RuntimeError("refusing to stage a non-frozen manifest")
    output_root.mkdir(parents=True, exist_ok=True)

    expected = set(manifest["objects"])
    existing = {path.stem for path in output_root.glob("*.npy")}
    unexpected = existing - expected
    if unexpected:
        raise RuntimeError("unexpected existing staged files: {}".format(sorted(unexpected)))
    for object_id, metadata in manifest["objects"].items():
        source = Path(metadata["preprocessed_source"]).expanduser().resolve()
        target = output_root / (object_id + ".npy")
        if not source.is_file():
            raise FileNotFoundError(source)
        if target.is_symlink():
            if target.resolve() != source:
                raise RuntimeError("staged link points to wrong source: {}".format(target))
            continue
        if target.exists():
            raise FileExistsError("non-symlink target already exists: {}".format(target))
        target.symlink_to(source)

    staged = list(output_root.glob("*.npy"))
    if len(staged) != 100:
        raise RuntimeError("expected 100 staged object files, found {}".format(len(staged)))
    base_count = sum(
        metadata["uses_frozen_base_preprocessing"]
        for metadata in manifest["objects"].values())
    print("[PASS] staged {} symlinks; frozen base sources={}".format(
        len(staged), base_count))
    print("FINAL_PREPROCESSED_STAGE_RESULT=READY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
