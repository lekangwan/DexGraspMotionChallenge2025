#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter


METHOD_SUFFIX = "_smoothed_v1"


def smooth_frames(frames, window=5, poly=2):
    result = frames.copy()
    for i in range(result.shape[0]):
        for d in range(result.shape[-1]):
            result[i, :, d] = savgol_filter(result[i, :, d], window, poly)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--poly", type=int, default=2)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for entry in manifest["entries"]:
        name = entry["object_name"]
        source_path = args.input_dir / f"{name}.npy"
        if not source_path.is_file():
            raise FileNotFoundError(f"候选文件不存在: {source_path}")
        data = np.load(source_path, allow_pickle=True).item()
        data["grasp_seqs"] = smooth_frames(
            np.asarray(data["grasp_seqs"], dtype=np.float32),
            window=args.window, poly=args.poly)
        data["retarget_method"] = str(data.get("retarget_method", "unknown")) + METHOD_SUFFIX
        np.save(args.output_dir / f"{name}.npy", data, allow_pickle=True)
    print(f"smoothed={len(manifest['entries'])}")
    print(f"output={args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
