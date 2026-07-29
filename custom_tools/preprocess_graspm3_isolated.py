"""Resume GraspM3 preprocessing with one fresh CUDA process per object.

Isaac Gym may retain native PhysX allocations after a simulation is destroyed.
Running each object in a child process guarantees that all native CUDA state is
released before the next mesh is loaded.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from custom_tools import preprocess_graspm3 as preprocessing


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default=str(REPO_ROOT / "custom_tools" / "configs" /
                    "scaled_category_split_candidates_v1.json"))
    parser.add_argument(
        "--manifest-split", action="append", choices=("train", "test", "backups"),
        default=[])
    parser.add_argument(
        "--output-root",
        default=str(REPO_ROOT / "dexgrasp" / "dataset" /
                    "scaled_category_candidates_v1_preprocessed"))
    parser.add_argument(
        "--input-root", default=str(REPO_ROOT / "external_data" / "dataset"))
    parser.add_argument(
        "--selection", default="official_final",
        choices=("official_final", "ever_task_success", "lift_30cm", "all"))
    parser.add_argument("--trajectories-per-chunk", type=int, default=10)
    parser.add_argument("--min-free-vram-mb", type=int, default=4500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def requested_ids(manifest, split_names):
    object_ids = []
    for category in manifest["criteria"]["categories"]:
        for split_name in split_names:
            object_ids.extend(manifest["categories"][category][split_name])
    return list(dict.fromkeys(object_ids))


def run_chunk_with_fallback(
        args, worker, environment, input_root, chunk_dir,
        object_id, start, end):
    """Run one chunk, recursively halving it after a PhysX/CUDA failure."""
    chunk_path = chunk_dir / "chunk_{:04d}_{:04d}.npy".format(start, end)
    if chunk_path.is_file():
        print("[SKIP CHUNK] {} [{}, {})".format(object_id, start, end))
        return [chunk_path]
    command = [
        sys.executable,
        "-u",
        str(worker),
        "--object-id", object_id,
        "--trajectory-start", str(start),
        "--trajectory-end", str(end),
        "--input-root", str(input_root),
        "--output-file", str(chunk_path),
        "--selection", args.selection,
        "--min-free-vram-mb", str(args.min_free_vram_mb),
        "--seed", str(args.seed),
    ]
    print("[CHUNK] {} [{}, {}) in fresh CUDA process".format(
        object_id, start, end), flush=True)
    result = subprocess.run(
        command, cwd=str(REPO_ROOT), env=environment, check=False)
    if result.returncode == 0 and chunk_path.is_file():
        return [chunk_path]
    width = end - start
    if width <= 1:
        print("[FAIL] single trajectory {} [{}] cannot be replayed".format(
            object_id, start))
        return None
    middle = start + width // 2
    print("[FALLBACK] {} [{}, {}) failed; retry as [{}, {}) + [{}, {})".format(
        object_id, start, end, start, middle, middle, end), flush=True)
    left = run_chunk_with_fallback(
        args, worker, environment, input_root, chunk_dir,
        object_id, start, middle)
    if left is None:
        return None
    right = run_chunk_with_fallback(
        args, worker, environment, input_root, chunk_dir,
        object_id, middle, end)
    if right is None:
        return None
    return left + right


def main():
    args = parse_args()
    if args.trajectories_per_chunk < 1:
        raise ValueError("--trajectories-per-chunk must be positive")
    manifest_path = Path(args.manifest).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    input_root = Path(args.input_root).expanduser().resolve()
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    split_names = args.manifest_split or ["train", "test"]
    object_ids = requested_ids(manifest, split_names)
    completed = [
        object_id for object_id in object_ids
        if (output_root / (object_id + ".npy")).is_file()]
    pending = [object_id for object_id in object_ids if object_id not in completed]
    print("Requested objects: {}; completed: {}; pending: {}".format(
        len(object_ids), len(completed), len(pending)))
    if args.dry_run:
        for object_id in pending:
            print("[PENDING] {}".format(object_id))
        print("ISOLATED_PREPROCESS_DRY_RUN=PASS")
        return 0

    output_root.mkdir(parents=True, exist_ok=True)
    worker = REPO_ROOT / "custom_tools" / "preprocess_graspm3_chunk_worker.py"
    preprocessing.np = np
    environment = os.environ.copy()
    environment.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    for number, object_id in enumerate(pending, 1):
        print("\n[OBJECT {}/{}] {}".format(number, len(pending), object_id), flush=True)
        output_path = output_root / (object_id + ".npy")
        raw_path = input_root / (object_id + ".npy")
        raw = np.load(str(raw_path), allow_pickle=True).item()
        raw_count = int(len(raw["grasp_seqs"]))
        chunk_dir = output_root / ".isolated_chunks" / object_id
        chunk_dir.mkdir(parents=True, exist_ok=True)
        chunk_paths = []
        for start in range(0, raw_count, args.trajectories_per_chunk):
            end = min(start + args.trajectories_per_chunk, raw_count)
            completed_chunks = run_chunk_with_fallback(
                args, worker, environment, input_root, chunk_dir,
                object_id, start, end)
            if completed_chunks is None:
                print("[FAIL] chunk fallback was exhausted for {} [{}, {})"
                      .format(object_id, start, end))
                print("Completed chunk files were kept for the next resume.")
                return 1
            chunk_paths.extend(completed_chunks)

        chunk_outputs = [
            np.load(str(path), allow_pickle=True).item() for path in chunk_paths]
        output = preprocessing.merge_processed_chunks(chunk_outputs, raw_count)
        temporary_path = output_path.with_suffix(output_path.suffix + ".partial")
        with temporary_path.open("wb") as handle:
            np.save(handle, output, allow_pickle=True)
        os.replace(str(temporary_path), str(output_path))
        del output, chunk_outputs, raw
        for path in chunk_paths:
            path.unlink()
        chunk_dir.rmdir()
        print("[PASS] merged {}; every chunk used a fresh CUDA process".format(
            object_id))

    final_count = sum(
        (output_root / (object_id + ".npy")).is_file()
        for object_id in object_ids)
    if final_count != len(object_ids):
        raise RuntimeError("isolated preprocessing ended with missing outputs")
    print("ISOLATED_PREPROCESS_RESULT=COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
