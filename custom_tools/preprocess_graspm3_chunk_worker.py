"""Replay exactly one trajectory chunk in exactly one CUDA process."""

import argparse
import os
from pathlib import Path

from custom_tools import preprocess_graspm3 as preprocessing


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--trajectory-start", type=int, required=True)
    parser.add_argument("--trajectory-end", type=int, required=True)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument(
        "--selection", default="official_final", choices=preprocessing.SELECTIONS)
    parser.add_argument("--min-free-vram-mb", type=int, default=4500)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.trajectory_start < 0 or args.trajectory_end <= args.trajectory_start:
        raise ValueError("invalid trajectory chunk range")

    # Isaac Gym must be imported before torch/numpy in this process.
    preprocessing.initialize_cuda_runtime()
    preprocessing.require_free_vram(args.min_free_vram_mb)
    preprocessing.initialize_runtime()

    input_path = Path(args.input_root).expanduser().resolve() / (
        args.object_id + ".npy")
    output_path = Path(args.output_file).expanduser().resolve()
    raw = preprocessing.np.load(str(input_path), allow_pickle=True).item()
    raw_count = int(len(raw["grasp_seqs"]))
    if args.trajectory_end > raw_count:
        raise ValueError("chunk end exceeds raw trajectory count")
    raw["obj_code"] = args.object_id
    chunk = preprocessing.subset_raw_trajectories(
        raw, args.trajectory_start, args.trajectory_end)

    original_cwd = Path.cwd()
    os.chdir(str(preprocessing.DEXGRASP_ROOT))
    try:
        runtime = preprocessing.prepare_runtime(args.seed)
        outputs = preprocessing.process_batch(
            *runtime, npy_list=[chunk], selection=args.selection)
    finally:
        os.chdir(str(original_cwd))
    if len(outputs) != 1 or outputs[0][0] != args.object_id:
        raise RuntimeError("unexpected worker output")
    output = outputs[0][1]
    for key in preprocessing.INDEX_KEYS:
        output[key] = (
            preprocessing.np.asarray(output[key], dtype=preprocessing.np.int64)
            + args.trajectory_start)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".partial")
    with temporary_path.open("wb") as handle:
        preprocessing.np.save(handle, output, allow_pickle=True)
    os.replace(str(temporary_path), str(output_path))
    print("CHUNK_WORKER_RESULT=PASS {} [{}, {})".format(
        args.object_id, args.trajectory_start, args.trajectory_end))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
