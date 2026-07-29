"""Collect one Task-ID DAgger round with one fresh process per object."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = ("bottle", "mug", "bowl", "camera")
MANIFEST = (
    REPO_ROOT / "custom_tools/configs/scaled_category_split_final_v1.json"
)
TRAJECTORY_ROOT = (
    REPO_ROOT / "dexgrasp/dataset/scaled_category_final_v1_preprocessed"
)
SPLIT_ROOT = REPO_ROOT / "dexgrasp/dataset/scaled_bc20_train_v1"
BC_CONFIG = (
    REPO_ROOT
    / "custom_tools/configs/unified_student_taskid_scaled20_v1.yaml"
)
STUDENT = (
    REPO_ROOT / "custom_tools/runs/bc/"
    / "unified_student_taskid_scaled20_t100_seed2025_e20_v1/"
    / "epoch=014-step=14145.ckpt"
)
TEACHERS = {
    "bottle": (
        REPO_ROOT / "custom_tools/runs/bc/"
        / "category_expert_bottle_scaled20_noise005_soup_seed2025_e40_v1/"
        / "epoch=029-step=8790.ckpt"
    ),
    "mug": (
        REPO_ROOT / "custom_tools/runs/bc/"
        / "category_expert_mug_scaled20_noise005_soup_seed2025_e40_v1/"
        / "epoch=009-step=2250.ckpt"
    ),
    "bowl": (
        REPO_ROOT / "custom_tools/runs/bc/"
        / "category_expert_bowl_scaled20_noise005_soup_seed2025_e40_v1/"
        / "epoch=039-step=7440.ckpt"
    ),
    "camera": (
        REPO_ROOT / "custom_tools/runs/bc/"
        / "category_expert_camera_scaled20_noise005_soup_seed2025_e40_v1/"
        / "epoch=039-step=9520.ckpt"
    ),
}
OUTPUT = (
    REPO_ROOT / "custom_tools/data/distillation/"
    / "online_taskid_scaled20_r1_train4.npz"
)
PARTS = (
    REPO_ROOT / "custom_tools/data/distillation/"
    / "online_taskid_scaled20_r1_train4_parts"
)


def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-checkpoint", default="")
    parser.add_argument("--bc-config", default="")
    parser.add_argument("--output", default="")
    parser.add_argument("--parts-dir", default="")
    parser.add_argument("--trajectories-per-object", type=int, default=4)
    parser.add_argument(
        "--trajectory-start-offset",
        type=int,
        default=0,
        help="Zero-based offset within each object's staged training split.",
    )
    parser.add_argument("--horizon", type=int, default=69)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--min-free-vram-mb", type=int, default=4500)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_manifest_objects():
    with MANIFEST.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    selected = [
        (category, object_id)
        for category in CATEGORIES
        for object_id in manifest["categories"][category]["train"]
    ]
    if len(selected) != 80 or len({item[1] for item in selected}) != 80:
        raise RuntimeError("Expected exactly 20 unique training objects per category")
    return selected


def required_inputs(selected):
    paths = [
        MANIFEST, TRAJECTORY_ROOT, SPLIT_ROOT, BC_CONFIG, STUDENT,
        *TEACHERS.values(),
    ]
    paths.extend(
        TRAJECTORY_ROOT / (object_id + ".npy")
        for _, object_id in selected)
    paths.extend(
        SPLIT_ROOT / (object_id + ".npy")
        for _, object_id in selected)
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing online collection inputs: {}".format(
            missing[:10]))
    if any("objectbalanced" in str(path).lower() for path in TEACHERS.values()):
        raise RuntimeError("Object-balanced teachers are forbidden in this pool")


def worker_command(cli, object_id, output):
    command = [
        sys.executable,
        "-u",
        str(REPO_ROOT / "custom_tools/collect_online_imitation_data.py"),
        "--student-checkpoint",
        str(STUDENT),
        "--manifest",
        str(MANIFEST),
        "--trajectory-root",
        str(TRAJECTORY_ROOT),
        "--trajectory-split-root",
        str(SPLIT_ROOT),
        "--bc-config",
        str(BC_CONFIG),
        "--object-id",
        object_id,
        "--output",
        str(output),
        "--horizon",
        str(cli.horizon),
        "--max-trajectories-per-object",
        str(cli.trajectories_per_object),
        "--trajectory-start-offset",
        str(cli.trajectory_start_offset),
        "--seed",
        str(cli.seed),
        "--min-free-vram-mb",
        str(cli.min_free_vram_mb),
    ]
    for category in CATEGORIES:
        command.extend([
            "--teacher",
            "{}={}".format(category, TEACHERS[category]),
        ])
    return command


def validate_part(path, category, object_id, cli):
    summary_path = path.with_suffix(".yaml")
    if not path.is_file() or not summary_path.is_file():
        raise FileNotFoundError("Incomplete object part: {}".format(path))
    data = np.load(path, allow_pickle=False)
    expected_keys = {
        "observations", "teacher_actions", "student_actions",
        "category_indices", "object_indices", "trajectory_indices",
        "frame_indices", "object_ids",
    }
    if set(data.files) != expected_keys:
        raise RuntimeError("Unexpected part keys in {}".format(path))
    count = len(data["observations"])
    if data["observations"].shape != (count, 2460):
        raise RuntimeError("Invalid observation shape in {}".format(path))
    if data["teacher_actions"].shape != (count, 28):
        raise RuntimeError("Invalid teacher action shape in {}".format(path))
    if data["student_actions"].shape != (count, 28):
        raise RuntimeError("Invalid student action shape in {}".format(path))
    if not all(np.isfinite(data[key]).all() for key in (
            "observations", "teacher_actions", "student_actions")):
        raise RuntimeError("Non-finite online sample in {}".format(path))
    if data["object_ids"].tolist() != [object_id]:
        raise RuntimeError("Object ID mismatch in {}".format(path))
    if not np.all(data["category_indices"] == CATEGORIES.index(category)):
        raise RuntimeError("Category mismatch in {}".format(path))
    lower = cli.trajectory_start_offset
    upper = lower + cli.trajectories_per_object
    if (np.any(data["trajectory_indices"] < lower)
            or np.any(data["trajectory_indices"] >= upper)):
        raise RuntimeError(
            "Part is outside trajectory range [{}, {}): {}".format(
                lower, upper, path))
    summary = yaml.safe_load(summary_path.read_text())
    if not summary.get("training_split_only", False):
        raise RuntimeError("Part is not marked training-only: {}".format(path))
    if Path(summary["trajectory_split_root"]).resolve() != SPLIT_ROOT.resolve():
        raise RuntimeError("Part used a different split root: {}".format(path))
    if int(summary.get("trajectory_start_offset", 0)) != lower:
        raise RuntimeError("Part used a different trajectory offset: {}".format(
            path))
    return data, summary


def merge_parts(selected, cli):
    arrays = {
        key: [] for key in (
            "observations", "teacher_actions", "student_actions",
            "category_indices", "object_indices", "trajectory_indices",
            "frame_indices")
    }
    object_summaries = []
    for object_index, (category, object_id) in enumerate(selected):
        part = PARTS / (object_id + ".npz")
        data, summary = validate_part(
            part, category, object_id, cli)
        for key in arrays:
            if key == "object_indices":
                arrays[key].append(np.full(
                    len(data["observations"]), object_index,
                    dtype=np.int16))
            else:
                arrays[key].append(data[key])
        object_summary = dict(summary["objects"][0])
        object_summary["global_object_index"] = object_index
        object_summaries.append(object_summary)

    merged = {
        key: np.concatenate(values, axis=0)
        for key, values in arrays.items()}
    merged["object_ids"] = np.asarray(
        [object_id for _, object_id in selected])
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUTPUT, **merged)

    count = len(merged["observations"])
    disagreement = np.abs(
        merged["student_actions"] - merged["teacher_actions"])
    category_summary = {}
    for category_index, category in enumerate(CATEGORIES):
        mask = merged["category_indices"] == category_index
        category_summary[category] = {
            "sample_count": int(mask.sum()),
            "mean_student_teacher_action_mae": float(
                disagreement[mask].mean()),
        }
    summary = {
        "method": "isolated one-round pure-student DAgger collection",
        "training_split_only": True,
        "formal_final_holdout_used": False,
        "student_checkpoint": str(STUDENT),
        "teacher_checkpoints": {
            category: str(path) for category, path in TEACHERS.items()},
        "manifest": str(MANIFEST),
        "trajectory_root": str(TRAJECTORY_ROOT),
        "trajectory_split_root": str(SPLIT_ROOT),
        "objects": len(selected),
        "trajectories_per_object": cli.trajectories_per_object,
        "trajectory_start_offset": cli.trajectory_start_offset,
        "trajectory_index_range": [
            cli.trajectory_start_offset,
            cli.trajectory_start_offset + cli.trajectories_per_object,
        ],
        "horizon": cli.horizon,
        "sample_count": count,
        "mean_student_teacher_action_mae": float(disagreement.mean()),
        "category_summary": category_summary,
        "object_summaries": object_summaries,
    }
    with OUTPUT.with_suffix(".yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(summary, handle, allow_unicode=True, sort_keys=False)
    print(
        "TASKID_ONLINE_ISOLATED_COLLECTION=COMPLETE objects={} samples={} "
        "mae={:.6f}".format(
            len(selected), count, disagreement.mean()),
        flush=True)


def main():
    global STUDENT, BC_CONFIG, OUTPUT, PARTS
    cli = parse_cli()
    if cli.student_checkpoint:
        STUDENT = Path(cli.student_checkpoint).expanduser().resolve()
    if cli.bc_config:
        BC_CONFIG = Path(cli.bc_config).expanduser().resolve()
    if cli.output:
        OUTPUT = Path(cli.output).expanduser().resolve()
    if cli.parts_dir:
        PARTS = Path(cli.parts_dir).expanduser().resolve()
    if cli.trajectories_per_object < 1:
        raise ValueError("--trajectories-per-object must be positive")
    if cli.trajectory_start_offset < 0:
        raise ValueError("--trajectory-start-offset must be non-negative")
    if not 1 <= cli.horizon <= 122:
        raise ValueError("--horizon must be in [1, 122]")
    if cli.max_attempts < 1:
        raise ValueError("--max-attempts must be positive")
    selected = load_manifest_objects()
    required_inputs(selected)
    PARTS.mkdir(parents=True, exist_ok=True)
    if OUTPUT.exists() or OUTPUT.with_suffix(".yaml").exists():
        if not OUTPUT.exists() or not OUTPUT.with_suffix(".yaml").exists():
            raise RuntimeError("Merged output is incomplete; inspect before rerun")
        print("[REUSE] merged online collection: {}".format(OUTPUT))
        return

    for index, (category, object_id) in enumerate(selected, 1):
        part = PARTS / (object_id + ".npz")
        part_summary = part.with_suffix(".yaml")
        if part.exists() != part_summary.exists():
            # These are disposable worker outputs, not user data.  A worker
            # killed between its two atomic writes must be safe to retry.
            if part.exists():
                part.unlink()
            if part_summary.exists():
                part_summary.unlink()
            print(
                "[{}/80] DISCARD incomplete part {}".format(
                    index, object_id),
                flush=True)
        if part.exists() and part_summary.exists():
            validate_part(part, category, object_id, cli)
            print(
                "[{}/80] REUSE {}".format(index, object_id),
                flush=True)
            continue
        command = worker_command(cli, object_id, part)
        print("[{}/80] RUN {}".format(index, object_id), flush=True)
        if cli.dry_run:
            print(" ".join(command), flush=True)
            continue
        for attempt in range(1, cli.max_attempts + 1):
            completed = subprocess.run(
                command, cwd=str(REPO_ROOT), check=False)
            if completed.returncode == 0:
                break
            print(
                "{} attempt {}/{} failed".format(
                    object_id, attempt, cli.max_attempts),
                flush=True)
        else:
            raise RuntimeError("Online collection worker failed: {}".format(
                object_id))
        validate_part(part, category, object_id, cli)
    if not cli.dry_run:
        merge_parts(selected, cli)


if __name__ == "__main__":
    main()
