"""Evaluate locked Temporal3 on held-out trajectories of all 80 seen objects.

The object meshes appeared in training, but the trajectories in
``scaled_bc20_valid_v1`` did not participate in optimizer updates.  This is a
reporting-only seen-instance evaluation and cannot change the locked model.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "custom_tools/configs/taskid_final_model_lock_v1.yaml"
MANIFEST = ROOT / "custom_tools/configs/scaled_category_split_final_v1.json"
FULL_TRAJECTORY_ROOT = (
    ROOT / "dexgrasp/dataset/scaled_category_final_v1_preprocessed")
VALID_SPLIT_ROOT = ROOT / "dexgrasp/dataset/scaled_bc20_valid_v1"
RESIDUAL_CONFIG = ROOT / "custom_tools/configs/residual_ppo_stage1.yaml"
OUTPUT_ROOT = ROOT / "custom_tools/results/taskid_locked_seen80_validation_v1"
TRAJECTORY_ROOT = OUTPUT_ROOT / "seen80_sim_validation_trajectories"
SELECTION = OUTPUT_ROOT / "seen80_selection.yaml"
CATEGORIES = ("bottle", "mug", "bowl", "camera")


def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--min-free-vram-mb", type=int, default=4500)
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_yaml(path):
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stage_simulation_validation(object_ids):
    """Restore simulation metadata while preserving the frozen valid indices."""
    TRAJECTORY_ROOT.mkdir(parents=True, exist_ok=True)
    for object_id in object_ids:
        output = TRAJECTORY_ROOT / (object_id + ".npy")
        split_path = VALID_SPLIT_ROOT / (object_id + ".npy")
        full_path = FULL_TRAJECTORY_ROOT / (object_id + ".npy")
        split = np.load(split_path, allow_pickle=True).item()
        info = split.get("custom_split_info")
        if not isinstance(info, dict) or info.get("split") != "valid":
            raise RuntimeError(
                "Missing valid split metadata for {}".format(object_id))
        indices = np.asarray(
            info["selected_local_indices"], dtype=np.int64)
        full = np.load(full_path, allow_pickle=True).item()
        if len(indices) != len(split["grasp_seqs"]):
            raise RuntimeError(
                "Validation index count mismatch for {}".format(object_id))
        if np.any(indices < 0) or np.any(indices >= len(full["grasp_seqs"])):
            raise RuntimeError(
                "Validation indices out of range for {}".format(object_id))
        selected_grasps = full["grasp_seqs"][indices]
        if not np.array_equal(selected_grasps, split["grasp_seqs"]):
            raise RuntimeError(
                "Validation trajectories do not match source for {}".format(
                    object_id))
        expected = {
            "obj_scale": full["obj_scale"][indices].copy(),
            "obj_rotmat": full["obj_rotmat"][indices].copy(),
            "grasp_seqs": selected_grasps.copy(),
            "custom_split_info": dict(info),
            "staging_purpose": "seen80_closed_loop_validation",
        }
        if output.exists():
            actual = np.load(output, allow_pickle=True).item()
            for key in ("obj_scale", "obj_rotmat", "grasp_seqs"):
                if not np.array_equal(actual.get(key), expected[key]):
                    raise RuntimeError(
                        "Staged validation data changed: {}".format(output))
            if actual.get("custom_split_info") != expected["custom_split_info"]:
                raise RuntimeError(
                    "Staged validation provenance changed: {}".format(output))
        else:
            np.save(output, expected, allow_pickle=True)


def prepare_inputs():
    lock = load_yaml(LOCK)
    if lock["post_holdout_training_or_selection_allowed"] is not False:
        raise RuntimeError("The final lock does not prohibit further selection")
    primary = lock["models"]["primary"]
    if primary["label"] != "temporal3":
        raise RuntimeError("Locked primary model is not Temporal3")
    checkpoint = ROOT / primary["checkpoint"]
    config = ROOT / primary["config"]
    if sha256(checkpoint) != primary["checkpoint_sha256"]:
        raise RuntimeError("Locked Temporal3 checkpoint hash changed")

    with MANIFEST.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    object_ids = []
    test_ids = set()
    for category in CATEGORIES:
        train = list(manifest["categories"][category]["train"])
        test = list(manifest["categories"][category]["test"])
        if len(train) != 20 or len(set(train)) != 20:
            raise RuntimeError(
                "Expected 20 unique training objects for {}".format(category))
        object_ids.extend(train)
        test_ids.update(test)
    if len(object_ids) != 80 or len(set(object_ids)) != 80:
        raise RuntimeError("Expected 80 unique seen objects")
    if set(object_ids) & test_ids:
        raise RuntimeError("Seen-object validation overlaps unseen objects")
    missing_split = [
        object_id for object_id in object_ids
        if not (VALID_SPLIT_ROOT / (object_id + ".npy")).is_file()]
    missing_full = [
        object_id for object_id in object_ids
        if not (FULL_TRAJECTORY_ROOT / (object_id + ".npy")).is_file()]
    if missing_split or missing_full:
        raise FileNotFoundError(
            "Missing validation split={} or full trajectories={}".format(
                missing_split, missing_full))

    selection = {
        "status": "reporting_only_seen_instance_validation",
        "model_was_locked_before_evaluation": True,
        "uses_optimizer_training_trajectories": False,
        "uses_final_unseen_objects": False,
        "objects_per_category": 20,
        "object_ids": object_ids,
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    if SELECTION.exists():
        if load_yaml(SELECTION) != selection:
            raise RuntimeError("Seen80 selection changed")
    else:
        with SELECTION.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(selection, handle, sort_keys=False)
    stage_simulation_validation(object_ids)
    return primary, checkpoint, config


def main():
    cli = parse_cli()
    if cli.max_attempts < 1:
        raise ValueError("--max-attempts must be positive")
    primary, checkpoint, config = prepare_inputs()
    output = OUTPUT_ROOT / "seed{}".format(cli.seed) / "temporal3.yaml"
    if not output.exists():
        command = [
            sys.executable, "-u",
            str(ROOT / "custom_tools/evaluate_bc_checkpoints_isolated.py"),
            "--checkpoint", str(checkpoint),
            "--bc-config", str(config),
            "--residual-config", str(RESIDUAL_CONFIG),
            "--trajectory-root", str(TRAJECTORY_ROOT),
            "--object-selection", str(SELECTION),
            "--output", str(output),
            "--seed", str(cli.seed),
            "--min-free-vram-mb", str(cli.min_free_vram_mb),
            "--max-attempts", str(cli.max_attempts),
        ]
        print("[RUN] locked Temporal3 on 80 seen objects", flush=True)
        print(" ".join(command), flush=True)
        if cli.dry_run:
            print("TASKID_SEEN80_DRY_RUN=COMPLETE", flush=True)
            return
        subprocess.run(command, cwd=str(ROOT), check=True)
    else:
        print("[REUSE] {}".format(output), flush=True)

    result = load_yaml(output)
    rows = result.get("checkpoint_results", [])
    if len(rows) != 1:
        raise RuntimeError("Expected one checkpoint result")
    row = rows[0]
    if Path(row["checkpoint"]).resolve() != checkpoint.resolve():
        raise RuntimeError("Result checkpoint does not match locked Temporal3")
    summary = {
        "status": "complete",
        "stage": "locked_temporal3_seen80_validation",
        "reporting_only": True,
        "seed": cli.seed,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": primary["checkpoint_sha256"],
        "object_count": 80,
        "trajectory_count": row["total_trajectory_count"],
        "overall_success_rate": row["overall_official_peak_success_rate"],
        "object_macro_success_rate": row[
            "macro_official_peak_success_rate"],
        "mean_maximum_lift_m": row["macro_mean_maximum_lift_m"],
        "failure_rate": row["macro_failure_rate"],
        "category_macro_success_rates": row[
            "category_macro_success_rates"],
        "source": str(output),
    }
    with (OUTPUT_ROOT / "summary.yaml").open(
            "w", encoding="utf-8") as handle:
        yaml.safe_dump(summary, handle, sort_keys=False)
    print(
        "TASKID_LOCKED_SEEN80_VALIDATION=COMPLETE "
        "success={}/{} overall={:.2f}% macro={:.2f}%".format(
            row["total_success_count"], row["total_trajectory_count"],
            100 * row["overall_official_peak_success_rate"],
            100 * row["macro_official_peak_success_rate"]),
        flush=True)


if __name__ == "__main__":
    main()
