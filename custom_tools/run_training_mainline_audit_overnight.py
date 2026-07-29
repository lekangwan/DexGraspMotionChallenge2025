"""Three-seed training-set audit of every locked serial-mainline node."""

import argparse
import collections
import json
from pathlib import Path
import statistics
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
TRAJECTORY_ROOT = (
    ROOT / "dexgrasp/dataset/scaled_category_final_v1_preprocessed")
SELECTION = (
    ROOT / "custom_tools/configs/temporal3_residual_audit16_objects.yaml")
PROTOCOL = ROOT / "custom_tools/configs/scaled_evaluation_protocol_v1.json"
RESIDUAL_CONFIG = ROOT / "custom_tools/configs/residual_ppo_stage1.yaml"
OUTPUT_ROOT = (
    ROOT / "custom_tools/results/training_mainline_three_seed_v1")
SEEDS = (2025, 2026, 2027)
CATEGORIES = ("bottle", "mug", "bowl", "camera")

SINGLE_NODES = (
    {
        "label": "bc_soup",
        "checkpoint": (
            ROOT / "custom_tools/runs/bc/model_soups/"
            "noise005_s2025_s2026_weighted2to1.ckpt"),
        "config": (
            ROOT / "custom_tools/configs/category_expert_bc_scaled20_v1.yaml"),
    },
    {
        "label": "offline_taskid_student",
        "checkpoint": (
            ROOT / "custom_tools/runs/bc/"
            "unified_student_taskid_scaled20_t100_seed2025_e20_v1/"
            "epoch=014-step=14145.ckpt"),
        "config": (
            ROOT / "custom_tools/configs/"
            "unified_student_taskid_scaled20_v1.yaml"),
    },
    {
        "label": "online_r1_student",
        "checkpoint": (
            ROOT / "custom_tools/runs/bc/"
            "unified_student_taskid_online_r1_frac025_seed2025_e10_v1/"
            "epoch=001-step=2232.ckpt"),
        "config": (
            ROOT / "custom_tools/configs/"
            "unified_student_taskid_online_r1_scaled20_v1.yaml"),
    },
    {
        "label": "temporal3_student",
        "checkpoint": (
            ROOT / "custom_tools/runs/bc/"
            "unified_student_taskid_temporal3_seed2025_e4_v1/"
            "epoch=003-step=5152.ckpt"),
        "config": (
            ROOT / "custom_tools/configs/"
            "unified_student_taskid_temporal3_v1.yaml"),
    },
)
TEACHERS = {
    "bottle": (
        ROOT / "custom_tools/runs/bc/"
        "category_expert_bottle_scaled20_noise005_soup_seed2025_e40_v1/"
        "epoch=029-step=8790.ckpt"),
    "mug": (
        ROOT / "custom_tools/runs/bc/"
        "category_expert_mug_scaled20_noise005_soup_seed2025_e40_v1/"
        "epoch=009-step=2250.ckpt"),
    "bowl": (
        ROOT / "custom_tools/runs/bc/"
        "category_expert_bowl_scaled20_noise005_soup_seed2025_e40_v1/"
        "epoch=039-step=7440.ckpt"),
    "camera": (
        ROOT / "custom_tools/runs/bc/"
        "category_expert_camera_scaled20_noise005_soup_seed2025_e40_v1/"
        "epoch=039-step=9520.ckpt"),
}
TEACHER_CONFIG = (
    ROOT / "custom_tools/configs/category_expert_bc_scaled20_v1.yaml")
TEMPORAL_CHECKPOINT = SINGLE_NODES[-1]["checkpoint"]
TEMPORAL_CONFIG = SINGLE_NODES[-1]["config"]
RENDER_CASES = (
    ROOT / "custom_tools/configs/temporal3_training_render_cases.yaml")
RENDER_OUTPUT = ROOT / "custom_tools/results/temporal3_training_renders_v1"


def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-free-vram-mb", type=int, default=4500)
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--capture-stride", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_yaml(path):
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def execute(command, dry_run=False):
    command = [str(value) for value in command]
    print("RUN {}".format(" ".join(command)), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=str(ROOT), check=True)


def checkpoint_result(path, expected_checkpoint):
    result = load_yaml(path)
    rows = result.get("checkpoint_results", [])
    if len(rows) != 1:
        raise RuntimeError("Expected one checkpoint result in {}".format(path))
    if Path(rows[0]["checkpoint"]).resolve() != expected_checkpoint.resolve():
        raise RuntimeError("Checkpoint mismatch in {}".format(path))
    return rows[0]


def evaluate(checkpoint, config, selection, output, seed, cli):
    if output.exists():
        checkpoint_result(output, checkpoint)
        print("REUSE {}".format(output), flush=True)
        return
    execute([
        PYTHON, "-u",
        ROOT / "custom_tools/evaluate_bc_checkpoints_isolated.py",
        "--checkpoint", checkpoint,
        "--bc-config", config,
        "--residual-config", RESIDUAL_CONFIG,
        "--trajectory-root", TRAJECTORY_ROOT,
        "--object-selection", selection,
        "--output", output,
        "--seed", seed,
        "--min-free-vram-mb", cli.min_free_vram_mb,
        "--max-attempts", cli.max_attempts,
    ], cli.dry_run)


def validate_and_write_category_selections():
    selection = load_yaml(SELECTION)
    object_ids = list(selection["object_ids"])
    if len(object_ids) != 16:
        raise ValueError("Expected 16 frozen training objects")
    with PROTOCOL.open("r", encoding="utf-8") as handle:
        protocol = json.load(handle)
    final_ids = {
        object_id
        for category in CATEGORIES
        for object_id in protocol["categories"][category]["final_holdout"]}
    if set(object_ids) & final_ids:
        raise RuntimeError("Training audit overlaps the final holdout")
    selection_dir = OUTPUT_ROOT / "selections"
    selection_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for category in CATEGORIES:
        ids = [
            object_id for object_id in object_ids
            if object_id.split("-", 2)[1] == category]
        if len(ids) != 4:
            raise ValueError("Expected four {} objects".format(category))
        path = selection_dir / "{}_training4.yaml".format(category)
        expected = {
            "status": "frozen_training_audit_category",
            "category": category,
            "uses_unseen_test_objects": False,
            "object_ids": ids,
        }
        if path.exists():
            if load_yaml(path) != expected:
                raise RuntimeError("Selection changed: {}".format(path))
        else:
            with path.open("w", encoding="utf-8") as handle:
                yaml.safe_dump(expected, handle, sort_keys=False)
        paths[category] = path
    return paths


def combine_objects(objects):
    category_rates = collections.defaultdict(list)
    for item in objects:
        category_rates[item["category"]].append(
            float(item["official_peak_success_rate"]))
    return {
        "total_success_count": sum(
            int(item["official_peak_success_count"]) for item in objects),
        "total_trajectory_count": sum(
            int(item["trajectory_count"]) for item in objects),
        "macro_official_peak_success_rate": statistics.mean(
            float(item["official_peak_success_rate"]) for item in objects),
        "macro_mean_maximum_lift_m": statistics.mean(
            float(item["mean_maximum_lift_m"]) for item in objects),
        "macro_failure_rate": statistics.mean(
            float(item["failure_rate"]) for item in objects),
        "category_macro_success_rates": {
            category: statistics.mean(values)
            for category, values in sorted(category_rates.items())},
        "objects": objects,
    }


def summarize(single_paths, teacher_paths):
    seed_rows = collections.defaultdict(list)
    for node in SINGLE_NODES:
        for seed in SEEDS:
            result = checkpoint_result(
                single_paths[node["label"]][seed], node["checkpoint"])
            seed_rows[node["label"]].append(result)
    for seed in SEEDS:
        objects = []
        for category in CATEGORIES:
            result = checkpoint_result(
                teacher_paths[seed][category], TEACHERS[category])
            objects.extend(result["objects"])
        seed_rows["category_teacher_pool"].append(combine_objects(objects))

    order = (
        "bc_soup", "category_teacher_pool", "offline_taskid_student",
        "online_r1_student", "temporal3_student")
    nodes = []
    for label in order:
        rows = seed_rows[label]
        macros = [
            float(row["macro_official_peak_success_rate"]) for row in rows]
        nodes.append({
            "label": label,
            "success_counts": [
                int(row["total_success_count"]) for row in rows],
            "trajectory_count_per_seed": int(
                rows[0]["total_trajectory_count"]),
            "macro_success_mean": statistics.mean(macros),
            "macro_success_std": statistics.pstdev(macros),
            "macro_success_values": macros,
            "lift_mean_m": statistics.mean(
                float(row["macro_mean_maximum_lift_m"]) for row in rows),
            "failure_mean": statistics.mean(
                float(row["macro_failure_rate"]) for row in rows),
            "category_success": {
                category: {
                    "mean": statistics.mean(
                        float(row["category_macro_success_rates"][category])
                        for row in rows),
                    "std": statistics.pstdev(
                        float(row["category_macro_success_rates"][category])
                        for row in rows),
                }
                for category in CATEGORIES
            },
        })
    deltas = {
        "{}_minus_{}".format(nodes[index]["label"], nodes[index - 1]["label"]):
        nodes[index]["macro_success_mean"] - nodes[index - 1][
            "macro_success_mean"]
        for index in range(1, len(nodes))
    }
    summary = {
        "status": "complete",
        "stage": "three_seed_serial_mainline_training_set_audit",
        "selection": str(SELECTION),
        "training_objects": 16,
        "trajectories_per_seed": 495,
        "seeds": list(SEEDS),
        "final_holdout_accessed": False,
        "model_selection_allowed_from_this_result": False,
        "nodes": nodes,
        "serial_macro_deltas": deltas,
    }
    with (OUTPUT_ROOT / "summary.yaml").open(
            "w", encoding="utf-8") as handle:
        yaml.safe_dump(summary, handle, allow_unicode=True, sort_keys=False)
    print("\nnode,success_counts,macro_mean,std,lift_m,failure", flush=True)
    for node in nodes:
        print("{},{},{:.6f},{:.6f},{:.6f},{:.6f}".format(
            node["label"], node["success_counts"],
            node["macro_success_mean"], node["macro_success_std"],
            node["lift_mean_m"], node["failure_mean"]), flush=True)
    print("Saved summary: {}".format(
        OUTPUT_ROOT / "summary.yaml"), flush=True)


def main():
    cli = parse_cli()
    if cli.max_attempts < 1:
        raise ValueError("--max-attempts must be positive")
    required = [
        TRAJECTORY_ROOT, SELECTION, PROTOCOL, RESIDUAL_CONFIG,
        TEACHER_CONFIG, RENDER_CASES,
    ]
    required.extend(TEACHERS.values())
    for node in SINGLE_NODES:
        required.extend((node["checkpoint"], node["config"]))
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing inputs: {}".format(missing))
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    category_selections = validate_and_write_category_selections()

    single_paths = {
        node["label"]: {} for node in SINGLE_NODES}
    teacher_paths = {seed: {} for seed in SEEDS}
    for seed in SEEDS:
        for node in SINGLE_NODES:
            output = (
                OUTPUT_ROOT / "seed{}".format(seed)
                / "{}.yaml".format(node["label"]))
            single_paths[node["label"]][seed] = output
            evaluate(
                node["checkpoint"], node["config"], SELECTION,
                output, seed, cli)
        for category in CATEGORIES:
            output = (
                OUTPUT_ROOT / "seed{}".format(seed)
                / "teacher_{}.yaml".format(category))
            teacher_paths[seed][category] = output
            evaluate(
                TEACHERS[category], TEACHER_CONFIG,
                category_selections[category], output, seed, cli)

    if cli.dry_run:
        print("DRY_RUN=COMPLETE", flush=True)
        return
    summarize(single_paths, teacher_paths)
    execute([
        PYTHON, "-u", ROOT / "custom_tools/render_residual_cases.py",
        "--cases", RENDER_CASES,
        "--zero-residual",
        "--residual-config", RESIDUAL_CONFIG,
        "--bc-checkpoint", TEMPORAL_CHECKPOINT,
        "--bc-config", TEMPORAL_CONFIG,
        "--trajectory-root", TRAJECTORY_ROOT,
        "--output-dir", RENDER_OUTPUT,
        "--capture-stride", cli.capture_stride,
        "--min-free-vram-mb", cli.min_free_vram_mb,
    ])
    print("TRAINING_MAINLINE_OVERNIGHT=COMPLETE", flush=True)
    print("FINAL_HOLDOUT_ACCESSED=False", flush=True)


if __name__ == "__main__":
    main()
