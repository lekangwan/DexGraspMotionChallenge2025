"""Train and progressively screen four category-specific Temporal3 experts."""

import argparse
import collections
import json
from pathlib import Path
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = ("bottle", "mug", "bowl", "camera")
EPOCHS = (1, 2, 4)
CONFIG = (
    ROOT / "custom_tools/configs/"
    / "category_temporal3_demo_scaled20_v1.yaml")
INIT = (
    ROOT / "custom_tools/runs/bc/"
    / "unified_student_taskid_temporal3_seed2025_e4_v1/"
    / "epoch=003-step=5152.ckpt")
MANIFEST = (
    ROOT / "custom_tools/configs/scaled_category_split_final_v1.json")
DATA_SUMMARY = (
    ROOT / "custom_tools/results/scaled_bc20_dataset_summary_v1.json")
PROTOCOL = (
    ROOT / "custom_tools/configs/scaled_evaluation_protocol_v1.json")
SELECTIONS = (
    ROOT / "custom_tools/configs/temporal_target_screen_round1_4.yaml",
    ROOT / "custom_tools/configs/temporal_target_screen_round2_4.yaml",
    ROOT / "custom_tools/configs/temporal_target_screen_round3_4.yaml",
)
TRAJECTORY_ROOT = (
    ROOT / "dexgrasp/dataset/scaled_category_final_v1_preprocessed")
RESIDUAL_CONFIG = ROOT / "custom_tools/configs/residual_ppo_stage1.yaml"
LOCKED_RESULT = (
    ROOT / "custom_tools/results/taskid_temporal3_development_v1/"
    / "temporal3_epoch04.yaml")
SINGLE_FRAME_RESULTS = {
    "bottle": (
        ROOT / "custom_tools/results/scaled_category_development_v1/"
        / "seed2025/bottle/scale20_epoch30.yaml"),
    "mug": (
        ROOT / "custom_tools/results/scaled_category_development_v1/"
        / "seed2025/mug/scale20_epoch10.yaml"),
    "bowl": (
        ROOT / "custom_tools/results/scaled_category_development_v1/"
        / "seed2025/bowl/scale20_epoch40.yaml"),
    "camera": (
        ROOT / "custom_tools/results/scaled_category_development_v1/"
        / "seed2025/camera/scale20_epoch40.yaml"),
}
LABEL_ROOT = (
    ROOT / "custom_tools/data/distillation/category_temporal_demo")
OUTPUT_ROOT = (
    ROOT / "custom_tools/results/category_temporal_expert_screen_v1")
REPEAT_MARGIN = 0.02


def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-free-vram-mb", type=int, default=4500)
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_yaml(path):
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_json(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def run_name(category):
    return "category_temporal3_demo_{}_scaled20_seed2025_e4_v1".format(
        category)


def run_dir(category):
    return ROOT / "custom_tools/runs/bc" / run_name(category)


def label_path(category):
    return LABEL_ROOT / "{}_demo_train.npz".format(category)


def checkpoint(category, epoch):
    matches = list(run_dir(category).glob(
        "epoch={:03d}-step=*.ckpt".format(epoch - 1)))
    if len(matches) != 1:
        raise RuntimeError(
            "Expected one {} epoch {} checkpoint".format(category, epoch))
    return matches[0].resolve()


def counts_by_category():
    summary = load_json(DATA_SUMMARY)
    counts = {}
    for category in CATEGORIES:
        rows = [
            row for row in summary["objects"]
            if row["category"] == category]
        if len(rows) != 20:
            raise RuntimeError(
                "Expected 20 {} training objects".format(category))
        counts[category] = {
            "train": sum(int(row["train_count"]) for row in rows),
            "valid": sum(int(row["valid_count"]) for row in rows),
        }
    return counts


def prepare_labels(cli):
    command = [
        sys.executable, "-u",
        str(ROOT / "custom_tools/prepare_category_temporal_demo_labels.py"),
    ]
    print("RUN {}".format(" ".join(command)), flush=True)
    if not cli.dry_run:
        subprocess.run(command, cwd=str(ROOT), check=True)
        missing = [
            str(label_path(category)) for category in CATEGORIES
            if not label_path(category).is_file()]
        if missing:
            raise FileNotFoundError(missing)


def train(cli, category, counts):
    directory = run_dir(category)
    if (
        (directory / "last.ckpt").is_file()
        and (directory / "resource_summary.yaml").is_file()
    ):
        for epoch in EPOCHS:
            checkpoint(category, epoch)
        print("[REUSE] {} temporal expert".format(category), flush=True)
        return
    if list(directory.glob("*.ckpt")):
        raise RuntimeError(
            "Partial category temporal run: {}".format(directory))
    command = [
        sys.executable, "-u", str(ROOT / "custom_tools/train_bc.py"),
        "--config", str(CONFIG),
        "--run-name", run_name(category),
        "--seed", "2025",
        "--num-epochs", "4",
        "--learning-rate", "2e-5",
        "--teacher-weight", "1.0",
        "--teacher-action-file", str(label_path(category)),
        "--seq-num", str(counts["train"]),
        "--val-seq-num", str(counts["valid"]),
        "--train-category", category,
        "--category-train-size", "20",
        "--category-manifest", str(MANIFEST),
        "--init-checkpoint", str(INIT),
        "--min-free-vram-mb", str(cli.min_free_vram_mb),
    ]
    print("RUN {}".format(" ".join(command)), flush=True)
    if not cli.dry_run:
        subprocess.run(command, cwd=str(ROOT), check=True)
        for epoch in EPOCHS:
            checkpoint(category, epoch)


def selection_groups():
    groups = [load_yaml(path)["object_ids"] for path in SELECTIONS]
    for group in groups:
        if {
            object_id.split("-", 2)[1] for object_id in group
        } != set(CATEGORIES):
            raise RuntimeError("Each screen round needs all four categories")
    with PROTOCOL.open(encoding="utf-8") as handle:
        protocol = json.load(handle)
    development = {
        object_id
        for category in protocol["categories"].values()
        for object_id in category["development"]
    }
    final = {
        object_id
        for category in protocol["categories"].values()
        for object_id in category["final_holdout"]
    }
    flat = [object_id for group in groups for object_id in group]
    if len(flat) != 12 or set(flat) != development or set(flat) & final:
        raise RuntimeError("Category temporal screen violates protocol")
    return groups


def object_selection(round_index, category, object_id):
    path = (
        OUTPUT_ROOT / "selections"
        / "round{}_{}.yaml".format(round_index, category))
    path.parent.mkdir(parents=True, exist_ok=True)
    expected = {"object_ids": [object_id]}
    if path.is_file():
        if load_yaml(path) != expected:
            raise RuntimeError("Selection mismatch: {}".format(path))
    else:
        with path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(expected, handle, sort_keys=False)
    return path


def read_result(path, expected_checkpoint=None):
    rows = load_yaml(path).get("checkpoint_results", [])
    if len(rows) != 1:
        raise RuntimeError("Expected one result in {}".format(path))
    if (
        expected_checkpoint is not None
        and Path(rows[0]["checkpoint"]).resolve()
        != expected_checkpoint.resolve()
    ):
        raise RuntimeError("Checkpoint mismatch in {}".format(path))
    return rows[0]


def evaluation_path(epoch, round_index, category):
    return (
        OUTPUT_ROOT / "epoch{:02d}".format(epoch)
        / "round{}_{}.yaml".format(round_index, category))


def evaluate(
        cli, epoch, round_index, category, object_id):
    model = checkpoint(category, epoch)
    output = evaluation_path(epoch, round_index, category)
    if output.is_file():
        read_result(output, model)
        print("[REUSE] epoch{} round{} {}".format(
            epoch, round_index, category), flush=True)
        return
    command = [
        sys.executable, "-u",
        str(ROOT / "custom_tools/evaluate_bc_checkpoints_isolated.py"),
        "--checkpoint", str(model),
        "--bc-config", str(CONFIG),
        "--residual-config", str(RESIDUAL_CONFIG),
        "--trajectory-root", str(TRAJECTORY_ROOT),
        "--object-selection", str(
            object_selection(round_index, category, object_id)),
        "--output", str(output),
        "--seed", "2025",
        "--min-free-vram-mb", str(cli.min_free_vram_mb),
        "--max-attempts", str(cli.max_attempts),
    ]
    print("RUN {}".format(" ".join(command)), flush=True)
    subprocess.run(command, cwd=str(ROOT), check=True)
    read_result(output, model)


def aggregate(objects):
    categories = collections.defaultdict(list)
    for item in objects:
        categories[item["category"]].append(
            float(item["official_peak_success_rate"]))
    return {
        "object_count": len(objects),
        "object_ids": [item["object_id"] for item in objects],
        "success_count": sum(
            int(item["official_peak_success_count"]) for item in objects),
        "trajectory_count": sum(
            int(item["trajectory_count"]) for item in objects),
        "macro_success_rate": sum(
            float(item["official_peak_success_rate"])
            for item in objects) / len(objects),
        "mean_maximum_lift_m": sum(
            float(item["mean_maximum_lift_m"])
            for item in objects) / len(objects),
        "failure_rate": sum(
            float(item["failure_rate"]) for item in objects) / len(objects),
        "category_macro_success_rates": {
            category: sum(values) / len(values)
            for category, values in sorted(categories.items())
        },
    }


def temporal_row(epoch, rounds):
    objects = []
    outputs = []
    for round_index in rounds:
        for category in CATEGORIES:
            path = evaluation_path(epoch, round_index, category)
            model = checkpoint(category, epoch)
            objects.extend(read_result(path, model)["objects"])
            outputs.append(str(path))
    return {
        "label": "category_temporal_epoch{:02d}".format(epoch),
        "epoch": epoch,
        **aggregate(objects),
        "outputs": outputs,
    }


def subset_row(label, sources, object_ids):
    objects = []
    for source in sources:
        objects.extend(read_result(source)["objects"])
    by_id = {item["object_id"]: item for item in objects}
    missing = set(object_ids) - set(by_id)
    if missing:
        raise RuntimeError(
            "{} missing objects {}".format(label, sorted(missing)))
    return {
        "label": label,
        **aggregate([by_id[object_id] for object_id in object_ids]),
        "outputs": [str(source) for source in sources],
    }


def ranking_key(row):
    return (
        row["macro_success_rate"],
        row["mean_maximum_lift_m"],
        -row["failure_rate"],
    )


def print_rows(title, rows):
    print(title, flush=True)
    for rank, row in enumerate(
            sorted(rows, key=ranking_key, reverse=True), 1):
        print(
            "#{:02d} {} objects={} success={}/{} macro={:.2f}% "
            "lift={:.3f}m failure={:.2f}%".format(
                rank, row["label"], row["object_count"],
                row["success_count"], row["trajectory_count"],
                100 * row["macro_success_rate"],
                row["mean_maximum_lift_m"],
                100 * row["failure_rate"]),
            flush=True)


def save(summary):
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_ROOT / "summary.yaml").open(
        "w", encoding="utf-8"
    ) as handle:
        yaml.safe_dump(
            summary, handle, allow_unicode=True, sort_keys=False)


def main():
    cli = parse_cli()
    required = (
        CONFIG, INIT, MANIFEST, DATA_SUMMARY, PROTOCOL,
        TRAJECTORY_ROOT, RESIDUAL_CONFIG, LOCKED_RESULT,
        *SELECTIONS, *SINGLE_FRAME_RESULTS.values(),
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(missing)
    groups = selection_groups()
    counts = counts_by_category()
    prepare_labels(cli)
    for category in CATEGORIES:
        train(cli, category, counts[category])
    if cli.dry_run:
        print(
            "DRY_RUN: four full-data category Temporal3 experts; common "
            "epoch selected on round1, then progressive 8/12 comparison.",
            flush=True)
        return

    # Select one common epoch across the complete routed pool.
    for epoch in EPOCHS:
        for object_id in groups[0]:
            category = object_id.split("-", 2)[1]
            evaluate(cli, epoch, 1, category, object_id)
    rows4 = [temporal_row(epoch, (1,)) for epoch in EPOCHS]
    rows4.sort(key=ranking_key, reverse=True)
    best_epoch = rows4[0]["epoch"]

    for object_id in groups[1]:
        category = object_id.split("-", 2)[1]
        evaluate(cli, best_epoch, 2, category, object_id)
    ids8 = groups[0] + groups[1]
    temporal8 = temporal_row(best_epoch, (1, 2))
    locked8 = subset_row(
        "locked_unified_temporal3", [LOCKED_RESULT], ids8)
    single8 = subset_row(
        "single_frame_category_teacher_pool",
        list(SINGLE_FRAME_RESULTS.values()), ids8)
    advance = ranking_key(temporal8) > ranking_key(locked8)
    summary = {
        "status": (
            "round3_pending" if advance
            else "complete_stopped_after_round2"),
        "stage": "four category-specific Temporal3 demonstration experts",
        "formal_final_holdout_result": False,
        "final_holdout_accessed": False,
        "training_data_reduced": False,
        "category_counts": counts,
        "initialization": str(INIT),
        "supervision": (
            "successful demonstration actions; no single-frame teacher "
            "labels in the new fine-tuning data"),
        "round1_common_epoch_ranking": rows4,
        "selected_common_epoch": best_epoch,
        "cumulative8": [temporal8, locked8, single8],
        "advance_to_full": advance,
        "repeat_margin": REPEAT_MARGIN,
    }
    if not advance:
        summary["distill_temporal_experts"] = False
        save(summary)
        print_rows("CATEGORY_TEMPORAL_ROUND1", rows4)
        print_rows(
            "CATEGORY_TEMPORAL_CUMULATIVE8",
            [temporal8, locked8, single8])
        print("ADVANCE_TO_FULL=False", flush=True)
        print("DISTILL_TEMPORAL_EXPERTS=False", flush=True)
        print("CATEGORY_TEMPORAL_EXPERT_SCREEN=COMPLETE", flush=True)
        return

    for object_id in groups[2]:
        category = object_id.split("-", 2)[1]
        evaluate(cli, best_epoch, 3, category, object_id)
    ids12 = ids8 + groups[2]
    temporal12 = temporal_row(best_epoch, (1, 2, 3))
    locked12 = subset_row(
        "locked_unified_temporal3", [LOCKED_RESULT], ids12)
    single12 = subset_row(
        "single_frame_category_teacher_pool",
        list(SINGLE_FRAME_RESULTS.values()), ids12)
    improvement = (
        temporal12["macro_success_rate"]
        - locked12["macro_success_rate"])
    distill = improvement >= REPEAT_MARGIN
    summary.update({
        "status": "complete",
        "full12": [temporal12, locked12, single12],
        "temporal_experts_minus_locked": improvement,
        "distill_temporal_experts": distill,
    })
    save(summary)
    print_rows("CATEGORY_TEMPORAL_ROUND1", rows4)
    print_rows(
        "CATEGORY_TEMPORAL_CUMULATIVE8",
        [temporal8, locked8, single8])
    print_rows(
        "CATEGORY_TEMPORAL_FULL12",
        [temporal12, locked12, single12])
    print("ADVANCE_TO_FULL=True", flush=True)
    print("DISTILL_TEMPORAL_EXPERTS={}".format(distill), flush=True)
    print("CATEGORY_TEMPORAL_EXPERT_SCREEN=COMPLETE", flush=True)


if __name__ == "__main__":
    main()
