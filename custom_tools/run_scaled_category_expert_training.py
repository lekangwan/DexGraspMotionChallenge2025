"""Train the 10/20-object category experts as one controlled data-scale run.

The four categories are trained sequentially so that only one process uses the
GPU.  Per-category trajectory counts are read from the frozen dataset summary
instead of being copied by hand.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = ("bottle", "mug", "bowl", "camera")
MANIFEST = REPO_ROOT / "custom_tools/configs/scaled_category_split_final_v1.json"
INIT_CHECKPOINT = (
    REPO_ROOT
    / "custom_tools/runs/bc/model_soups/"
    / "noise005_s2025_s2026_weighted2to1.ckpt"
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def category_counts(scale, manifest):
    summary_path = (
        REPO_ROOT
        / "custom_tools/results"
        / "scaled_bc{}_dataset_summary_v1.json".format(scale)
    )
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    summary = load_json(summary_path)
    rows_by_category = {category: [] for category in CATEGORIES}
    for row in summary["objects"]:
        rows_by_category[row["category"]].append(row)

    counts = {}
    for category in CATEGORIES:
        rows = rows_by_category[category]
        expected_ids = manifest["categories"][category]["train_nested"][str(scale)]
        actual_ids = [row["object_id"] for row in rows]
        if set(actual_ids) != set(expected_ids) or len(actual_ids) != scale:
            raise RuntimeError(
                "Dataset summary does not match the frozen {}-object {} split"
                .format(scale, category)
            )
        train_count = sum(int(row["train_count"]) for row in rows)
        valid_count = sum(int(row["valid_count"]) for row in rows)
        if train_count <= 0 or valid_count <= 0:
            raise RuntimeError("Empty trajectory split for {}".format(category))
        counts[category] = {
            "object_count": len(rows),
            "train_trajectories": train_count,
            "valid_trajectories": valid_count,
        }
    return summary_path, counts


def make_run(scale, category, counts, seed, min_free_vram_mb):
    config = (
        REPO_ROOT
        / "custom_tools/configs"
        / "category_expert_bc_scaled{}_v1.yaml".format(scale)
    )
    run_name = (
        "category_expert_{}_scaled{}_noise005_soup_seed{}_e40_v1"
        .format(category, scale, seed)
    )
    run_dir = REPO_ROOT / "custom_tools/runs/bc" / run_name
    command = [
        sys.executable,
        "-u",
        str(REPO_ROOT / "custom_tools/train_bc.py"),
        "--config",
        str(config),
        "--run-name",
        run_name,
        "--seed",
        str(seed),
        "--num-epochs",
        "40",
        "--seq-num",
        str(counts["train_trajectories"]),
        "--val-seq-num",
        str(counts["valid_trajectories"]),
        "--train-category",
        category,
        "--category-train-size",
        str(scale),
        "--category-manifest",
        str(MANIFEST),
        "--init-checkpoint",
        str(INIT_CHECKPOINT),
        "--min-free-vram-mb",
        str(min_free_vram_mb),
    ]
    return run_name, run_dir, command


def parse_cli():
    parser = argparse.ArgumentParser(
        description="Sequentially train frozen 10/20-object category expert runs."
    )
    parser.add_argument(
        "--scales", type=int, nargs="+", choices=(10, 20), default=(10, 20)
    )
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--min-free-vram-mb", type=int, default=4500)
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate and print commands only."
    )
    return parser.parse_args()


def main():
    cli = parse_cli()
    if not MANIFEST.is_file():
        raise FileNotFoundError(MANIFEST)
    if not INIT_CHECKPOINT.is_file():
        raise FileNotFoundError(INIT_CHECKPOINT)
    manifest = load_json(MANIFEST)

    runs = []
    summary_paths = {}
    for scale in cli.scales:
        summary_path, counts_by_category = category_counts(scale, manifest)
        summary_paths[str(scale)] = str(summary_path)
        for category in CATEGORIES:
            counts = counts_by_category[category]
            run_name, run_dir, command = make_run(
                scale, category, counts, cli.seed, cli.min_free_vram_mb
            )
            runs.append(
                {
                    "scale": scale,
                    "category": category,
                    "run_name": run_name,
                    "run_dir": str(run_dir),
                    **counts,
                    "command": command,
                }
            )

    plan = {
        "status": "planned",
        "seed": cli.seed,
        "scales": list(cli.scales),
        "categories": list(CATEGORIES),
        "manifest": str(MANIFEST),
        "dataset_summaries": summary_paths,
        "init_checkpoint": str(INIT_CHECKPOINT),
        "init_checkpoint_sha256": sha256(INIT_CHECKPOINT),
        "controlled_constants": {
            "epochs": 40,
            "batch_size": 128,
            "learning_rate": 5e-5,
            "proprioceptive_noise": 0.05,
            "feature_encoder_frozen": False,
        },
        "runs": runs,
    }
    plan_path = (
        REPO_ROOT
        / "custom_tools/results"
        / "scaled_category_training_plan_seed{}.json".format(cli.seed)
    )
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    with plan_path.open("w", encoding="utf-8") as handle:
        json.dump(plan, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print("Validated {} runs; plan: {}".format(len(runs), plan_path))
    for index, run in enumerate(runs, start=1):
        print(
            "[{}/{}] scale={} category={} objects={} train={} valid={} run={}".format(
                index,
                len(runs),
                run["scale"],
                run["category"],
                run["object_count"],
                run["train_trajectories"],
                run["valid_trajectories"],
                run["run_name"],
            )
        )
        if cli.dry_run:
            print("  " + " ".join(run["command"]))
            continue

        run_dir = Path(run["run_dir"])
        last_checkpoint = run_dir / "last.ckpt"
        existing_checkpoints = list(run_dir.glob("*.ckpt"))
        if last_checkpoint.is_file():
            print("  [SKIP] completed checkpoint exists: {}".format(last_checkpoint))
            continue
        if existing_checkpoints:
            raise RuntimeError(
                "Partial run contains checkpoints but no last.ckpt: {}. "
                "Inspect it before deciding whether to resume or use a new run name."
                .format(run_dir)
            )

        environment = os.environ.copy()
        environment["PYTHONUNBUFFERED"] = "1"
        subprocess.run(
            run["command"], cwd=str(REPO_ROOT), env=environment, check=True
        )
        if not last_checkpoint.is_file():
            raise RuntimeError("Training returned without {}".format(last_checkpoint))

    if not cli.dry_run:
        plan["status"] = "complete"
        with plan_path.open("w", encoding="utf-8") as handle:
            json.dump(plan, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        print("All requested category expert runs are complete.")


if __name__ == "__main__":
    main()
