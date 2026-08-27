"""Run the locked ShadowHand baseline/component ablation on Final8."""

import argparse
from pathlib import Path
import statistics
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT.parent / "DexGraspMotionChallenge2025"
OUTPUT = ROOT / "custom_tools/results/restart_shadow_final_ablation_v1"
SEEDS = (2025, 2026, 2027)
FINAL_EXISTING = ROOT / "custom_tools/results/restart_shadow_final_v1"
PLAIN_CONFIG = ROOT / "custom_tools/configs/official_bc_scaled20_matched_v1.yaml"
CHUNK_CONFIG = ROOT / "custom_tools/configs/unified_student_temporal_chunk8_demo80_v1.yaml"

METHODS = (
    {
        "label": "released_official_bc",
        "checkpoint": OLD / "ActionDiffusion/bc/saved_models/1obj_seq2000_DexRep_pro100_start_uniform_vis_action_dsam_mod/last.ckpt",
        "config": PLAIN_CONFIG,
    },
    {
        "label": "matched_official_bc",
        "checkpoint": OLD / "custom_tools/runs/bc/official_bc_scaled20_matched_seed2025_e100_v1/epoch=079-step=75440.ckpt",
        "config": PLAIN_CONFIG,
    },
    {
        "label": "temporal3",
        "checkpoint": OLD / "custom_tools/runs/bc/unified_student_taskid_temporal3_seed2025_e4_v1/epoch=003-step=5152.ckpt",
        "config": ROOT / "custom_tools/configs/unified_student_taskid_temporal3_v1.yaml",
        "existing": "temporal3_seed{seed}.yaml",
    },
    {
        "label": "temporal3_demo80",
        "checkpoint": ROOT / "custom_tools/runs/bc/restart_shadow_temporal3_demo80_v1/epoch=000-step=1288.ckpt",
        "config": ROOT / "custom_tools/configs/unified_student_temporal3_demo80_v1.yaml",
    },
    {
        "label": "chunk8_no_ensemble",
        "checkpoint": ROOT / "custom_tools/runs/bc/restart_shadow_chunk8_demo80_v1/epoch=001-step=2576.ckpt",
        "config": ROOT / "custom_tools/configs/unified_student_temporal_chunk8_demo80_noensemble_v1.yaml",
    },
    {
        "label": "chunk8_equal",
        "checkpoint": ROOT / "custom_tools/runs/bc/restart_shadow_chunk8_demo80_v1/epoch=001-step=2576.ckpt",
        "config": CHUNK_CONFIG,
        "decay": 0.0,
    },
    {
        "label": "chunk8_equal_lift",
        "checkpoint": ROOT / "custom_tools/runs/bc/restart_shadow_chunk8_demo80_v1/epoch=001-step=2576.ckpt",
        "config": CHUNK_CONFIG,
        "decay": 0.0,
        "lift_boost": 0.20,
        "lift_start": 40,
        "existing": "chunk8_equal_seed{seed}.yaml",
    },
)


def load_result(path):
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)["checkpoint_results"][0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-free-vram-mb", type=int, default=4500)
    parser.add_argument("--max-attempts", type=int, default=3)
    cli = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = []

    for method in METHODS:
        for path in (method["checkpoint"], method["config"]):
            if not path.exists():
                raise FileNotFoundError(path)
        for seed in SEEDS:
            if "existing" in method:
                result_path = FINAL_EXISTING / method["existing"].format(seed=seed)
            else:
                result_path = OUTPUT / "{}_seed{}.yaml".format(method["label"], seed)
            if not result_path.exists():
                command = [
                    sys.executable, "-u",
                    str(ROOT / "custom_tools/evaluate_bc_checkpoints_isolated.py"),
                    "--checkpoint", str(method["checkpoint"]),
                    "--bc-config", str(method["config"]),
                    "--residual-config", str(ROOT / "custom_tools/configs/residual_ppo_stage1.yaml"),
                    "--trajectory-root", str(OLD / "dexgrasp/dataset/scaled_category_final_v1_preprocessed"),
                    "--meshdata-root", str(OLD / "assets/meshdata"),
                    "--object-selection", str(ROOT / "custom_tools/configs/scaled_final_holdout_all8.yaml"),
                    "--output", str(result_path),
                    "--policy-motion-steps", "70",
                    "--seed", str(seed),
                    "--min-free-vram-mb", str(cli.min_free_vram_mb),
                    "--max-attempts", str(cli.max_attempts),
                ]
                if "decay" in method:
                    command.extend(["--temporal-ensemble-decay", str(method["decay"])])
                if method.get("lift_boost", 0.0):
                    command.extend([
                        "--late-lift-z-boost", str(method["lift_boost"]),
                        "--late-lift-start-step", str(method["lift_start"]),
                    ])
                print("RUN: " + " ".join(command), flush=True)
                subprocess.run(command, cwd=str(ROOT), check=True)

            result = load_result(result_path)
            rows.append({
                "method": method["label"],
                "seed": seed,
                "official_count": result["total_success_count"],
                "trajectory_count": result["total_trajectory_count"],
                "official_overall_rate": result["overall_official_peak_success_rate"],
                "official_macro_rate": result["macro_official_peak_success_rate"],
                "stable_official_rate": result["overall_stable_official_success_rate"],
                "category_official_rates": result["category_macro_success_rates"],
                "source": str(result_path),
            })

    summary = {
        "status": "reporting_only_no_further_model_selection",
        "primary_metric": "object_macro_official_peak_success_rate",
        "secondary_metric": "overall_stable_official_success_rate",
        "rows": rows,
        "methods": {},
    }
    for method in METHODS:
        label = method["label"]
        selected = [row for row in rows if row["method"] == label]
        item = {
            "checkpoint": str(method["checkpoint"]),
            "official_counts": [row["official_count"] for row in selected],
        }
        for key in ("official_macro_rate", "official_overall_rate", "stable_official_rate"):
            values = [row[key] for row in selected]
            item[key] = {
                "mean": statistics.mean(values),
                "std": statistics.pstdev(values),
                "values": values,
            }
        item["category_official_rate_mean"] = {
            category: statistics.mean([
                row["category_official_rates"][category] for row in selected])
            for category in ("bottle", "mug", "bowl", "camera")
        }
        summary["methods"][label] = item

    with (OUTPUT / "summary.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(summary, handle, allow_unicode=True, sort_keys=False)
    print(yaml.safe_dump(summary["methods"], allow_unicode=True, sort_keys=False))


if __name__ == "__main__":
    main()
