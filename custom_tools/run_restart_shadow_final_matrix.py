"""按冻结配置顺序评测旧Temporal3与最终Chunk8，汇总三种子结果。"""

import argparse
from pathlib import Path
import statistics
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
OLD_ROOT = ROOT.parent / "DexGraspMotionChallenge2025"
OUTPUT = ROOT / "custom_tools/results/restart_shadow_final_v1"
SEEDS = (2025, 2026, 2027)
METHODS = (
    (
        "temporal3",
        OLD_ROOT / "custom_tools/runs/bc/unified_student_taskid_temporal3_seed2025_e4_v1/epoch=003-step=5152.ckpt",
        ROOT / "custom_tools/configs/unified_student_taskid_temporal3_v1.yaml",
        None,
        0.0,
        40,
    ),
    (
        "chunk8_equal",
        ROOT / "custom_tools/runs/bc/restart_shadow_chunk8_demo80_v1/epoch=001-step=2576.ckpt",
        ROOT / "custom_tools/configs/unified_student_temporal_chunk8_demo80_v1.yaml",
        0.0,
        0.20,
        40,
    ),
)


def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-free-vram-mb", type=int, default=4500)
    parser.add_argument("--max-attempts", type=int, default=3)
    return parser.parse_args()


def load_result(path):
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)["checkpoint_results"][0]


def main():
    cli = parse_cli()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for label, checkpoint, config, decay, lift_boost, lift_start in METHODS:
        for seed in SEEDS:
            output = OUTPUT / "{}_seed{}.yaml".format(label, seed)
            if not output.exists():
                command = [
                    sys.executable, "-u",
                    str(ROOT / "custom_tools/evaluate_bc_checkpoints_isolated.py"),
                    "--checkpoint", str(checkpoint),
                    "--bc-config", str(config),
                    "--residual-config", str(ROOT / "custom_tools/configs/residual_ppo_stage1.yaml"),
                    "--trajectory-root", str(OLD_ROOT / "dexgrasp/dataset/scaled_category_final_v1_preprocessed"),
                    "--meshdata-root", str(OLD_ROOT / "assets/meshdata"),
                    "--object-selection", str(ROOT / "custom_tools/configs/scaled_final_holdout_all8.yaml"),
                    "--output", str(output),
                    "--policy-motion-steps", "70",
                    "--seed", str(seed),
                    "--min-free-vram-mb", str(cli.min_free_vram_mb),
                    "--max-attempts", str(cli.max_attempts),
                ]
                if decay is not None:
                    command.extend(["--temporal-ensemble-decay", str(decay)])
                if lift_boost:
                    command.extend([
                        "--late-lift-z-boost", str(lift_boost),
                        "--late-lift-start-step", str(lift_start),
                    ])
                subprocess.run(command, cwd=str(ROOT), check=True)
            result = load_result(output)
            rows.append({
                "method": label,
                "seed": seed,
                "official_rate": result["overall_official_peak_success_rate"],
                "stable_official_rate": result["overall_stable_official_success_rate"],
                "object_macro_stable_official_rate": result["macro_strict_terminal_success_rate"],
            })
    summary = {"rows": rows, "methods": {}}
    for label, _, _, _, _, _ in METHODS:
        selected = [row for row in rows if row["method"] == label]
        summary["methods"][label] = {}
        for key in ("official_rate", "stable_official_rate", "object_macro_stable_official_rate"):
            values = [row[key] for row in selected]
            summary["methods"][label][key] = {
                "mean": statistics.mean(values),
                "std": statistics.pstdev(values),
            }
    with (OUTPUT / "summary.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(summary, handle, allow_unicode=True, sort_keys=False)
    print(yaml.safe_dump(summary["methods"], sort_keys=False))


if __name__ == "__main__":
    main()
