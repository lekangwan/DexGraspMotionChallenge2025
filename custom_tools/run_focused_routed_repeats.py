"""Repeat the selected focused route against its pre-finetuning control."""

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "custom_tools/results/focused_routed_repeats_v1"
BC_CONFIG = ROOT / "custom_tools/configs/multicategory_bc_noise005.yaml"
RESIDUAL_CONFIG = ROOT / "custom_tools/configs/residual_ppo_stage1.yaml"

OBJECTS = {
    "bottle_dc005": {
        "control": ROOT / "custom_tools/runs/bc/category_expert_bottle_noise005_soup_seed2025_e40_v1/epoch=039-step=2560.ckpt",
        "candidate": ROOT / "custom_tools/runs/bc/focused_anchor_interpolation_v1/bottle_dc005_focused_w025.ckpt",
        "train_selection": ROOT / "custom_tools/configs/focused_bottle_dc005_train_all.yaml",
        "valid_selection": ROOT / "custom_tools/configs/focused_bottle_dc005_valid_all.yaml",
    },
    "mug_b4ae": {
        "control": ROOT / "custom_tools/runs/bc/model_soups/noise005_s2025_s2026_weighted2to1.ckpt",
        "candidate": ROOT / "custom_tools/runs/bc/focused_mug_b4ae_lr1e5_seed2025_e40_v1/epoch=019-step=320.ckpt",
        "train_selection": ROOT / "custom_tools/configs/focused_mug_b4ae_train_all.yaml",
        "valid_selection": ROOT / "custom_tools/configs/focused_mug_b4ae_valid_all.yaml",
    },
    "camera_82819": {
        "control": ROOT / "custom_tools/runs/bc/category_expert_camera_noise005_soup_seed2025_e40_v1/epoch=009-step=500.ckpt",
        "candidate": ROOT / "custom_tools/runs/bc/focused_camera_82819_lr1e5_seed2025_e40_v1/epoch=019-step=320.ckpt",
        "train_selection": ROOT / "custom_tools/configs/focused_camera_82819_train_all.yaml",
        "valid_selection": ROOT / "custom_tools/configs/focused_camera_82819_valid_all.yaml",
    },
}


def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--min-free-vram-mb", type=int, default=4500)
    return parser.parse_args()


def run(command):
    print("RUN: {}".format(" ".join(map(str, command))), flush=True)
    subprocess.run(list(map(str, command)), cwd=str(ROOT), check=True)


def main():
    cli = parse_cli()
    if cli.repeats < 1:
        raise ValueError("--repeats must be positive")
    for repeat in range(1, cli.repeats + 1):
        repeat_root = OUTPUT_ROOT / "repeat_{:02d}".format(repeat)
        repeat_root.mkdir(parents=True, exist_ok=True)
        for object_label, item in OBJECTS.items():
            for split in ("train", "valid"):
                output = repeat_root / "{}_{}.yaml".format(object_label, split)
                if output.is_file():
                    print("REUSE {}".format(output), flush=True)
                    continue
                run([
                    sys.executable, ROOT / "custom_tools/screen_bc_sweep_fresh.py",
                    "--bc-config", BC_CONFIG,
                    "--residual-config", RESIDUAL_CONFIG,
                    "--trajectory-root", ROOT / "dexgrasp/dataset/bc_multicategory_{}".format(split),
                    "--object-selection", item[split + "_selection"],
                    "--output", output,
                    "--seed", "2025",
                    "--min-free-vram-mb", cli.min_free_vram_mb,
                    "--checkpoint", "control={}".format(item["control"]),
                    "--checkpoint", "candidate={}".format(item["candidate"]),
                ])
    print("FOCUSED_ROUTED_REPEATS=COMPLETE", flush=True)
    print("RESULT_DIR={}".format(OUTPUT_ROOT), flush=True)


if __name__ == "__main__":
    main()
