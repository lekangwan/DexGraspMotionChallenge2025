"""Interpolate focused specialists back toward their pre-finetuning policies."""

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "custom_tools/results/focused_anchor_interpolation_v1"
SOUP_ROOT = ROOT / "custom_tools/runs/bc/focused_anchor_interpolation_v1"
BC_CONFIG = ROOT / "custom_tools/configs/multicategory_bc_noise005.yaml"
RESIDUAL_CONFIG = ROOT / "custom_tools/configs/residual_ppo_stage1.yaml"

OBJECTS = {
    "bottle_dc005": {
        "initial": ROOT / "custom_tools/runs/bc/category_expert_bottle_noise005_soup_seed2025_e40_v1/epoch=039-step=2560.ckpt",
        "focused": ROOT / "custom_tools/runs/bc/focused_bottle_dc005_lr5e5_seed2025_e40_v1/epoch=019-step=340.ckpt",
        "train_selection": ROOT / "custom_tools/configs/focused_bottle_dc005_train_all.yaml",
        "valid_selection": ROOT / "custom_tools/configs/focused_bottle_dc005_valid_all.yaml",
    },
    "mug_b4ae": {
        "initial": ROOT / "custom_tools/runs/bc/model_soups/noise005_s2025_s2026_weighted2to1.ckpt",
        "focused": ROOT / "custom_tools/runs/bc/focused_mug_b4ae_lr1e5_seed2025_e40_v1/epoch=019-step=320.ckpt",
        "train_selection": ROOT / "custom_tools/configs/focused_mug_b4ae_train_all.yaml",
        "valid_selection": ROOT / "custom_tools/configs/focused_mug_b4ae_valid_all.yaml",
    },
    "camera_82819": {
        "initial": ROOT / "custom_tools/runs/bc/category_expert_camera_noise005_soup_seed2025_e40_v1/epoch=009-step=500.ckpt",
        "focused": ROOT / "custom_tools/runs/bc/focused_camera_82819_lr1e5_seed2025_e40_v1/epoch=019-step=320.ckpt",
        "train_selection": ROOT / "custom_tools/configs/focused_camera_82819_train_all.yaml",
        "valid_selection": ROOT / "custom_tools/configs/focused_camera_82819_valid_all.yaml",
    },
}

FOCUSED_WEIGHTS = (0.25, 0.50, 0.75)


def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-free-vram-mb", type=int, default=4500)
    return parser.parse_args()


def run(command):
    print("RUN: {}".format(" ".join(map(str, command))), flush=True)
    subprocess.run(list(map(str, command)), cwd=str(ROOT), check=True)


def main():
    cli = parse_cli()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    SOUP_ROOT.mkdir(parents=True, exist_ok=True)
    for object_label, item in OBJECTS.items():
        candidates = {
            "initial_w000": item["initial"],
            "focused_w100": item["focused"],
        }
        for focused_weight in FOCUSED_WEIGHTS:
            percentage = int(round(focused_weight * 100))
            output = SOUP_ROOT / "{}_focused_w{:03d}.ckpt".format(
                object_label, percentage)
            if not output.is_file():
                run([
                    sys.executable, ROOT / "custom_tools/make_bc_model_soup.py",
                    "--ingredient", item["initial"],
                    "--ingredient", item["focused"],
                    "--weight", 1.0 - focused_weight,
                    "--weight", focused_weight,
                    "--output", output,
                ])
            candidates["focused_w{:03d}".format(percentage)] = output

        for split in ("train", "valid"):
            output = OUTPUT_ROOT / "{}_{}.yaml".format(object_label, split)
            if output.is_file():
                print("REUSE SCREEN: {}".format(output), flush=True)
                continue
            command = [
                sys.executable, ROOT / "custom_tools/screen_bc_sweep_fresh.py",
                "--bc-config", BC_CONFIG,
                "--residual-config", RESIDUAL_CONFIG,
                "--trajectory-root", ROOT / "dexgrasp/dataset/bc_multicategory_{}".format(split),
                "--object-selection", item[split + "_selection"],
                "--output", output,
                "--seed", "2025",
                "--min-free-vram-mb", cli.min_free_vram_mb,
            ]
            for label, checkpoint in candidates.items():
                command.extend([
                    "--checkpoint", "{}={}".format(label, checkpoint)])
            run(command)

    print("FOCUSED_ANCHOR_INTERPOLATION=COMPLETE", flush=True)
    print("RESULT_DIR={}".format(OUTPUT_ROOT), flush=True)


if __name__ == "__main__":
    main()
