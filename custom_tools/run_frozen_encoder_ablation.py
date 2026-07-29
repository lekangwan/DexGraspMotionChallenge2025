"""Train and evaluate a controlled true-encoder-freeze ablation on 3 objects."""

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "custom_tools/runs/bc"
OUTPUT_ROOT = ROOT / "custom_tools/results/frozen_encoder_ablation_v1"
BC_CONFIG = ROOT / "custom_tools/configs/multicategory_bc_noise005.yaml"
RESIDUAL_CONFIG = ROOT / "custom_tools/configs/residual_ppo_stage1.yaml"

OBJECTS = {
    "bottle_dc005": {
        "config": ROOT / "custom_tools/configs/focused_specialist_bottle_dc005.yaml",
        "lr": "0.00005",
        "init": ROOT / "custom_tools/runs/bc/category_expert_bottle_noise005_soup_seed2025_e40_v1/epoch=039-step=2560.ckpt",
        "category": ROOT / "custom_tools/runs/bc/category_expert_bottle_noise005_soup_seed2025_e40_v1/epoch=039-step=2560.ckpt",
        "unfrozen": ROOT / "custom_tools/runs/bc/focused_bottle_dc005_lr5e5_seed2025_e40_v1/epoch=019-step=340.ckpt",
        "train_selection": ROOT / "custom_tools/configs/focused_bottle_dc005_train_all.yaml",
        "valid_selection": ROOT / "custom_tools/configs/focused_bottle_dc005_valid_all.yaml",
    },
    "mug_b4ae": {
        "config": ROOT / "custom_tools/configs/focused_specialist_mug_b4ae.yaml",
        "lr": "0.00001",
        "init": ROOT / "custom_tools/runs/bc/model_soups/noise005_s2025_s2026_weighted2to1.ckpt",
        "category": ROOT / "custom_tools/runs/bc/category_expert_mug_noise005_soup_seed2025_e40_v1/epoch=009-step=570.ckpt",
        "unfrozen": ROOT / "custom_tools/runs/bc/focused_mug_b4ae_lr1e5_seed2025_e40_v1/epoch=019-step=320.ckpt",
        "train_selection": ROOT / "custom_tools/configs/focused_mug_b4ae_train_all.yaml",
        "valid_selection": ROOT / "custom_tools/configs/focused_mug_b4ae_valid_all.yaml",
    },
    "camera_82819": {
        "config": ROOT / "custom_tools/configs/focused_specialist_camera_82819.yaml",
        "lr": "0.00001",
        "init": ROOT / "custom_tools/runs/bc/category_expert_camera_noise005_soup_seed2025_e40_v1/epoch=009-step=500.ckpt",
        "category": ROOT / "custom_tools/runs/bc/category_expert_camera_noise005_soup_seed2025_e40_v1/epoch=009-step=500.ckpt",
        "unfrozen": ROOT / "custom_tools/runs/bc/focused_camera_82819_lr1e5_seed2025_e40_v1/epoch=019-step=320.ckpt",
        "train_selection": ROOT / "custom_tools/configs/focused_camera_82819_train_all.yaml",
        "valid_selection": ROOT / "custom_tools/configs/focused_camera_82819_valid_all.yaml",
    },
}


def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-free-vram-mb", type=int, default=4500)
    return parser.parse_args()


def run(command):
    print("RUN: {}".format(" ".join(map(str, command))), flush=True)
    subprocess.run(list(map(str, command)), cwd=str(ROOT), check=True)


def one_checkpoint(run_dir, epoch):
    matches = sorted(run_dir.glob("epoch={:03d}-step=*.ckpt".format(epoch - 1)))
    if len(matches) != 1:
        raise RuntimeError(
            "Expected one epoch-{} checkpoint in {}, found {}".format(
                epoch, run_dir, matches))
    return matches[0]


def main():
    cli = parse_cli()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for label, item in OBJECTS.items():
        run_name = "focused_{}_truefreeze_seed2025_e20_v1".format(label)
        run_dir = RUN_ROOT / run_name
        if not list(run_dir.glob("epoch=019-step=*.ckpt")):
            run([
                sys.executable, ROOT / "custom_tools/train_bc.py",
                "--config", item["config"],
                "--run-name", run_name,
                "--num-epochs", "20",
                "--learning-rate", item["lr"],
                "--init-checkpoint", item["init"],
                "--freeze-feature-encoder",
                "--min-free-vram-mb", cli.min_free_vram_mb,
            ])
        else:
            print("REUSE TRAINING: {}".format(run_dir), flush=True)

        candidates = {
            "initial_policy": item["init"],
            "unfrozen_focused_e20": item["unfrozen"],
            "truefreeze_e10": one_checkpoint(run_dir, 10),
            "truefreeze_e20": one_checkpoint(run_dir, 20),
        }
        if item["category"] != item["init"]:
            candidates["category_expert"] = item["category"]
        for split in ("train", "valid"):
            output = OUTPUT_ROOT / "{}_{}.yaml".format(label, split)
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
            for candidate_label, checkpoint in candidates.items():
                command.extend([
                    "--checkpoint", "{}={}".format(candidate_label, checkpoint)])
            run(command)

    print("FROZEN_ENCODER_ABLATION=COMPLETE", flush=True)
    print("RESULT_DIR={}".format(OUTPUT_ROOT), flush=True)


if __name__ == "__main__":
    main()
