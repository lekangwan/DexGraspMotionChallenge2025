"""Train and screen two learning rates for three focused-object specialists."""

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "custom_tools/runs/bc"
OUTPUT_ROOT = ROOT / "custom_tools/results/focused_specialist_search"
BC_CONFIG = ROOT / "custom_tools/configs/multicategory_bc_noise005.yaml"
RESIDUAL_CONFIG = ROOT / "custom_tools/configs/residual_ppo_stage1.yaml"
TRAIN_ROOT = ROOT / "dexgrasp/dataset/bc_multicategory_train"

OBJECTS = {
    "bottle_dc005": {
        "config": ROOT / "custom_tools/configs/focused_specialist_bottle_dc005.yaml",
        "selection": ROOT / "custom_tools/configs/focused_bottle_dc005_train_all.yaml",
        "init": ROOT / "custom_tools/runs/bc/category_expert_bottle_noise005_soup_seed2025_e40_v1/epoch=039-step=2560.ckpt",
    },
    "mug_b4ae": {
        "config": ROOT / "custom_tools/configs/focused_specialist_mug_b4ae.yaml",
        "selection": ROOT / "custom_tools/configs/focused_mug_b4ae_train_all.yaml",
        "init": ROOT / "custom_tools/runs/bc/model_soups/noise005_s2025_s2026_weighted2to1.ckpt",
    },
    "camera_82819": {
        "config": ROOT / "custom_tools/configs/focused_specialist_camera_82819.yaml",
        "selection": ROOT / "custom_tools/configs/focused_camera_82819_train_all.yaml",
        "init": ROOT / "custom_tools/runs/bc/category_expert_camera_noise005_soup_seed2025_e40_v1/epoch=009-step=500.ckpt",
    },
}

LEARNING_RATES = {
    "lr1e5": "0.00001",
    "lr5e5": "0.00005",
}


def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-free-vram-mb", type=int, default=4500)
    return parser.parse_args()


def epoch_40_checkpoint(run_dir):
    matches = sorted(run_dir.glob("epoch=039-step=*.ckpt"))
    if len(matches) == 1:
        return matches[0]
    if matches:
        raise RuntimeError("Ambiguous epoch-40 checkpoints: {}".format(matches))
    return None


def run(command):
    print("RUN: {}".format(" ".join(map(str, command))), flush=True)
    subprocess.run(list(map(str, command)), cwd=str(ROOT), check=True)


def main():
    cli = parse_cli()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    for object_label, item in OBJECTS.items():
        if not item["init"].is_file():
            raise FileNotFoundError(item["init"])
        for rate_label, learning_rate in LEARNING_RATES.items():
            run_name = "focused_{}_{}_seed2025_e40_v1".format(
                object_label, rate_label)
            run_dir = RUN_ROOT / run_name
            if epoch_40_checkpoint(run_dir) is None:
                run([
                    sys.executable, ROOT / "custom_tools/train_bc.py",
                    "--config", item["config"],
                    "--run-name", run_name,
                    "--learning-rate", learning_rate,
                    "--init-checkpoint", item["init"],
                    "--min-free-vram-mb", cli.min_free_vram_mb,
                ])
            else:
                print("REUSE TRAINING: {}".format(run_dir), flush=True)

    for object_label, item in OBJECTS.items():
        output = OUTPUT_ROOT / (object_label + "_train_screen.yaml")
        if output.is_file():
            print("REUSE SCREEN: {}".format(output), flush=True)
            continue
        command = [
            sys.executable, ROOT / "custom_tools/screen_bc_sweep_fresh.py",
            "--epochs", "10,20,40",
            "--bc-config", BC_CONFIG,
            "--residual-config", RESIDUAL_CONFIG,
            "--trajectory-root", TRAIN_ROOT,
            "--object-selection", item["selection"],
            "--output", output,
            "--seed", "2025",
            "--min-free-vram-mb", cli.min_free_vram_mb,
        ]
        for rate_label in LEARNING_RATES:
            run_name = "focused_{}_{}_seed2025_e40_v1".format(
                object_label, rate_label)
            command.extend([
                "--run", "{}={}".format(rate_label, RUN_ROOT / run_name)])
        run(command)

    print("FOCUSED_SPECIALIST_SEARCH=COMPLETE", flush=True)
    print("RESULT_DIR={}".format(OUTPUT_ROOT), flush=True)


if __name__ == "__main__":
    main()
