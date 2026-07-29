"""Compare focused specialists with their initial models on held-out motions."""

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "custom_tools/results/focused_specialist_validation"
BC_CONFIG = ROOT / "custom_tools/configs/multicategory_bc_noise005.yaml"
RESIDUAL_CONFIG = ROOT / "custom_tools/configs/residual_ppo_stage1.yaml"
VALID_ROOT = ROOT / "dexgrasp/dataset/bc_multicategory_valid"

OBJECTS = {
    "bottle_dc005": {
        "selection": ROOT / "custom_tools/configs/focused_bottle_dc005_valid_all.yaml",
        "baseline": ROOT / "custom_tools/runs/bc/category_expert_bottle_noise005_soup_seed2025_e40_v1/epoch=039-step=2560.ckpt",
        "candidate": ROOT / "custom_tools/runs/bc/focused_bottle_dc005_lr5e5_seed2025_e40_v1/epoch=019-step=340.ckpt",
    },
    "mug_b4ae": {
        "selection": ROOT / "custom_tools/configs/focused_mug_b4ae_valid_all.yaml",
        "baseline": ROOT / "custom_tools/runs/bc/model_soups/noise005_s2025_s2026_weighted2to1.ckpt",
        "candidate": ROOT / "custom_tools/runs/bc/focused_mug_b4ae_lr1e5_seed2025_e40_v1/epoch=019-step=320.ckpt",
    },
    "camera_82819": {
        "selection": ROOT / "custom_tools/configs/focused_camera_82819_valid_all.yaml",
        "baseline": ROOT / "custom_tools/runs/bc/category_expert_camera_noise005_soup_seed2025_e40_v1/epoch=009-step=500.ckpt",
        "candidate": ROOT / "custom_tools/runs/bc/focused_camera_82819_lr1e5_seed2025_e40_v1/epoch=019-step=320.ckpt",
    },
}


def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-free-vram-mb", type=int, default=4500)
    return parser.parse_args()


def main():
    cli = parse_cli()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for label, item in OBJECTS.items():
        output = OUTPUT_ROOT / (label + ".yaml")
        if output.is_file():
            print("REUSE {}".format(output), flush=True)
            continue
        for key in ("baseline", "candidate"):
            if not item[key].is_file():
                raise FileNotFoundError(item[key])
        command = [
            sys.executable, ROOT / "custom_tools/screen_bc_sweep_fresh.py",
            "--checkpoint", "baseline={}".format(item["baseline"]),
            "--checkpoint", "candidate={}".format(item["candidate"]),
            "--bc-config", BC_CONFIG,
            "--residual-config", RESIDUAL_CONFIG,
            "--trajectory-root", VALID_ROOT,
            "--object-selection", item["selection"],
            "--output", output,
            "--seed", "2025",
            "--min-free-vram-mb", str(cli.min_free_vram_mb),
        ]
        print("VALIDATE {}".format(label), flush=True)
        subprocess.run(list(map(str, command)), cwd=str(ROOT), check=True)
    print("FOCUSED_SPECIALIST_VALIDATION=COMPLETE", flush=True)
    print("RESULT_DIR={}".format(OUTPUT_ROOT), flush=True)


if __name__ == "__main__":
    main()
