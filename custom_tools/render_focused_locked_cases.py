"""Render held-out successes and failures for the locked focused policies."""

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "dexgrasp/dataset/bc_multicategory_train"
VALID = ROOT / "dexgrasp/dataset/bc_multicategory_valid"
OUTPUT = ROOT / "custom_tools/results/focused_locked_renders"

CASES = (
    ("bottle_success", "core-bottle-dc005c019fbfb32c90071898148dca0e", "bc", 2,
     ROOT / "custom_tools/runs/bc/category_expert_bottle_noise005_soup_seed2025_e40_v1/epoch=039-step=2560.ckpt"),
    ("bottle_failure", "core-bottle-dc005c019fbfb32c90071898148dca0e", "bc", 0,
     ROOT / "custom_tools/runs/bc/category_expert_bottle_noise005_soup_seed2025_e40_v1/epoch=039-step=2560.ckpt"),
    ("mug_success", "core-mug-b4ae56d6638d5338de671f28c83d2dcb", "bc", 1,
     ROOT / "custom_tools/runs/bc/model_soups/noise005_s2025_s2026_weighted2to1.ckpt"),
    ("mug_failure", "core-mug-b4ae56d6638d5338de671f28c83d2dcb", "bc", 0,
     ROOT / "custom_tools/runs/bc/model_soups/noise005_s2025_s2026_weighted2to1.ckpt"),
    ("camera_success", "core-camera-82819e1201d2dc583a3e53900c6cbba", "framewise_k1", 0,
     ROOT / "custom_tools/runs/bc/category_expert_camera_noise005_soup_seed2025_e40_v1/epoch=009-step=500.ckpt"),
    ("camera_failure", "core-camera-82819e1201d2dc583a3e53900c6cbba", "framewise_k1", 6,
     ROOT / "custom_tools/runs/bc/category_expert_camera_noise005_soup_seed2025_e40_v1/epoch=009-step=500.ckpt"),
)


def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-stride", type=int, default=2)
    parser.add_argument("--min-free-vram-mb", type=int, default=4500)
    return parser.parse_args()


def main():
    cli = parse_cli()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for label, object_id, method, env_index, checkpoint in CASES:
        result = OUTPUT / (label + ".yaml")
        if result.is_file():
            print("REUSE {}".format(label), flush=True)
            continue
        capture = OUTPUT / label
        command = [
            sys.executable, ROOT / "custom_tools/evaluate_phase_retrieval_bc.py",
            "--object-id", object_id,
            "--reference-root", TRAIN,
            "--evaluation-root", VALID,
            "--bc-checkpoint", checkpoint,
            "--candidate-profile", "paired",
            "--only-candidate", method,
            "--capture-dir", capture,
            "--capture-env", str(env_index),
            "--capture-stride", str(cli.capture_stride),
            "--output", result,
            "--seed", "2025",
            "--min-free-vram-mb", str(cli.min_free_vram_mb),
        ]
        print("RENDER {} method={} env={}".format(
            label, method, env_index), flush=True)
        subprocess.run(list(map(str, command)), cwd=str(ROOT), check=True)
    print("FOCUSED_LOCKED_RENDERS=COMPLETE", flush=True)
    print("OUTPUT={}".format(OUTPUT), flush=True)


if __name__ == "__main__":
    main()
