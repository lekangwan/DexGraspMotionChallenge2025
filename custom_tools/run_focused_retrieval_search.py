"""Run retrieval-imitation screening on train and held-out motions."""

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "dexgrasp/dataset/bc_multicategory_train"
VALID = ROOT / "dexgrasp/dataset/bc_multicategory_valid"
OUTPUT = ROOT / "custom_tools/results/focused_retrieval_search"

OBJECTS = {
    "bottle_dc005": (
        "core-bottle-dc005c019fbfb32c90071898148dca0e",
        ROOT / "custom_tools/runs/bc/category_expert_bottle_noise005_soup_seed2025_e40_v1/epoch=039-step=2560.ckpt"),
    "mug_b4ae": (
        "core-mug-b4ae56d6638d5338de671f28c83d2dcb",
        ROOT / "custom_tools/runs/bc/model_soups/noise005_s2025_s2026_weighted2to1.ckpt"),
    "camera_82819": (
        "core-camera-82819e1201d2dc583a3e53900c6cbba",
        ROOT / "custom_tools/runs/bc/category_expert_camera_noise005_soup_seed2025_e40_v1/epoch=009-step=500.ckpt"),
}


def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-free-vram-mb", type=int, default=4500)
    return parser.parse_args()


def main():
    cli = parse_cli()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for label, (object_id, checkpoint) in OBJECTS.items():
        for split, evaluation_root in (("train", TRAIN), ("valid", VALID)):
            output = OUTPUT / "{}_{}.yaml".format(label, split)
            if output.is_file():
                print("REUSE {}".format(output), flush=True)
                continue
            command = [
                sys.executable, ROOT / "custom_tools/evaluate_phase_retrieval_bc.py",
                "--object-id", object_id,
                "--reference-root", TRAIN,
                "--evaluation-root", evaluation_root,
                "--bc-checkpoint", checkpoint,
                "--output", output,
                "--seed", "2025",
                "--min-free-vram-mb", str(cli.min_free_vram_mb),
            ]
            print("RUN {} {}".format(label, split), flush=True)
            subprocess.run(list(map(str, command)), cwd=str(ROOT), check=True)
    print("FOCUSED_RETRIEVAL_SEARCH=COMPLETE", flush=True)
    print("RESULT_DIR={}".format(OUTPUT), flush=True)


if __name__ == "__main__":
    main()
