"""Screen the frozen non-temporal neural pipeline on identical focused splits."""

import argparse
from pathlib import Path
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "custom_tools/configs/focused_neural_comparison_v1.yaml"
OUTPUT = ROOT / "custom_tools/results/focused_neural_matrix_v1"
BC_CONFIG = ROOT / "custom_tools/configs/multicategory_bc_noise005.yaml"
RESIDUAL_CONFIG = ROOT / "custom_tools/configs/residual_ppo_stage1.yaml"


def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-free-vram-mb", type=int, default=4500)
    return parser.parse_args()


def resolved(path):
    return (ROOT / path).resolve()


def main():
    cli = parse_cli()
    with MANIFEST.open(encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle)
    if manifest.get("status") != "frozen_before_focused_neural_matrix":
        raise ValueError("Focused comparison manifest is not frozen")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    shared = manifest["shared_candidates"]
    for object_label, data in manifest["data"]["objects"].items():
        candidates = dict(shared)
        candidates.update(manifest["object_candidates"][object_label])
        for label, item in candidates.items():
            checkpoint = resolved(item["checkpoint"])
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
        for split in ("train", "valid"):
            output = OUTPUT / "{}_{}.yaml".format(object_label, split)
            if output.is_file():
                print("REUSE {}".format(output), flush=True)
                continue
            command = [
                sys.executable, ROOT / "custom_tools/screen_bc_sweep_fresh.py",
                "--bc-config", BC_CONFIG,
                "--residual-config", RESIDUAL_CONFIG,
                "--trajectory-root", resolved(
                    manifest["data"][split + "_root"]),
                "--object-selection", resolved(data[split + "_selection"]),
                "--output", output,
                "--seed", "2025",
                "--min-free-vram-mb", str(cli.min_free_vram_mb),
            ]
            for label, item in candidates.items():
                command.extend([
                    "--checkpoint", "{}={}".format(
                        label, resolved(item["checkpoint"]))])
            print("SCREEN object={} split={} candidates={}".format(
                object_label, split, len(candidates)), flush=True)
            subprocess.run(list(map(str, command)), cwd=str(ROOT), check=True)
    print("FOCUSED_NEURAL_MATRIX=COMPLETE", flush=True)
    print("RESULT_DIR={}".format(OUTPUT), flush=True)


if __name__ == "__main__":
    main()
