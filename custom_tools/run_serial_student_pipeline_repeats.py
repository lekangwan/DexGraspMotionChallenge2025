"""Repeat every checkpoint in the fixed serial teacher-student pipeline."""

import argparse
from pathlib import Path
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "custom_tools/configs/serial_student_pipeline_v1.yaml"
OUTPUT_ROOT = ROOT / "custom_tools/results/serial_student_pipeline_v1"
BC_CONFIG = ROOT / "custom_tools/configs/multicategory_bc_noise005.yaml"
RESIDUAL_CONFIG = ROOT / "custom_tools/configs/residual_ppo_stage1.yaml"


def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--min-free-vram-mb", type=int, default=4500)
    return parser.parse_args()


def resolved(path):
    return (ROOT / path).resolve()


def run(command):
    print("RUN: {}".format(" ".join(map(str, command))), flush=True)
    subprocess.run(list(map(str, command)), cwd=str(ROOT), check=True)


def main():
    cli = parse_cli()
    if cli.repeats < 1:
        raise ValueError("--repeats must be positive")
    with MANIFEST.open(encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle)
    if manifest.get("status") != "frozen_before_serial_repeat":
        raise ValueError("Serial pipeline manifest is not frozen")

    shared = {
        label: resolved(item["checkpoint"])
        for label, item in manifest["nodes"].items()
        if item["checkpoint"] != "per_object"
    }
    for repeat in range(1, cli.repeats + 1):
        repeat_root = OUTPUT_ROOT / "repeat_{:02d}".format(repeat)
        repeat_root.mkdir(parents=True, exist_ok=True)
        for object_label, item in manifest["data"]["objects"].items():
            candidates = dict(shared)
            candidates["routed_teacher"] = resolved(item["routed_teacher"])
            for checkpoint in candidates.values():
                if not checkpoint.is_file():
                    raise FileNotFoundError(checkpoint)
            for split in ("train", "valid"):
                output = repeat_root / "{}_{}.yaml".format(object_label, split)
                if output.is_file():
                    print("REUSE {}".format(output), flush=True)
                    continue
                command = [
                    sys.executable, ROOT / "custom_tools/screen_bc_sweep_fresh.py",
                    "--bc-config", BC_CONFIG,
                    "--residual-config", RESIDUAL_CONFIG,
                    "--trajectory-root", resolved(manifest["data"][split + "_root"]),
                    "--object-selection", resolved(item[split + "_selection"]),
                    "--output", output,
                    "--seed", "2025",
                    "--min-free-vram-mb", cli.min_free_vram_mb,
                ]
                for label in (
                        "base_soup", "routed_teacher", "offline_student",
                        "online_student_r1"):
                    command.extend([
                        "--checkpoint", "{}={}".format(label, candidates[label])])
                run(command)
    print("SERIAL_STUDENT_PIPELINE_REPEATS=COMPLETE", flush=True)
    print("RESULT_DIR={}".format(OUTPUT_ROOT), flush=True)


if __name__ == "__main__":
    main()
