"""Render report cases with the locked category-routed BC teacher pool."""

import argparse
from pathlib import Path
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", required=True)
    parser.add_argument("--lock", required=True)
    parser.add_argument("--trajectory-root", required=True)
    parser.add_argument("--residual-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--capture-stride", type=int, default=2)
    parser.add_argument("--min-free-vram-mb", type=int, default=4500)
    cli = parser.parse_args()

    with Path(cli.cases).open(encoding="utf-8") as handle:
        cases = yaml.safe_load(handle)["cases"]
    with Path(cli.lock).open(encoding="utf-8") as handle:
        lock = yaml.safe_load(handle)
    if lock.get("status") != "locked_before_final_unseen_v2_evaluation":
        raise ValueError("Final lock is invalid")
    teachers = {category: (ROOT / checkpoint).resolve()
                for category, checkpoint in lock["routed_teacher_pool"].items()}
    for path in teachers.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    output = Path(cli.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    for number, case in enumerate(cases, 1):
        tag = "{:02d}_{}_{}_{}".format(
            number, case["category"], case["outcome"],
            case["trajectory_index"])
        result = output / (tag + ".yaml")
        if result.exists():
            print("reuse render case: {}".format(tag), flush=True)
            continue
        command = [
            sys.executable,
            str(ROOT / "custom_tools/evaluate_residual_ppo.py"),
            "--object-id", case["object_id"],
            "--trajectory-root", str(Path(cli.trajectory_root).resolve()),
            "--trajectory-indices", str(case["trajectory_index"]),
            "--residual-config", str(Path(cli.residual_config).resolve()),
            "--bc-checkpoint", str(teachers[case["category"]]),
            "--zero-residual",
            "--capture-dir", str(output / tag),
            "--capture-stride", str(cli.capture_stride),
            "--min-free-vram-mb", str(cli.min_free_vram_mb),
            "--output", str(result),
        ]
        print("render case {}/{}: {}".format(
            number, len(cases), tag), flush=True)
        subprocess.run(command, cwd=str(ROOT), check=True)
    print("ROUTED_REPORT_RENDERS=COMPLETE output={}".format(output), flush=True)


if __name__ == "__main__":
    main()
