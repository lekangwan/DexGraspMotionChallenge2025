"""Freeze development and final subsets before any scaled policy is trained."""

import argparse
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", default=str(
            REPO_ROOT / "custom_tools/configs/scaled_category_split_final_v1.json"))
    parser.add_argument(
        "--output", default=str(
            REPO_ROOT / "custom_tools/configs/scaled_evaluation_protocol_v1.json"))
    return parser.parse_args()


def main():
    args = parse_args()
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite a frozen evaluation protocol")
    manifest_path = Path(args.manifest).expanduser().resolve()
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    categories = {}
    for category, split in manifest["categories"].items():
        test = list(split["test"])
        if len(test) != 5:
            raise RuntimeError("expected five test objects per category")
        # Candidate order was geometry-center first, then farthest-point
        # expansion.  Keeping the first two gives the final set one central
        # and one strongly different object; the other three form development.
        categories[category] = {
            "final_holdout": test[:2],
            "development": test[2:],
        }
    protocol = {
        "status": "frozen_before_scaled_training",
        "source_manifest": str(manifest_path),
        "selection_rule": (
            "Within each policy-blind five-object test list, reserve the first "
            "geometry-center/farthest pair as final holdout and use the remaining "
            "three for development. No scaled policy existed at freeze time."),
        "categories": categories,
        "counts": {"development": 12, "final_holdout": 8},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(protocol, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print("[PASS] development objects=12; final holdout objects=8")
    print("SCALED_EVALUATION_PROTOCOL=FROZEN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
