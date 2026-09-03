#!/usr/bin/env python3
"""把候选矩阵展开成训练脚本可直接读取的九个JSON。"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    root = Path(matrix["output_root"])
    for hand, data_dir in matrix["hands"].items():
        for model_type in matrix["models"]:
            config = dict(matrix["common"])
            config.update({
                "hand": hand,
                "model_type": model_type,
                "data_dir": data_dir,
                "output_dir": str(root / hand / model_type),
            })
            path = args.output_dir / f"{hand}_{model_type}.json"
            path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(path)


if __name__ == "__main__":
    main()

