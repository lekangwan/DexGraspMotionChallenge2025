#!/usr/bin/env python3
"""从一份矩阵配置生成三手×三模型的独立训练JSON。

输入：training matrix JSON和输出目录。
输出：9个可直接交给`advanced_policy/train.py`的配置文件及索引JSON。
内部逻辑：合并common、模型专有参数和手专有data/output路径，不运行训练。
作用：避免手工复制配置导致某只手使用不同学习率或错误数据目录。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def generate_configs(matrix):
    """展开训练矩阵。

    输入：含common/models/hands/root的字典。
    输出：实验名到完整配置的字典。
    内部逻辑：对手和模型做笛卡尔积，模型字段覆盖common字段。
    作用：形成可测试的纯配置生成逻辑。
    """
    result = {}
    for hand in matrix["hands"]:
        for model_type, model_values in matrix["models"].items():
            name = f"{hand}_{model_type}_v1"
            result[name] = {
                "experiment_name": name,
                "hand": hand,
                "model_type": model_type,
                "seed": int(matrix["seed"]),
                "device": matrix.get("device", "cuda"),
                **matrix["common"],
                **model_values,
                "data_dir": str(Path(matrix["data_root"]) / hand),
                "output_dir": str(Path(matrix["output_root"]) / name),
            }
    return result


def main():
    """解析矩阵、生成配置并打印数量。

    输入：`--matrix`和`--output-dir`。
    输出：独立JSON及`config_index.json`。
    内部逻辑：逐实验稳定排序写盘，并在索引中记录路径。
    作用：正式训练前一次生成全部命令输入，便于审计和冻结哈希。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
    configs = generate_configs(matrix)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    index = {}
    for name, config in sorted(configs.items()):
        path = args.output_dir / f"{name}.json"
        path.write_text(
            json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        index[name] = str(path.resolve())
    (args.output_dir / "config_index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"training_configs={len(configs)}")
    print("TRAINING_CONFIGS=READY")


if __name__ == "__main__":
    main()
