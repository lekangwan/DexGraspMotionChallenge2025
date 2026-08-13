#!/usr/bin/env python3
"""按索引顺序运行或续跑三手策略训练矩阵。

输入：`config_index.json`、可选实验名过滤和设备。
输出：逐实验训练产物；已有last.pt时自动续训。
内部逻辑：一次只启动一个训练子进程，避免九个GPU任务同时抢显存。
作用：把大规模训练变成可中断恢复的一条命令，同时允许只跑指定手或模型。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


TRAIN_SCRIPT = Path(__file__).resolve().parent / "train.py"


def selected_names(index, filters):
    """按子串过滤并排序实验名。

    输入：配置索引和零个或多个过滤字符串。
    输出：所有过滤字符串都命中的实验名列表。
    内部逻辑：无过滤返回全部；有过滤时使用AND语义。
    作用：支持`--filter xhand --filter diffusion`精确运行一个分支。
    """
    return sorted(
        name
        for name in index
        if not filters or all(value in name for value in filters)
    )


def main():
    """顺序调用训练入口并安全续跑。

    输入：索引、过滤器和可选设备。
    输出：子进程日志；任一失败立即退出非零。
    内部逻辑：从配置读取output_dir，存在last.pt则追加`--resume`。
    作用：长训练在系统重启后执行同一命令即可从各自checkpoint继续。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--filter", action="append", default=[])
    parser.add_argument("--device")
    args = parser.parse_args()
    index = json.loads(args.index.read_text(encoding="utf-8"))
    names = selected_names(index, args.filter)
    if not names:
        raise ValueError("过滤后没有训练实验")
    for name in names:
        config_path = Path(index[name])
        config = json.loads(config_path.read_text(encoding="utf-8"))
        last = Path(config["output_dir"]) / "last.pt"
        command = [sys.executable, str(TRAIN_SCRIPT), "--config", str(config_path)]
        if last.is_file():
            command.extend(["--resume", str(last)])
        if args.device:
            command.extend(["--device", args.device])
        print(f"=== train {name} ===", flush=True)
        subprocess.run(command, check=True)
    print("TRAINING_MATRIX=COMPLETE")


if __name__ == "__main__":
    main()
