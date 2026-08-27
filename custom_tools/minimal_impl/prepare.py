"""训练前的两个必要工具：BC Soup和四教师离线路由标签。"""

import argparse
from pathlib import Path

import numpy as np
import torch

from .model import (
    CATEGORIES, load_project_checkpoint, save_checkpoint, weighted_model_soup)
from .project_data import load_offline_trajectories


def parse_teacher(values):
    """输入多条``类别=checkpoint``字符串，输出四类到路径的字典。

    内部检查类别是否完整且不重复；作用是防止教师标签路由到错误网络。
    """
    result = {}
    for value in values:
        category, path = value.split("=", 1)
        if category not in CATEGORIES or category in result:
            raise ValueError(f"教师类别无效或重复：{category}")
        result[category] = path
    if set(result) != set(CATEGORIES):
        raise ValueError("必须提供四个类别教师")
    return result


def soup(cli):
    """输入命令行中的同结构checkpoint和权重，无返回值并写出Soup权重。

    内部逐参数加权平均；作用是用一个模型吸收多个BC checkpoint的参数。
    """
    models = [load_project_checkpoint(
        path, use_task_id=False, history_steps=0, chunk_horizon=1)
        for path in cli.checkpoint]
    weights = cli.weight or [1.0] * len(models)
    state = weighted_model_soup([model.state_dict() for model in models], weights)
    result = models[0]
    result.load_state_dict(state)
    output = Path(cli.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(str(output), result, soup_inputs=cli.checkpoint, soup_weights=weights)
    print(f"LEAN_SOUP_COMPLETE={output}")


@torch.no_grad()
def labels(cli):
    """输入离线轨迹和四类教师，无返回值并写出逐帧教师动作NPZ。

    内部按物体类别选择对应教师批量推理；作用是准备0.20权重的蒸馏监督。
    """
    teachers = {
        category: load_project_checkpoint(
            path, use_task_id=False, history_steps=0,
            chunk_horizon=1, device=cli.device)
        for category, path in parse_teacher(cli.teacher).items()
    }
    data = load_offline_trajectories(cli.offline_dir, cli.sequence_limit)
    observations = data.observations.reshape(-1, data.observations.shape[-1])
    frame_categories = data.category_indices[:, None].repeat(
        1, data.observations.shape[1]).reshape(-1)
    actions = torch.empty(len(observations), 28)
    for start in range(0, len(observations), cli.batch_size):
        stop = min(start + cli.batch_size, len(observations))
        batch_obs = observations[start:stop].to(cli.device)
        batch_category = frame_categories[start:stop]
        for index, category in enumerate(CATEGORIES):
            mask = batch_category == index
            if torch.any(mask):
                actions[start:stop][mask] = teachers[category](
                    batch_obs[mask.to(cli.device)]).cpu()
    output = Path(cli.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(output), teacher_actions=actions.numpy(),
        teacher_categories=np.asarray(CATEGORIES),
        teacher_checkpoints=np.asarray([parse_teacher(cli.teacher)[c] for c in CATEGORIES]),
    )
    print(f"LEAN_TEACHER_LABELS_COMPLETE={output}")


def main():
    """输入命令行，输出Soup或教师标签文件；作用是统一两个训练前准备入口。"""
    parser = argparse.ArgumentParser(description="精简版训练准备工具")
    subparsers = parser.add_subparsers(dest="command", required=True)
    soup_parser = subparsers.add_parser("soup")
    soup_parser.add_argument("--checkpoint", action="append", required=True)
    soup_parser.add_argument("--weight", action="append", type=float, default=[])
    soup_parser.add_argument("--output", required=True)
    label_parser = subparsers.add_parser("labels")
    label_parser.add_argument("--offline-dir", required=True)
    label_parser.add_argument("--sequence-limit", type=int, default=1726)
    label_parser.add_argument("--teacher", action="append", required=True)
    label_parser.add_argument("--batch-size", type=int, default=512)
    label_parser.add_argument("--device", default="cuda")
    label_parser.add_argument("--output", required=True)
    cli = parser.parse_args()
    if cli.command == "soup":
        soup(cli)
    else:
        labels(cli)


if __name__ == "__main__":
    main()
