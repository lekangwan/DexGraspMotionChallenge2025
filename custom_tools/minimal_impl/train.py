"""最终 Chunk8 策略的普通 PyTorch 训练入口。"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader

from .data import FrameDataset, balanced_sampler
from .model import Chunk8Policy, chunk_loss, load_project_checkpoint, save_checkpoint


@dataclass
class TrainConfig:
    """训练超参数输入，供训练循环读取；作用是集中记录最终配置。"""

    learning_rate: float = 2e-5
    batch_size: int = 128
    epochs: int = 4
    online_fraction: float = 0.25
    teacher_weight: float = 0.20
    demo_weight: float = 0.80
    seed: int = 2025


def train_policy(model: Chunk8Policy, dataset: FrameDataset,
                 config: TrainConfig, device: str = "cuda") -> Dict[str, float]:
    """输入模型、数据、配置和设备，输出最后一轮训练指标。

    内部执行类别/来源均衡采样，分别计算教师动作块和示范动作块损失，再按0.2/0.8
    加权反向传播；作用是复现最终Chunk8的核心监督训练。
    """
    torch.manual_seed(config.seed)
    model = model.to(device)
    loader = DataLoader(
        dataset, batch_size=config.batch_size,
        sampler=balanced_sampler(dataset, config.online_fraction, config.seed),
        drop_last=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    metrics = {}
    for epoch in range(config.epochs):
        model.train()
        totals = {"loss": 0.0, "demo": 0.0, "teacher": 0.0}
        steps = 0
        for batch in loader:
            observation = batch["observation"].to(device)
            prediction = model.forward_action_chunk(
                observation, batch["task_one_hot"].to(device), batch["history"].to(device))
            mask = batch["action_chunk_mask"].to(device)
            demo = chunk_loss(prediction, batch["demo_action_chunk"].to(device), mask)
            teacher = chunk_loss(prediction, batch["teacher_action_chunk"].to(device), mask)
            loss = config.demo_weight * demo + config.teacher_weight * teacher
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            totals["loss"] += float(loss.detach())
            totals["demo"] += float(demo.detach())
            totals["teacher"] += float(teacher.detach())
            steps += 1
        metrics = {name: value / steps for name, value in totals.items()}
        print("epoch={:02d} loss={:.6f} demo={:.6f} teacher={:.6f}".format(
            epoch + 1, metrics["loss"], metrics["demo"], metrics["teacher"]))
    return metrics


def parse_args() -> argparse.Namespace:
    """输入命令行参数，输出解析结果；作用是提供唯一、简洁的主线训练入口。"""
    parser = argparse.ArgumentParser(description="训练DexRep-Temporal3-Chunk8策略")
    parser.add_argument("--offline-dir", required=True)
    parser.add_argument("--teacher-actions", required=True)
    parser.add_argument("--online-actions", required=True)
    parser.add_argument("--init-checkpoint", required=True,
                        help="用于warm-start的Temporal3或Chunk8 checkpoint")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--online-fraction", type=float, default=0.25)
    parser.add_argument("--sequence-limit", type=int, default=1726)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    """输入来自命令行，输出训练checkpoint。

    内部加载真实离线/在线数据和初始化模型，执行训练并保存；作用是串起完整训练闭环。
    """
    cli = parse_args()
    paths = (cli.offline_dir, cli.teacher_actions, cli.online_actions, cli.init_checkpoint)
    missing = [path for path in paths if not Path(path).expanduser().exists()]
    if missing:
        raise FileNotFoundError(f"缺少训练输入：{missing}")
    if cli.dry_run:
        print("MINIMAL_TRAIN_DRY_RUN=READY")
        return
    from .project_data import build_training_dataset
    dataset = build_training_dataset(
        cli.offline_dir, cli.sequence_limit, cli.teacher_actions,
        cli.online_actions, history_steps=2)
    model = load_project_checkpoint(cli.init_checkpoint, device="cpu")
    config = TrainConfig(cli.learning_rate, cli.batch_size, cli.epochs,
                         cli.online_fraction)
    metrics = train_policy(model, dataset, config, cli.device)
    output = Path(cli.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(str(output), model, train_metrics=metrics, command=vars(cli))
    print(f"MINIMAL_TRAIN_COMPLETE={output}")


if __name__ == "__main__":
    main()
