#!/usr/bin/env python3
"""训练目标手单帧BC、Temporal3或条件Diffusion策略。

输入：JSON配置、prepare生成的数据目录和可选断点路径。
输出：last/best checkpoint、逐epoch CSV、配置快照和训练曲线PNG。
内部逻辑：自动读取观测/动作/类别维度，按模型类型计算MSE或DDPM噪声损失，
使用验证loss选best并完整保存优化器/调度器/RNG状态以支持真正续训。
作用：为三只目标手提供不依赖Shadow专用DexRep代码的可复现离线训练入口。
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

try:
    from .dataset import TargetHandPolicyDataset
    from .models import (
        ConditionalDiffusionPolicy,
        MLPBCPolicy,
        Temporal3BCPolicy,
        linear_beta_schedule,
    )
except ImportError:
    from dataset import TargetHandPolicyDataset
    from models import (
        ConditionalDiffusionPolicy,
        MLPBCPolicy,
        Temporal3BCPolicy,
        linear_beta_schedule,
    )


def set_seed(seed):
    """固定Python、NumPy和PyTorch随机数。

    输入：整数seed。
    输出：无返回值。
    内部逻辑：同时设置CPU/CUDA种子并要求cuDNN确定性。
    作用：让同一数据和配置尽量得到可复现实验曲线。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_checkpoint_file(path, map_location):
    """加载本项目自己生成的完整训练checkpoint。

    输入：checkpoint路径和目标设备。
    输出：包含模型、优化器和RNG状态的字典。
    内部逻辑：显式声明需要完整pickle状态；旧PyTorch不支持该参数时兼容回退。
    作用：区分本地可信续训文件与只加载权重的外部模型，并消除未来默认值歧义。
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def build_model(config, observation_dim, action_dim, category_count):
    """按配置构造三种策略之一。

    输入：配置及数据自动发现的三个维度。
    输出：策略网络。
    内部逻辑：共享隐藏层/embedding/dropout配置，Diffusion另读取片段和时间维度。
    作用：把模型选择集中在一处，避免训练与推理实例化不一致。
    """
    common = {
        "observation_dim": observation_dim,
        "action_dim": action_dim,
        "category_count": category_count,
        "category_embedding_dim": config.get("category_embedding_dim", 16),
        "hidden_dims": tuple(config.get("hidden_dims", [256, 256, 256])),
        "dropout": config.get("dropout", 0.0),
    }
    model_type = config["model_type"]
    if model_type == "bc":
        return MLPBCPolicy(**common)
    if model_type == "temporal3":
        return Temporal3BCPolicy(**common)
    if model_type == "diffusion":
        return ConditionalDiffusionPolicy(
            **common,
            action_horizon=config.get("action_horizon", 8),
            observation_history=config.get("history", 3),
            time_embedding_dim=config.get("time_embedding_dim", 64),
        )
    raise ValueError(f"未知model_type: {model_type}")


def compute_loss(model, batch, model_type, device, betas=None):
    """计算一个batch的监督损失。

    输入：模型、batch、类型、设备和可选beta表。
    输出：标量loss及用于日志的未约简绝对误差代理。
    内部逻辑：BC/Temporal直接回归动作；Diffusion随机采样时间和噪声并预测噪声。
    作用：保持三个实验共享同一训练循环但数学目标清晰分离。
    """
    batch = {name: value.to(device) for name, value in batch.items()}
    if model_type == "bc":
        prediction = model(batch["observations"], batch["category_id"])
        target = batch["actions"]
    elif model_type == "temporal3":
        prediction = model(
            batch["observation_history"],
            batch["previous_actions"],
            batch["category_id"],
        )
        target = batch["actions"]
    else:
        if betas is None:
            raise ValueError("Diffusion训练缺少beta表")
        actions = batch["action_sequence"]
        timesteps = torch.randint(
            0, len(betas), (len(actions),), device=device
        )
        alpha_bars = torch.cumprod(1.0 - betas, dim=0)
        selected = alpha_bars[timesteps].view(-1, 1, 1)
        noise = torch.randn_like(actions)
        noisy = torch.sqrt(selected) * actions + torch.sqrt(1.0 - selected) * noise
        prediction = model(
            noisy,
            batch["observation_history"],
            batch["category_id"],
            timesteps,
        )
        target = noise
    loss = nn.functional.mse_loss(prediction, target)
    mae = nn.functional.l1_loss(prediction, target)
    return loss, mae


def run_epoch(
    model,
    loader,
    model_type,
    device,
    optimizer=None,
    betas=None,
    grad_clip=1.0,
):
    """训练或验证一个epoch。

    输入：模型、loader、类型、设备、可选优化器/beta和梯度裁剪。
    输出：按样本加权的MSE与MAE。
    内部逻辑：有optimizer时反向传播和裁剪；验证时使用no_grad且不更新状态。
    作用：保证训练/验证统计口径一致并正确处理最后一个小batch。
    """
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "mae": 0.0, "samples": 0}
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            if training:
                optimizer.zero_grad(set_to_none=True)
            loss, mae = compute_loss(model, batch, model_type, device, betas)
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
                optimizer.step()
            batch_size = len(next(iter(batch.values())))
            totals["loss"] += float(loss.item()) * batch_size
            totals["mae"] += float(mae.item()) * batch_size
            totals["samples"] += batch_size
    return {
        "loss": totals["loss"] / totals["samples"],
        "mae": totals["mae"] / totals["samples"],
    }


def checkpoint_payload(
    model,
    optimizer,
    scheduler,
    epoch,
    best_valid_loss,
    config,
    dimensions,
    stale_epochs,
    data_loader_generator,
):
    """构造可精确续训的checkpoint字典。

    输入：训练状态、epoch、最佳loss、配置、维度、早停计数和采样生成器。
    输出：可交给`torch.save`的字典。
    内部逻辑：包含模型/优化器/调度器和三套RNG状态。
    作用：长训练中断后从下一epoch继续，而不是只加载权重重新开始优化器。
    """
    return {
        "epoch": int(epoch),
        "best_valid_loss": float(best_valid_loss),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "config": config,
        "dimensions": dimensions,
        "stale_epochs": int(stale_epochs),
        "data_loader_generator_state": data_loader_generator.get_state(),
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
        "cuda_random_state": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
    }


def restore_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    device,
    expected_config,
    expected_dimensions,
    data_loader_generator,
):
    """恢复模型、优化状态和随机状态。

    输入：checkpoint路径、已构造对象、设备、当前配置/维度和DataLoader生成器。
    输出：下一epoch编号和历史最佳验证loss。
    内部逻辑：逐项load_state_dict并恢复Python/NumPy/Torch RNG。
    作用：保证`--resume`是真正续训而非只做warm start。
    """
    payload = load_checkpoint_file(path, device)
    if payload.get("config") != expected_config:
        raise ValueError("续训checkpoint的完整配置与当前JSON不同")
    if payload.get("dimensions") != expected_dimensions:
        raise ValueError("续训checkpoint的数据维度与当前数据集不同")
    model.load_state_dict(payload["model_state"])
    optimizer.load_state_dict(payload["optimizer_state"])
    scheduler.load_state_dict(payload["scheduler_state"])
    random.setstate(payload["python_random_state"])
    np.random.set_state(payload["numpy_random_state"])
    torch.set_rng_state(payload["torch_random_state"])
    if torch.cuda.is_available() and payload.get("cuda_random_state") is not None:
        torch.cuda.set_rng_state_all(payload["cuda_random_state"])
    if "data_loader_generator_state" not in payload:
        raise ValueError("旧checkpoint缺少DataLoader随机状态，不能作为严格续训输入")
    data_loader_generator.set_state(payload["data_loader_generator_state"])
    return (
        int(payload["epoch"]) + 1,
        float(payload["best_valid_loss"]),
        int(payload.get("stale_epochs", 0)),
    )


def write_metrics(path, rows):
    """把epoch指标原子式重写为CSV。

    输入：CSV路径和字典行列表。
    输出：无返回值。
    内部逻辑：先写同目录`.tmp`再替换正式文件。
    作用：避免训练中断留下半行CSV，并支持续训追加后统一重写。
    """
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def save_checkpoint(path, payload):
    """原子式保存训练checkpoint。

    输入：目标路径和checkpoint字典。
    输出：无返回值。
    内部逻辑：先写同目录临时文件，再用原子replace替换正式文件。
    作用：断电或中断最多丢失当前epoch，不会把上一份可续训checkpoint写成半个文件。
    """
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def plot_metrics(csv_path, output_path):
    """从CSV生成训练/验证loss曲线。

    输入：指标CSV与PNG输出路径。
    输出：成功时写PNG；matplotlib不可用时静默跳过。
    内部逻辑：使用Agg后端读取CSV两列并绘图。
    作用：直接满足进阶报告需要的训练loss曲线素材。
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    epochs = [int(row["epoch"]) for row in rows]
    figure, axis = plt.subplots(figsize=(6.0, 3.8))
    axis.plot(epochs, [float(row["train_loss"]) for row in rows], label="train")
    axis.plot(epochs, [float(row["valid_loss"]) for row in rows], label="valid")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Normalized loss")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main():
    """读取配置、训练并保存全部产物。

    输入：`--config`、可选`--resume`和设备覆盖。
    输出：配置快照、checkpoint、CSV、PNG和完成摘要。
    内部逻辑：数据维度自动推断，按验证loss早停；每epoch保存last，改善时保存best。
    作用：作为大规模策略训练的统一可续跑命令。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    set_seed(int(config.get("seed", 20260813)))
    device_name = args.device or config.get("device", "cuda")
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("配置要求CUDA，但当前PyTorch不可见GPU；请显式--device cpu做冒烟")
    device = torch.device(device_name)
    data_dir = Path(config["data_dir"]).expanduser().resolve()
    output_dir = Path(config["output_dir"]).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, output_dir / "config.json")
    mode = config["model_type"]
    history = int(config.get("history", 3))
    horizon = int(config.get("action_horizon", 8))
    train_set = TargetHandPolicyDataset(
        data_dir / "train.npz", data_dir / "normalization.npz", mode, history, horizon
    )
    valid_set = TargetHandPolicyDataset(
        data_dir / "valid.npz", data_dir / "normalization.npz", mode, history, horizon
    )
    mappings = json.loads((data_dir / "mappings.json").read_text(encoding="utf-8"))
    observation_dim = int(train_set.observations.shape[1])
    action_dim = int(train_set.actions.shape[1])
    category_count = len(mappings["category_to_id"])
    dimensions = {
        "observation_dim": observation_dim,
        "action_dim": action_dim,
        "category_count": category_count,
    }
    model = build_model(config, observation_dim, action_dim, category_count).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("learning_rate", 3e-4)),
        weight_decay=float(config.get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(config.get("epochs", 100)), eta_min=1e-6
    )
    betas = (
        linear_beta_schedule(config.get("diffusion_steps", 50)).to(device)
        if mode == "diffusion"
        else None
    )
    loader_args = {
        "batch_size": int(config.get("batch_size", 512)),
        "num_workers": int(config.get("num_workers", 4)),
        "pin_memory": device.type == "cuda",
    }
    generator = torch.Generator().manual_seed(int(config.get("seed", 20260813)))
    train_loader = DataLoader(
        train_set, shuffle=True, generator=generator, drop_last=False, **loader_args
    )
    valid_loader = DataLoader(valid_set, shuffle=False, drop_last=False, **loader_args)
    start_epoch, best_valid, stale_epochs = 1, float("inf"), 0
    metrics_path = output_dir / "metrics.csv"
    rows = list(csv.DictReader(metrics_path.open(encoding="utf-8"))) if metrics_path.exists() else []
    if args.resume is not None:
        start_epoch, best_valid, stale_epochs = restore_checkpoint(
            args.resume,
            model,
            optimizer,
            scheduler,
            device,
            config,
            dimensions,
            generator,
        )
        rows = [row for row in rows if int(row["epoch"]) < start_epoch]
    patience = int(config.get("early_stopping_patience", 20))
    started = time.perf_counter()
    for epoch in range(start_epoch, int(config.get("epochs", 100)) + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            mode,
            device,
            optimizer,
            betas,
            config.get("gradient_clip", 1.0),
        )
        valid_metrics = run_epoch(model, valid_loader, mode, device, None, betas)
        scheduler.step()
        improved = valid_metrics["loss"] < best_valid - float(
            config.get("minimum_improvement", 1e-6)
        )
        if improved:
            best_valid = valid_metrics["loss"]
            stale_epochs = 0
        else:
            stale_epochs += 1
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_mae": train_metrics["mae"],
            "valid_loss": valid_metrics["loss"],
            "valid_mae": valid_metrics["mae"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "elapsed_seconds": time.perf_counter() - started,
        }
        rows.append(row)
        write_metrics(metrics_path, rows)
        payload = checkpoint_payload(
            model,
            optimizer,
            scheduler,
            epoch,
            best_valid,
            config,
            dimensions,
            stale_epochs,
            generator,
        )
        save_checkpoint(output_dir / "last.pt", payload)
        if improved:
            save_checkpoint(output_dir / "best.pt", payload)
        print(
            f"epoch={epoch} train={train_metrics['loss']:.6f} "
            f"valid={valid_metrics['loss']:.6f} best={best_valid:.6f}",
            flush=True,
        )
        if stale_epochs >= patience:
            print(f"early_stop_patience={patience}")
            break
    if not rows:
        raise RuntimeError("没有可用于总结的epoch指标")
    plot_metrics(metrics_path, output_dir / "loss_curve.png")
    summary = {
        "status": "complete",
        "model_type": mode,
        "dimensions": dimensions,
        "best_valid_loss": best_valid,
        "last_epoch": int(rows[-1]["epoch"]),
        "wall_time_seconds": time.perf_counter() - started,
        "best_checkpoint": str((output_dir / "best.pt").resolve()),
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"TRAINING_COMPLETE={output_dir}")


if __name__ == "__main__":
    main()
