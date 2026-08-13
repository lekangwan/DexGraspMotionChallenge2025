"""目标手策略网络：单帧BC、Temporal3和条件动作Diffusion。

输入：标准化状态观测、历史动作、类别ID及可选带噪动作片段。
输出：标准化目标位置动作，或Diffusion预测的动作噪声。
内部逻辑：三种模型共享小型MLP积木；类别embedding提供类别条件，Temporal3显式拼接
三帧，Diffusion用正弦时间编码和DDPM噪声回归学习多模态动作片段。
作用：在不依赖旧Shadow专用ActorCritic/DexRep维度的情况下，为12/18/26维目标手提供基线。
"""

from __future__ import annotations

import math

import torch
from torch import nn


def make_mlp(input_dim, output_dim, hidden_dims, dropout=0.0):
    """构造带LayerNorm和SiLU的多层感知机。

    输入：输入/输出维度、隐藏层列表和dropout。
    输出：`nn.Sequential`网络。
    内部逻辑：每层线性映射后接LayerNorm、SiLU和可选Dropout，末层不激活。
    作用：为三类策略提供数值稳定且配置统一的基础模块。
    """
    layers = []
    current = int(input_dim)
    for width in hidden_dims:
        layers.extend([nn.Linear(current, width), nn.LayerNorm(width), nn.SiLU()])
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        current = width
    layers.append(nn.Linear(current, int(output_dim)))
    return nn.Sequential(*layers)


class CategoryConditioner(nn.Module):
    """把类别ID转换为可学习向量。"""

    def __init__(self, category_count, embedding_dim):
        """输入类别数和embedding维度，创建嵌入表。"""
        super().__init__()
        self.embedding = nn.Embedding(int(category_count), int(embedding_dim))

    def forward(self, category_id):
        """输入`(B,)`类别ID，输出`(B,E)`向量。"""
        return self.embedding(category_id.long())


class MLPBCPolicy(nn.Module):
    """根据单帧状态和类别直接回归一帧动作。"""

    def __init__(
        self,
        observation_dim,
        action_dim,
        category_count,
        category_embedding_dim=16,
        hidden_dims=(256, 256, 256),
        dropout=0.0,
    ):
        """保存维度并建立类别条件MLP。"""
        super().__init__()
        self.category = CategoryConditioner(category_count, category_embedding_dim)
        self.actor = make_mlp(
            observation_dim + category_embedding_dim,
            action_dim,
            hidden_dims,
            dropout,
        )

    def forward(self, observations, category_id):
        """输入`(B,O)`观测和类别，输出`(B,A)`标准化动作。"""
        return self.actor(torch.cat([observations, self.category(category_id)], dim=-1))


class Temporal3BCPolicy(nn.Module):
    """使用当前及前两帧状态和前两步已执行动作的显式短历史策略。"""

    def __init__(
        self,
        observation_dim,
        action_dim,
        category_count,
        category_embedding_dim=16,
        hidden_dims=(384, 384, 256),
        dropout=0.0,
    ):
        """根据三帧观测、两帧历史动作和类别建立MLP。"""
        super().__init__()
        self.category = CategoryConditioner(category_count, category_embedding_dim)
        input_dim = observation_dim * 3 + action_dim * 2 + category_embedding_dim
        self.actor = make_mlp(input_dim, action_dim, hidden_dims, dropout)

    def forward(self, observation_history, previous_actions, category_id):
        """输入`(B,3,O)`、`(B,2,A)`和类别，输出下一帧动作。"""
        features = torch.cat(
            [
                observation_history.flatten(1),
                previous_actions.flatten(1),
                self.category(category_id),
            ],
            dim=-1,
        )
        return self.actor(features)


def sinusoidal_time_embedding(timesteps, dimension):
    """把DDPM离散时间步编码为正余弦向量。

    输入：`(B,)`时间步和偶数/奇数embedding维度。
    输出：`(B,D)`连续时间特征。
    内部逻辑：使用Transformer式对数频率；奇数维在末尾补零。
    作用：让同一去噪网络区分强噪声早期和弱噪声后期。
    """
    half = dimension // 2
    exponent = -math.log(10000.0) * torch.arange(
        half, device=timesteps.device, dtype=torch.float32
    ) / max(half - 1, 1)
    angles = timesteps.float()[:, None] * torch.exp(exponent)[None, :]
    embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    if dimension % 2:
        embedding = torch.nn.functional.pad(embedding, (0, 1))
    return embedding


class ConditionalDiffusionPolicy(nn.Module):
    """在状态条件下对固定长度动作片段执行DDPM噪声预测。"""

    def __init__(
        self,
        observation_dim,
        action_dim,
        category_count,
        action_horizon=8,
        observation_history=3,
        category_embedding_dim=16,
        time_embedding_dim=64,
        hidden_dims=(512, 512, 384),
        dropout=0.0,
    ):
        """建立类别条件、时间编码和动作片段去噪MLP。"""
        super().__init__()
        self.action_dim = int(action_dim)
        self.action_horizon = int(action_horizon)
        self.observation_history = int(observation_history)
        self.time_embedding_dim = int(time_embedding_dim)
        self.category = CategoryConditioner(category_count, category_embedding_dim)
        input_dim = (
            self.action_horizon * self.action_dim
            + self.observation_history * observation_dim
            + category_embedding_dim
            + self.time_embedding_dim
        )
        self.denoiser = make_mlp(
            input_dim,
            self.action_horizon * self.action_dim,
            hidden_dims,
            dropout,
        )

    def forward(self, noisy_actions, observations, category_id, timesteps):
        """输入带噪`(B,H,A)`动作、历史观测、类别和时间，输出同形状噪声。"""
        features = torch.cat(
            [
                noisy_actions.flatten(1),
                observations.flatten(1),
                self.category(category_id),
                sinusoidal_time_embedding(timesteps, self.time_embedding_dim),
            ],
            dim=-1,
        )
        return self.denoiser(features).view_as(noisy_actions)


def linear_beta_schedule(step_count, start=1e-4, end=0.02):
    """生成简单稳定的线性DDPM beta表。

    输入：扩散步数和beta上下界。
    输出：`(T,)`float32张量。
    内部逻辑：在给定边界线性插值；函数名保留接口语义但当前明确采用linear。
    作用：集中管理训练和采样必须完全一致的噪声日程。
    """
    return torch.linspace(float(start), float(end), int(step_count), dtype=torch.float32)


@torch.no_grad()
def sample_diffusion(model, observation_history, category_id, betas):
    """从高斯噪声反向采样一个动作片段。

    输入：条件Diffusion模型、历史观测、类别ID和beta表。
    输出：`(B,H,A)`标准化动作片段。
    内部逻辑：按DDPM均值公式从T-1迭代到0，非末步加入标准高斯噪声。
    作用：提供闭环评测器可直接调用的无第三方diffusion依赖推理入口。
    """
    device = observation_history.device
    betas = betas.to(device)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    batch = observation_history.shape[0]
    actions = torch.randn(
        batch, model.action_horizon, model.action_dim, device=device
    )
    for step in reversed(range(len(betas))):
        timesteps = torch.full((batch,), step, device=device, dtype=torch.long)
        predicted_noise = model(
            actions, observation_history, category_id, timesteps
        )
        alpha = alphas[step]
        alpha_bar = alpha_bars[step]
        mean = (actions - (1.0 - alpha) * predicted_noise / torch.sqrt(1.0 - alpha_bar)) / torch.sqrt(alpha)
        if step > 0:
            actions = mean + torch.sqrt(betas[step]) * torch.randn_like(actions)
        else:
            actions = mean
    return actions
