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


class SharedCategoryExpertPolicy(nn.Module):
    """共享状态理解、但为每个物体类别保留轻量动作修正的类别教师。

    与旧项目“每类一个完整网络”表达的目标相同：不同类别可以形成不同抓取习惯。
    当前任务有50类且每类最多8条训练轨迹，因此只分开最后的残差动作头，前面的
    状态特征提取器由所有类别共同训练。某类别没有成功轨迹时，它的残差保持为零，
    自动退回共享策略，而不会调用一个从未训练过的随机专家。
    """

    def __init__(
        self,
        observation_dim,
        action_dim,
        category_count,
        hidden_dims=(256, 256, 256),
        dropout=0.0,
    ):
        """建立共享主干、共享动作头和50个按类别路由的残差头。

        输入：观测/动作/类别维度、隐藏层宽度和dropout。
        输出：构造完成的PyTorch模块。
        内部逻辑：主干只读物理观测；共享头给出通用动作，每个类别再加一个
        零初始化线性残差。类别头参数用一个三维张量保存，便于混合类别batch并行。
        作用：在极少的单类数据下保留类别专门化，同时让无数据类别安全回退。
        """
        super().__init__()
        hidden_dims = tuple(int(value) for value in hidden_dims)
        if not hidden_dims:
            raise ValueError("类别教师至少需要一个隐藏层")
        full = make_mlp(
            observation_dim,
            action_dim,
            hidden_dims,
            dropout,
        )
        self.trunk = nn.Sequential(*list(full.children())[:-1])
        self.shared_head = list(full.children())[-1]
        feature_dim = hidden_dims[-1]
        self.category_head_weight = nn.Parameter(
            torch.zeros(int(category_count), int(action_dim), feature_dim)
        )
        self.category_head_bias = nn.Parameter(
            torch.zeros(int(category_count), int(action_dim))
        )

    def forward(self, observations, category_id):
        """输入`(B,O)`观测和`(B,)`类别，输出`(B,A)`标准化教师动作。

        内部逻辑：先算共享动作，再用每个样本的类别ID选出对应残差矩阵做批量乘法。
        """
        features = self.trunk(observations)
        category_id = category_id.long()
        weights = self.category_head_weight[category_id]
        residual = torch.bmm(weights, features.unsqueeze(-1)).squeeze(-1)
        residual = residual + self.category_head_bias[category_id]
        return self.shared_head(features) + residual


def initialize_category_expert_from_bc(expert, bc_state, observation_dim):
    """用BC Soup初始化共享类别教师，并把类别残差严格置零。

    输入：`SharedCategoryExpertPolicy`、BC的state_dict和观测维度。
    输出：原地初始化后的expert。
    内部逻辑：BC首层原本读取“观测+Task-ID embedding”，这里只复制观测列；
    后续共享层和最终动作头完整复制，类别embedding列被有意丢弃。残差头保持零。
    作用：让教师从已会基本动作的Soup出发，而缺样本类别不会继承随机embedding。
    """
    expert_state = expert.state_dict()
    actor_indices = sorted(
        int(key.split(".")[1])
        for key in bc_state
        if key.startswith("actor.") and key.endswith(".weight")
    )
    if not actor_indices:
        raise ValueError("BC checkpoint中没有actor线性层")
    final_actor_index = actor_indices[-1]
    for key in list(expert_state):
        if key.startswith("trunk."):
            source_key = "actor." + key[len("trunk."):]
            if source_key not in bc_state:
                raise ValueError(f"BC缺少可初始化参数: {source_key}")
            source = bc_state[source_key]
            target = expert_state[key]
            if key == "trunk.0.weight":
                if source.shape[0] != target.shape[0] or source.shape[1] < observation_dim:
                    raise ValueError("BC首层尺寸与类别教师不兼容")
                source = source[:, :observation_dim]
            if source.shape != target.shape:
                raise ValueError(f"初始化尺寸不一致: {source_key} -> {key}")
            expert_state[key] = source.clone()
        elif key.startswith("shared_head."):
            suffix = key.split(".", 1)[1]
            source_key = f"actor.{final_actor_index}.{suffix}"
            if source_key not in bc_state or bc_state[source_key].shape != expert_state[key].shape:
                raise ValueError(f"BC最终动作头与类别教师不兼容: {source_key}")
            expert_state[key] = bc_state[source_key].clone()
        elif key in {"category_head_weight", "category_head_bias"}:
            expert_state[key] = torch.zeros_like(expert_state[key])
    expert.load_state_dict(expert_state, strict=True)
    return expert


def initialize_temporal_from_single_frame(
    temporal,
    single_frame_state,
    observation_dim,
    action_dim,
    history=3,
):
    """把单帧学生无损嵌入Temporal3网络的“当前帧”输入位置。

    输入：Temporal3模型、单帧学生state_dict、观测/动作维度和历史长度。
    输出：原地初始化后的Temporal模型。
    内部逻辑：复制类别embedding和全部同形隐藏/输出层；首层权重先清零，再把
    单帧观测列放入最新一帧位置，把Task-ID embedding列放到时序输入末尾。
    作用：刚初始化时Temporal3的输出等于Online-R1单帧学生，之后再学习历史修正，
    避免切换结构时丢掉前面所有阶段的能力。
    """
    if int(history) != 3:
        raise ValueError("当前初始化只支持Temporal3")
    state = temporal.state_dict()
    for key in list(state):
        if key.startswith("category."):
            source_key = key
            if source_key not in single_frame_state or single_frame_state[source_key].shape != state[key].shape:
                raise ValueError(f"单帧学生类别参数不兼容: {key}")
            state[key] = single_frame_state[source_key].clone()
        elif key.startswith("actor."):
            if key not in single_frame_state:
                raise ValueError(f"单帧学生缺少actor参数: {key}")
            source = single_frame_state[key]
            target = state[key]
            if key == "actor.0.weight":
                expected_source = observation_dim + temporal.category.embedding.embedding_dim
                if source.shape[1] != expected_source or source.shape[0] != target.shape[0]:
                    raise ValueError("单帧学生首层与Temporal3不兼容")
                mapped = torch.zeros_like(target)
                current_start = (history - 1) * observation_dim
                mapped[:, current_start:current_start + observation_dim] = source[:, :observation_dim]
                mapped[:, history * observation_dim + (history - 1) * action_dim:] = source[:, observation_dim:]
                state[key] = mapped
            else:
                if source.shape != target.shape:
                    raise ValueError(
                        f"Temporal warm start要求相同隐藏层宽度: {key} {source.shape}!={target.shape}"
                    )
                state[key] = source.clone()
    temporal.load_state_dict(state, strict=True)
    return temporal


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
