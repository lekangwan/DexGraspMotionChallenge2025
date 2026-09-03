"""几何条件的单步、动作块和Temporal动作块策略。"""

import math

import torch
from torch import nn


def mlp(input_dim, output_dim, hidden_dims):
    layers = []
    current = int(input_dim)
    for width in hidden_dims:
        layers.extend([nn.Linear(current, width), nn.LayerNorm(width), nn.SiLU()])
        current = int(width)
    layers.append(nn.Linear(current, int(output_dim)))
    return nn.Sequential(*layers)


def phase_features(phase, frequencies=4):
    values = torch.arange(1, frequencies + 1, device=phase.device, dtype=phase.dtype)
    angles = 2.0 * math.pi * phase * values.view(1, -1)
    return torch.cat([phase, torch.sin(angles), torch.cos(angles)], dim=-1)


class PointNetEncoder(nn.Module):
    """用共享点MLP和对称池化编码无序物体表面点。"""

    def __init__(self, output_dim=128):
        super().__init__()
        self.point_mlp = mlp(3, 128, (64, 128))
        self.output = mlp(256, output_dim, (256,))

    def forward(self, points):
        features = self.point_mlp(points)
        pooled = torch.cat([features.max(dim=1).values, features.mean(dim=1)], dim=-1)
        return self.output(pooled)


class GeometryTrajectoryPCA(nn.Module):
    """由初始任务状态和物体点云预测整条轨迹的PCA系数。"""

    def __init__(
        self, task_observation_dim, action_dim, rank, hidden_dim=192,
        point_feature_dim=64, interaction_dim=0, interaction_feature_dim=64,
    ):
        super().__init__()
        self.points = PointNetEncoder(point_feature_dim)
        self.interaction = (
            mlp(interaction_dim, interaction_feature_dim, (128,))
            if interaction_dim > 0 else None
        )
        self.regressor = mlp(
            task_observation_dim + action_dim + point_feature_dim
            + (interaction_feature_dim if interaction_dim > 0 else 0),
            rank, (hidden_dim, hidden_dim),
        )

    def forward(
        self, task_observation, initial_command, object_points,
        initial_interaction=None,
    ):
        geometry = self.points(object_points)
        features = [task_observation, initial_command, geometry]
        if self.interaction is not None:
            if initial_interaction is None:
                raise ValueError("当前PCA模型需要初始手物交互特征")
            features.append(self.interaction(initial_interaction))
        return self.regressor(torch.cat(features, dim=-1))


class GeometryKeyposePolicy(nn.Module):
    """由初始几何预测预抓取、抓稳和运输结束三个关键状态。"""

    def __init__(
        self, task_observation_dim, action_dim, keypose_dim,
        hidden_dim=256, point_feature_dim=64, interaction_dim=75,
        interaction_feature_dim=64,
    ):
        super().__init__()
        self.points = PointNetEncoder(point_feature_dim)
        self.interaction = mlp(
            interaction_dim, interaction_feature_dim, (128,)
        )
        condition_dim = (
            task_observation_dim + action_dim + point_feature_dim
            + interaction_feature_dim
        )
        self.regressor = mlp(
            condition_dim, keypose_dim, (hidden_dim, hidden_dim, hidden_dim)
        )

    def forward(
        self, task_observation, initial_command, object_points,
        initial_interaction,
    ):
        geometry = self.points(object_points)
        interaction = self.interaction(initial_interaction)
        return self.regressor(torch.cat([
            task_observation, initial_command, geometry, interaction,
        ], dim=-1))


class GeometryPCAMixture(nn.Module):
    """生成多个PCA轨迹候选，并预测每个候选的物理质量。"""

    def __init__(
        self, task_observation_dim, action_dim, rank, mode_count=4,
        hidden_dim=192, point_feature_dim=64,
    ):
        super().__init__()
        self.rank = int(rank)
        self.mode_count = int(mode_count)
        self.points = PointNetEncoder(point_feature_dim)
        condition_dim = task_observation_dim + action_dim + point_feature_dim
        self.condition = mlp(condition_dim, hidden_dim, (hidden_dim,))
        self.mode_logits = nn.Linear(hidden_dim, self.mode_count)
        self.mode_coefficients = mlp(
            hidden_dim, self.mode_count * self.rank, (hidden_dim,)
        )
        self.quality = mlp(hidden_dim + self.rank, 1, (hidden_dim, 96))

    def encode(self, task_observation, initial_command, object_points):
        geometry = self.points(object_points)
        return self.condition(torch.cat([task_observation, initial_command, geometry], dim=-1))

    def generate(self, task_observation, initial_command, object_points):
        condition = self.encode(task_observation, initial_command, object_points)
        logits = self.mode_logits(condition)
        coefficients = self.mode_coefficients(condition).reshape(
            -1, self.mode_count, self.rank
        )
        return condition, logits, coefficients

    def score(self, condition, coefficients):
        if coefficients.ndim == 2:
            return self.quality(torch.cat([condition, coefficients], dim=-1)).squeeze(-1)
        expanded = condition[:, None].expand(-1, coefficients.shape[1], -1)
        return self.quality(torch.cat([expanded, coefficients], dim=-1)).squeeze(-1)


class GeometryPCALatentDiffusion(nn.Module):
    """在PCA潜空间生成多种完整抓取轨迹，并学习条件—轨迹匹配分数。"""

    def __init__(
        self, task_observation_dim, action_dim, rank, hidden_dim=256,
        point_feature_dim=64, time_frequencies=8,
    ):
        super().__init__()
        self.rank = int(rank)
        self.time_frequencies = int(time_frequencies)
        self.points = PointNetEncoder(point_feature_dim)
        condition_input = task_observation_dim + action_dim + point_feature_dim
        self.condition = mlp(condition_input, hidden_dim, (hidden_dim,))
        time_dim = 1 + 2 * self.time_frequencies
        self.denoiser = mlp(
            hidden_dim + self.rank + time_dim,
            self.rank, (hidden_dim, hidden_dim),
        )
        self.regression = mlp(hidden_dim, self.rank, (hidden_dim,))
        self.compatibility = mlp(
            hidden_dim + self.rank, 1, (hidden_dim, 96)
        )

    def encode(self, task_observation, initial_command, object_points):
        geometry = self.points(object_points)
        return self.condition(torch.cat([
            task_observation, initial_command, geometry,
        ], dim=-1))

    def predict_noise(self, condition, noisy_coefficients, time):
        time_feature = phase_features(time, self.time_frequencies)
        return self.denoiser(torch.cat([
            condition, noisy_coefficients, time_feature,
        ], dim=-1))

    def score(self, condition, coefficients):
        if coefficients.ndim == 2:
            return self.compatibility(torch.cat([
                condition, coefficients,
            ], dim=-1)).squeeze(-1)
        expanded = condition[:, None].expand(-1, coefficients.shape[1], -1)
        return self.compatibility(torch.cat([
            expanded, coefficients,
        ], dim=-1)).squeeze(-1)


def sample_pca_latent_diffusion(model, condition, alpha_bars, candidate_count,
                                seed):
    """从固定随机噪声出发，用确定性DDIM生成可复现的PCA系数候选。"""
    batch = condition.shape[0]
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    coefficients = torch.randn(
        batch, int(candidate_count), model.rank, generator=generator,
        dtype=condition.dtype,
    ).to(condition.device)
    flattened = coefficients.reshape(-1, model.rank)
    repeated = condition[:, None].expand(
        -1, int(candidate_count), -1
    ).reshape(flattened.shape[0], -1)
    steps = len(alpha_bars)
    for index in reversed(range(steps)):
        time = torch.full(
            (len(flattened), 1), index / float(max(steps - 1, 1)),
            dtype=flattened.dtype, device=flattened.device,
        )
        noise = model.predict_noise(repeated, flattened, time)
        alpha = alpha_bars[index]
        clean = (
            flattened - torch.sqrt(1.0 - alpha) * noise
        ) / torch.sqrt(alpha)
        if index:
            previous = alpha_bars[index - 1]
            flattened = (
                torch.sqrt(previous) * clean
                + torch.sqrt(1.0 - previous) * noise
            )
        else:
            flattened = clean
    return flattened.reshape(batch, int(candidate_count), model.rank)


class InteractionResidualPolicy(nn.Module):
    """根据当前手—物交互状态修正PCA名义动作。"""

    def __init__(self, task_dim, action_dim, interaction_dim=75, hidden_dim=256,
                 phase_frequencies=4):
        super().__init__()
        self.phase_frequencies = int(phase_frequencies)
        time_dim = 1 + 2 * self.phase_frequencies
        self.network = mlp(
            task_dim + action_dim + interaction_dim + time_dim,
            action_dim, (hidden_dim, hidden_dim, hidden_dim),
        )

    def forward(self, current_task, nominal_delta, interaction, phase):
        time = phase_features(phase, self.phase_frequencies)
        features = torch.cat([current_task, nominal_delta, interaction, time], dim=-1)
        return torch.tanh(self.network(features))


class DirectInteractionTemporalPolicy(nn.Module):
    """不用PCA，依据动态手物关系和最近状态直接预测动作块。"""

    def __init__(
        self, observation_dim, action_dim, interaction_dim=75,
        action_horizon=1, hidden_dim=256, point_feature_dim=96,
        state_feature_dim=192, recurrent_layers=2, phase_frequencies=4,
    ):
        super().__init__()
        self.action_horizon = int(action_horizon)
        self.phase_frequencies = int(phase_frequencies)
        self.points = PointNetEncoder(point_feature_dim)
        self.task = mlp(
            observation_dim + action_dim + point_feature_dim,
            hidden_dim, (hidden_dim,),
        )
        self.state = mlp(
            observation_dim + interaction_dim + action_dim,
            state_feature_dim, (state_feature_dim,),
        )
        self.recurrent_layers = int(recurrent_layers)
        self.temporal = nn.GRU(
            state_feature_dim + hidden_dim, hidden_dim,
            num_layers=self.recurrent_layers, batch_first=True,
        )
        time_dim = 1 + 2 * self.phase_frequencies
        self.head = mlp(
            hidden_dim * 2 + time_dim,
            self.action_horizon * action_dim,
            (hidden_dim, hidden_dim),
        )
        self.action_dim = int(action_dim)

    def forward(
        self, initial_observation, initial_command, object_points,
        observation_history, interaction_history,
        previous_delta_history, phase,
    ):
        task = self.task(torch.cat([
            initial_observation, initial_command, self.points(object_points),
        ], dim=-1))
        states = self.state(torch.cat([
            observation_history, interaction_history, previous_delta_history,
        ], dim=-1))
        repeated_task = task[:, None].expand(-1, states.shape[1], -1)
        encoded, _ = self.temporal(torch.cat([states, repeated_task], dim=-1))
        time = phase_features(phase, self.phase_frequencies)
        output = self.head(torch.cat([task, encoded[:, -1], time], dim=-1))
        return output.reshape(-1, self.action_horizon, self.action_dim)


class GeometryPolicy(nn.Module):
    """共享三种候选的物体编码和动作块输出网络。"""

    def __init__(
        self, observation_dim, action_dim, model_type="geometry_chunk",
        history=3, action_horizon=8, hidden_dim=384, point_feature_dim=128,
        transformer_layers=2, phase_frequencies=4,
    ):
        super().__init__()
        if model_type not in {
            "geometry_phase", "geometry_chunk", "geometry_plan_chunk",
            "geometry_temporal_chunk",
        }:
            raise ValueError(f"未知模型类型: {model_type}")
        self.model_type = model_type
        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        self.history = int(history)
        self.phase_frequencies = int(phase_frequencies)
        self.action_horizon = 1 if model_type == "geometry_phase" else int(action_horizon)
        self.points = PointNetEncoder(point_feature_dim)
        task_dim = observation_dim + action_dim + point_feature_dim
        self.task = mlp(task_dim, hidden_dim, (hidden_dim,))
        time_dim = 1 + 2 * self.phase_frequencies
        if model_type in {"geometry_phase", "geometry_plan_chunk"}:
            self.head = mlp(hidden_dim + time_dim, self.action_horizon * action_dim, (hidden_dim, hidden_dim))
        elif model_type == "geometry_chunk":
            state_dim = observation_dim + action_dim
            self.head = mlp(
                hidden_dim + state_dim + time_dim,
                self.action_horizon * action_dim,
                (hidden_dim, hidden_dim),
            )
        else:
            token_dim = 192
            self.state_token = nn.Linear(observation_dim + action_dim, token_dim)
            self.task_token = nn.Linear(hidden_dim, token_dim)
            layer = nn.TransformerEncoderLayer(
                d_model=token_dim, nhead=4, dim_feedforward=512,
                dropout=0.0, activation="gelu", batch_first=True, norm_first=True,
            )
            self.temporal = nn.TransformerEncoder(layer, num_layers=int(transformer_layers))
            self.head = mlp(
                token_dim + time_dim,
                self.action_horizon * action_dim,
                (hidden_dim, hidden_dim),
            )

    def forward(
        self, initial_observation, initial_command, object_points,
        observation_history, previous_delta_history, phase,
    ):
        geometry = self.points(object_points)
        task = self.task(torch.cat([initial_observation, initial_command, geometry], dim=-1))
        time = phase_features(phase, self.phase_frequencies)
        if self.model_type in {"geometry_phase", "geometry_plan_chunk"}:
            features = torch.cat([task, time], dim=-1)
        elif self.model_type == "geometry_chunk":
            current = torch.cat(
                [observation_history[:, -1], previous_delta_history[:, -1]], dim=-1
            )
            features = torch.cat([task, current, time], dim=-1)
        else:
            states = torch.cat([observation_history, previous_delta_history], dim=-1)
            tokens = torch.cat([
                self.task_token(task).unsqueeze(1), self.state_token(states)
            ], dim=1)
            encoded = self.temporal(tokens)[:, -1]
            features = torch.cat([encoded, time], dim=-1)
        return self.head(features).reshape(-1, self.action_horizon, self.action_dim)


def build_model(config, observation_dim, action_dim):
    return GeometryPolicy(
        observation_dim=observation_dim,
        action_dim=action_dim,
        model_type=config["model_type"],
        history=config.get("history", 3),
        action_horizon=config.get("action_horizon", 8),
        hidden_dim=config.get("hidden_dim", 384),
        point_feature_dim=config.get("point_feature_dim", 128),
        transformer_layers=config.get("transformer_layers", 2),
        phase_frequencies=config.get("phase_frequencies", 4),
    )


def build_pca_model(config, task_observation_dim, action_dim):
    return GeometryTrajectoryPCA(
        task_observation_dim=task_observation_dim,
        action_dim=action_dim,
        rank=config["pca_rank"],
        hidden_dim=config.get("hidden_dim", 192),
        point_feature_dim=config.get("point_feature_dim", 64),
        interaction_dim=config.get("interaction_dim", 0),
        interaction_feature_dim=config.get("interaction_feature_dim", 64),
    )


def build_pca_mixture(config, task_observation_dim, action_dim):
    return GeometryPCAMixture(
        task_observation_dim=task_observation_dim,
        action_dim=action_dim,
        rank=config["pca_rank"],
        mode_count=config.get("mode_count", 4),
        hidden_dim=config.get("hidden_dim", 192),
        point_feature_dim=config.get("point_feature_dim", 64),
    )


def build_pca_latent_diffusion(config, task_observation_dim, action_dim):
    return GeometryPCALatentDiffusion(
        task_observation_dim=task_observation_dim,
        action_dim=action_dim,
        rank=config["pca_rank"],
        hidden_dim=config.get("hidden_dim", 256),
        point_feature_dim=config.get("point_feature_dim", 64),
        time_frequencies=config.get("time_frequencies", 8),
    )


def build_interaction_residual(config, task_dim, action_dim, interaction_dim=75):
    return InteractionResidualPolicy(
        task_dim=task_dim,
        action_dim=action_dim,
        interaction_dim=interaction_dim,
        hidden_dim=config.get("hidden_dim", 256),
        phase_frequencies=config.get("phase_frequencies", 4),
    )


def build_direct_interaction(config, observation_dim, action_dim, interaction_dim=75):
    return DirectInteractionTemporalPolicy(
        observation_dim=observation_dim,
        action_dim=action_dim,
        interaction_dim=interaction_dim,
        action_horizon=config.get("action_horizon", 1),
        hidden_dim=config.get("hidden_dim", 256),
        point_feature_dim=config.get("point_feature_dim", 96),
        state_feature_dim=config.get("state_feature_dim", 192),
        recurrent_layers=config.get("recurrent_layers", 2),
        phase_frequencies=config.get("phase_frequencies", 4),
    )


def build_keypose_model(config, task_observation_dim, action_dim, keypose_dim):
    return GeometryKeyposePolicy(
        task_observation_dim=task_observation_dim,
        action_dim=action_dim,
        keypose_dim=keypose_dim,
        hidden_dim=config.get("hidden_dim", 256),
        point_feature_dim=config.get("point_feature_dim", 64),
        interaction_dim=config.get("interaction_dim", 75),
        interaction_feature_dim=config.get("interaction_feature_dim", 64),
    )
