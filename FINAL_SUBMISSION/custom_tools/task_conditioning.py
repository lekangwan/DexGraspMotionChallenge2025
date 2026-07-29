"""Explicit category conditioning for the custom unified BC student.

The official DexRep policy is left unchanged.  This module subclasses it and
appends a one-hot task identifier to the 384-dimensional encoded observation
before the actor MLP.
"""

from collections import OrderedDict

import torch
import torch.nn as nn

from ActionDiffusion.bc.model.policy.lqt_policy import ActorCriticDexRep


TASK_CATEGORIES = ("bottle", "mug", "bowl", "camera")
ACTOR_INPUT_WEIGHT_KEY = "model.actor.0.weight"


def category_from_object_id(object_id):
    parts = str(object_id).split("-", 2)
    if len(parts) < 2 or parts[1] not in TASK_CATEGORIES:
        raise ValueError("Cannot infer supported category from {!r}".format(
            object_id))
    return parts[1]


def task_categories(args):
    config = args.get("task_conditioning")
    if config is None:
        return TASK_CATEGORIES
    categories = tuple(config.get("categories", TASK_CATEGORIES))
    if categories != TASK_CATEGORIES:
        raise ValueError(
            "Task category order must be {}, got {}".format(
                TASK_CATEGORIES, categories))
    return categories


def task_conditioning_enabled(args):
    config = args.get("task_conditioning")
    return config is not None and bool(config.get("enabled", False))


def task_input_enabled(args):
    """Whether the category one-hot is actually appended to the actor input."""
    config = args.get("task_conditioning")
    return (
        config is not None
        and bool(config.get("enabled", False))
        and bool(config.get("input_enabled", True))
    )


def category_onehot(category, categories=TASK_CATEGORIES):
    if category not in categories:
        raise ValueError("Unsupported task category: {}".format(category))
    result = torch.zeros(len(categories), dtype=torch.float32)
    result[categories.index(category)] = 1.0
    return result


class TaskConditionedActorCriticDexRep(ActorCriticDexRep):
    """DexRep actor whose action head also receives a category one-hot."""

    def __init__(
            self, *args, task_category_names=TASK_CATEGORIES,
            use_task_input=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_category_names = tuple(task_category_names)
        self.use_task_input = bool(use_task_input)
        self.task_dim = (
            len(self.task_category_names) if self.use_task_input else 0)
        if self.use_task_input and self.task_dim < 2:
            raise ValueError("Task conditioning requires at least two tasks")

        first = self.actor[0]
        if not isinstance(first, nn.Linear):
            raise TypeError("Expected actor[0] to be nn.Linear")
        if self.use_task_input:
            expanded = nn.Linear(
                first.in_features + self.task_dim,
                first.out_features,
                bias=first.bias is not None)
            # Preserve the original random initialization and start with zero
            # dependence on Task ID.  This also defines the exact Soup expansion.
            with torch.no_grad():
                expanded.weight[:, :first.in_features].copy_(first.weight)
                expanded.weight[:, first.in_features:].zero_()
                if first.bias is not None:
                    expanded.bias.copy_(first.bias)
            self.actor[0] = expanded
        self._inference_task_onehot = None

    def set_inference_task_categories(self, categories):
        categories = list(categories)
        if not categories:
            raise ValueError("At least one inference category is required")
        indices = []
        for category in categories:
            if category not in self.task_category_names:
                raise ValueError("Unsupported task category: {}".format(
                    category))
            indices.append(self.task_category_names.index(category))
        device = next(self.parameters()).device
        if self.use_task_input:
            self._inference_task_onehot = torch.nn.functional.one_hot(
                torch.as_tensor(indices, device=device),
                num_classes=self.task_dim).to(dtype=torch.float32)
        else:
            self._inference_task_onehot = torch.empty(
                len(indices), 0, device=device, dtype=torch.float32)

    def _task_tensor(self, observations, task_onehot):
        if not self.use_task_input:
            return torch.empty(
                observations.shape[0], 0,
                device=observations.device, dtype=observations.dtype)
        if task_onehot is None:
            task_onehot = self._inference_task_onehot
        if task_onehot is None:
            raise RuntimeError(
                "Task-conditioned policy requires an explicit Task ID")
        task_onehot = torch.as_tensor(
            task_onehot, device=observations.device,
            dtype=observations.dtype)
        if task_onehot.ndim == 1:
            task_onehot = task_onehot.unsqueeze(0)
        if task_onehot.shape[-1] != self.task_dim:
            raise ValueError(
                "Expected Task ID dimension {}, got {}".format(
                    self.task_dim, task_onehot.shape[-1]))
        if task_onehot.shape[0] == 1 and observations.shape[0] != 1:
            task_onehot = task_onehot.expand(observations.shape[0], -1)
        if task_onehot.shape[0] != observations.shape[0]:
            raise ValueError(
                "Task ID batch {} does not match observation batch {}".format(
                    task_onehot.shape[0], observations.shape[0]))
        return task_onehot

    def forward(self, observations, task_onehot=None):
        encoded = self.encode(observations)
        task_onehot = self._task_tensor(observations, task_onehot)
        return self.actor(torch.cat((encoded, task_onehot), dim=-1))

    @torch.no_grad()
    def act_inference(self, observations, task_onehot=None):
        return self.forward(observations, task_onehot)


class TemporalTaskConditionedActorCriticDexRep(
        TaskConditionedActorCriticDexRep):
    """Task-conditioned policy with two previous states/actions by default."""

    def __init__(
            self, *args, history_frames=3, history_prop_dim=100,
            history_action_dim=28, history_lags=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.history_frames = int(history_frames)
        self.history_prop_dim = int(history_prop_dim)
        self.history_action_dim = int(history_action_dim)
        if self.history_frames < 2:
            raise ValueError("Temporal policy requires at least two frames")
        history_steps = self.history_frames - 1
        if history_lags is None:
            history_lags = tuple(range(history_steps, 0, -1))
        self.history_lags = tuple(int(lag) for lag in history_lags)
        if len(self.history_lags) != history_steps:
            raise ValueError(
                "Expected {} history lags, got {}".format(
                    history_steps, self.history_lags))
        if (
            any(lag <= 0 for lag in self.history_lags)
            or tuple(sorted(self.history_lags, reverse=True))
            != self.history_lags
            or len(set(self.history_lags)) != len(self.history_lags)
        ):
            raise ValueError(
                "History lags must be unique positive integers in "
                "oldest-to-newest order")
        self.history_feature_dim = history_steps * (
            self.history_prop_dim + self.history_action_dim)

        first = self.actor[0]
        expanded = nn.Linear(
            first.in_features + self.history_feature_dim,
            first.out_features,
            bias=first.bias is not None)
        with torch.no_grad():
            expanded.weight[:, :first.in_features].copy_(first.weight)
            expanded.weight[:, first.in_features:].zero_()
            if first.bias is not None:
                expanded.bias.copy_(first.bias)
        self.actor[0] = expanded
        self._inference_prop_history = None
        self._inference_action_history = None

    def reset_inference_history(self):
        self._inference_prop_history = None
        self._inference_action_history = None

    @torch.no_grad()
    def act_inference_for_reset(
            self, observations, env_ids, task_onehot=None):
        """Reset and evaluate only selected vector-environment histories."""
        env_ids = torch.as_tensor(
            env_ids, device=observations.device, dtype=torch.long)
        if env_ids.ndim != 1 or env_ids.numel() == 0:
            raise ValueError("env_ids must be a non-empty one-dimensional set")
        batch = observations.shape[0]
        buffer_steps = max(self.history_lags)
        expected_prop = (
            batch, buffer_steps, self.history_prop_dim)
        expected_action = (
            batch, buffer_steps, self.history_action_dim)
        if (
            self._inference_prop_history is None
            or tuple(self._inference_prop_history.shape) != expected_prop
            or self._inference_action_history is None
            or tuple(self._inference_action_history.shape) != expected_action
        ):
            raise RuntimeError(
                "Partial history reset requires an initialized full batch")

        all_tasks = self._task_tensor(observations, task_onehot)
        reset_obs = observations[env_ids]
        reset_prop = reset_obs[:, :self.history_prop_dim]
        history_steps = len(self.history_lags)
        history_features = torch.cat((
            reset_prop[:, None, :].repeat(
                1, history_steps, 1).reshape(env_ids.numel(), -1),
            torch.zeros(
                env_ids.numel(),
                history_steps * self.history_action_dim,
                device=observations.device,
                dtype=observations.dtype),
        ), dim=-1)
        action = self.forward(
            reset_obs, all_tasks[env_ids], history_features)

        self._inference_prop_history[env_ids] = reset_prop[
            :, None, :].repeat(1, buffer_steps, 1)
        self._inference_action_history[env_ids] = 0.0
        self._inference_action_history[env_ids, -1] = action
        return action

    def _validate_history(self, observations, history_features):
        if history_features is None:
            raise RuntimeError(
                "Temporal training requires explicit history features")
        history_features = torch.as_tensor(
            history_features, device=observations.device,
            dtype=observations.dtype)
        if history_features.ndim == 1:
            history_features = history_features.unsqueeze(0)
        expected = (observations.shape[0], self.history_feature_dim)
        if tuple(history_features.shape) != expected:
            raise ValueError(
                "Expected history shape {}, got {}".format(
                    expected, tuple(history_features.shape)))
        return history_features

    def forward(
            self, observations, task_onehot=None, history_features=None):
        encoded = self.encode(observations)
        task_onehot = self._task_tensor(observations, task_onehot)
        history_features = self._validate_history(
            observations, history_features)
        return self.actor(torch.cat(
            (encoded, task_onehot, history_features), dim=-1))

    @torch.no_grad()
    def act_inference(self, observations, task_onehot=None):
        batch = observations.shape[0]
        prop = observations[:, :self.history_prop_dim]
        buffer_steps = max(self.history_lags)
        if (self._inference_prop_history is None
                or self._inference_prop_history.shape[0] != batch):
            self._inference_prop_history = prop[:, None, :].repeat(
                1, buffer_steps, 1)
            self._inference_action_history = torch.zeros(
                (batch, buffer_steps, self.history_action_dim),
                device=observations.device, dtype=observations.dtype)
        selected_indices = [-lag for lag in self.history_lags]
        history_features = torch.cat((
            self._inference_prop_history[
                :, selected_indices].reshape(batch, -1),
            self._inference_action_history[
                :, selected_indices].reshape(batch, -1),
        ), dim=-1)
        action = self.forward(
            observations, task_onehot, history_features)
        if buffer_steps > 1:
            self._inference_prop_history = torch.cat((
                self._inference_prop_history[:, 1:],
                prop[:, None, :],
            ), dim=1)
            self._inference_action_history = torch.cat((
                self._inference_action_history[:, 1:],
                action[:, None, :],
            ), dim=1)
        else:
            self._inference_prop_history[:, 0] = prop
            self._inference_action_history[:, 0] = action
        return action


class PhaseConditionedTemporalActorCriticDexRep(
        TemporalTaskConditionedActorCriticDexRep):
    """Temporal3 plus one explicit normalized rollout-progress feature."""

    def __init__(
            self, *args, phase_max_frame_index=69, **kwargs):
        super().__init__(*args, **kwargs)
        self.phase_max_frame_index = int(phase_max_frame_index)
        if self.phase_max_frame_index < 1:
            raise ValueError("phase_max_frame_index must be positive")
        first = self.actor[0]
        expanded = nn.Linear(
            first.in_features + 1,
            first.out_features,
            bias=first.bias is not None)
        with torch.no_grad():
            expanded.weight[:, :first.in_features].copy_(first.weight)
            expanded.weight[:, first.in_features:].zero_()
            if first.bias is not None:
                expanded.bias.copy_(first.bias)
        self.actor[0] = expanded
        self._inference_phase_steps = None
        self._inference_phase_override = None

    def reset_inference_history(self):
        super().reset_inference_history()
        self._inference_phase_steps = None
        self._inference_phase_override = None

    def _phase_tensor(self, observations, phase_feature):
        if phase_feature is None:
            phase_feature = self._inference_phase_override
        if phase_feature is None:
            if (
                self._inference_phase_steps is None
                or self._inference_phase_steps.shape[0]
                != observations.shape[0]
            ):
                raise RuntimeError(
                    "Phase-conditioned policy requires rollout progress")
            phase_feature = (
                2.0 * self._inference_phase_steps.to(
                    dtype=observations.dtype)
                / float(self.phase_max_frame_index)
                - 1.0
            ).clamp(-1.0, 1.0).unsqueeze(-1)
        phase_feature = torch.as_tensor(
            phase_feature, device=observations.device,
            dtype=observations.dtype)
        if phase_feature.ndim == 1:
            phase_feature = phase_feature.unsqueeze(-1)
        expected = (observations.shape[0], 1)
        if tuple(phase_feature.shape) != expected:
            raise ValueError(
                "Expected phase feature {}, got {}".format(
                    expected, tuple(phase_feature.shape)))
        return phase_feature

    def forward(
            self, observations, task_onehot=None, history_features=None,
            phase_feature=None):
        encoded = self.encode(observations)
        task_onehot = self._task_tensor(observations, task_onehot)
        history_features = self._validate_history(
            observations, history_features)
        phase_feature = self._phase_tensor(
            observations, phase_feature)
        return self.actor(torch.cat((
            encoded, task_onehot, history_features, phase_feature), dim=-1))

    @torch.no_grad()
    def act_inference(self, observations, task_onehot=None):
        batch = observations.shape[0]
        if (
            self._inference_phase_steps is None
            or self._inference_phase_steps.shape[0] != batch
        ):
            self._inference_phase_steps = torch.zeros(
                batch, device=observations.device, dtype=torch.long)
        action = super().act_inference(observations, task_onehot)
        self._inference_phase_steps.add_(1).clamp_(
            max=self.phase_max_frame_index)
        return action

    @torch.no_grad()
    def act_inference_for_reset(
            self, observations, env_ids, task_onehot=None):
        env_ids = torch.as_tensor(
            env_ids, device=observations.device, dtype=torch.long)
        if (
            self._inference_phase_steps is None
            or self._inference_phase_steps.shape[0]
            != observations.shape[0]
        ):
            raise RuntimeError(
                "Partial phase reset requires an initialized full batch")
        self._inference_phase_override = -torch.ones(
            (env_ids.numel(), 1),
            device=observations.device, dtype=observations.dtype)
        try:
            action = super().act_inference_for_reset(
                observations, env_ids, task_onehot)
        finally:
            self._inference_phase_override = None
        self._inference_phase_steps[env_ids] = 1
        return action


class FullObservationGRUTemporalActorCriticDexRep(
        TemporalTaskConditionedActorCriticDexRep):
    """Temporal3 plus a shared-encoder GRU over three full observations."""

    def __init__(
            self, *args, full_history_observation_dim=2460,
            full_history_gru_hidden_dim=128, **kwargs):
        super().__init__(*args, **kwargs)
        self.full_history_observation_dim = int(
            full_history_observation_dim)
        self.full_history_gru_hidden_dim = int(
            full_history_gru_hidden_dim)
        if self.full_history_observation_dim <= 0:
            raise ValueError(
                "full_history_observation_dim must be positive")
        if self.full_history_gru_hidden_dim <= 0:
            raise ValueError(
                "full_history_gru_hidden_dim must be positive")
        encoded_dim = self.actor[0].in_features
        encoded_dim -= self.task_dim + self.history_feature_dim
        token_dim = encoded_dim + self.history_action_dim
        self.full_history_gru = nn.GRU(
            input_size=token_dim,
            hidden_size=self.full_history_gru_hidden_dim,
            num_layers=1,
            batch_first=True)
        self.full_history_norm = nn.LayerNorm(
            self.full_history_gru_hidden_dim)
        self.full_history_action_head = nn.Linear(
            self.full_history_gru_hidden_dim,
            self.history_action_dim)
        # Exact Temporal3 warm start: the new sequence branch initially adds
        # zero to every action while all loaded Temporal3 weights are retained.
        with torch.no_grad():
            self.full_history_action_head.weight.zero_()
            self.full_history_action_head.bias.zero_()
        self._inference_full_observation_history = None

    def reset_inference_history(self):
        super().reset_inference_history()
        self._inference_full_observation_history = None

    def _validate_full_history(
            self, observations, full_history_observations):
        if full_history_observations is None:
            raise RuntimeError(
                "Full-observation GRU requires explicit observation history")
        full_history_observations = torch.as_tensor(
            full_history_observations,
            device=observations.device,
            dtype=observations.dtype)
        expected = (
            observations.shape[0],
            len(self.history_lags),
            self.full_history_observation_dim,
        )
        if tuple(full_history_observations.shape) != expected:
            raise ValueError(
                "Expected full observation history {}, got {}".format(
                    expected, tuple(full_history_observations.shape)))
        return full_history_observations

    def forward(
            self, observations, task_onehot=None, history_features=None,
            full_history_observations=None):
        history_features = self._validate_history(
            observations, history_features)
        full_history_observations = self._validate_full_history(
            observations, full_history_observations)
        task_onehot = self._task_tensor(observations, task_onehot)
        current_encoded = self.encode(observations)
        base_action = self.actor(torch.cat((
            current_encoded, task_onehot, history_features), dim=-1))

        batch = observations.shape[0]
        history_steps = len(self.history_lags)
        previous_encoded = self.encode(
            full_history_observations.reshape(
                batch * history_steps,
                self.full_history_observation_dim)
        ).reshape(batch, history_steps, -1)
        prop_width = history_steps * self.history_prop_dim
        previous_actions = history_features[:, prop_width:].reshape(
            batch, history_steps, self.history_action_dim)
        previous_tokens = torch.cat(
            (previous_encoded, previous_actions), dim=-1)
        current_action_placeholder = torch.zeros(
            (batch, self.history_action_dim),
            device=observations.device, dtype=observations.dtype)
        current_token = torch.cat(
            (current_encoded, current_action_placeholder),
            dim=-1).unsqueeze(1)
        tokens = torch.cat((previous_tokens, current_token), dim=1)
        _, hidden = self.full_history_gru(tokens)
        residual = self.full_history_action_head(
            self.full_history_norm(hidden[-1]))
        return base_action + residual

    @torch.no_grad()
    def act_inference(self, observations, task_onehot=None):
        batch = observations.shape[0]
        prop = observations[:, :self.history_prop_dim]
        buffer_steps = max(self.history_lags)
        if (
            self._inference_prop_history is None
            or self._inference_prop_history.shape[0] != batch
        ):
            self._inference_prop_history = prop[:, None, :].repeat(
                1, buffer_steps, 1)
            self._inference_action_history = torch.zeros(
                (batch, buffer_steps, self.history_action_dim),
                device=observations.device, dtype=observations.dtype)
            self._inference_full_observation_history = observations[
                :, None, :].repeat(1, buffer_steps, 1)
        expected_full = (
            batch, buffer_steps, self.full_history_observation_dim)
        if (
            self._inference_full_observation_history is None
            or tuple(self._inference_full_observation_history.shape)
            != expected_full
        ):
            raise RuntimeError(
                "Full observation inference history is inconsistent")
        selected_indices = [-lag for lag in self.history_lags]
        history_features = torch.cat((
            self._inference_prop_history[
                :, selected_indices].reshape(batch, -1),
            self._inference_action_history[
                :, selected_indices].reshape(batch, -1),
        ), dim=-1)
        full_history = self._inference_full_observation_history[
            :, selected_indices]
        action = self.forward(
            observations, task_onehot, history_features, full_history)
        if buffer_steps > 1:
            self._inference_prop_history = torch.cat((
                self._inference_prop_history[:, 1:],
                prop[:, None, :],
            ), dim=1)
            self._inference_action_history = torch.cat((
                self._inference_action_history[:, 1:],
                action[:, None, :],
            ), dim=1)
            self._inference_full_observation_history = torch.cat((
                self._inference_full_observation_history[:, 1:],
                observations[:, None, :],
            ), dim=1)
        else:
            self._inference_prop_history[:, 0] = prop
            self._inference_action_history[:, 0] = action
            self._inference_full_observation_history[:, 0] = observations
        return action

    @torch.no_grad()
    def act_inference_for_reset(
            self, observations, env_ids, task_onehot=None):
        env_ids = torch.as_tensor(
            env_ids, device=observations.device, dtype=torch.long)
        if env_ids.ndim != 1 or env_ids.numel() == 0:
            raise ValueError(
                "env_ids must be a non-empty one-dimensional set")
        batch = observations.shape[0]
        buffer_steps = max(self.history_lags)
        expected_full = (
            batch, buffer_steps, self.full_history_observation_dim)
        if (
            self._inference_prop_history is None
            or self._inference_action_history is None
            or self._inference_full_observation_history is None
            or tuple(self._inference_full_observation_history.shape)
            != expected_full
        ):
            raise RuntimeError(
                "Partial reset requires initialized full histories")
        all_tasks = self._task_tensor(observations, task_onehot)
        reset_obs = observations[env_ids]
        reset_prop = reset_obs[:, :self.history_prop_dim]
        history_steps = len(self.history_lags)
        history_features = torch.cat((
            reset_prop[:, None, :].repeat(
                1, history_steps, 1).reshape(env_ids.numel(), -1),
            torch.zeros(
                env_ids.numel(),
                history_steps * self.history_action_dim,
                device=observations.device,
                dtype=observations.dtype),
        ), dim=-1)
        full_history = reset_obs[:, None, :].repeat(
            1, history_steps, 1)
        action = self.forward(
            reset_obs, all_tasks[env_ids],
            history_features, full_history)
        self._inference_prop_history[env_ids] = reset_prop[
            :, None, :].repeat(1, buffer_steps, 1)
        self._inference_action_history[env_ids] = 0.0
        self._inference_action_history[env_ids, -1] = action
        self._inference_full_observation_history[env_ids] = reset_obs[
            :, None, :].repeat(1, buffer_steps, 1)
        return action


class AttentionResidualTemporalActorCriticDexRep(
        TemporalTaskConditionedActorCriticDexRep):
    """Frozen Temporal3 base plus a trainable attention action residual."""

    def __init__(
            self, *args, attention_model_dim=128, attention_heads=4,
            attention_layers=2, attention_feedforward_dim=256,
            attention_dropout=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.attention_model_dim = int(attention_model_dim)
        self.attention_heads = int(attention_heads)
        self.attention_layers = int(attention_layers)
        self.attention_feedforward_dim = int(attention_feedforward_dim)
        self.attention_dropout = float(attention_dropout)
        if self.attention_model_dim <= 0:
            raise ValueError("attention_model_dim must be positive")
        if self.attention_heads <= 0:
            raise ValueError("attention_heads must be positive")
        if self.attention_model_dim % self.attention_heads:
            raise ValueError(
                "attention_model_dim must be divisible by attention_heads")
        if self.attention_layers <= 0:
            raise ValueError("attention_layers must be positive")
        if self.attention_feedforward_dim <= 0:
            raise ValueError("attention_feedforward_dim must be positive")
        if not 0.0 <= self.attention_dropout < 1.0:
            raise ValueError("attention_dropout must be in [0, 1)")

        token_input_dim = self.history_prop_dim + self.history_action_dim
        self.history_token_projection = nn.Linear(
            token_input_dim, self.attention_model_dim)
        self.history_position_embedding = nn.Parameter(torch.empty(
            1, self.history_frames, self.attention_model_dim))
        nn.init.normal_(self.history_position_embedding, mean=0.0, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=self.attention_model_dim,
            nhead=self.attention_heads,
            dim_feedforward=self.attention_feedforward_dim,
            dropout=self.attention_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False)
        self.history_transformer = nn.TransformerEncoder(
            layer, num_layers=self.attention_layers)
        self.history_attention_norm = nn.LayerNorm(self.attention_model_dim)
        self.history_attention_action_head = nn.Linear(
            self.attention_model_dim, self.history_action_dim)
        # Exact functional warm start: before training the new branch adds
        # precisely zero to every Temporal3 action.
        with torch.no_grad():
            self.history_attention_action_head.weight.zero_()
            self.history_attention_action_head.bias.zero_()

    def _attention_tokens(self, observations, history_features):
        batch = observations.shape[0]
        history_steps = self.history_frames - 1
        prop_width = history_steps * self.history_prop_dim
        previous_props = history_features[:, :prop_width].reshape(
            batch, history_steps, self.history_prop_dim)
        previous_actions = history_features[:, prop_width:].reshape(
            batch, history_steps, self.history_action_dim)
        previous_tokens = torch.cat(
            (previous_props, previous_actions), dim=-1)
        current_prop = observations[:, :self.history_prop_dim]
        current_action_placeholder = torch.zeros(
            (batch, self.history_action_dim),
            device=observations.device, dtype=observations.dtype)
        current_token = torch.cat(
            (current_prop, current_action_placeholder), dim=-1).unsqueeze(1)
        return torch.cat((previous_tokens, current_token), dim=1)

    def _attention_residual(self, observations, history_features):
        tokens = self._attention_tokens(observations, history_features)
        encoded = (
            self.history_token_projection(tokens)
            + self.history_position_embedding.to(
                device=tokens.device, dtype=tokens.dtype))
        causal_mask = torch.triu(torch.ones(
            (self.history_frames, self.history_frames),
            dtype=torch.bool, device=tokens.device), diagonal=1)
        encoded = self.history_transformer(encoded, mask=causal_mask)
        current_feature = self.history_attention_norm(encoded[:, -1])
        return self.history_attention_action_head(current_feature)

    def forward(
            self, observations, task_onehot=None, history_features=None):
        history_features = self._validate_history(
            observations, history_features)
        base_action = super().forward(
            observations, task_onehot, history_features)
        return base_action + self._attention_residual(
            observations, history_features)

    def freeze_temporal_base(self):
        """Freeze all pre-existing policy weights and expose attention only."""
        self.requires_grad_(False)
        self.history_token_projection.requires_grad_(True)
        self.history_transformer.requires_grad_(True)
        self.history_attention_norm.requires_grad_(True)
        self.history_attention_action_head.requires_grad_(True)
        self.history_position_embedding.requires_grad_(True)
        return [
            module for module in (
                self.state_enc,
                self.dexrep_sensor_enc,
                self.dexrep_pointL_enc,
                self.bn_pnl,
                self.actor,
                self.critic,
            )
            if module is not None
        ]


class ActionChunkAuxTemporalActorCriticDexRep(
        TemporalTaskConditionedActorCriticDexRep):
    """Temporal3 policy with future-action prediction used only in training."""

    def __init__(self, *args, action_chunk_horizon=4, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_chunk_horizon = int(action_chunk_horizon)
        if self.action_chunk_horizon < 2:
            raise ValueError("action_chunk_horizon must be at least two")
        final = self.actor[-1]
        if not isinstance(final, nn.Linear):
            raise TypeError("Expected final actor module to be nn.Linear")
        if final.out_features != self.history_action_dim:
            raise ValueError(
                "Actor output {} does not match history action dimension {}"
                .format(final.out_features, self.history_action_dim))
        self.future_action_head = nn.Linear(
            final.in_features,
            (self.action_chunk_horizon - 1) * self.history_action_dim)
        # A fresh model initially repeats its current-action prediction for
        # each future step. Checkpoint expansion repeats the loaded final head.
        with torch.no_grad():
            self.future_action_head.weight.copy_(
                final.weight.repeat(self.action_chunk_horizon - 1, 1))
            self.future_action_head.bias.copy_(
                final.bias.repeat(self.action_chunk_horizon - 1))

    def _actor_input(
            self, observations, task_onehot=None, history_features=None):
        encoded = self.encode(observations)
        task_onehot = self._task_tensor(observations, task_onehot)
        history_features = self._validate_history(
            observations, history_features)
        return torch.cat(
            (encoded, task_onehot, history_features), dim=-1)

    def forward(
            self, observations, task_onehot=None, history_features=None):
        return self.actor(self._actor_input(
            observations, task_onehot, history_features))

    def forward_action_chunk(
            self, observations, task_onehot=None, history_features=None):
        actor_input = self._actor_input(
            observations, task_onehot, history_features)
        hidden = self.actor[:-1](actor_input)
        current = self.actor[-1](hidden).unsqueeze(1)
        future = self.future_action_head(hidden).reshape(
            observations.shape[0],
            self.action_chunk_horizon - 1,
            self.history_action_dim)
        return torch.cat((current, future), dim=1)


def temporal_history_enabled(args):
    config = args.get("temporal_history")
    return config is not None and bool(config.get("enabled", False))


def temporal_history_dimensions(args):
    config = args.get("temporal_history", {})
    history_frames = int(config.get("history_frames", 3))
    prop_dim = int(config.get("prop_dim", 100))
    action_dim = int(config.get("action_dim", 28))
    if history_frames < 2:
        raise ValueError("history_frames must be at least two")
    return history_frames, prop_dim, action_dim


def temporal_history_lags(args):
    history_frames, _, _ = temporal_history_dimensions(args)
    history_steps = history_frames - 1
    config = args.get("temporal_history", {})
    configured = config.get("history_lags")
    if configured is None:
        return tuple(range(history_steps, 0, -1))
    lags = tuple(int(lag) for lag in configured)
    if (
        len(lags) != history_steps
        or any(lag <= 0 for lag in lags)
        or tuple(sorted(lags, reverse=True)) != lags
        or len(set(lags)) != len(lags)
    ):
        raise ValueError(
            "temporal_history.history_lags must contain {} unique positive "
            "integers in oldest-to-newest order".format(history_steps))
    return lags


def temporal_attention_enabled(args):
    config = args.get("temporal_attention")
    return config is not None and bool(config.get("enabled", False))


def full_observation_gru_enabled(args):
    config = args.get("full_observation_gru")
    return config is not None and bool(config.get("enabled", False))


def phase_conditioning_enabled(args):
    config = args.get("phase_conditioning")
    return config is not None and bool(config.get("enabled", False))


def phase_conditioning_parameters(args):
    config = args.get("phase_conditioning", {})
    max_frame_index = int(config.get("max_frame_index", 69))
    if max_frame_index < 1:
        raise ValueError(
            "phase_conditioning.max_frame_index must be positive")
    return {"phase_max_frame_index": max_frame_index}


def full_observation_gru_parameters(args):
    config = args.get("full_observation_gru", {})
    parameters = {
        "full_history_observation_dim": int(
            config.get("observation_dim", 2460)),
        "full_history_gru_hidden_dim": int(
            config.get("hidden_dim", 128)),
    }
    if parameters["full_history_observation_dim"] <= 0:
        raise ValueError(
            "full_observation_gru.observation_dim must be positive")
    if parameters["full_history_gru_hidden_dim"] <= 0:
        raise ValueError(
            "full_observation_gru.hidden_dim must be positive")
    return parameters


def temporal_attention_parameters(args):
    config = args.get("temporal_attention", {})
    parameters = {
        "attention_model_dim": int(config.get("model_dim", 128)),
        "attention_heads": int(config.get("heads", 4)),
        "attention_layers": int(config.get("layers", 2)),
        "attention_feedforward_dim": int(
            config.get("feedforward_dim", 256)),
        "attention_dropout": float(config.get("dropout", 0.0)),
    }
    return parameters


def temporal_attention_freeze_base(args):
    config = args.get("temporal_attention", {})
    return temporal_attention_enabled(args) and bool(
        config.get("freeze_temporal_base", False))


def action_chunk_aux_enabled(args):
    config = args.get("action_chunk_aux")
    return config is not None and bool(config.get("enabled", False))


def action_chunk_aux_parameters(args):
    config = args.get("action_chunk_aux", {})
    horizon = int(config.get("horizon", 4))
    auxiliary_weight = float(config.get("auxiliary_weight", 0.25))
    if horizon < 2:
        raise ValueError("action_chunk_aux.horizon must be at least two")
    if auxiliary_weight <= 0:
        raise ValueError(
            "action_chunk_aux.auxiliary_weight must be positive")
    return horizon, auxiliary_weight


def _base_actor_input_dim(args, env_cfg):
    return len(env_cfg["obs_dim"]) * int(args.encoder.emb_dim)


def checkpoint_uses_task_conditioning(args, env_cfg, state_dict):
    """Distinguish a Task-ID student from a standard expert checkpoint."""
    if not task_conditioning_enabled(args):
        return False
    weight = state_dict.get(ACTOR_INPUT_WEIGHT_KEY)
    if weight is None:
        raise KeyError(
            "Checkpoint is missing {}".format(ACTOR_INPUT_WEIGHT_KEY))
    base_dim = _base_actor_input_dim(args, env_cfg)
    task_dim = (
        len(task_categories(args)) if task_input_enabled(args) else 0)
    if weight.shape[1] == base_dim:
        return False
    conditioned_widths = {base_dim + task_dim}
    if temporal_history_enabled(args):
        history_frames, prop_dim, action_dim = temporal_history_dimensions(args)
        temporal_width = (
            base_dim + task_dim
            + (history_frames - 1) * (prop_dim + action_dim))
        conditioned_widths.add(temporal_width)
        if phase_conditioning_enabled(args):
            conditioned_widths.add(temporal_width + 1)
    if weight.shape[1] in conditioned_widths:
        return True
    raise ValueError(
        "Unexpected actor input width {}; expected {} (standard) or one of "
        "{} (conditioned)".format(
            weight.shape[1], base_dim, sorted(conditioned_widths)))


def enable_task_conditioning(lit_model, args, env_cfg):
    """Replace only the inner policy of an already-created LitBCModel."""
    if not task_conditioning_enabled(args):
        return lit_model
    extension_count = sum((
        temporal_attention_enabled(args),
        action_chunk_aux_enabled(args),
        full_observation_gru_enabled(args),
        phase_conditioning_enabled(args),
    ))
    if extension_count > 1:
        raise ValueError(
            "Temporal extensions are separate controlled experiments and "
            "cannot be enabled together")
    if phase_conditioning_enabled(args):
        if not temporal_history_enabled(args):
            raise ValueError(
                "Phase conditioning requires temporal_history.enabled=true")
        policy_class = PhaseConditionedTemporalActorCriticDexRep
    elif full_observation_gru_enabled(args):
        if not temporal_history_enabled(args):
            raise ValueError(
                "Full-observation GRU requires temporal_history.enabled=true")
        policy_class = FullObservationGRUTemporalActorCriticDexRep
    elif action_chunk_aux_enabled(args):
        if not temporal_history_enabled(args):
            raise ValueError(
                "Action-chunk auxiliary requires temporal_history.enabled=true")
        policy_class = ActionChunkAuxTemporalActorCriticDexRep
    elif temporal_attention_enabled(args):
        if not temporal_history_enabled(args):
            raise ValueError(
                "Temporal attention requires temporal_history.enabled=true")
        policy_class = AttentionResidualTemporalActorCriticDexRep
    elif temporal_history_enabled(args):
        policy_class = TemporalTaskConditionedActorCriticDexRep
    else:
        policy_class = TaskConditionedActorCriticDexRep
    temporal_kwargs = {}
    if temporal_history_enabled(args):
        history_frames, prop_dim, action_dim = temporal_history_dimensions(args)
        temporal_kwargs = {
            "history_frames": history_frames,
            "history_prop_dim": prop_dim,
            "history_action_dim": action_dim,
            "history_lags": temporal_history_lags(args),
        }
    if temporal_attention_enabled(args):
        temporal_kwargs.update(temporal_attention_parameters(args))
    if action_chunk_aux_enabled(args):
        horizon, _ = action_chunk_aux_parameters(args)
        temporal_kwargs["action_chunk_horizon"] = horizon
    if full_observation_gru_enabled(args):
        temporal_kwargs.update(full_observation_gru_parameters(args))
    if phase_conditioning_enabled(args):
        temporal_kwargs.update(phase_conditioning_parameters(args))
    lit_model.model = policy_class(
        None, lit_model.actions_shape, lit_model.init_noise_std,
        lit_model.model_cfg, lit_model.encoder_cfg, env_cfg,
        task_category_names=task_categories(args),
        use_task_input=task_input_enabled(args), **temporal_kwargs)
    return lit_model


def expand_standard_state_dict_for_task_model(model, state_dict):
    """Expand a standard 384-D actor checkpoint to the 388-D Task-ID actor."""
    target = model.state_dict()
    if ACTOR_INPUT_WEIGHT_KEY not in state_dict:
        raise KeyError("Checkpoint is missing {}".format(
            ACTOR_INPUT_WEIGHT_KEY))
    source_weight = state_dict[ACTOR_INPUT_WEIGHT_KEY]
    target_weight = target[ACTOR_INPUT_WEIGHT_KEY]
    expanded = OrderedDict(state_dict)
    changed = False
    allowed_prefixes = []
    if temporal_attention_enabled(model.args):
        allowed_prefixes.extend((
            "model.history_token_projection.",
            "model.history_position_embedding",
            "model.history_transformer.",
            "model.history_attention_norm.",
            "model.history_attention_action_head.",
        ))
    if action_chunk_aux_enabled(model.args):
        allowed_prefixes.append("model.future_action_head.")
    if full_observation_gru_enabled(model.args):
        allowed_prefixes.extend((
            "model.full_history_gru.",
            "model.full_history_norm.",
            "model.full_history_action_head.",
        ))
    if allowed_prefixes:
        allowed_prefixes = tuple(allowed_prefixes)
        missing = [key for key in target if key not in expanded]
        unexpected_missing = [
            key for key in missing
            if not key.startswith(allowed_prefixes)
        ]
        if unexpected_missing:
            raise KeyError(
                "Checkpoint is missing non-attention parameters: {}".format(
                    unexpected_missing))
        for key in missing:
            expanded[key] = target[key].detach().clone()
        if action_chunk_aux_enabled(model.args):
            future_weight_key = "model.future_action_head.weight"
            future_bias_key = "model.future_action_head.bias"
            if future_weight_key in missing:
                actor_weight_keys = [
                    key for key in state_dict
                    if key.startswith("model.actor.")
                    and key.endswith(".weight")
                ]
                if not actor_weight_keys:
                    raise KeyError("Checkpoint has no actor linear weights")
                final_actor_weight_key = max(
                    actor_weight_keys,
                    key=lambda key: int(key.split(".")[2]))
                final_actor_bias_key = final_actor_weight_key.replace(
                    ".weight", ".bias")
                source_final_weight = state_dict[final_actor_weight_key]
                source_final_bias = state_dict[final_actor_bias_key]
                repeat = (
                    target[future_weight_key].shape[0]
                    // source_final_weight.shape[0])
                if (
                    repeat * source_final_weight.shape[0]
                    != target[future_weight_key].shape[0]
                ):
                    raise ValueError(
                        "Future action head is not an integer action multiple")
                expanded[future_weight_key] = source_final_weight.repeat(
                    repeat, 1).detach().clone()
                expanded[future_bias_key] = source_final_bias.repeat(
                    repeat).detach().clone()
        changed = changed or bool(missing)
    if source_weight.shape == target_weight.shape:
        return expanded, changed
    if (source_weight.shape[0] != target_weight.shape[0]
            or source_weight.shape[1] >= target_weight.shape[1]):
        raise ValueError(
            "Cannot expand actor input weight {} to {}".format(
                tuple(source_weight.shape), tuple(target_weight.shape)))
    base_dim = target_weight.shape[1]
    if temporal_history_enabled(model.args):
        history_frames, prop_dim, action_dim = temporal_history_dimensions(
            model.args)
        base_dim -= (history_frames - 1) * (prop_dim + action_dim)
    valid_source_widths = {
        base_dim,
        base_dim - len(TASK_CATEGORIES),
    }
    if phase_conditioning_enabled(model.args):
        valid_source_widths.add(target_weight.shape[1] - 1)
    if source_weight.shape[1] not in valid_source_widths:
        raise ValueError(
            "Cannot expand actor input width {} to {}; expected source width "
            "in {}".format(
                source_weight.shape[1], target_weight.shape[1],
                sorted(valid_source_widths)))
    new_weight = target_weight.detach().clone()
    new_weight[:, :source_weight.shape[1]].copy_(source_weight)
    new_weight[:, source_weight.shape[1]:].zero_()
    expanded[ACTOR_INPUT_WEIGHT_KEY] = new_weight
    return expanded, True


def set_inference_tasks(lit_model, object_ids, object_indices=None):
    """Attach per-environment Task IDs; no-op for a standard BC policy."""
    policy = getattr(lit_model, "model", None)
    if not isinstance(policy, TaskConditionedActorCriticDexRep):
        return False
    object_ids = list(object_ids)
    if object_indices is None:
        if len(object_ids) != 1:
            raise ValueError(
                "Multiple objects require per-environment object indices")
        categories = [category_from_object_id(object_ids[0])]
    else:
        if torch.is_tensor(object_indices):
            object_indices = object_indices.detach().cpu().tolist()
        categories = [
            category_from_object_id(object_ids[int(index)])
            for index in object_indices]
    policy.set_inference_task_categories(categories)
    reset_history = getattr(policy, "reset_inference_history", None)
    if reset_history is not None:
        reset_history()
    return True
