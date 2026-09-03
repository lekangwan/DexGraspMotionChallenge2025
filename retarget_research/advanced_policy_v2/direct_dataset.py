"""读取动态手物表征，构造Temporal3直接动作监督样本。"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class DirectInteractionDataset(Dataset):
    """每个样本包含初始任务、最近三步真实状态和未来动作块。"""

    def __init__(self, data_dir, split, history=3, action_horizon=1):
        data_dir = Path(data_dir)
        with np.load(data_dir / f"{split}.npz", allow_pickle=False) as archive:
            self.data = {name: archive[name].copy() for name in archive.files}
        with np.load(data_dir / f"geometry_{split}.npz", allow_pickle=False) as archive:
            geometry = {name: archive[name].copy() for name in archive.files}
        with np.load(
            data_dir / f"direct_interaction_{split}.npz", allow_pickle=False
        ) as archive:
            interaction = archive["interaction"].copy()
        with np.load(data_dir / "geometry_normalization.npz", allow_pickle=False) as archive:
            self.normalization = {
                name: archive[name].astype(np.float32) for name in archive.files
            }
        with np.load(
            data_dir / "direct_interaction_normalization.npz", allow_pickle=False
        ) as archive:
            self.interaction_mean = archive["interaction_mean"].astype(np.float32)
            self.interaction_std = archive["interaction_std"].astype(np.float32)

        self.history = int(history)
        self.action_horizon = int(action_horizon)
        self.observations = (
            self.data["observations"] - self.normalization["observation_mean"]
        ) / self.normalization["observation_std"]
        self.interactions = (
            interaction - self.interaction_mean
        ) / self.interaction_std
        self.actions = self.data["actions"].astype(np.float32)
        trajectory_ids = self.data["trajectory_id"].astype(np.int64)
        self.trajectory_indices = {
            int(value): np.flatnonzero(trajectory_ids == value)
            for value in np.unique(trajectory_ids)
        }
        geometry_rows = {
            int(value): row for row, value in enumerate(geometry["trajectory_id"])
        }
        self.initial_commands = (
            geometry["initial_command"] - self.normalization["initial_command_mean"]
        ) / self.normalization["initial_command_std"]
        self.object_points = (
            geometry["object_points"] - self.normalization["point_mean"]
        ) / self.normalization["point_std"]
        self.geometry_row = geometry_rows
        self.local_position = np.empty(len(self.actions), dtype=np.int64)
        self.initial_observations = np.empty_like(self.observations)
        self.previous_deltas = np.empty_like(self.actions)
        self.phase = np.empty((len(self.actions), 1), dtype=np.float32)
        is_hold = self.data["is_hold"].astype(bool)
        for trajectory_id, indices in self.trajectory_indices.items():
            row = geometry_rows[trajectory_id]
            initial = geometry["initial_command"][row]
            self.local_position[indices] = np.arange(len(indices))
            self.initial_observations[indices] = self.observations[indices[0]]
            previous = np.vstack([initial[None], self.actions[indices[:-1]]])
            self.previous_deltas[indices] = (
                previous - initial - self.normalization["initial_delta_mean"]
            ) / self.normalization["initial_delta_std"]
            motion_steps = max(int(np.count_nonzero(~is_hold[indices])), 2)
            self.phase[indices, 0] = np.minimum(
                np.arange(len(indices), dtype=np.float32) / (motion_steps - 1), 1.0
            )

    def __len__(self):
        return len(self.actions)

    def _window(self, index, before, after):
        trajectory_id = int(self.data["trajectory_id"][index])
        indices = self.trajectory_indices[trajectory_id]
        local = int(self.local_position[index])
        offsets = np.arange(-before, after + 1)
        return indices[np.clip(local + offsets, 0, len(indices) - 1)]

    def __getitem__(self, index):
        trajectory_id = int(self.data["trajectory_id"][index])
        row = self.geometry_row[trajectory_id]
        history = self._window(index, self.history - 1, 0)
        future = self._window(index, 0, self.action_horizon - 1)
        initial = (
            self.initial_commands[row] * self.normalization["initial_command_std"]
            + self.normalization["initial_command_mean"]
        )
        target = (
            self.actions[future] - initial - self.normalization["initial_delta_mean"]
        ) / self.normalization["initial_delta_std"]
        return {
            "initial_observation": torch.from_numpy(self.initial_observations[index]).float(),
            "initial_command": torch.from_numpy(self.initial_commands[row]).float(),
            "object_points": torch.from_numpy(self.object_points[row]).float(),
            "observation_history": torch.from_numpy(self.observations[history]).float(),
            "interaction_history": torch.from_numpy(self.interactions[history]).float(),
            "previous_delta_history": torch.from_numpy(self.previous_deltas[history]).float(),
            "phase": torch.from_numpy(self.phase[index]).float(),
            "action_chunk": torch.from_numpy(target).float(),
        }

    @property
    def observation_dim(self):
        return int(self.observations.shape[1])

    @property
    def action_dim(self):
        return int(self.actions.shape[1])

    @property
    def interaction_dim(self):
        return int(self.interactions.shape[1])

    @property
    def point_count(self):
        return int(self.object_points.shape[1])
