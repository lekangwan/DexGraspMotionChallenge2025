"""读取第二版几何策略数据，并安全构造历史与未来动作块。"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class GeometryPolicyDataset(Dataset):
    """以物理步为样本的几何条件动作块数据集。"""

    def __init__(self, data_dir, split, model_type, history=3, action_horizon=8):
        data_dir = Path(data_dir)
        with np.load(data_dir / f"{split}.npz", allow_pickle=False) as archive:
            self.data = {name: archive[name].copy() for name in archive.files}
        with np.load(data_dir / f"geometry_{split}.npz", allow_pickle=False) as archive:
            geometry = {name: archive[name].copy() for name in archive.files}
        with np.load(data_dir / "geometry_normalization.npz", allow_pickle=False) as archive:
            self.normalization = {name: archive[name].astype(np.float32) for name in archive.files}
        if model_type not in {
            "geometry_phase", "geometry_chunk", "geometry_plan_chunk",
            "geometry_temporal_chunk",
        }:
            raise ValueError(f"未知模型类型: {model_type}")
        self.model_type = model_type
        self.history = int(history)
        self.action_horizon = 1 if model_type == "geometry_phase" else int(action_horizon)
        self.observations = (
            self.data["observations"] - self.normalization["observation_mean"]
        ) / self.normalization["observation_std"]
        self.actions = self.data["actions"].astype(np.float32)
        trajectory_ids = self.data["trajectory_id"].astype(np.int64)
        self.trajectory_indices = {
            int(value): np.flatnonzero(trajectory_ids == value)
            for value in np.unique(trajectory_ids)
        }
        geometry_ids = geometry["trajectory_id"].astype(np.int64)
        self.geometry_row = {int(value): index for index, value in enumerate(geometry_ids)}
        if set(self.geometry_row) != set(self.trajectory_indices):
            raise ValueError(f"{split}的几何sidecar与轨迹ID不一致")
        self.initial_commands = (
            geometry["initial_command"] - self.normalization["initial_command_mean"]
        ) / self.normalization["initial_command_std"]
        self.object_points = (
            geometry["object_points"] - self.normalization["point_mean"]
        ) / self.normalization["point_std"]
        self.local_position = np.empty(len(trajectory_ids), dtype=np.int64)
        self.initial_observations = np.empty_like(self.observations)
        self.previous_deltas = np.empty_like(self.actions)
        self.phase = np.empty((len(self.actions), 1), dtype=np.float32)
        is_hold = self.data.get("is_hold", np.zeros(len(self.actions), dtype=bool)).astype(bool)
        for trajectory_id, indices in self.trajectory_indices.items():
            if not np.array_equal(indices, np.arange(indices[0], indices[-1] + 1)):
                raise ValueError("同一轨迹的物理步必须连续存储")
            row = self.geometry_row[trajectory_id]
            raw_initial = geometry["initial_command"][row]
            self.local_position[indices] = np.arange(len(indices))
            self.initial_observations[indices] = self.observations[indices[0]]
            previous = np.vstack([raw_initial[None], self.actions[indices[:-1]]])
            self.previous_deltas[indices] = (
                previous - raw_initial - self.normalization["initial_delta_mean"]
            ) / self.normalization["initial_delta_std"]
            motion_steps = max(int(np.count_nonzero(~is_hold[indices])), 2)
            self.phase[indices, 0] = np.minimum(
                np.arange(len(indices), dtype=np.float32) / float(motion_steps - 1), 1.0
            )

    @property
    def observation_dim(self):
        return int(self.observations.shape[1])

    @property
    def action_dim(self):
        return int(self.actions.shape[1])

    @property
    def point_count(self):
        return int(self.object_points.shape[1])

    def __len__(self):
        return len(self.actions)

    def _indices(self, index, before, after):
        trajectory_id = int(self.data["trajectory_id"][index])
        indices = self.trajectory_indices[trajectory_id]
        local = int(self.local_position[index])
        offsets = np.arange(-before, after + 1)
        return indices[np.clip(local + offsets, 0, len(indices) - 1)]

    def __getitem__(self, index):
        trajectory_id = int(self.data["trajectory_id"][index])
        row = self.geometry_row[trajectory_id]
        history = self._indices(index, self.history - 1, 0)
        future = self._indices(index, 0, self.action_horizon - 1)
        initial_raw = (
            self.initial_commands[row] * self.normalization["initial_command_std"]
            + self.normalization["initial_command_mean"]
        )
        target = (
            self.actions[future] - initial_raw - self.normalization["initial_delta_mean"]
        ) / self.normalization["initial_delta_std"]
        return {
            "initial_observation": torch.from_numpy(self.initial_observations[index]).float(),
            "initial_command": torch.from_numpy(self.initial_commands[row]).float(),
            "object_points": torch.from_numpy(self.object_points[row]).float(),
            "observation_history": torch.from_numpy(self.observations[history]).float(),
            "previous_delta_history": torch.from_numpy(self.previous_deltas[history]).float(),
            "phase": torch.from_numpy(self.phase[index]).float(),
            "action_chunk": torch.from_numpy(target).float(),
        }
