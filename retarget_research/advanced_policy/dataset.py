"""读取标准策略NPZ并构造单帧、Temporal3或动作片段样本。

输入：`prepare_policy_dataset.py`生成的split与normalization NPZ。
输出：PyTorch Dataset样本，保证历史和未来窗口不跨轨迹边界。
内部逻辑：建立每个步骤在自身轨迹内的位置索引，开头复制首帧，末尾复制末帧。
作用：避免时间窗口把相邻文件误拼在一起，并统一三类策略的归一化口径。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class TargetHandPolicyDataset(Dataset):
    """目标手状态—动作监督数据集。"""

    def __init__(
        self,
        data_path,
        normalization_path,
        mode="bc",
        history=3,
        action_horizon=8,
        teacher_labels_path=None,
    ):
        """读取数组、标准化观测/动作并预计算每条轨迹边界。

        输入：split路径、归一化路径、模式、历史长度、动作片段长度和可选教师标签。
        输出：构造完成的数据集对象。
        内部逻辑：按trajectory_id收集全局索引；拒绝非连续重复ID和未知模式。
        作用：训练循环无需了解NPZ布局即可安全抽样；统一学生可在同一索引读取教师动作。
        """
        if mode not in {"bc", "phase_residual", "phase_residual_temporal", "temporal3", "diffusion"}:
            raise ValueError(f"未知数据模式: {mode}")
        if mode == "temporal3" and int(history) != 3:
            raise ValueError("Temporal3的history必须严格等于3")
        with np.load(Path(data_path), allow_pickle=False) as archive:
            self.data = {name: archive[name].copy() for name in archive.files}
        with np.load(Path(normalization_path), allow_pickle=False) as archive:
            self.normalization = {
                name: archive[name].astype(np.float32) for name in archive.files
            }
        self.mode = mode
        self.history = int(history)
        self.action_horizon = int(action_horizon)
        self.observations = (
            self.data["observations"] - self.normalization["observation_mean"]
        ) / self.normalization["observation_std"]
        self.actions = (
            self.data["actions"] - self.normalization["action_mean"]
        ) / self.normalization["action_std"]
        executed = self.data.get("executed_actions", self.data["actions"])
        if executed.shape != self.data["actions"].shape:
            raise ValueError("executed_actions与监督actions尺寸不一致")
        self.history_actions = (
            executed - self.normalization["action_mean"]
        ) / self.normalization["action_std"]
        self.teacher_actions = None
        if teacher_labels_path is not None:
            with np.load(Path(teacher_labels_path), allow_pickle=False) as archive:
                if "teacher_actions" not in archive.files:
                    raise ValueError("教师标签文件缺少teacher_actions")
                labels = archive["teacher_actions"].astype(np.float32)
                label_trajectory_ids = archive["trajectory_id"].astype(np.int64)
            if labels.shape != self.actions.shape:
                raise ValueError(
                    f"教师标签尺寸{labels.shape}与动作尺寸{self.actions.shape}不一致"
                )
            if not np.array_equal(label_trajectory_ids, self.data["trajectory_id"]):
                raise ValueError("教师标签的轨迹顺序与策略数据不一致")
            self.teacher_actions = labels
        trajectory_ids = self.data["trajectory_id"].astype(np.int64)
        self.trajectory_indices = {
            int(trajectory_id): np.flatnonzero(trajectory_ids == trajectory_id)
            for trajectory_id in np.unique(trajectory_ids)
        }
        self.local_position = np.empty(len(trajectory_ids), dtype=np.int64)
        for indices in self.trajectory_indices.values():
            if not np.array_equal(indices, np.arange(indices[0], indices[-1] + 1)):
                raise ValueError("同一trajectory_id的步骤必须连续存储")
            self.local_position[indices] = np.arange(len(indices))
        self.previous_commands = None
        self.action_deltas = None
        self.phase = None
        if self.mode in ("phase_residual", "phase_residual_temporal"):
            required = {"action_delta_mean", "action_delta_std"}
            missing = required - set(self.normalization)
            if missing:
                raise ValueError(f"残差策略缺少训练增量统计: {sorted(missing)}")
            if "previous_commands" in self.data:
                previous_raw = self.data["previous_commands"].astype(np.float32)
                if previous_raw.shape != self.data["actions"].shape:
                    raise ValueError("previous_commands与actions形状不一致")
            else:
                previous_raw = np.empty_like(self.data["actions"], dtype=np.float32)
            phase = np.empty((len(self.actions), 1), dtype=np.float32)
            is_hold = self.data.get(
                "is_hold", np.zeros(len(self.actions), dtype=bool)
            ).astype(bool)
            if "previous_commands" not in self.data:
                for indices in self.trajectory_indices.values():
                    if len(indices) < 2:
                        raise ValueError("残差策略每条轨迹至少需要两个动作")
                    raw = self.data["actions"][indices]
                    # 专家trace首步是张开初态到第0帧的1/3线性插值，故由前两步
                    # `open = 2*a0-a1`精确恢复episode真正的上一命令。
                    previous_raw[indices[0]] = 2.0 * raw[0] - raw[1]
                    previous_raw[indices[1:]] = raw[:-1]
            for indices in self.trajectory_indices.values():
                non_hold = int(np.count_nonzero(~is_hold[indices]))
                motion_denominator = max(non_hold - 1, 1)
                phase[indices, 0] = np.minimum(
                    np.arange(len(indices), dtype=np.float32) / motion_denominator,
                    1.0,
                )
            self.previous_commands = (
                previous_raw - self.normalization["action_mean"]
            ) / self.normalization["action_std"]
            raw_delta = self.data["actions"] - previous_raw
            self.action_deltas = (
                raw_delta - self.normalization["action_delta_mean"]
            ) / self.normalization["action_delta_std"]
            self.phase = phase

    def __len__(self):
        """返回可作为当前时刻的物理步数量。"""
        return len(self.observations)

    def _window_indices(self, index, before, after):
        """生成不跨轨迹且端点复制的窗口全局索引。

        输入：当前全局index、向前/向后步数。
        输出：长度`before+after+1`的全局索引。
        内部逻辑：在当前trajectory局部坐标上clip，再映射回全局数组。
        作用：Temporal3开头与Diffusion片段末尾都无需特殊丢样本。
        """
        trajectory_id = int(self.data["trajectory_id"][index])
        indices = self.trajectory_indices[trajectory_id]
        local = int(self.local_position[index])
        offsets = np.arange(-before, after + 1)
        local_indices = np.clip(local + offsets, 0, len(indices) - 1)
        return indices[local_indices]

    def __getitem__(self, index):
        """按模式返回当前或时序监督样本。

        输入：全局步骤索引。
        输出：含观测、目标动作、类别和可选历史/动作片段的张量字典。
        内部逻辑：BC取当前；Temporal3取三观测、前两动作；Diffusion取历史观测和未来片段。
        作用：让训练脚本只根据model_type选择loss，而不重复窗口逻辑。
        """
        category = torch.tensor(self.data["category_id"][index], dtype=torch.long)
        if self.mode in ("phase_residual",):
            return {
                "observations": torch.from_numpy(self.observations[index]).float(),
                "previous_actions": torch.from_numpy(
                    self.previous_commands[index]
                ).float(),
                "phase": torch.from_numpy(self.phase[index]).float(),
                "action_deltas": torch.from_numpy(self.action_deltas[index]).float(),
                "category_id": category,
            }
        if self.mode == "bc":
            sample = {
                "observations": torch.from_numpy(self.observations[index]).float(),
                "actions": torch.from_numpy(self.actions[index]).float(),
                "category_id": category,
            }
            if self.teacher_actions is not None:
                sample["teacher_actions"] = torch.from_numpy(
                    self.teacher_actions[index]
                ).float()
            return sample
        history_indices = self._window_indices(index, self.history - 1, 0)
        if self.mode == "phase_residual_temporal":
            trajectory_id = int(self.data["trajectory_id"][index])
            indices = self.trajectory_indices[trajectory_id]
            local = int(self.local_position[index])
            previous_actions = []
            for offset in range(-(self.history - 1), 0):
                previous_local = local + offset
                previous_actions.append(
                    np.zeros(self.actions.shape[1], dtype=np.float32)
                    if previous_local < 0
                    else self.previous_commands[indices[previous_local]]
                )
            return {
                "observation_history": torch.from_numpy(
                    self.observations[history_indices]
                ).float(),
                "previous_actions": torch.from_numpy(
                    np.stack(previous_actions)
                ).float(),
                "phase": torch.from_numpy(self.phase[index]).float(),
                "action_deltas": torch.from_numpy(self.action_deltas[index]).float(),
                "category_id": category,
            }
        if self.mode == "temporal3":
            trajectory_id = int(self.data["trajectory_id"][index])
            indices = self.trajectory_indices[trajectory_id]
            local = int(self.local_position[index])
            previous_actions = []
            for offset in range(-(self.history - 1), 0):
                previous_local = local + offset
                previous_actions.append(
                    np.zeros(self.actions.shape[1], dtype=np.float32)
                    if previous_local < 0
                    else self.history_actions[indices[previous_local]]
                )
            return {
                "observation_history": torch.from_numpy(
                    self.observations[history_indices]
                ).float(),
                "previous_actions": torch.from_numpy(
                    np.stack(previous_actions)
                ).float(),
                "actions": torch.from_numpy(self.actions[index]).float(),
                "category_id": category,
            }
        future_indices = self._window_indices(index, 0, self.action_horizon - 1)
        return {
            "observation_history": torch.from_numpy(
                self.observations[history_indices]
            ).float(),
            "action_sequence": torch.from_numpy(self.actions[future_indices]).float(),
            "category_id": category,
        }
