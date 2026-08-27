"""最终 Chunk8 训练所需的离线、在线、历史和动作块数据。"""

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from .model import ACTION_DIM, CHUNK_HORIZON, PROP_DIM, category_one_hot


@dataclass
class TrajectoryBatch:
    """统一保存一组等长轨迹。

    输入字段前两维均为[轨迹,帧]；输出是供FrameDataset读取的数据对象。内部同时保存
    原示范、教师标签和真正执行动作；作用是明确“监督动作”和“推进环境动作”的区别。
    """

    observations: torch.Tensor
    demo_actions: torch.Tensor
    teacher_actions: torch.Tensor
    executed_actions: torch.Tensor
    category_indices: torch.Tensor
    source: int

    def validate(self) -> None:
        """无额外输入输出，检查所有张量是否属于同一批轨迹。

        内部核对动作形状、类别数和来源编号；作用是在训练前尽早暴露数据错位。
        """
        n, t = self.observations.shape[:2]
        for name in ("demo_actions", "teacher_actions", "executed_actions"):
            if tuple(getattr(self, name).shape) != (n, t, ACTION_DIM):
                raise ValueError(f"{name}形状错误")
        if tuple(self.category_indices.shape) != (n,) or self.source not in (0, 1):
            raise ValueError("类别或数据来源格式错误")


class FrameDataset(Dataset):
    """把轨迹展开为逐帧样本，同时构造Temporal3历史和未来8步标签。"""

    def __init__(self, trajectory_groups: Sequence[TrajectoryBatch],
                 history_steps: int = 2, chunk_horizon: int = CHUNK_HORIZON,
                 proprioception_noise: float = 0.05):
        """输入轨迹组、历史长度、块长度和噪声，输出可索引数据集。

        内部建立(组,轨迹,帧)索引；作用是让DataLoader逐帧采样但不跨episode取历史。
        """
        self.groups = list(trajectory_groups)
        self.history_steps = history_steps
        self.chunk_horizon = chunk_horizon
        self.noise = proprioception_noise
        self.index, self.sample_categories, self.sample_sources = [], [], []
        for group_index, group in enumerate(self.groups):
            group.validate()
            n, t = group.observations.shape[:2]
            for trajectory in range(n):
                for frame in range(t):
                    self.index.append((group_index, trajectory, frame))
                    self.sample_categories.append(int(group.category_indices[trajectory]))
                    self.sample_sources.append(group.source)

    def __len__(self) -> int:
        """无输入，输出逐帧样本总数；作用是供PyTorch DataLoader确定epoch长度。"""
        return len(self.index)

    def _history(self, group: TrajectoryBatch, trajectory: int,
                 frame: int) -> torch.Tensor:
        """输入轨迹位置，输出前两步本体与实际动作的256维历史。

        episode开头本体复制首帧、动作置零；作用是复现闭环HistoryBuffer的语义。
        """
        props, actions = [], []
        for lag in range(self.history_steps, 0, -1):
            previous = frame - lag
            props.append(group.observations[trajectory, max(0, previous), :PROP_DIM].clone())
            actions.append(
                torch.zeros(ACTION_DIM, dtype=group.observations.dtype)
                if previous < 0 else group.executed_actions[trajectory, previous].clone())
        return torch.cat(props + actions) if props else torch.empty(0)

    def _action_chunk(self, actions: torch.Tensor, trajectory: int,
                      frame: int) -> tuple:
        """输入动作轨迹和当前位置，输出8步动作块及有效掩码。

        越过末帧的位置用最后动作补齐但标为无效；作用是形成固定形状且无伪标签的监督。
        """
        length = actions.shape[1]
        indices = [min(frame + offset, length - 1) for offset in range(self.chunk_horizon)]
        mask = torch.tensor([frame + offset < length for offset in range(self.chunk_horizon)])
        return actions[trajectory, indices].clone(), mask

    def __getitem__(self, item: int) -> dict:
        """输入样本编号，输出模型训练所需的一个字典。

        内部加入本体均匀噪声并构造类别、历史、示范块、教师块和掩码；作用是完整描述
        最终Chunk8的一次监督样本。
        """
        group_index, trajectory, frame = self.index[item]
        group = self.groups[group_index]
        observation = group.observations[trajectory, frame].clone()
        if self.noise:
            observation[:PROP_DIM] += torch.empty(PROP_DIM).uniform_(-self.noise, self.noise)
        demo_chunk, mask = self._action_chunk(group.demo_actions, trajectory, frame)
        teacher_chunk, _ = self._action_chunk(group.teacher_actions, trajectory, frame)
        category = group.category_indices[trajectory]
        return {
            "observation": observation,
            "task_one_hot": category_one_hot(category),
            "history": self._history(group, trajectory, frame),
            "demo_action_chunk": demo_chunk,
            "teacher_action_chunk": teacher_chunk,
            "action_chunk_mask": mask,
            "category": category,
            "source": torch.tensor(group.source),
        }


def balanced_sampler(dataset: FrameDataset,
                     online_fraction: Optional[float] = 0.25,
                     seed: int = 2025) -> WeightedRandomSampler:
    """输入数据集、期望在线占比和随机种子，输出带权重采样器。

    内部先给离线/在线来源分配75%/25%概率，再令各来源中的四类等概率；作用是避免
    类别数量和在线样本数量差异主导训练。
    """
    categories = torch.tensor(dataset.sample_categories)
    sources = torch.tensor(dataset.sample_sources)
    weights = torch.empty(len(dataset), dtype=torch.double)
    source_probabilities = ({0: 1.0} if online_fraction is None or not torch.any(sources == 1)
                            else {0: 1.0 - online_fraction, 1: online_fraction})
    for source, probability in source_probabilities.items():
        source_mask = sources == source
        present = torch.unique(categories[source_mask]).tolist()
        for category in present:
            mask = source_mask & (categories == category)
            weights[mask] = probability / len(present) / int(mask.sum())
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(weights, len(weights), replacement=True, generator=generator)
