"""无需Isaac Gym的最小实现回归测试。"""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from custom_tools.minimal_impl.data import balanced_sampler
from custom_tools.minimal_impl.model import (
    ACTION_DIM, CHUNK_HORIZON, FILTERED_OBS_DIM, RAW_OBS_DIM,
    Chunk8Policy, ChunkEnsembler, HistoryBuffer, category_one_hot,
    filter_observation, weighted_model_soup,
)
from custom_tools.minimal_impl.project_data import build_training_dataset


def test_model_and_temporal_logic() -> None:
    """无输入输出，检查模型维度、历史、动作块集成和Soup。

    内部完全使用随机张量；作用是快速发现模型结构或时间对齐被意外改坏。
    """
    torch.manual_seed(7)
    raw = torch.randn(3, RAW_OBS_DIM)
    filtered = filter_observation(raw)
    assert filtered.shape == (3, FILTERED_OBS_DIM)

    model = Chunk8Policy(hidden_dims=(32, 32))
    task = category_one_hot(torch.tensor([0, 1, 2]))
    history = HistoryBuffer(2)
    chunk = model.forward_action_chunk(filtered, task, history.features(filtered))
    assert chunk.shape == (3, CHUNK_HORIZON, ACTION_DIM)

    ensemble = ChunkEnsembler(decay=0.0)
    first = ensemble.select(torch.ones_like(chunk))
    second = ensemble.select(torch.cat((2 * torch.ones_like(chunk[:, :1]), chunk[:, 1:]), dim=1))
    torch.testing.assert_close(first, torch.ones_like(first))
    torch.testing.assert_close(second, 1.5 * torch.ones_like(second))

    soup = weighted_model_soup((model.state_dict(), model.state_dict()), (2, 1))
    for name, value in model.state_dict().items():
        torch.testing.assert_close(soup[name], value)


def test_dataset_logic() -> None:
    """无输入输出，检查逐帧数据集、末帧修复及在线轨迹恢复。

    内部在临时目录创建最小NPY/NPZ；作用是覆盖真实训练入口曾遗漏的数据连接层。
    """
    n, t = 2, 4
    observations = np.random.randn(n, t, FILTERED_OBS_DIM).astype(np.float32)
    actions = np.random.randn(n, t, ACTION_DIM).astype(np.float32)
    teacher = (actions + 0.1).reshape(n * t, ACTION_DIM)

    with TemporaryDirectory() as directory:
        root = Path(directory)
        offline = root / "offline"
        offline.mkdir()
        np.save(offline / "core-bottle-example.npy", {
            "obs": observations,
            "vis_unscale_actions": actions,
        })
        np.savez(root / "teacher.npz", teacher_actions=teacher)

        online_observations = np.random.randn(t, FILTERED_OBS_DIM).astype(np.float32)
        online_teacher = np.random.randn(t, ACTION_DIM).astype(np.float32)
        np.savez(
            root / "online.npz",
            observations=online_observations,
            teacher_actions=online_teacher,
            student_actions=online_teacher - 0.1,
            category_indices=np.ones(t, dtype=np.int64),
            object_indices=np.zeros(t, dtype=np.int64),
            trajectory_indices=np.zeros(t, dtype=np.int64),
            frame_indices=np.arange(t, dtype=np.int64),
        )
        dataset = build_training_dataset(
            str(offline), n, str(root / "teacher.npz"),
            str(root / "online.npz"), history_steps=2, noise=0.0)

    assert len(dataset) == n * t + t
    assert dataset[0]["history"].shape == (2 * (100 + 28),)
    assert dataset[t - 1]["action_chunk_mask"].tolist() == [True] + [False] * 7
    torch.testing.assert_close(
        dataset.groups[0].demo_actions[:, -1],
        dataset.groups[0].demo_actions[:, -2])
    torch.testing.assert_close(
        dataset.groups[0].teacher_actions[:, -1],
        dataset.groups[0].demo_actions[:, -1])
    assert len(list(balanced_sampler(dataset, 0.25))) == len(dataset)


def main() -> None:
    """无输入，输出PASS文本。

    内部测试观测裁剪、模型形状、时间集成、历史边界、动作块mask和Soup；作用是用CPU
    快速发现整理代码时最容易产生的维度与时间对齐错误。
    """
    test_model_and_temporal_logic()
    test_dataset_logic()
    print("MINIMAL_IMPL_TEST=PASS")


if __name__ == "__main__":
    main()
