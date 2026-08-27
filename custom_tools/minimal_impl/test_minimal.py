"""无需Isaac Gym的最小实现回归测试。"""

import torch

from custom_tools.minimal_impl.data import FrameDataset, TrajectoryBatch, balanced_sampler
from custom_tools.minimal_impl.model import (
    ACTION_DIM, CHUNK_HORIZON, FILTERED_OBS_DIM, RAW_OBS_DIM,
    Chunk8Policy, ChunkEnsembler, HistoryBuffer, category_one_hot,
    filter_observation, weighted_model_soup,
)


def main() -> None:
    """无输入，输出PASS文本。

    内部测试观测裁剪、模型形状、时间集成、历史边界、动作块mask和Soup；作用是用CPU
    快速发现整理代码时最容易产生的维度与时间对齐错误。
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

    n, t = 4, 5
    observations = torch.randn(n, t, FILTERED_OBS_DIM)
    actions = torch.randn(n, t, ACTION_DIM)
    offline = TrajectoryBatch(
        observations, actions, actions + 0.1, actions,
        torch.tensor([0, 1, 2, 3]), source=0)
    online = TrajectoryBatch(
        observations, actions, actions - 0.1, actions,
        torch.tensor([0, 1, 2, 3]), source=1)
    dataset = FrameDataset((offline, online), proprioception_noise=0.0)
    assert dataset[0]["history"].shape == (2 * (100 + 28),)
    assert dataset[t - 1]["action_chunk_mask"].tolist() == [True] + [False] * 7
    assert len(list(balanced_sampler(dataset, 0.25))) == len(dataset)

    soup = weighted_model_soup((model.state_dict(), model.state_dict()), (2, 1))
    for name, value in model.state_dict().items():
        torch.testing.assert_close(soup[name], value)
    print("MINIMAL_IMPL_TEST=PASS")


if __name__ == "__main__":
    main()
