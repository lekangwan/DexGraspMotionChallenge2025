"""直接读取本项目真实预处理数据和在线采集NPZ。

该文件只依赖 ``dexgrasp/dataset`` 中的数据文件，不依赖原 ``custom_tools`` 的Dataset。
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .data import FrameDataset, TrajectoryBatch
from .model import CATEGORIES, filter_observation


def category_index(object_id: str) -> int:
    """输入形如``core-bottle-...``的物体ID，输出0到3类别编号。

    内部解析第二段类别名；作用是为共享策略生成固定Task-ID。
    """
    parts = object_id.split("-", 2)
    if len(parts) < 3 or parts[1] not in CATEGORIES:
        raise ValueError(f"无法从物体ID识别四类任务：{object_id}")
    return CATEGORIES.index(parts[1])


def load_offline_trajectories(
    directory: str,
    sequence_limit: int,
    teacher_action_file: Optional[str] = None,
    category_filter: Optional[str] = None,
) -> TrajectoryBatch:
    """读取 ``<object-id>.npy``，得到实际离线训练轨迹。

    文件按物体ID排序，和正式代码生成路由教师标签后的稳定顺序一致。观测和动作保持
    ``[轨迹, 帧, 维度]``，因此构造历史时不会跨越轨迹边界。
    """
    root = Path(directory).expanduser().resolve()
    paths = sorted(path for path in root.glob("*.npy") if path.stat().st_size >= 1024)
    observations, demos, categories = [], [], []
    used = 0
    for path in paths:
        if category_filter is not None and path.stem.split("-", 2)[1] != category_filter:
            continue
        data = np.load(str(path), allow_pickle=True).item()
        if "obs" not in data or "vis_unscale_actions" not in data:
            raise KeyError(f"{path} 尚未包含预处理obs和vis_unscale_actions")
        take = len(data["obs"])
        if sequence_limit > 0:
            take = min(take, sequence_limit - used)
        if take <= 0:
            break
        observations.append(data["obs"][:take].astype(np.float32, copy=False))
        demos.append(data["vis_unscale_actions"][:take].astype(np.float32, copy=False))
        categories.extend([category_index(path.stem)] * take)
        used += take
        if sequence_limit > 0 and used >= sequence_limit:
            break

    if not observations:
        raise FileNotFoundError(f"没有找到可训练的预处理轨迹：{root}")
    observation = torch.from_numpy(np.concatenate(observations))
    observation = filter_observation(observation)
    demo_array = np.concatenate(demos).copy()
    # 预处理只真实写入了0到68帧动作；第69帧应保持第68帧目标，而不是回到零位。
    demo_array[:, -1] = demo_array[:, -2]
    demo = torch.from_numpy(demo_array)
    if teacher_action_file:
        labels = np.load(str(Path(teacher_action_file).expanduser().resolve()), allow_pickle=False)
        flat_teacher = labels["teacher_actions"].astype(np.float32, copy=True)
        expected = observation.shape[0] * observation.shape[1]
        if flat_teacher.shape != (expected, 28):
            raise ValueError(f"教师标签应为 {(expected, 28)}，实际为 {flat_teacher.shape}")
        teacher_array = flat_teacher.reshape(observation.shape[0], observation.shape[1], 28)
        # 旧教师文件最后一帧标签错位，因此主线训练也用示范的保持动作覆盖它。
        teacher_array[:, -1] = demo_array[:, -1]
        teacher = torch.from_numpy(teacher_array)
    else:
        teacher = demo.clone()
    return TrajectoryBatch(
        observations=observation,
        demo_actions=demo,
        teacher_actions=teacher,
        executed_actions=demo,
        category_indices=torch.tensor(categories, dtype=torch.long),
        source=0,
    )


def load_online_trajectories(path: str) -> TrajectoryBatch:
    """把正式采集的逐帧在线NPZ恢复为轨迹，供Temporal3正确构造历史。"""
    data = np.load(str(Path(path).expanduser().resolve()), allow_pickle=False)
    required = (
        "observations", "teacher_actions", "student_actions", "category_indices",
        "object_indices", "trajectory_indices", "frame_indices",
    )
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"在线数据缺少字段：{missing}")

    groups: Dict[Tuple[int, int], List[int]] = {}
    for index, key in enumerate(zip(data["object_indices"], data["trajectory_indices"])):
        groups.setdefault((int(key[0]), int(key[1])), []).append(index)
    ordered = []
    for key in sorted(groups):
        indices = sorted(groups[key], key=lambda i: int(data["frame_indices"][i]))
        frames = data["frame_indices"][indices].astype(np.int64)
        if not np.array_equal(frames, np.arange(len(frames))):
            raise ValueError(f"在线轨迹{key}帧不连续：{frames.tolist()}")
        ordered.append(indices)
    lengths = {len(indices) for indices in ordered}
    if len(lengths) != 1:
        raise ValueError(f"精简版要求在线轨迹等长，检测到长度：{sorted(lengths)}")

    def stack(name):
        """输入NPZ字段名，输出按轨迹和帧恢复的浮点张量。

        内部按已排序索引逐轨迹堆叠；作用是避免在线历史跨越episode。
        """
        return torch.from_numpy(np.stack([data[name][indices] for indices in ordered]).astype(np.float32))

    observation = filter_observation(stack("observations"))
    teacher = stack("teacher_actions")
    student = stack("student_actions")
    categories = torch.tensor(
        [int(data["category_indices"][indices[0]]) for indices in ordered], dtype=torch.long
    )
    return TrajectoryBatch(
        observations=observation,
        demo_actions=teacher.clone(),  # 在线状态没有对应示范，目标始终使用教师动作。
        teacher_actions=teacher,
        executed_actions=student,
        category_indices=categories,
        source=1,
    )


def build_training_dataset(
    offline_directory: str,
    sequence_limit: int,
    teacher_action_file: Optional[str],
    online_action_file: Optional[str],
    history_steps: int,
    noise: float = 0.05,
    category_filter: Optional[str] = None,
) -> FrameDataset:
    """输入真实数据路径和训练设置，输出最终FrameDataset。

    内部加载离线轨迹并可选追加在线rollout；作用是为训练入口提供统一数据对象。
    """
    groups = [load_offline_trajectories(
        offline_directory, sequence_limit, teacher_action_file, category_filter)]
    if online_action_file:
        groups.append(load_online_trajectories(online_action_file))
    return FrameDataset(
        groups, history_steps=history_steps, proprioception_noise=noise)
