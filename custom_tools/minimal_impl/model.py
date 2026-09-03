"""ShadowHand 最终 Chunk8 策略的教学用最小模型。"""

from collections import OrderedDict
from pathlib import Path
from typing import Optional, Sequence

import torch
from torch import nn
from torch.nn import functional as F


CATEGORIES = ("bottle", "mug", "bowl", "camera")
RAW_OBS_DIM = 2582
PROP_DIM = 100
DEXREP_SENSOR_DIM = 1080
DEXREP_PNL_DIM = 1280
FILTERED_OBS_DIM = PROP_DIM + DEXREP_SENSOR_DIM + DEXREP_PNL_DIM
ENCODED_OBS_DIM = 384
ACTION_DIM = 28
HISTORY_STEPS = 2
CHUNK_HORIZON = 8


def _observation_mask(device: torch.device) -> torch.Tensor:
    """输入设备，输出2582维观测的保留掩码。

    内部删除关节速度、力、部分指尖速度和物体速度共122维；作用是严格复现
    正式训练中 ``pro_dim=100`` 的观测过滤方式。
    """
    fingertip_velocity = torch.arange(84, 149, device=device).reshape(5, 13)[:, -6:].reshape(-1)
    removed = torch.cat((
        torch.arange(28, 56, device=device), torch.arange(56, 84, device=device),
        fingertip_velocity, torch.arange(149, 179, device=device),
        torch.arange(216, 222, device=device),
    ))
    mask = torch.ones(RAW_OBS_DIM, dtype=torch.bool, device=device)
    mask[removed] = False
    return mask


def filter_observation(observation: torch.Tensor) -> torch.Tensor:
    """输入末维2582或2460的观测，输出末维2460的策略观测。

    原始观测按固定掩码裁剪，已裁剪观测直接返回；作用是统一离线数据和
    Isaac Gym在线观测的格式。
    """
    if observation.shape[-1] == FILTERED_OBS_DIM:
        return observation
    if observation.shape[-1] != RAW_OBS_DIM:
        raise ValueError(f"观测维度应为{RAW_OBS_DIM}或{FILTERED_OBS_DIM}")
    return observation[..., _observation_mask(observation.device)]


def category_one_hot(category_indices: torch.Tensor) -> torch.Tensor:
    """输入类别编号0到3，输出固定顺序的4维one-hot。

    内部按 bottle、mug、bowl、camera 编码；作用是告诉共享策略当前物体类别。
    """
    indices = torch.as_tensor(category_indices, dtype=torch.long)
    if torch.any(indices < 0) or torch.any(indices >= len(CATEGORIES)):
        raise ValueError("类别编号必须位于[0, 3]")
    return F.one_hot(indices, num_classes=len(CATEGORIES)).float()


class DexRepEncoder(nn.Module):
    """把2460维观测编码为384维当前状态特征。

    100维本体、1080维Sensor和1280维PNL分别编码成128维，再拼接；两路
    DexRep几何特征做L2归一化，本体特征不做。作用是提取手物几何表示。
    """

    def __init__(self, embedding_dim: int = 128):
        """输入每路嵌入维数，输出初始化后的编码器。

        内部建立三条线性支路和PNL批归一化；作用是兼容正式DexRep权重。
        """
        super().__init__()
        self.state_encoder = nn.Linear(PROP_DIM, embedding_dim)
        self.sensor_encoder = nn.Linear(DEXREP_SENSOR_DIM, embedding_dim)
        self.pnl_norm = nn.BatchNorm1d(DEXREP_PNL_DIM)
        self.pnl_encoder = nn.Linear(DEXREP_PNL_DIM, embedding_dim)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """输入一批观测，输出一批384维编码。

        内部执行过滤、切分、编码和归一化；作用是完成策略的特征提取阶段。
        """
        prop, sensor, pnl = torch.split(
            filter_observation(observation),
            (PROP_DIM, DEXREP_SENSOR_DIM, DEXREP_PNL_DIM), dim=-1)
        return torch.cat((
            self.state_encoder(prop),
            F.normalize(self.sensor_encoder(sensor), dim=-1),
            F.normalize(self.pnl_encoder(self.pnl_norm(pnl)), dim=-1),
        ), dim=-1)


def _mlp(input_dim: int, hidden_dims: Sequence[int], output_dim: int) -> nn.Sequential:
    """输入网络维数，输出ELU多层感知机。

    内部在每个隐藏线性层后接ELU；作用是建立1024-1024-512-512动作网络。
    """
    layers = []
    width = input_dim
    for hidden in hidden_dims:
        layers.extend((nn.Linear(width, hidden), nn.ELU()))
        width = hidden
    layers.append(nn.Linear(width, output_dim))
    return nn.Sequential(*layers)


class Chunk8Policy(nn.Module):
    """由当前观测、类别和前两步历史预测未来8步28维动作。

    输入为384维DexRep编码、4维类别和256维历史。共享MLP后，当前头预测第0步，
    未来头预测后7步。作用是实现最终DexRep + Task-ID + Temporal3 + Chunk8网络。
    """

    def __init__(self, use_task_id: bool = True, history_steps: int = HISTORY_STEPS,
                 chunk_horizon: int = CHUNK_HORIZON,
                 hidden_dims: Sequence[int] = (1024, 1024, 512, 512)):
        """输入类别开关、历史长度、动作块长度和隐藏层，输出策略实例。

        内部计算拼接维数并建立共享动作干路与未来头；作用是定义可训练网络。
        """
        super().__init__()
        self.use_task_id = use_task_id
        self.history_steps = history_steps
        self.chunk_horizon = chunk_horizon
        self.task_dim = len(CATEGORIES) if use_task_id else 0
        self.history_dim = history_steps * (PROP_DIM + ACTION_DIM)
        input_dim = ENCODED_OBS_DIM + self.task_dim + self.history_dim
        self.encoder = DexRepEncoder()
        self.actor = _mlp(input_dim, hidden_dims, ACTION_DIM)
        self.future_action_head = nn.Linear(hidden_dims[-1], (chunk_horizon - 1) * ACTION_DIM)

    def _actor_input(self, observation: torch.Tensor,
                     task_one_hot: Optional[torch.Tensor],
                     history: Optional[torch.Tensor]) -> torch.Tensor:
        """输入当前观测、类别和历史，输出644维动作网络输入。

        内部检查维数后与DexRep编码拼接；作用是统一训练和推理的数据排列。
        """
        features = [self.encoder(observation)]
        if self.use_task_id:
            if task_one_hot is None or task_one_hot.shape[-1] != self.task_dim:
                raise ValueError("缺少4维Task-ID")
            features.append(task_one_hot.to(observation))
        if self.history_steps:
            if history is None or history.shape[-1] != self.history_dim:
                raise ValueError(f"历史特征应为{self.history_dim}维")
            features.append(history.to(observation))
        return torch.cat(features, dim=-1)

    def forward_action_chunk(self, observation: torch.Tensor,
                             task_one_hot: Optional[torch.Tensor] = None,
                             history: Optional[torch.Tensor] = None) -> torch.Tensor:
        """输入策略三项输入，输出形状[批量,8,28]的动作块。

        内部让当前头和未来头共享隐藏特征；作用是一次预测短期连续动作。
        """
        hidden = self.actor[:-1](self._actor_input(observation, task_one_hot, history))
        current = self.actor[-1](hidden).unsqueeze(1)
        future = self.future_action_head(hidden).reshape(
            observation.shape[0], self.chunk_horizon - 1, ACTION_DIM)
        return torch.cat((current, future), dim=1)

    def forward(self, observation: torch.Tensor,
                task_one_hot: Optional[torch.Tensor] = None,
                history: Optional[torch.Tensor] = None) -> torch.Tensor:
        """输入策略三项输入，输出新动作块的第一个28维动作。

        内部调用动作块前向并取索引0；作用是兼容单步教师和诊断接口。
        """
        return self.forward_action_chunk(observation, task_one_hot, history)[:, 0]


class HistoryBuffer:
    """将连续观测和实际动作整理为Temporal3的两步历史。"""

    def __init__(self, history_steps: int = HISTORY_STEPS):
        """输入历史步数，输出空缓存；张量在看到首帧后创建。"""
        self.history_steps = history_steps
        self.props = None
        self.actions = None

    def reset(self, observation: torch.Tensor) -> None:
        """输入episode首帧，无返回值。

        内部用首帧本体填充历史、动作置零；作用是阻止跨episode信息泄漏。
        """
        prop = filter_observation(observation)[:, :PROP_DIM]
        self.props = prop[:, None].repeat(1, self.history_steps, 1)
        self.actions = torch.zeros(
            observation.shape[0], self.history_steps, ACTION_DIM,
            device=observation.device, dtype=observation.dtype)

    def features(self, observation: torch.Tensor) -> torch.Tensor:
        """输入当前观测，输出形状[批量,256]的历史特征。

        内部按两步本体、两步动作展平拼接；作用是构造Temporal3输入。
        """
        if self.props is None or self.props.shape[0] != observation.shape[0]:
            self.reset(observation)
        return torch.cat((self.props.flatten(1), self.actions.flatten(1)), dim=-1)

    def append(self, observation: torch.Tensor, action: torch.Tensor) -> None:
        """输入当前观测和实际动作，无返回值。

        内部丢弃最旧槽并追加当前值；作用是为下一次决策更新历史。
        """
        prop = filter_observation(observation)[:, :PROP_DIM]
        self.props = torch.cat((self.props[:, 1:], prop[:, None]), dim=1)
        self.actions = torch.cat((self.actions[:, 1:], action[:, None]), dim=1)


class ChunkEnsembler:
    """融合最近多个动作块对当前时刻的重叠预测。"""

    def __init__(self, horizon: int = CHUNK_HORIZON, decay: float = 0.0):
        """输入块长度和年龄衰减，输出空集成器；最终衰减0代表等权。"""
        self.horizon = horizon
        self.decay = decay
        self.chunks = []

    def reset(self) -> None:
        """无输入输出，清空动作块；作用是隔离不同episode。"""
        self.chunks = []

    def select(self, new_chunk: torch.Tensor) -> torch.Tensor:
        """输入[批量,8,28]新动作块，输出[批量,28]当前动作。

        年龄a的块取第a项，按exp(-decay*a)归一化平均；作用是降低单次预测抖动。
        """
        self.chunks.append(new_chunk)
        self.chunks = self.chunks[-self.horizon:]
        predictions = [chunk[:, age] for age, chunk in enumerate(reversed(self.chunks))]
        stacked = torch.stack(predictions, dim=1)
        ages = torch.arange(len(predictions), device=stacked.device, dtype=stacked.dtype)
        weights = torch.exp(-self.decay * ages)
        return (stacked * (weights / weights.sum())[None, :, None]).sum(dim=1)


class PolicyRuntime:
    """把模型、Temporal3历史、Chunk8集成和后期抬升组成最终闭环策略。"""

    def __init__(self, policy: Chunk8Policy, decay: float = 0.0,
                 lift_start_step: int = 40, lift_z_boost: float = 0.20,
                 policy_steps: int = 70):
        """输入策略和推理超参数，输出运行时包装器。

        内部创建历史与集成缓存；作用是集中保存所有有状态推理逻辑。
        """
        self.policy = policy
        self.history = HistoryBuffer(policy.history_steps)
        self.ensemble = ChunkEnsembler(policy.chunk_horizon, decay)
        self.lift_start_step = lift_start_step
        self.lift_z_boost = lift_z_boost
        self.policy_steps = policy_steps

    def reset(self, observation: torch.Tensor) -> None:
        """输入episode首帧，无返回值；初始化历史并清空旧动作块。"""
        self.history.reset(observation)
        self.ensemble.reset()

    @torch.no_grad()
    def act(self, observation: torch.Tensor, task_one_hot: torch.Tensor,
            step: int) -> torch.Tensor:
        """输入当前观测、类别和步号，输出裁剪到[-1,1]的28维动作。

        内部生成动作块、等权集成并先写入模型历史；第40到69步把腕部z补偿从0线性
        增加到0.20。历史保存未加外部补偿的策略动作，与正式评测器一致。
        作用是执行冻结的最终推理协议。
        """
        processed = filter_observation(observation)
        history = self.history.features(processed)
        raw_action = self.ensemble.select(
            self.policy.forward_action_chunk(processed, task_one_hot, history))
        self.history.append(processed, raw_action)
        action = raw_action
        if step >= self.lift_start_step:
            action = action.clone()
            ramp = max(self.policy_steps - 1 - self.lift_start_step, 1)
            fraction = min((step - self.lift_start_step) / float(ramp), 1.0)
            action[:, 2] += self.lift_z_boost * fraction
        action = action.clamp(-1.0, 1.0)
        return action


def action_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """输入[N,28]预测和目标，输出标量监督损失。

    内部计算2×腕平移MSE、腕姿态MSE、手指MSE和全动作L1；作用是复现正式BC损失。
    """
    return (2.0 * F.mse_loss(prediction[:, :3], target[:, :3])
            + F.mse_loss(prediction[:, 3:6], target[:, 3:6])
            + F.mse_loss(prediction[:, 6:], target[:, 6:])
            + F.l1_loss(prediction, target))


def chunk_loss(prediction: torch.Tensor, target: torch.Tensor,
               mask: torch.Tensor) -> torch.Tensor:
    """输入预测/目标动作块和有效帧掩码，输出当前损失加未来平均损失。

    第0步始终计算，未来各步只计算未越过轨迹末端的样本；作用是避免补齐帧污染监督。
    """
    future = []
    for offset in range(1, prediction.shape[1]):
        valid = mask[:, offset].bool()
        if torch.any(valid):
            future.append(action_loss(prediction[valid, offset], target[valid, offset]))
    return action_loss(prediction[:, 0], target[:, 0]) + (
        torch.stack(future).mean() if future else prediction.sum() * 0.0)


def load_project_checkpoint(path: str, use_task_id: bool = True,
                            history_steps: int = HISTORY_STEPS,
                            chunk_horizon: int = CHUNK_HORIZON,
                            device: str = "cpu") -> Chunk8Policy:
    """输入正式/最小版checkpoint和结构参数，输出可推理Chunk8策略。

    最小版权重直接加载；正式Lightning权重去掉参数名前缀和无关critic。若载入的是
    Temporal3单步权重，未来7步头先复制当前动作头；作用是读取主线冻结权重或完成
    Temporal3到Chunk8的初始化。
    """
    checkpoint = torch.load(str(Path(path).expanduser().resolve()), map_location=device)
    source = checkpoint.get("state_dict", checkpoint)
    model = Chunk8Policy(use_task_id, history_steps, chunk_horizon)
    if "encoder.state_encoder.weight" in source:
        if "future_action_head.weight" not in source and chunk_horizon > 1:
            source = OrderedDict(source)
            current_weight = source[f"actor.{len(model.actor) - 1}.weight"]
            current_bias = source[f"actor.{len(model.actor) - 1}.bias"]
            source["future_action_head.weight"] = current_weight.repeat(chunk_horizon - 1, 1)
            source["future_action_head.bias"] = current_bias.repeat(chunk_horizon - 1)
        model.load_state_dict(source, strict=(chunk_horizon > 1))
        return model.to(device).eval()
    mapping = {
        "model.state_enc.": "encoder.state_encoder.",
        "model.dexrep_sensor_enc.": "encoder.sensor_encoder.",
        "model.bn_pnl.": "encoder.pnl_norm.",
        "model.dexrep_pointL_enc.": "encoder.pnl_encoder.",
        "model.actor.": "actor.",
        "model.future_action_head.": "future_action_head.",
    }
    converted = {}
    for old_name, tensor in source.items():
        for old_prefix, new_prefix in mapping.items():
            if old_name.startswith(old_prefix):
                converted[new_prefix + old_name[len(old_prefix):]] = tensor
                break
    if chunk_horizon > 1 and "future_action_head.weight" not in converted:
        current_weight = converted[f"actor.{len(model.actor) - 1}.weight"]
        current_bias = converted[f"actor.{len(model.actor) - 1}.bias"]
        converted["future_action_head.weight"] = current_weight.repeat(chunk_horizon - 1, 1)
        converted["future_action_head.bias"] = current_bias.repeat(chunk_horizon - 1)
    model.load_state_dict(converted, strict=(chunk_horizon > 1))
    return model.to(device).eval()


def save_checkpoint(path: str, model: Chunk8Policy, **metadata) -> None:
    """输入路径、模型和元数据，无返回值；保存简单且自描述的教学版checkpoint。"""
    torch.save({"state_dict": model.state_dict(), "use_task_id": model.use_task_id,
                "history_steps": model.history_steps, "chunk_horizon": model.chunk_horizon,
                "metadata": metadata}, str(Path(path).expanduser().resolve()))


def weighted_model_soup(states, weights) -> OrderedDict:
    """输入同结构参数字典和权重，输出逐参数加权平均结果。

    浮点参数加权平均，整数BatchNorm计数取最大值；作用是保留BC Soup最小实现。
    """
    if len(states) != len(weights) or not states or sum(weights) <= 0:
        raise ValueError("模型和正权重必须非空且数量相同")
    normalized = [float(weight) / sum(weights) for weight in weights]
    result = OrderedDict()
    for name in states[0]:
        tensors = [state[name] for state in states]
        if tensors[0].is_floating_point():
            averaged = torch.zeros_like(tensors[0], dtype=torch.float64)
            for weight, tensor in zip(normalized, tensors):
                averaged.add_(tensor.to(torch.float64), alpha=weight)
            result[name] = averaged.to(tensors[0].dtype)
        else:
            # 模型中唯一的整数状态是BatchNorm计数，推理不用它，保留最大训练步数即可。
            result[name] = torch.stack(tensors).max(dim=0).values
    return result
