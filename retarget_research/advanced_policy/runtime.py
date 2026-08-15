"""加载训练checkpoint并提供跨模型统一的单步策略推理器。

输入：best/last checkpoint、训练数据归一化文件、类别名和原始单步观测。
输出：未标准化的目标手绝对位置动作。
内部逻辑：BC直接前向；Temporal3维护三帧观测与两步动作；Diffusion生成并缓存动作片段。
作用：把训练模型可靠接到Isaac闭环，而不是在每个评测脚本里重复历史和反归一化逻辑。
"""

from __future__ import annotations

from collections import deque
import json
from pathlib import Path

import numpy as np
import torch

try:
    from .models import linear_beta_schedule, sample_diffusion
    from .train import build_model, load_checkpoint_file
except ImportError:
    from models import linear_beta_schedule, sample_diffusion
    from train import build_model, load_checkpoint_file


class PolicyRunner:
    """持有一个已训练策略及其单条episode时序状态。"""

    def __init__(
        self,
        checkpoint,
        data_dir,
        device="cpu",
        diffusion_execute_steps=2,
        normalized_action_clip=5.0,
        action_rate_limit_scale=0.0,
    ):
        """加载模型、维度、归一化参数和类别映射。

        输入：checkpoint、数据目录、设备、Diffusion片段长度、标准化裁剪和动作限速倍率。
        输出：可调用`reset/act`的推理器。
        内部逻辑：完全使用checkpoint内配置构造网络，拒绝数据维度或类别数错配。
        作用：确保推理结构与训练结构一致，并对异常大动作提供统一安全边界。
        """
        self.device = torch.device(device)
        payload = load_checkpoint_file(Path(checkpoint), self.device)
        self.config = payload["config"]
        self.dimensions = payload["dimensions"]
        self.model_type = self.config["model_type"]
        self.model = build_model(
            self.config,
            self.dimensions["observation_dim"],
            self.dimensions["action_dim"],
            self.dimensions["category_count"],
        ).to(self.device)
        self.model.load_state_dict(payload["model_state"])
        self.model.eval()
        data_dir = Path(data_dir)
        with np.load(data_dir / "normalization.npz", allow_pickle=False) as archive:
            self.normalization = {name: archive[name].astype(np.float32) for name in archive.files}
        self.mappings = json.loads((data_dir / "mappings.json").read_text(encoding="utf-8"))
        if len(self.normalization["observation_mean"]) != self.dimensions["observation_dim"]:
            raise ValueError("checkpoint观测维度与normalization不一致")
        if len(self.normalization["action_mean"]) != self.dimensions["action_dim"]:
            raise ValueError("checkpoint动作维度与normalization不一致")
        if len(self.mappings["category_to_id"]) != self.dimensions["category_count"]:
            raise ValueError("checkpoint类别数量与mappings不一致")
        if len(self.mappings.get("policy_action_order", [])) != self.dimensions["action_dim"]:
            raise ValueError("mappings缺少与checkpoint维度一致的policy_action_order")
        self.history = int(self.config.get("history", 3))
        if self.model_type == "temporal3" and self.history != 3:
            raise ValueError("当前Temporal3网络只接受history=3")
        self.diffusion_execute_steps = int(diffusion_execute_steps)
        if self.diffusion_execute_steps <= 0:
            raise ValueError("diffusion_execute_steps必须为正数")
        self.action_clip = float(normalized_action_clip)
        self.action_rate_limit_scale = float(action_rate_limit_scale)
        if self.action_rate_limit_scale < 0.0:
            raise ValueError("动作限速倍率不能为负数")
        if self.action_rate_limit_scale > 0.0:
            required_limits = {"action_delta_limit", "action_delta_norm_limit"}
            missing_limits = required_limits - set(self.normalization)
            if missing_limits:
                raise ValueError(f"normalization缺少动作限速统计: {sorted(missing_limits)}")
            self.action_delta_limit = (
                self.normalization["action_delta_limit"] * self.action_rate_limit_scale
            )
            self.action_delta_norm_limit = float(
                self.normalization["action_delta_norm_limit"]
            ) * self.action_rate_limit_scale
        else:
            self.action_delta_limit = None
            self.action_delta_norm_limit = None
        self.betas = (
            linear_beta_schedule(int(self.config.get("diffusion_steps", 50))).to(self.device)
            if self.model_type == "diffusion"
            else None
        )
        self.observation_history = deque(maxlen=self.history)
        self.previous_actions = deque(maxlen=max(self.history - 1, 1))
        self.action_cache = deque()
        self.category_id = None
        self.previous_command = None

    def normalize_observation(self, observation):
        """输入原始一维观测，输出按训练集统计标准化的一维数组。"""
        observation = np.asarray(observation, dtype=np.float32)
        if observation.shape != (self.dimensions["observation_dim"],):
            raise ValueError(f"观测维度错误: {observation.shape}")
        return (observation - self.normalization["observation_mean"]) / self.normalization["observation_std"]

    def denormalize_action(self, action):
        """输入标准化动作，裁剪后恢复为目标手绝对位置命令。"""
        clipped = np.clip(np.asarray(action, dtype=np.float32), -self.action_clip, self.action_clip)
        return clipped * self.normalization["action_std"] + self.normalization["action_mean"]

    def normalize_action(self, action):
        """输入未标准化动作，输出供Temporal历史使用的训练标准化动作。"""
        action = np.asarray(action, dtype=np.float32)
        return (action - self.normalization["action_mean"]) / self.normalization["action_std"]

    def apply_action_rate_limit(self, action):
        """按train专家相邻动作分布限制当前绝对位置命令的变化速度。

        输入：反归一化后的原始策略动作。
        输出：相对上一条实际命令同时满足逐维和L2范围的动作。
        内部逻辑：先逐维截断delta，再在总L2超限时等比例缩放；倍率0保持原动作。
        作用：阻止闭环分布偏移导致手腕和手指在相邻60 Hz步产生非专家式跳变。
        """
        action = np.asarray(action, dtype=np.float32)
        if self.action_rate_limit_scale <= 0.0:
            return action
        if self.previous_command is None:
            raise RuntimeError("启用动作限速时reset必须提供initial_action")
        delta = np.clip(
            action - self.previous_command,
            -self.action_delta_limit,
            self.action_delta_limit,
        )
        norm = float(np.linalg.norm(delta))
        if norm > self.action_delta_norm_limit:
            delta *= self.action_delta_norm_limit / max(norm, 1e-12)
        return self.previous_command + delta

    def reset(self, category_name, initial_observation, initial_action=None):
        """开始新episode并用首观测初始化历史。

        输入：官方类别名、当前原始观测，以及可选的episode实际初始动作命令。
        输出：无返回值。
        内部逻辑：首观测复制history次，历史动作置零，清空Diffusion动作缓存。
        作用：阻止上一条轨迹的历史状态泄漏到下一条物体。
        """
        if category_name not in self.mappings["category_to_id"]:
            raise KeyError(f"训练映射中没有类别: {category_name}")
        self.category_id = int(self.mappings["category_to_id"][category_name])
        normalized = self.normalize_observation(initial_observation)
        self.observation_history.clear()
        self.previous_actions.clear()
        self.action_cache.clear()
        self.previous_command = (
            None if initial_action is None else np.asarray(initial_action, dtype=np.float32).copy()
        )
        if self.previous_command is not None and self.previous_command.shape != (
            self.dimensions["action_dim"],
        ):
            raise ValueError(f"初始动作维度错误: {self.previous_command.shape}")
        if self.action_rate_limit_scale > 0.0 and self.previous_command is None:
            raise ValueError("启用动作限速时必须提供initial_action")
        for _ in range(self.history):
            self.observation_history.append(normalized.copy())
        for _ in range(max(self.history - 1, 1)):
            self.previous_actions.append(np.zeros(self.dimensions["action_dim"], dtype=np.float32))

    @torch.no_grad()
    def act(self, observation):
        """根据当前观测返回下一条未标准化动作。

        输入：一维原始观测。
        输出：一维目标手绝对位置动作。
        内部逻辑：更新观测历史，按模型分支推理；Temporal记录已输出动作，Diffusion按片段缓存执行。
        作用：统一三类模型在60 Hz物理循环中的调用协议。
        """
        if self.category_id is None:
            raise RuntimeError("调用act前必须先reset")
        normalized = self.normalize_observation(observation)
        self.observation_history.append(normalized.copy())
        category = torch.tensor([self.category_id], dtype=torch.long, device=self.device)
        if self.model_type in {"bc", "category_teacher", "student", "online_student"}:
            tensor = torch.from_numpy(normalized[None]).to(self.device)
            action = self.model(tensor, category)[0].cpu().numpy()
        elif self.model_type == "temporal3":
            observations = torch.from_numpy(np.stack(self.observation_history)[None]).to(self.device)
            previous = torch.from_numpy(np.stack(self.previous_actions)[None]).to(self.device)
            action = self.model(observations, previous, category)[0].cpu().numpy()
        else:
            if not self.action_cache:
                observations = torch.from_numpy(np.stack(self.observation_history)[None]).to(self.device)
                sequence = sample_diffusion(self.model, observations, category, self.betas)[0].cpu().numpy()
                for value in sequence[: self.diffusion_execute_steps]:
                    self.action_cache.append(value.copy())
            action = self.action_cache.popleft()
        command = self.apply_action_rate_limit(self.denormalize_action(action))
        self.previous_command = command.copy()
        if self.model_type == "temporal3":
            self.previous_actions.append(self.normalize_action(command))
        return command
