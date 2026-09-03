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
from scipy.spatial.transform import Rotation

try:
    from .models import linear_beta_schedule, make_mlp, sample_diffusion
    from .train import build_model, load_checkpoint_file
except ImportError:
    from models import linear_beta_schedule, make_mlp, sample_diffusion
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
        if self.model_type == "parametric_blend":
            self.model = None
        elif self.model_type in {"trajectory_retrieval", "trajectory_blend"}:
            self.retrieval_initial = np.asarray(
                payload["retrieval_initial_observations"], dtype=np.float32
            )
            self.retrieval_deltas = np.asarray(
                payload["retrieval_action_deltas"], dtype=np.float32
            )
            self.retrieval_features = np.asarray(
                payload["retrieval_feature_indices"], dtype=np.int64
            )
            self.retrieval_k = int(self.config["retrieval_k"])
            self.finger_retrieval_k = int(
                self.config.get("finger_retrieval_k", self.retrieval_k)
            )
            if self.model_type == "trajectory_blend":
                base_config = payload["base_model_config"]
                self.model = build_model(
                    base_config, self.dimensions["observation_dim"],
                    self.dimensions["action_dim"], self.dimensions["category_count"],
                ).to(self.device)
                self.model.load_state_dict(payload["model_state"])
                self.model.eval()
                common_alpha = float(self.config.get("blend_alpha", 0.0))
                self.wrist_blend_alpha = float(
                    self.config.get("wrist_blend_alpha", common_alpha)
                )
                self.finger_blend_alpha = float(
                    self.config.get("finger_blend_alpha", common_alpha)
                )
            else:
                self.model = None
        elif self.model_type in {"trajectory_se3_retrieval", "trajectory_se3_blend"}:
            self.retrieval_initial = np.asarray(
                payload["retrieval_initial_observations"], dtype=np.float32
            )
            self.retrieval_local_translation = np.asarray(
                payload["retrieval_local_translation"], dtype=np.float32
            )
            self.retrieval_relative_rotvec = np.asarray(
                payload["retrieval_relative_rotvec"], dtype=np.float32
            )
            self.retrieval_fingers = np.asarray(
                payload["retrieval_finger_actions"], dtype=np.float32
            )
            self.retrieval_k = int(self.config["retrieval_k"])
            self.finger_retrieval_k = int(
                self.config.get("finger_retrieval_k", self.retrieval_k)
            )
            if self.model_type == "trajectory_se3_blend":
                base_config = payload["base_model_config"]
                self.model = build_model(
                    base_config, self.dimensions["observation_dim"],
                    self.dimensions["action_dim"], self.dimensions["category_count"],
                ).to(self.device)
                self.model.load_state_dict(payload["model_state"])
                self.model.eval()
                self.wrist_blend_alpha = float(self.config["blend_alpha"])
                self.finger_blend_alpha = float(self.config["blend_alpha"])
            else:
                self.model = None
        elif self.model_type == "trajectory_pca_rbf":
            self.model = None
            self.pca_train_features = np.asarray(payload["rbf_train_features"], dtype=np.float32)
            self.pca_dual = np.asarray(payload["rbf_dual"], dtype=np.float32)
            self.pca_mean = np.asarray(payload["pca_mean"], dtype=np.float32)
            self.pca_components = np.asarray(payload["pca_components"], dtype=np.float32)
            self.pca_sigma = float(self.config["rbf_sigma"])
            self.pca_sequence_shape = tuple(payload["sequence_shape"])
        elif self.model_type == "trajectory_local_ridge":
            self.model = None
            self.local_features = np.asarray(payload["train_features"], dtype=np.float32)
            self.local_sequences = np.asarray(payload["train_sequences"], dtype=np.float32)
            self.local_k = int(self.config["local_k"])
            self.local_ridge = float(self.config["ridge"])
        elif self.model_type == "trajectory_pca_mlp":
            self.model = make_mlp(
                self.dimensions["observation_dim"], self.config["pca_rank"],
                self.config["hidden_dims"], self.config.get("dropout", 0.0),
            ).to(self.device)
            self.model.load_state_dict(payload["model_state"])
            self.model.eval()
            self.pca_mean = np.asarray(payload["pca_mean"], dtype=np.float32)
            self.pca_components = np.asarray(payload["pca_components"], dtype=np.float32)
            self.coefficient_mean = np.asarray(payload["coefficient_mean"], dtype=np.float32)
            self.coefficient_std = np.asarray(payload["coefficient_std"], dtype=np.float32)
            self.pca_sequence_shape = tuple(payload["sequence_shape"])
        else:
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
        if self.model_type == "parametric_blend":
            self.blend_alpha = float(self.config["blend_alpha"])
            self.blend_first = PolicyRunner(
                self.config["first_checkpoint"], data_dir, device,
                diffusion_execute_steps, normalized_action_clip, action_rate_limit_scale,
            )
            self.blend_second = PolicyRunner(
                self.config["second_checkpoint"], data_dir, device,
                diffusion_execute_steps, normalized_action_clip, action_rate_limit_scale,
            )
        if self.model_type == "phase_residual":
            required_delta = {"action_delta_mean", "action_delta_std"}
            missing_delta = required_delta - set(self.normalization)
            if missing_delta:
                raise ValueError(f"残差策略缺少动作增量统计: {sorted(missing_delta)}")
        if self.model_type in {"initial_phase_delta", "initial_fourier_delta", "initial_phase_feedback", "initial_temporal_feedback", "parametric_blend", "trajectory_blend", "trajectory_se3_blend", "trajectory_pca_rbf", "trajectory_local_ridge", "trajectory_pca_mlp"}:
            required = {"initial_delta_mean", "initial_delta_std"}
            missing = required - set(self.normalization)
            if missing:
                raise ValueError(f"初态相对策略缺少统计量: {sorted(missing)}")
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
        self.initial_observation = None
        self.initial_command = None
        self.previous_command = None
        self.phase_step = 0
        self.retrieval_sequence = None
        self.motion_steps = int(self.config.get("motion_steps", 210))
        if self.motion_steps < 2:
            raise ValueError("motion_steps必须至少为2")

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

    def denormalize_action_delta(self, delta):
        """输入标准化动作增量，按train delta统计恢复为一帧真实命令变化。"""
        clipped = np.clip(
            np.asarray(delta, dtype=np.float32), -self.action_clip, self.action_clip
        )
        return (
            clipped * self.normalization["action_delta_std"]
            + self.normalization["action_delta_mean"]
        )

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
        self.initial_observation = normalized.copy()
        self.initial_command = (
            None if initial_action is None else np.asarray(initial_action, dtype=np.float32).copy()
        )
        self.observation_history.clear()
        self.previous_actions.clear()
        self.action_cache.clear()
        self.previous_command = (
            None if initial_action is None else np.asarray(initial_action, dtype=np.float32).copy()
        )
        self.phase_step = 0
        if self.model_type == "parametric_blend":
            self.blend_first.reset(category_name, initial_observation, initial_action)
            self.blend_second.reset(category_name, initial_observation, initial_action)
        if self.model_type in {"trajectory_retrieval", "trajectory_blend"}:
            delta = self.retrieval_initial[:, self.retrieval_features] \
                - normalized[self.retrieval_features][None]
            distance = np.sum(delta * delta, axis=1)
            nearest = np.argsort(distance)[:self.retrieval_k]
            weights = 1.0 / (distance[nearest] + 1e-3)
            weights /= weights.sum()
            self.retrieval_sequence = np.tensordot(
                weights, self.retrieval_deltas[nearest], axes=(0, 0)
            ).astype(np.float32)
            finger_nearest = np.argsort(distance)[:self.finger_retrieval_k]
            finger_weights = 1.0 / (distance[finger_nearest] + 1e-3)
            finger_weights /= finger_weights.sum()
            self.retrieval_sequence[:, 6:] = np.tensordot(
                finger_weights, self.retrieval_deltas[finger_nearest, :, 6:], axes=(0, 0)
            )
        elif self.model_type == "trajectory_pca_rbf":
            query = normalized[:6]
            distance = np.sum((self.pca_train_features - query[None]) ** 2, axis=1)
            kernel = np.exp(-distance / (2.0 * self.pca_sigma ** 2))
            coefficients = kernel @ self.pca_dual
            normalized_sequence = self.pca_mean + coefficients @ self.pca_components
            normalized_sequence = np.clip(normalized_sequence, -5.0, 5.0)
            normalized_sequence = normalized_sequence.reshape(self.pca_sequence_shape)
            self.retrieval_sequence = (
                normalized_sequence * self.normalization["initial_delta_std"]
                + self.normalization["initial_delta_mean"]
            ).astype(np.float32)
        elif self.model_type in {"trajectory_se3_retrieval", "trajectory_se3_blend"}:
            distance = np.sum((self.retrieval_initial[:, :6] - normalized[:6]) ** 2, axis=1)
            nearest = np.argsort(distance)[:self.retrieval_k]
            weights = 1.0 / (distance[nearest] + 1e-3)
            weights /= weights.sum()
            local_translation = np.tensordot(
                weights, self.retrieval_local_translation[nearest], axes=(0, 0)
            )
            relative_rotvec = np.tensordot(
                weights, self.retrieval_relative_rotvec[nearest], axes=(0, 0)
            )
            finger_nearest = np.argsort(distance)[:self.finger_retrieval_k]
            finger_weights = 1.0 / (distance[finger_nearest] + 1e-3)
            finger_weights /= finger_weights.sum()
            fingers = np.tensordot(
                finger_weights, self.retrieval_fingers[finger_nearest], axes=(0, 0)
            )
            initial_rotation = Rotation.from_euler("xyz", self.initial_command[3:6])
            if self.config.get("translation_frame", "local") == "world":
                positions = self.initial_command[:3] + local_translation
            else:
                positions = self.initial_command[:3] + initial_rotation.apply(local_translation)
            rotations = initial_rotation * Rotation.from_rotvec(relative_rotvec)
            wrist_delta = np.concatenate([
                positions - self.initial_command[:3],
                rotations.as_euler("xyz") - self.initial_command[3:6],
            ], axis=1)
            self.retrieval_sequence = np.concatenate([wrist_delta, fingers], axis=1).astype(np.float32)
        elif self.model_type == "trajectory_local_ridge":
            query = normalized[:6]
            distance = np.sum((self.local_features - query[None]) ** 2, axis=1)
            nearest = np.argsort(distance)[:self.local_k]
            centered = self.local_features[nearest] - query[None]
            design = np.concatenate([
                np.ones((len(nearest), 1), dtype=np.float32), centered
            ], axis=1)
            weights = 1.0 / (distance[nearest] + 1e-3)
            normal = design.T @ (weights[:, None] * design)
            normal += self.local_ridge * np.diag([0.0] + [1.0] * 6)
            target = self.local_sequences[nearest].reshape(len(nearest), -1)
            coefficients = np.linalg.pinv(normal) @ (
                design.T @ (weights[:, None] * target)
            )
            normalized_sequence = np.clip(coefficients[0], -5.0, 5.0)
            normalized_sequence = normalized_sequence.reshape(self.local_sequences.shape[1:])
            self.retrieval_sequence = (
                normalized_sequence * self.normalization["initial_delta_std"]
                + self.normalization["initial_delta_mean"]
            ).astype(np.float32)
        elif self.model_type == "trajectory_pca_mlp":
            input_tensor = torch.from_numpy(normalized[None]).to(self.device)
            with torch.no_grad():
                normalized_coefficients = self.model(input_tensor)[0].cpu().numpy()
            coefficients = (
                normalized_coefficients * self.coefficient_std + self.coefficient_mean
            )
            normalized_sequence = self.pca_mean + coefficients @ self.pca_components
            normalized_sequence = normalized_sequence.reshape(self.pca_sequence_shape)
            self.retrieval_sequence = (
                normalized_sequence * self.normalization["initial_delta_std"]
                + self.normalization["initial_delta_mean"]
            ).astype(np.float32)
        if self.previous_command is not None and self.previous_command.shape != (
            self.dimensions["action_dim"],
        ):
            raise ValueError(f"初始动作维度错误: {self.previous_command.shape}")
        if self.action_rate_limit_scale > 0.0 and self.previous_command is None:
            raise ValueError("启用动作限速时必须提供initial_action")
        if self.model_type in ("phase_residual", "phase_residual_temporal", "initial_phase_delta", "initial_fourier_delta", "initial_phase_feedback", "initial_temporal_feedback", "parametric_blend", "trajectory_retrieval", "trajectory_blend", "trajectory_pca_rbf", "trajectory_local_ridge", "trajectory_se3_retrieval", "trajectory_se3_blend", "trajectory_pca_mlp") and self.previous_command is None:
            raise ValueError(f"{self.model_type}必须提供episode初始动作")
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
        if self.model_type == "parametric_blend":
            first = self.blend_first.act(observation)
            second = self.blend_second.act(observation)
            command = (1.0 - self.blend_alpha) * first + self.blend_alpha * second
            self.previous_command = command.copy()
            self.phase_step += 1
            return command
        normalized = self.normalize_observation(observation)
        self.observation_history.append(normalized.copy())
        if self.model_type in {"trajectory_retrieval", "trajectory_pca_rbf", "trajectory_local_ridge", "trajectory_se3_retrieval", "trajectory_pca_mlp"}:
            index = min(self.phase_step, len(self.retrieval_sequence) - 1)
            command = self.initial_command + self.retrieval_sequence[index]
            self.phase_step += 1
            self.previous_command = command.copy()
            return command
        category = torch.tensor([self.category_id], dtype=torch.long, device=self.device)
        if self.model_type in {"trajectory_blend", "trajectory_se3_blend"}:
            index = min(self.phase_step, len(self.retrieval_sequence) - 1)
            phase_value = min(self.phase_step / float(self.motion_steps - 1), 1.0)
            initial = torch.from_numpy(self.initial_observation[None]).to(self.device)
            phase = torch.tensor([[phase_value]], dtype=torch.float32, device=self.device)
            normalized_delta = self.model(initial, phase, category)[0].cpu().numpy()
            learned_delta = (
                np.clip(normalized_delta, -self.action_clip, self.action_clip)
                * self.normalization["initial_delta_std"]
                + self.normalization["initial_delta_mean"]
            )
            alpha = np.full(
                self.dimensions["action_dim"], self.finger_blend_alpha,
                dtype=np.float32,
            )
            alpha[:6] = self.wrist_blend_alpha
            delta = (1.0 - alpha) * self.retrieval_sequence[index] + alpha * learned_delta
            command = self.initial_command + delta
            self.phase_step += 1
            self.previous_command = command.copy()
            return command
        if self.model_type in {"initial_phase", "initial_phase_delta", "initial_fourier_delta", "initial_phase_feedback", "initial_temporal_feedback"}:
            phase_value = min(
                self.phase_step / float(self.motion_steps - 1), 1.0
            )
            initial = torch.from_numpy(self.initial_observation[None]).to(self.device)
            phase = torch.tensor(
                [[phase_value]], dtype=torch.float32, device=self.device
            )
            if self.model_type == "initial_phase_feedback":
                action = self.model(
                    initial, torch.from_numpy(normalized[None]).to(self.device),
                    phase, category,
                )[0].cpu().numpy()
            elif self.model_type == "initial_temporal_feedback":
                history = torch.from_numpy(
                    np.stack(self.observation_history)[None]
                ).to(self.device)
                action = self.model(initial, history, phase, category)[0].cpu().numpy()
            else:
                action = self.model(initial, phase, category)[0].cpu().numpy()
            self.phase_step += 1
            if self.model_type in {"initial_phase_delta", "initial_fourier_delta", "initial_phase_feedback", "initial_temporal_feedback"}:
                clipped = np.clip(action, -self.action_clip, self.action_clip)
                delta = (
                    clipped * self.normalization["initial_delta_std"]
                    + self.normalization["initial_delta_mean"]
                )
                command = self.apply_action_rate_limit(self.initial_command + delta)
                self.previous_command = command.copy()
                return command
        elif self.model_type in {"bc", "category_teacher", "student", "online_student"}:
            tensor = torch.from_numpy(normalized[None]).to(self.device)
            action = self.model(tensor, category)[0].cpu().numpy()
        elif self.model_type == "phase_residual":
            tensor = torch.from_numpy(normalized[None]).to(self.device)
            if self.previous_command is None:
                self.previous_command = self.normalization["action_mean"].copy()
            previous = torch.from_numpy(
                self.normalize_action(self.previous_command)[None]
            ).to(self.device)
            phase_value = min(
                self.phase_step / float(self.motion_steps - 1), 1.0
            )
            phase = torch.tensor(
                [[phase_value]], dtype=torch.float32, device=self.device
            )
            delta = self.model(tensor, previous, phase, category)[0].cpu().numpy()
            raw_delta = self.denormalize_action_delta(delta)
            if "action_phase_delta_limits" in self.normalization:
                limits = self.normalization["action_phase_delta_limits"]
                bin_count = len(limits) - 1
                bin_index = min(
                    int(phase_value * bin_count), bin_count
                )
                raw_delta = np.clip(raw_delta, -limits[bin_index], limits[bin_index])
            command = self.previous_command + raw_delta
            if "action_phase_position_envelope" in self.normalization:
                envelope = self.normalization["action_phase_position_envelope"]
                bin_count = len(envelope) - 1
                bin_index = min(int(phase_value * bin_count), bin_count)
                command = command.copy()
                command[:3] = np.clip(
                    command[:3], envelope[bin_index, 0], envelope[bin_index, 1])
            self.previous_command = command.copy()
            self.phase_step += 1
            if self.model_type == "phase_residual_temporal":
                self.previous_actions.append(self.normalize_action(command))
            return command
        elif self.model_type == "phase_residual_temporal":
            tensor = torch.from_numpy(
                np.stack(self.observation_history)[None]).to(self.device)
            previous = torch.from_numpy(
                np.stack(self.previous_actions)[None]).to(self.device)
            phase_value = min(
                self.phase_step / float(self.motion_steps - 1), 1.0
            )
            phase = torch.tensor(
                [[phase_value]], dtype=torch.float32, device=self.device
            )
            delta = self.model(tensor, previous, phase, category)[0].cpu().numpy()
            raw_delta = self.denormalize_action_delta(delta)
            if "action_phase_delta_limits" in self.normalization:
                limits = self.normalization["action_phase_delta_limits"]
                bin_count = len(limits) - 1
                bin_index = min(int(phase_value * bin_count), bin_count)
                raw_delta = np.clip(raw_delta, -limits[bin_index], limits[bin_index])
            command = self.previous_command + raw_delta
            if "action_phase_position_envelope" in self.normalization:
                envelope = self.normalization["action_phase_position_envelope"]
                bin_count = len(envelope) - 1
                bin_index = min(int(phase_value * bin_count), bin_count)
                command = command.copy()
                command[:3] = np.clip(
                    command[:3], envelope[bin_index, 0], envelope[bin_index, 1])
            self.previous_command = command.copy()
            self.phase_step += 1
            self.previous_actions.append(self.normalize_action(command))
            return command
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
