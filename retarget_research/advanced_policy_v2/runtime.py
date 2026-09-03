"""在Isaac闭环中执行几何条件动作块，并做时间集成。"""

from collections import deque
import json
from pathlib import Path

import numpy as np
import torch

from .geometry import object_points_in_initial_wrist
from .models import (
    build_direct_interaction, build_interaction_residual, build_keypose_model, build_model,
    build_pca_latent_diffusion, build_pca_model, build_pca_mixture,
    sample_pca_latent_diffusion,
)


FINGER_GROUPS = {
    "linker": ((0, 1), (2,), (3,), (4,), (5,)),
    "xhand": ((0, 1, 2), (3, 4, 5), (6, 7), (8, 9), (10, 11)),
    "wuji": ((0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 10, 11),
             (12, 13, 14, 15), (16, 17, 18, 19)),
}

TIP_LINKS = {
    "linker": ("rh_thumb_distal", "rh_index_distal", "rh_middle_distal",
               "rh_ring_distal", "rh_pinky_distal"),
    "xhand": ("right_hand_thumb_rota_link2", "right_hand_index_rota_link2",
              "right_hand_mid_link2", "right_hand_ring_link2",
              "right_hand_pinky_link2"),
    "wuji": ("finger1_tip_link", "finger2_tip_link", "finger3_tip_link",
             "finger4_tip_link", "finger5_tip_link"),
}


class GeometryPolicyRunner:
    """保持旧评测器调用接口的第二版策略运行器。"""

    def __init__(
        self, checkpoint_path, data_dir, device,
        diffusion_execute_steps=1, normalized_action_clip=5.0,
        action_rate_limit_scale=0.0,
    ):
        del diffusion_execute_steps
        self.device = torch.device(device)
        payload = torch.load(checkpoint_path, map_location=self.device)
        schema = payload.get("schema")
        if schema not in {
            "geometry_action_chunk_policy_v1", "geometry_composite_policy_v1",
            "geometry_trajectory_pca_policy_v1", "geometry_pca_mixture_policy_v1",
            "geometry_pca_interaction_residual_v1",
            "geometry_pca_contact_feedback_v1",
            "geometry_keypose_lift_policy_v1",
            "geometry_pca_latent_diffusion_policy_v1",
            "geometry_pca_surface_ik_policy_v1",
            "direct_interaction_temporal_policy_v1",
        }:
            raise ValueError("不是第二版几何策略checkpoint")
        self.interaction_residual = False
        self.contact_feedback = False
        self.keypose_policy = False
        self.trajectory_diffusion = False
        self.surface_ik = False
        self.direct_interaction = False
        if schema == "direct_interaction_temporal_policy_v1":
            self._init_direct_interaction(
                payload, data_dir, normalized_action_clip,
                action_rate_limit_scale,
            )
            return
        if schema == "geometry_pca_surface_ik_policy_v1":
            self._init_surface_ik(
                payload, data_dir, device, normalized_action_clip,
                action_rate_limit_scale,
            )
            return
        if schema == "geometry_pca_latent_diffusion_policy_v1":
            self._init_latent_diffusion(
                payload, data_dir, normalized_action_clip,
                action_rate_limit_scale,
            )
            return
        if schema == "geometry_keypose_lift_policy_v1":
            self._init_keypose(
                payload, data_dir, normalized_action_clip,
                action_rate_limit_scale,
            )
            return
        if schema == "geometry_pca_contact_feedback_v1":
            self._init_contact_feedback(
                payload, data_dir, device, normalized_action_clip,
                action_rate_limit_scale,
            )
            return
        if schema == "geometry_pca_interaction_residual_v1":
            self._init_interaction(
                payload, data_dir, device, normalized_action_clip,
                action_rate_limit_scale,
            )
            return
        if schema == "geometry_pca_mixture_policy_v1":
            self._init_mixture(
                payload, data_dir, normalized_action_clip,
                action_rate_limit_scale,
            )
            return
        if schema == "geometry_trajectory_pca_policy_v1":
            self._init_pca(
                payload, data_dir, normalized_action_clip,
                action_rate_limit_scale,
            )
            return
        if schema == "geometry_composite_policy_v1":
            self._init_composite(
                payload, data_dir, device, normalized_action_clip,
                action_rate_limit_scale,
            )
            return
        self.composite_type = None
        self.trajectory_pca = False
        self.trajectory_mixture = False
        self.config = payload["config"]
        self.dimensions = payload["dimensions"]
        self.model_type = self.config["model_type"]
        self.model = build_model(
            self.config, self.dimensions["observation_dim"], self.dimensions["action_dim"]
        ).to(self.device)
        self.model.load_state_dict(payload["model_state"])
        self.model.eval()
        data_dir = Path(data_dir)
        with np.load(data_dir / "geometry_normalization.npz", allow_pickle=False) as archive:
            self.normalization = {name: archive[name].astype(np.float32) for name in archive.files}
        self.mappings = json.loads((data_dir / "mappings.json").read_text(encoding="utf-8"))
        self.history = int(self.config.get("history", 3))
        self.action_horizon = 1 if self.model_type == "geometry_phase" else int(self.config.get("action_horizon", 8))
        self.motion_steps = int(self.config.get("motion_steps", 210))
        self.ensemble_decay = float(self.config.get("temporal_ensemble_decay", 1.0))
        self.action_clip = float(normalized_action_clip)
        self.rate_scale = float(action_rate_limit_scale)
        self.action_delta_norm_limit = float(
            self.normalization.get("action_delta_norm_limit", np.asarray(0.0))
        )
        self.observation_history = deque(maxlen=self.history)
        self.previous_delta_history = deque(maxlen=self.history)
        self.pending_chunks = []
        self.object_points = None
        self.initial_command = None

    def _init_direct_interaction(
        self, payload, data_dir, normalized_action_clip,
        action_rate_limit_scale,
    ):
        """加载完全不依赖PCA的动态手物Temporal策略。"""
        from .interaction import TargetHandGeometry

        self.direct_interaction = True
        self.composite_type = None
        self.trajectory_pca = False
        self.trajectory_mixture = False
        self.config = payload["config"]
        self.dimensions = payload["dimensions"]
        self.model_type = self.config["model_type"]
        self.model = build_direct_interaction(
            self.config, self.dimensions["observation_dim"],
            self.dimensions["action_dim"], self.dimensions["interaction_dim"],
        ).to(self.device)
        self.model.load_state_dict(payload["model_state"])
        self.model.eval()
        data_dir = Path(data_dir)
        with np.load(data_dir / "geometry_normalization.npz", allow_pickle=False) as archive:
            self.normalization = {
                name: archive[name].astype(np.float32) for name in archive.files
            }
        self.mappings = json.loads(
            (data_dir / "mappings.json").read_text(encoding="utf-8")
        )
        self.interaction_mean = np.asarray(
            payload["interaction_mean"], dtype=np.float32
        )
        self.interaction_std = np.asarray(
            payload["interaction_std"], dtype=np.float32
        )
        self.hand = self.config["hand"]
        self.hand_geometry = TargetHandGeometry(self.hand)
        self.history = int(self.config.get("history", 3))
        self.action_horizon = int(self.config.get("action_horizon", 1))
        self.motion_steps = int(self.config.get("motion_steps", 210))
        self.ensemble_decay = float(
            self.config.get("temporal_ensemble_decay", 1.0)
        )
        self.action_clip = float(normalized_action_clip)
        self.rate_scale = float(action_rate_limit_scale)
        self.action_delta_norm_limit = float(
            self.normalization.get("action_delta_norm_limit", np.asarray(0.0))
        )
        self.observation_history = deque(maxlen=self.history)
        self.interaction_history = deque(maxlen=self.history)
        self.previous_delta_history = deque(maxlen=self.history)
        self.pending_chunks = []
        self.object_points = None
        self.object_local_points = None
        self.initial_command = None

    def _init_pca(self, payload, data_dir, normalized_action_clip, action_rate_limit_scale):
        """加载只保存PCA基和网络参数的整轨迹生成策略。"""
        self.composite_type = None
        self.trajectory_pca = True
        self.trajectory_mixture = False
        self.config = payload["config"]
        self.dimensions = payload["dimensions"]
        self.model_type = self.config["model_type"]
        self.model = build_pca_model(
            self.config, self.dimensions["task_observation_dim"],
            self.dimensions["action_dim"],
        ).to(self.device)
        self.model.load_state_dict(payload["model_state"])
        self.model.eval()
        self.pca_mean = np.asarray(payload["pca_mean"], dtype=np.float32)
        self.pca_components = np.asarray(payload["pca_components"], dtype=np.float32)
        self.coefficient_mean = np.asarray(payload["coefficient_mean"], dtype=np.float32)
        self.coefficient_std = np.asarray(payload["coefficient_std"], dtype=np.float32)
        self.sequence_shape = tuple(payload["sequence_shape"])
        data_dir = Path(data_dir)
        with np.load(data_dir / "geometry_normalization.npz", allow_pickle=False) as archive:
            self.normalization = {name: archive[name].astype(np.float32) for name in archive.files}
        self.mappings = json.loads((data_dir / "mappings.json").read_text(encoding="utf-8"))
        self.history = 1
        self.action_horizon = self.sequence_shape[0]
        self.motion_steps = self.sequence_shape[0]
        self.ensemble_decay = 0.0
        self.action_clip = float(normalized_action_clip)
        self.rate_scale = float(action_rate_limit_scale)
        self.action_delta_norm_limit = float(
            self.normalization.get("action_delta_norm_limit", np.asarray(0.0))
        )
        self.object_points = None
        self.initial_command = None
        self.phase_step = 0
        self.initial_interaction_dim = int(self.config.get("interaction_dim", 0))
        if self.initial_interaction_dim:
            from .interaction import TargetHandGeometry

            self.hand_geometry = TargetHandGeometry(self.config["hand"])
            self.interaction_mean = np.asarray(
                payload["interaction_mean"], dtype=np.float32
            )
            self.interaction_std = np.asarray(
                payload["interaction_std"], dtype=np.float32
            )

    def _init_latent_diffusion(
        self, payload, data_dir, normalized_action_clip, action_rate_limit_scale,
    ):
        """加载条件Diffusion、冻结PCA基线和确定性候选选择器。"""
        self.composite_type = None
        self.trajectory_pca = False
        self.trajectory_mixture = False
        self.trajectory_diffusion = True
        self.config = payload["config"]
        self.dimensions = payload["dimensions"]
        self.model_type = self.config["model_type"]
        task_dim = self.dimensions["task_observation_dim"]
        action_dim = self.dimensions["action_dim"]
        self.model = build_pca_latent_diffusion(
            self.config, task_dim, action_dim
        ).to(self.device)
        self.model.load_state_dict(payload["model_state"])
        self.model.eval()
        self.base_model = build_pca_model(
            payload["base_model_config"], task_dim, action_dim
        ).to(self.device)
        self.base_model.load_state_dict(payload["base_model_state"])
        self.base_model.eval()
        self.pca_mean = np.asarray(payload["pca_mean"], dtype=np.float32)
        self.pca_components = np.asarray(payload["pca_components"], dtype=np.float32)
        self.coefficient_mean = np.asarray(
            payload["coefficient_mean"], dtype=np.float32
        )
        self.coefficient_std = np.asarray(
            payload["coefficient_std"], dtype=np.float32
        )
        self.alpha_bars = torch.as_tensor(
            payload["alpha_bars"], dtype=torch.float32, device=self.device
        )
        self.sequence_shape = tuple(payload["sequence_shape"])
        data_dir = Path(data_dir)
        with np.load(data_dir / "geometry_normalization.npz", allow_pickle=False) as archive:
            self.normalization = {
                name: archive[name].astype(np.float32) for name in archive.files
            }
        self.mappings = json.loads(
            (data_dir / "mappings.json").read_text(encoding="utf-8")
        )
        self.history = 1
        self.action_horizon = self.sequence_shape[0]
        self.motion_steps = self.sequence_shape[0]
        self.ensemble_decay = 0.0
        self.action_clip = float(normalized_action_clip)
        self.rate_scale = float(action_rate_limit_scale)
        self.action_delta_norm_limit = float(
            self.normalization.get("action_delta_norm_limit", np.asarray(0.0))
        )
        self.object_points = None
        self.initial_command = None
        self.phase_step = 0
        self.initial_interaction_dim = 0

    def _initial_interaction(self):
        """由张开手初态和初始物体点云计算标准化15点手—物关系。"""
        from scipy.spatial.transform import Rotation
        from .interaction import interaction_features

        rotation = Rotation.from_euler(
            "xyz", self.initial_command[3:6]
        ).as_matrix().astype(np.float32)
        object_world = (
            self.object_points @ rotation.T + self.initial_command[:3]
        )
        hand_points = self.hand_geometry.points(self.initial_command)[0]
        feature = interaction_features(
            hand_points[None], object_world[None],
            self.initial_command[None, 3:6],
        )[0]
        return ((feature - self.interaction_mean) / self.interaction_std).astype(
            np.float32
        )

    def _init_contact_feedback(
        self, payload, data_dir, device, normalized_action_clip,
        action_rate_limit_scale,
    ):
        """加载自主PCA名义轨迹和无学习参数的接触反馈控制器。"""
        self.contact_feedback = True
        self.interaction_residual = False
        self.composite_type = None
        self.trajectory_pca = False
        self.trajectory_mixture = False
        self.config = payload["config"]
        self.dimensions = payload["dimensions"]
        self.model_type = self.config["model_type"]
        self.base = GeometryPolicyRunner(
            payload["base_checkpoint"], data_dir, device,
            normalized_action_clip=normalized_action_clip,
            action_rate_limit_scale=action_rate_limit_scale,
        )
        self.normalization = self.base.normalization
        self.mappings = self.base.mappings
        self.motion_steps = self.base.motion_steps
        self.action_horizon = self.base.action_horizon
        self.history = 1
        self.ensemble_decay = 0.0
        self.action_clip = float(normalized_action_clip)
        self.rate_scale = float(action_rate_limit_scale)
        self.action_delta_norm_limit = self.base.action_delta_norm_limit
        self.hand = self.config["hand"]
        self.finger_groups = FINGER_GROUPS[self.hand]
        self.tip_links = TIP_LINKS[self.hand]
        self.contact_threshold = float(self.config.get("contact_threshold", 0.02))
        self.contact_stable_steps = int(self.config.get("contact_stable_steps", 2))
        self.release_steps = int(self.config.get("release_steps", 2))
        self.grip_step = float(self.config.get("grip_step", 0.003))
        self.max_grip = float(self.config.get("max_grip", 0.15))
        self.pause_for_grasp = bool(self.config.get("pause_for_grasp", False))
        self.max_grasp_hold_steps = int(self.config.get("max_grasp_hold_steps", 20))
        self.initial_command = None
        self.phase_step = 0

    def _init_surface_ik(
        self, payload, data_dir, device, normalized_action_clip,
        action_rate_limit_scale,
    ):
        """加载自主PCA，并准备只优化手指的可微表面IK。"""
        from .interaction import TargetHandGeometry

        self.surface_ik = True
        self.composite_type = None
        self.trajectory_pca = False
        self.trajectory_mixture = False
        self.config = payload["config"]
        self.dimensions = payload["dimensions"]
        self.model_type = self.config["model_type"]
        self.base = GeometryPolicyRunner(
            payload["base_checkpoint"], data_dir, device,
            normalized_action_clip=normalized_action_clip,
            action_rate_limit_scale=action_rate_limit_scale,
        )
        self.normalization = self.base.normalization
        self.mappings = self.base.mappings
        self.motion_steps = self.base.motion_steps
        self.action_horizon = self.motion_steps
        self.history = 1
        self.ensemble_decay = 0.0
        self.action_clip = float(normalized_action_clip)
        self.rate_scale = float(action_rate_limit_scale)
        self.action_delta_norm_limit = self.base.action_delta_norm_limit
        self.hand_geometry = TargetHandGeometry(self.config["hand"])
        self.finger_lower = torch.as_tensor(
            payload["finger_lower"], dtype=torch.float32
        )
        self.finger_upper = torch.as_tensor(
            payload["finger_upper"], dtype=torch.float32
        )
        self.initial_command = None
        self.phase_step = 0

    def _init_keypose(
        self, payload, data_dir, normalized_action_clip,
        action_rate_limit_scale,
    ):
        """加载只预测三个关键状态、再确定性插值的自主策略。"""
        from .interaction import TargetHandGeometry

        self.keypose_policy = True
        self.composite_type = None
        self.trajectory_pca = False
        self.trajectory_mixture = False
        self.config = payload["config"]
        self.dimensions = payload["dimensions"]
        self.model_type = self.config["model_type"]
        self.model = build_keypose_model(
            self.config, self.dimensions["task_observation_dim"],
            self.dimensions["action_dim"], self.dimensions["keypose_dim"],
        ).to(self.device)
        self.model.load_state_dict(payload["model_state"])
        self.model.eval()
        data_dir = Path(data_dir)
        with np.load(data_dir / "geometry_normalization.npz", allow_pickle=False) as archive:
            self.normalization = {
                name: archive[name].astype(np.float32) for name in archive.files
            }
        self.mappings = json.loads(
            (data_dir / "mappings.json").read_text(encoding="utf-8")
        )
        self.target_mean = np.asarray(payload["target_mean"], dtype=np.float32)
        self.target_std = np.asarray(payload["target_std"], dtype=np.float32)
        self.interaction_mean = np.asarray(payload["interaction_mean"], dtype=np.float32)
        self.interaction_std = np.asarray(payload["interaction_std"], dtype=np.float32)
        self.initial_interaction_dim = self.dimensions["interaction_dim"]
        self.hand_geometry = TargetHandGeometry(self.config["hand"])
        self.pregrasp_frame = int(payload["pregrasp_frame"])
        self.grasp_frame = int(payload["grasp_frame"])
        self.lift_end_frame = int(payload["lift_end_frame"])
        self.motion_steps = int(payload["sequence_length"])
        self.action_horizon = self.motion_steps
        self.history = 1
        self.ensemble_decay = 0.0
        self.action_clip = float(normalized_action_clip)
        self.rate_scale = float(action_rate_limit_scale)
        self.action_delta_norm_limit = float(
            self.normalization.get("action_delta_norm_limit", np.asarray(0.0))
        )
        self.object_points = None
        self.initial_command = None
        self.phase_step = 0

    @staticmethod
    def _smooth_interpolate(start, end, count):
        """生成含终点的smoothstep关键状态插值。"""
        alpha = np.linspace(0.0, 1.0, count, dtype=np.float32)
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)
        return start[None] * (1.0 - alpha[:, None]) + end[None] * alpha[:, None]

    def _keypose_sequence(self, keypose):
        """把网络输出的三个关键状态展开为240步完整命令。"""
        action_dim = self.dimensions["action_dim"]
        pre = self.initial_command.copy()
        pre[:6] += keypose[:6]
        grasp = self.initial_command + keypose[6:6 + action_dim]
        final = grasp.copy()
        final[:6] = self.initial_command[:6] + keypose[6 + action_dim:]
        sequence = np.empty((self.motion_steps, action_dim), dtype=np.float32)
        sequence[:self.pregrasp_frame + 1] = self._smooth_interpolate(
            self.initial_command, pre, self.pregrasp_frame + 1
        )
        sequence[self.pregrasp_frame:self.grasp_frame + 1] = self._smooth_interpolate(
            pre, grasp, self.grasp_frame - self.pregrasp_frame + 1
        )
        sequence[self.grasp_frame:self.lift_end_frame + 1] = self._smooth_interpolate(
            grasp, final, self.lift_end_frame - self.grasp_frame + 1
        )
        sequence[self.lift_end_frame:] = final
        return sequence

    @staticmethod
    def _phase_frames(frames):
        """由自主轨迹的腕部高度最低点定位“抓稳后、抬升前”阶段。"""
        frame_count = len(frames)
        wrist_z = np.asarray(frames[:, 2], dtype=np.float32)
        search_start = int(0.15 * frame_count)
        search_end = max(search_start + 1, int(0.75 * frame_count))
        grasp_frame = search_start + int(np.argmin(wrist_z[search_start:search_end]))
        close_frame = max(0, grasp_frame - 30)
        return close_frame, grasp_frame

    def _closure_directions(self, frames, close_frame, grasp_frame):
        """从PCA名义动作自身提取五根手指的闭合方向。"""
        start = frames[close_frame, 6:]
        end = frames[grasp_frame, 6:]
        fallback = frames[-1, 6:]
        directions = []
        for group in self.finger_groups:
            group = np.asarray(group, dtype=np.int64)
            direction = end[group] - start[group]
            norm = float(np.linalg.norm(direction))
            if norm < 1e-4:
                direction = fallback[group] - start[group]
                norm = float(np.linalg.norm(direction))
            directions.append(
                direction / norm if norm >= 1e-4 else np.zeros_like(direction)
            )
        return directions

    def set_runtime_contacts(self, body_loads):
        """接收评测器在当前物理步测得的各刚体法向接触冲量。"""
        if not self.contact_feedback:
            return
        self.tip_loads = np.asarray(
            [float(body_loads.get(name, 0.0)) for name in self.tip_links],
            dtype=np.float32,
        )

    def _init_mixture(self, payload, data_dir, normalized_action_clip, action_rate_limit_scale):
        """加载多轨迹生成器与质量判别器，不携带训练轨迹。"""
        self.composite_type = None
        self.trajectory_pca = False
        self.trajectory_mixture = True
        self.config = payload["config"]
        self.dimensions = payload["dimensions"]
        self.model_type = self.config["model_type"]
        self.model = build_pca_mixture(
            self.config, self.dimensions["task_observation_dim"],
            self.dimensions["action_dim"],
        ).to(self.device)
        self.model.load_state_dict(payload["model_state"])
        self.model.eval()
        self.pca_mean = np.asarray(payload["pca_mean"], dtype=np.float32)
        self.pca_components = np.asarray(payload["pca_components"], dtype=np.float32)
        self.coefficient_mean = np.asarray(payload["coefficient_mean"], dtype=np.float32)
        self.coefficient_std = np.asarray(payload["coefficient_std"], dtype=np.float32)
        self.sequence_shape = tuple(payload["sequence_shape"])
        data_dir = Path(data_dir)
        with np.load(data_dir / "geometry_normalization.npz", allow_pickle=False) as archive:
            self.normalization = {name: archive[name].astype(np.float32) for name in archive.files}
        self.mappings = json.loads((data_dir / "mappings.json").read_text(encoding="utf-8"))
        self.history = 1
        self.action_horizon = self.sequence_shape[0]
        self.motion_steps = self.sequence_shape[0]
        self.ensemble_decay = 0.0
        self.action_clip = float(normalized_action_clip)
        self.rate_scale = float(action_rate_limit_scale)
        self.action_delta_norm_limit = float(
            self.normalization.get("action_delta_norm_limit", np.asarray(0.0))
        )
        self.object_points = None
        self.initial_command = None
        self.phase_step = 0
        self.initial_observation = None
        self.previous_command = None
        self.phase_step = 0

    def _init_interaction(
        self, payload, data_dir, device, normalized_action_clip,
        action_rate_limit_scale,
    ):
        """加载PCA名义策略和只依赖当前手—物状态的有界残差网络。"""
        from .interaction import TargetHandGeometry

        self.interaction_residual = True
        self.composite_type = None
        self.trajectory_pca = False
        self.trajectory_mixture = False
        self.config = payload["config"]
        self.dimensions = payload["dimensions"]
        self.model_type = self.config["model_type"]
        self.model = build_interaction_residual(
            self.config, self.dimensions["task_dim"], self.dimensions["action_dim"],
            self.dimensions["interaction_dim"],
        ).to(self.device)
        self.model.load_state_dict(payload["model_state"])
        self.model.eval()
        self.base = GeometryPolicyRunner(
            payload["base_checkpoint"], data_dir, device,
            normalized_action_clip=normalized_action_clip,
            action_rate_limit_scale=action_rate_limit_scale,
        )
        self.normalization = self.base.normalization
        self.mappings = self.base.mappings
        self.motion_steps = self.base.motion_steps
        self.action_horizon = self.base.action_horizon
        self.history = 1
        self.ensemble_decay = 0.0
        self.action_clip = float(normalized_action_clip)
        self.rate_scale = float(action_rate_limit_scale)
        self.action_delta_norm_limit = self.base.action_delta_norm_limit
        self.residual_limit = np.asarray(payload["residual_limit"], dtype=np.float32)
        self.hand_geometry = TargetHandGeometry(self.config["hand"])
        self.initial_command = None
        self.object_points = None
        self.object_local_points = None
        self.phase_step = 0

    def _init_composite(
        self, payload, data_dir, device, normalized_action_clip,
        action_rate_limit_scale,
    ):
        """加载只组合自主网络输出、不读取参考轨迹的复合策略。"""
        self.config = payload["config"]
        self.composite_type = self.config["composite_type"]
        self.trajectory_pca = False
        self.trajectory_mixture = False
        self.primary = GeometryPolicyRunner(
            payload["primary_checkpoint"], data_dir, device,
            normalized_action_clip=normalized_action_clip,
            action_rate_limit_scale=action_rate_limit_scale,
        )
        self.secondary = GeometryPolicyRunner(
            payload.get("secondary_checkpoint", payload["primary_checkpoint"]),
            data_dir, device,
            normalized_action_clip=normalized_action_clip,
            action_rate_limit_scale=action_rate_limit_scale,
        )
        self.dimensions = self.primary.dimensions
        self.model_type = self.config["model_type"]
        self.mappings = self.primary.mappings
        self.history = self.primary.history
        self.action_horizon = self.primary.action_horizon
        self.motion_steps = self.primary.motion_steps
        self.ensemble_decay = self.primary.ensemble_decay
        self.action_clip = self.primary.action_clip
        self.rate_scale = self.primary.rate_scale
        self.action_delta_norm_limit = self.primary.action_delta_norm_limit
        self.phase_step = 0
        self.initial_command = None

    def set_task_geometry(self, object_dir, scale, rotation, initial_action):
        if self.surface_ik:
            self.base.set_task_geometry(object_dir, scale, rotation, initial_action)
            self.object_points = self.base.object_points.copy()
            return
        if self.contact_feedback:
            self.base.set_task_geometry(object_dir, scale, rotation, initial_action)
            return
        if self.interaction_residual:
            self.base.set_task_geometry(object_dir, scale, rotation, initial_action)
            self.object_points = self.base.object_points.copy()
            return
        if self.composite_type is not None:
            self.primary.set_task_geometry(object_dir, scale, rotation, initial_action)
            self.secondary.set_task_geometry(object_dir, scale, rotation, initial_action)
            return
        self.object_points = object_points_in_initial_wrist(
            object_dir, scale, rotation, initial_action,
            self.dimensions.get("point_count", 128),
        )

    def normalize_observation(self, observation):
        return (
            np.asarray(observation, dtype=np.float32) - self.normalization["observation_mean"]
        ) / self.normalization["observation_std"]

    def _runtime_interaction(self, observation):
        """由当前真实手姿和物体姿态计算标准化75维手物关系。"""
        from scipy.spatial.transform import Rotation
        from .interaction import interaction_features, policy_pose_from_observations

        observation = np.asarray(observation, dtype=np.float32)
        pose = policy_pose_from_observations(self.hand, observation)[0]
        dof_count = (len(observation) - 32) // 2
        object_position = observation[2 * dof_count:2 * dof_count + 3]
        object_quaternion = observation[2 * dof_count + 3:2 * dof_count + 7]
        object_rotation = Rotation.from_quat(
            object_quaternion
        ).as_matrix().astype(np.float32)
        object_points = self.object_local_points @ object_rotation.T + object_position
        hand_points = self.hand_geometry.points(pose)[0]
        feature = interaction_features(
            hand_points[None], object_points[None], pose[None, 3:6]
        )[0]
        return ((feature - self.interaction_mean) / self.interaction_std).astype(
            np.float32
        )

    def reset(self, category_name, initial_observation, initial_action=None):
        if self.direct_interaction:
            from scipy.spatial.transform import Rotation

            del category_name
            self.initial_command = np.asarray(initial_action, dtype=np.float32).copy()
            observation = np.asarray(initial_observation, dtype=np.float32)
            dof_count = (len(observation) - 32) // 2
            object_position = observation[2 * dof_count:2 * dof_count + 3]
            object_quaternion = observation[2 * dof_count + 3:2 * dof_count + 7]
            wrist_rotation = Rotation.from_euler(
                "xyz", self.initial_command[3:6]
            ).as_matrix().astype(np.float32)
            initial_world = (
                self.object_points @ wrist_rotation.T + self.initial_command[:3]
            )
            object_rotation = Rotation.from_quat(
                object_quaternion
            ).as_matrix().astype(np.float32)
            self.object_local_points = (
                initial_world - object_position
            ) @ object_rotation
            self.previous_command = self.initial_command.copy()
            initial_normalized = self.normalize_observation(observation)
            initial_interaction = self._runtime_interaction(observation)
            zero_delta = (
                -self.normalization["initial_delta_mean"]
                / self.normalization["initial_delta_std"]
            ).astype(np.float32)
            self.initial_observation = initial_normalized
            self.observation_history.clear()
            self.interaction_history.clear()
            self.previous_delta_history.clear()
            for _ in range(self.history):
                self.observation_history.append(initial_normalized.copy())
                self.interaction_history.append(initial_interaction.copy())
                self.previous_delta_history.append(zero_delta.copy())
            self.pending_chunks = []
            self.phase_step = 0
            return
        if self.surface_ik:
            from scipy.spatial.transform import Rotation

            self.base.reset(category_name, initial_observation, initial_action)
            self.initial_command = np.asarray(initial_action, dtype=np.float32).copy()
            frames = self.initial_command[None] + self.base.generated_sequence
            close_frame, grasp_frame = self._phase_frames(frames)
            nominal = frames[grasp_frame].copy()
            rotation = Rotation.from_euler(
                "xyz", self.initial_command[3:6]
            ).as_matrix().astype(np.float32)
            object_world = (
                self.object_points @ rotation.T + self.initial_command[:3]
            )
            tip_indices = torch.as_tensor([3, 6, 9, 12, 14])
            nominal_tensor = torch.from_numpy(nominal)
            with torch.no_grad():
                nominal_tips = self.hand_geometry.points_tensor(
                    nominal_tensor
                )[0, tip_indices]
            surface = torch.from_numpy(object_world)
            distance = torch.cdist(nominal_tips, surface)
            targets = surface[distance.argmin(dim=1)]
            center = surface.mean(dim=0)
            outward = targets - center
            outward = outward / outward.norm(dim=1, keepdim=True).clamp_min(1e-6)
            targets = targets + float(self.config.get("surface_offset_m", 0.002)) * outward
            finger = nominal_tensor[6:].clone().requires_grad_(True)
            translation = torch.zeros(3, dtype=torch.float32, requires_grad=True)
            lower = torch.maximum(
                self.finger_lower,
                finger.detach() - float(self.config.get("joint_delta_bound", 0.35)),
            )
            upper = torch.minimum(
                self.finger_upper,
                finger.detach() + float(self.config.get("joint_delta_bound", 0.35)),
            )
            optimizer = torch.optim.Adam(
                [finger, translation],
                lr=float(self.config.get("ik_learning_rate", 0.03)),
            )
            nominal_finger = finger.detach().clone()
            scale = (upper - lower).clamp_min(0.1)
            for _ in range(int(self.config.get("ik_steps", 60))):
                command = torch.cat([
                    nominal_tensor[:3] + translation,
                    nominal_tensor[3:6], finger,
                ])
                tips = self.hand_geometry.points_tensor(command)[0, tip_indices]
                contact_loss = (tips - targets).square().sum(dim=1).mean() / 1e-4
                anchor = ((finger - nominal_finger) / scale).square().mean()
                translation_anchor = translation.square().mean() / (0.02 ** 2)
                loss = (
                    contact_loss
                    + float(self.config.get("anchor_weight", 0.02)) * anchor
                    + float(self.config.get("translation_anchor_weight", 0.02))
                    * translation_anchor
                )
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                with torch.no_grad():
                    finger.clamp_(lower, upper)
                    translation_limit = float(
                        self.config.get("wrist_translation_bound_m", 0.03)
                    )
                    translation_norm = translation.norm().clamp_min(1e-9)
                    translation.mul_(min(1.0, translation_limit / float(translation_norm)))
            correction = np.zeros(frames.shape[1], dtype=np.float32)
            correction[:3] = translation.detach().numpy()
            correction[6:] = finger.detach().numpy() - nominal[6:]
            alpha = np.zeros(len(frames), dtype=np.float32)
            ramp = np.linspace(0.0, 1.0, grasp_frame - close_frame + 1, dtype=np.float32)
            alpha[close_frame:grasp_frame + 1] = ramp * ramp * (3.0 - 2.0 * ramp)
            alpha[grasp_frame + 1:] = 1.0
            adjusted = frames + alpha[:, None] * correction[None]
            with torch.no_grad():
                final_tips = self.hand_geometry.points_tensor(
                    torch.from_numpy(adjusted[grasp_frame])
                )[0, tip_indices]
            self.ik_diagnostic = {
                "close_frame": close_frame, "grasp_frame": grasp_frame,
                "before_tip_distance_m": float(
                    torch.linalg.norm(nominal_tips - targets, dim=1).mean()
                ),
                "after_tip_distance_m": float(
                    torch.linalg.norm(final_tips - targets, dim=1).mean()
                ),
                "joint_correction_l2_rad": float(np.linalg.norm(correction[6:])),
                "wrist_translation_l2_m": float(np.linalg.norm(correction[:3])),
            }
            self.generated_sequence = adjusted - self.initial_command[None]
            self.phase_step = 0
            return
        if self.trajectory_diffusion:
            del category_name
            self.initial_command = np.asarray(initial_action, dtype=np.float32).copy()
            task = self.normalize_observation(initial_observation)[-32:]
            command = (
                self.initial_command - self.normalization["initial_command_mean"]
            ) / self.normalization["initial_command_std"]
            points = (
                self.object_points - self.normalization["point_mean"]
            ) / self.normalization["point_std"]
            with torch.no_grad():
                task_tensor = torch.from_numpy(task[None]).to(self.device)
                command_tensor = torch.from_numpy(command[None]).to(self.device)
                point_tensor = torch.from_numpy(points[None]).to(self.device)
                condition = self.model.encode(
                    task_tensor, command_tensor, point_tensor
                )
                baseline = self.base_model(
                    task_tensor, command_tensor, point_tensor, None
                )
                regression = self.model.regression(condition)
                generated = sample_pca_latent_diffusion(
                    self.model, condition, self.alpha_bars,
                    self.config.get("candidate_count", 8),
                    self.config.get("sample_seed", 20260902),
                )
                candidates = torch.cat([
                    baseline[:, None], regression[:, None], generated,
                ], dim=1)
                scores = self.model.score(condition, candidates)[0]
                alternative_score, alternative_index = scores[1:].max(dim=0)
                margin = float(self.config.get("selection_margin", 0.1))
                selected = (
                    int(alternative_index.item()) + 1
                    if float(alternative_score - scores[0]) > margin else 0
                )
                normalized_coefficients = candidates[0, selected].cpu().numpy()
            coefficients = (
                normalized_coefficients * self.coefficient_std
                + self.coefficient_mean
            )
            normalized_sequence = self.pca_mean + coefficients @ self.pca_components
            normalized_sequence = np.clip(
                normalized_sequence.reshape(self.sequence_shape),
                -self.action_clip, self.action_clip,
            )
            self.generated_sequence = (
                normalized_sequence * self.normalization["initial_delta_std"]
                + self.normalization["initial_delta_mean"]
            ).astype(np.float32)
            self.selected_candidate = selected
            self.candidate_scores = scores.cpu().numpy()
            self.phase_step = 0
            return
        if self.keypose_policy:
            del category_name
            self.initial_command = np.asarray(initial_action, dtype=np.float32).copy()
            normalized_observation = self.normalize_observation(initial_observation)[-32:]
            normalized_command = (
                self.initial_command - self.normalization["initial_command_mean"]
            ) / self.normalization["initial_command_std"]
            normalized_points = (
                self.object_points - self.normalization["point_mean"]
            ) / self.normalization["point_std"]
            normalized_interaction = self._initial_interaction()
            with torch.no_grad():
                prediction = self.model(
                    torch.from_numpy(normalized_observation[None]).to(self.device),
                    torch.from_numpy(normalized_command[None]).to(self.device),
                    torch.from_numpy(normalized_points[None]).to(self.device),
                    torch.from_numpy(normalized_interaction[None]).to(self.device),
                )[0].cpu().numpy()
            keypose = prediction * self.target_std + self.target_mean
            self.generated_sequence = self._keypose_sequence(keypose)
            self.phase_step = 0
            return
        if self.contact_feedback:
            self.base.reset(category_name, initial_observation, initial_action)
            self.initial_command = np.asarray(initial_action, dtype=np.float32).copy()
            frames = self.initial_command[None] + self.base.generated_sequence
            close_frame, grasp_frame = self._phase_frames(frames)
            self.feedback_start = grasp_frame
            self.closure_directions = self._closure_directions(
                frames, close_frame, grasp_frame
            )
            self.finger_residual = np.zeros(5, dtype=np.float32)
            self.tip_loads = np.zeros(5, dtype=np.float32)
            self.contact_streak = np.zeros(5, dtype=np.int32)
            self.release_streak = np.zeros(5, dtype=np.int32)
            self.finger_engaged = np.zeros(5, dtype=bool)
            self.contact_seen = False
            self.grasp_hold_steps = 0
            self.grasp_released = False
            self.phase_step = 0
            return
        if self.interaction_residual:
            from scipy.spatial.transform import Rotation

            self.base.reset(category_name, initial_observation, initial_action)
            self.initial_command = np.asarray(initial_action, dtype=np.float32).copy()
            observation = np.asarray(initial_observation, dtype=np.float32)
            dof_count = (len(observation) - 32) // 2
            object_position = observation[2 * dof_count:2 * dof_count + 3]
            object_quaternion = observation[2 * dof_count + 3:2 * dof_count + 7]
            wrist_rotation = Rotation.from_euler(
                "xyz", self.initial_command[3:6]
            ).as_matrix().astype(np.float32)
            initial_world = self.object_points @ wrist_rotation.T + self.initial_command[:3]
            object_rotation = Rotation.from_quat(object_quaternion).as_matrix().astype(np.float32)
            self.object_local_points = (initial_world - object_position) @ object_rotation
            self.phase_step = 0
            return
        if self.composite_type is not None:
            self.primary.reset(category_name, initial_observation, initial_action)
            self.secondary.reset(category_name, initial_observation, initial_action)
            self.initial_command = np.asarray(initial_action, dtype=np.float32).copy()
            if self.composite_type == "phase_lead":
                lead = float(self.config["finger_phase_lead"])
                self.secondary.phase_step = int(round(lead * (self.motion_steps - 1)))
            self.phase_step = 0
            return
        if self.trajectory_mixture:
            del category_name
            if initial_action is None or self.object_points is None:
                raise ValueError("多候选策略reset前必须提供初始动作并设置物体点云")
            self.initial_command = np.asarray(initial_action, dtype=np.float32).copy()
            normalized_observation = self.normalize_observation(initial_observation)[-32:]
            normalized_command = (
                self.initial_command - self.normalization["initial_command_mean"]
            ) / self.normalization["initial_command_std"]
            normalized_points = (
                self.object_points - self.normalization["point_mean"]
            ) / self.normalization["point_std"]
            with torch.no_grad():
                condition, logits, candidates = self.model.generate(
                    torch.from_numpy(normalized_observation[None]).to(self.device),
                    torch.from_numpy(normalized_command[None]).to(self.device),
                    torch.from_numpy(normalized_points[None]).to(self.device),
                )
                selection = self.config.get("selection", "critic")
                if selection == "fixed":
                    selected = int(self.config["fixed_mode"])
                elif selection == "gate":
                    selected = int(torch.argmax(logits[0]).item())
                else:
                    quality = self.model.score(condition, candidates)[0]
                    prior = torch.log_softmax(logits[0], dim=-1)
                    score = quality + float(self.config.get("critic_prior_weight", 0.15)) * prior
                    selected = int(torch.argmax(score).item())
                normalized_coefficients = candidates[0, selected].cpu().numpy()
            coefficients = normalized_coefficients * self.coefficient_std + self.coefficient_mean
            normalized_sequence = self.pca_mean + coefficients @ self.pca_components
            normalized_sequence = np.clip(
                normalized_sequence.reshape(self.sequence_shape), -self.action_clip, self.action_clip
            )
            self.generated_sequence = (
                normalized_sequence * self.normalization["initial_delta_std"]
                + self.normalization["initial_delta_mean"]
            ).astype(np.float32)
            self.phase_step = 0
            return
        if self.trajectory_pca:
            del category_name
            if initial_action is None or self.object_points is None:
                raise ValueError("PCA策略reset前必须提供初始动作并设置物体点云")
            self.initial_command = np.asarray(initial_action, dtype=np.float32).copy()
            normalized_observation = self.normalize_observation(initial_observation)[-32:]
            normalized_command = (
                self.initial_command - self.normalization["initial_command_mean"]
            ) / self.normalization["initial_command_std"]
            normalized_points = (
                self.object_points - self.normalization["point_mean"]
            ) / self.normalization["point_std"]
            normalized_interaction = (
                self._initial_interaction()
                if self.initial_interaction_dim else None
            )
            with torch.no_grad():
                normalized_coefficients = self.model(
                    torch.from_numpy(normalized_observation[None]).to(self.device),
                    torch.from_numpy(normalized_command[None]).to(self.device),
                    torch.from_numpy(normalized_points[None]).to(self.device),
                    (
                        torch.from_numpy(normalized_interaction[None]).to(self.device)
                        if normalized_interaction is not None else None
                    ),
                )[0].cpu().numpy()
            coefficients = normalized_coefficients * self.coefficient_std + self.coefficient_mean
            normalized_sequence = self.pca_mean + coefficients @ self.pca_components
            normalized_sequence = np.clip(
                normalized_sequence.reshape(self.sequence_shape),
                -self.action_clip, self.action_clip,
            )
            self.generated_sequence = (
                normalized_sequence * self.normalization["initial_delta_std"]
                + self.normalization["initial_delta_mean"]
            ).astype(np.float32)
            self.phase_step = 0
            return
        del category_name
        if initial_action is None or self.object_points is None:
            raise ValueError("几何策略reset前必须提供初始动作并设置物体点云")
        self.initial_command = np.asarray(initial_action, dtype=np.float32).copy()
        self.previous_command = self.initial_command.copy()
        self.initial_observation = self.normalize_observation(initial_observation)
        zero_delta = (
            -self.normalization["initial_delta_mean"]
            / self.normalization["initial_delta_std"]
        ).astype(np.float32)
        self.observation_history.clear()
        self.previous_delta_history.clear()
        for _ in range(self.history):
            self.observation_history.append(self.initial_observation.copy())
            self.previous_delta_history.append(zero_delta.copy())
        self.pending_chunks = []
        self.phase_step = 0

    def _ensemble(self, chunk):
        self.pending_chunks.append((self.phase_step, chunk))
        predictions, weights = [], []
        retained = []
        for start, values in self.pending_chunks:
            offset = self.phase_step - start
            if offset < len(values):
                retained.append((start, values))
                predictions.append(values[offset])
                weights.append(self.ensemble_decay ** offset)
        self.pending_chunks = retained
        weights = np.asarray(weights, dtype=np.float32)
        weights /= weights.sum()
        return np.sum(np.asarray(predictions) * weights[:, None], axis=0)

    @torch.no_grad()
    def act(self, observation):
        if self.direct_interaction:
            normalized_observation = self.normalize_observation(observation)
            normalized_interaction = self._runtime_interaction(observation)
            previous_delta = (
                self.previous_command - self.initial_command
                - self.normalization["initial_delta_mean"]
            ) / self.normalization["initial_delta_std"]
            self.observation_history.append(normalized_observation)
            self.interaction_history.append(normalized_interaction)
            self.previous_delta_history.append(previous_delta.astype(np.float32))
            phase = min(
                self.phase_step / float(max(self.motion_steps - 1, 1)), 1.0
            )
            chunk = self.model(
                torch.from_numpy(self.initial_observation[None]).to(self.device),
                torch.from_numpy((
                    (self.initial_command - self.normalization["initial_command_mean"])
                    / self.normalization["initial_command_std"]
                )[None]).to(self.device),
                torch.from_numpy((
                    (self.object_points - self.normalization["point_mean"])
                    / self.normalization["point_std"]
                )[None]).to(self.device),
                torch.from_numpy(np.asarray(self.observation_history)[None]).to(self.device),
                torch.from_numpy(np.asarray(self.interaction_history)[None]).to(self.device),
                torch.from_numpy(np.asarray(self.previous_delta_history)[None]).to(self.device),
                torch.tensor([[phase]], dtype=torch.float32, device=self.device),
            )[0].cpu().numpy()
            chunk = np.clip(chunk, -self.action_clip, self.action_clip)
            normalized_delta = self._ensemble(chunk)
            command = self.initial_command + (
                normalized_delta * self.normalization["initial_delta_std"]
                + self.normalization["initial_delta_mean"]
            )
            if self.rate_scale > 0:
                limits = self.normalization["action_delta_limit"] * self.rate_scale
                command = self.previous_command + np.clip(
                    command - self.previous_command, -limits, limits
                )
            self.previous_command = command.astype(np.float32)
            self.phase_step += 1
            return self.previous_command.copy()
        if self.keypose_policy:
            del observation
            index = min(self.phase_step, len(self.generated_sequence) - 1)
            command = self.generated_sequence[index]
            self.phase_step += 1
            return command.copy()
        if self.contact_feedback:
            loaded = self.tip_loads >= self.contact_threshold
            self.contact_streak = np.where(loaded, self.contact_streak + 1, 0)
            self.release_streak = np.where(loaded, 0, self.release_streak + 1)
            self.finger_engaged |= self.contact_streak >= self.contact_stable_steps
            self.finger_engaged &= self.release_streak < self.release_steps
            self.contact_seen = bool(self.contact_seen or np.any(self.finger_engaged))
            opposition = bool(
                self.finger_engaged[0] and np.any(self.finger_engaged[1:])
            )
            holding = (
                self.pause_for_grasp
                and not self.grasp_released
                and self.base.phase_step >= self.feedback_start
            )
            if holding and (
                opposition or self.grasp_hold_steps >= self.max_grasp_hold_steps
            ):
                self.grasp_released = True
                holding = False
            if holding:
                index = min(self.base.phase_step, len(self.base.generated_sequence) - 1)
                nominal = self.initial_command + self.base.generated_sequence[index]
                self.grasp_hold_steps += 1
                for finger in range(5):
                    if not self.finger_engaged[finger]:
                        self.finger_residual[finger] = min(
                            self.finger_residual[finger] + self.grip_step,
                            self.max_grip,
                        )
            else:
                nominal = self.base.act(observation)
                if (
                    not self.pause_for_grasp
                    and self.base.phase_step - 1 >= self.feedback_start
                    and self.contact_seen and not opposition
                ):
                    for finger in range(5):
                        if self.release_streak[finger] >= self.release_steps:
                            self.finger_residual[finger] = min(
                                self.finger_residual[finger] + self.grip_step,
                                self.max_grip,
                            )
            command = nominal.copy()
            for finger, group in enumerate(self.finger_groups):
                indices = 6 + np.asarray(group, dtype=np.int64)
                command[indices] += (
                    self.closure_directions[finger] * self.finger_residual[finger]
                )
            self.phase_step = self.base.phase_step
            return command.astype(np.float32)
        if self.interaction_residual:
            from scipy.spatial.transform import Rotation
            from .interaction import interaction_features, policy_pose_from_observations

            nominal = self.base.act(observation)
            observation = np.asarray(observation, dtype=np.float32)
            pose = policy_pose_from_observations(self.config["hand"], observation)[0]
            hand_points = self.hand_geometry.points(pose)[0]
            dof_count = (len(observation) - 32) // 2
            object_position = observation[2 * dof_count:2 * dof_count + 3]
            object_quaternion = observation[2 * dof_count + 3:2 * dof_count + 7]
            object_rotation = Rotation.from_quat(object_quaternion).as_matrix().astype(np.float32)
            object_points = self.object_local_points @ object_rotation.T + object_position
            interaction = interaction_features(
                hand_points[None], object_points[None], pose[None, 3:6]
            )[0]
            current_task = self.normalize_observation(observation)[-32:]
            nominal_delta = (
                nominal - self.initial_command - self.normalization["initial_delta_mean"]
            ) / self.normalization["initial_delta_std"]
            phase = min(
                max(self.base.phase_step - 1, 0) / float(max(self.motion_steps - 1, 1)), 1.0
            )
            residual = self.model(
                torch.from_numpy(current_task[None]).to(self.device),
                torch.from_numpy(nominal_delta[None].astype(np.float32)).to(self.device),
                torch.from_numpy(interaction[None]).to(self.device),
                torch.tensor([[phase]], dtype=torch.float32, device=self.device),
            )[0].cpu().numpy()
            correction = self.residual_limit * residual
            if bool(self.config.get("finger_residual_only", False)):
                correction[:6] = 0.0
            correction *= float(self.config.get("residual_gain", 1.0))
            self.phase_step = self.base.phase_step
            return (nominal + correction).astype(np.float32)
        if self.composite_type is not None:
            primary = self.primary.act(observation)
            secondary = self.secondary.act(observation)
            command = primary.copy()
            command[6:] = secondary[6:]
            scale = float(self.config.get("finger_scale", 1.0))
            command[6:] = self.initial_command[6:] + scale * (
                command[6:] - self.initial_command[6:]
            )
            self.phase_step += 1
            return command.astype(np.float32)
        if (self.trajectory_pca or self.trajectory_mixture
                or self.trajectory_diffusion or self.surface_ik):
            del observation
            index = min(self.phase_step, len(self.generated_sequence) - 1)
            command = self.initial_command + self.generated_sequence[index]
            self.phase_step += 1
            return command.astype(np.float32)
        normalized = self.normalize_observation(observation)
        previous_delta = (
            self.previous_command - self.initial_command - self.normalization["initial_delta_mean"]
        ) / self.normalization["initial_delta_std"]
        self.observation_history.append(normalized)
        self.previous_delta_history.append(previous_delta.astype(np.float32))
        phase = min(self.phase_step / float(max(self.motion_steps - 1, 1)), 1.0)
        tensors = {
            "initial_observation": torch.from_numpy(self.initial_observation[None]).to(self.device),
            "initial_command": torch.from_numpy((
                (self.initial_command - self.normalization["initial_command_mean"])
                / self.normalization["initial_command_std"]
            )[None]).to(self.device),
            "object_points": torch.from_numpy((
                (self.object_points - self.normalization["point_mean"])
                / self.normalization["point_std"]
            )[None]).to(self.device),
            "observation_history": torch.from_numpy(np.asarray(self.observation_history)[None]).to(self.device),
            "previous_delta_history": torch.from_numpy(np.asarray(self.previous_delta_history)[None]).to(self.device),
            "phase": torch.tensor([[phase]], dtype=torch.float32, device=self.device),
        }
        chunk = self.model(**tensors)[0].cpu().numpy()
        chunk = np.clip(chunk, -self.action_clip, self.action_clip)
        normalized_delta = self._ensemble(chunk)
        command = self.initial_command + (
            normalized_delta * self.normalization["initial_delta_std"]
            + self.normalization["initial_delta_mean"]
        )
        if self.rate_scale > 0:
            limits = self.normalization["action_delta_limit"] * self.rate_scale
            command = self.previous_command + np.clip(command - self.previous_command, -limits, limits)
        self.previous_command = command.astype(np.float32)
        self.phase_step += 1
        return self.previous_command.copy()


def is_geometry_checkpoint(path):
    try:
        payload = torch.load(path, map_location="cpu")
    except Exception:
        return False
    return payload.get("schema") in {
        "geometry_action_chunk_policy_v1", "geometry_composite_policy_v1",
        "geometry_trajectory_pca_policy_v1", "geometry_pca_mixture_policy_v1",
        "geometry_pca_interaction_residual_v1",
        "geometry_pca_contact_feedback_v1",
        "geometry_keypose_lift_policy_v1",
        "geometry_pca_latent_diffusion_policy_v1",
        "geometry_pca_surface_ik_policy_v1",
        "direct_interaction_temporal_policy_v1",
    }
