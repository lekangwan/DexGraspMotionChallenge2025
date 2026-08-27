"""Train a BC policy without changing the challenge's official trainer.

Run this script from any directory.  Dataset-relative paths are resolved by
temporarily using ``dexgrasp/`` as the working directory, matching the layout
assumed by the original project.
"""

import argparse
import collections
import hashlib
import json
import os
import pathlib
import socket
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEXGRASP_ROOT = REPO_ROOT / "dexgrasp"
for import_root in (str(REPO_ROOT), str(DEXGRASP_ROOT)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

import isaacgym  # Isaac Gym must be imported before torch.  # noqa: E402,F401
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402


import pytorch_lightning as pl
from ActionDiffusion.bc.model.policy.lhm_policy import LitBCModel
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from custom_tools.graspm3_dexrep_dataset import (
    GraspM3DexRepDataset, add_observation_noise)
from custom_tools.task_conditioning import (
    TASK_CATEGORIES,
    action_diffusion_enabled,
    action_chunk_aux_enabled,
    action_chunk_aux_parameters,
    enable_task_conditioning,
    expand_standard_state_dict_for_task_model,
    full_observation_gru_enabled,
    full_observation_gru_freeze_base,
    full_observation_transformer_enabled,
    full_observation_transformer_freeze_base,
    multi_candidate_chunk_enabled,
    phase_conditioning_enabled,
    phase_conditioning_parameters,
    relative_action_chunk_enabled,
    task_conditioning_enabled,
    temporal_attention_freeze_base,
    temporal_history_dimensions,
    temporal_history_enabled,
    temporal_history_lags,
    wrist_residual_chunk_enabled,
)


FEATURE_ENCODER_MODULES = (
    'state_enc',
    'dexrep_sensor_enc',
    'dexrep_pointL_enc',
    'bn_pnl',
)


def freeze_feature_encoder(bc_model):
    """Freeze only the loaded DexRep feature encoder, leaving the actor trainable."""
    policy = bc_model.model
    frozen_modules = []
    for name in FEATURE_ENCODER_MODULES:
        module = getattr(policy, name, None)
        if module is None:
            continue
        module.requires_grad_(False)
        frozen_modules.append(module)
    if len(frozen_modules) < 3:
        raise RuntimeError(
            'Could not identify the DexRep feature encoder; found {}/{} modules'
            .format(len(frozen_modules), len(FEATURE_ENCODER_MODULES)))
    trainable = sum(p.numel() for p in bc_model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in bc_model.parameters() if not p.requires_grad)
    if trainable == 0 or frozen == 0:
        raise RuntimeError(
            'Invalid feature freeze: trainable={} frozen={}'.format(trainable, frozen))
    print(
        'Feature encoder frozen: modules={} trainable_parameters={} '
        'frozen_parameters={}'.format(
            ','.join(FEATURE_ENCODER_MODULES), trainable, frozen))
    return frozen_modules


class FrozenEncoderModeCallback(Callback):
    """Keep frozen BatchNorm statistics fixed when Lightning enters train mode."""

    def __init__(self, modules):
        super().__init__()
        self.modules = list(modules)

    def _set_eval(self):
        for module in self.modules:
            module.eval()

    def on_train_start(self, trainer, pl_module):
        self._set_eval()

    def on_train_epoch_start(self, trainer, pl_module):
        self._set_eval()


def require_free_vram(min_free_vram_mb):
    """Stop before allocating a model when another GPU job leaves too little room."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; BC training requires the NVIDIA GPU.")
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    free_mb = free_bytes / (1024 ** 2)
    total_mb = total_bytes / (1024 ** 2)
    print("GPU memory before training: {:.0f}/{:.0f} MiB free".format(free_mb, total_mb))
    if free_mb < min_free_vram_mb:
        raise RuntimeError(
            "Only {:.0f} MiB VRAM is free, below the safety threshold of {} MiB. "
            "Wait for the other GPU process to finish instead of competing for memory."
            .format(free_mb, min_free_vram_mb)
        )


def checkpoint_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def build_category_balanced_sampler(dataset, seed, online_sample_fraction=None):
    """Give bottle/mug/bowl/camera equal expected sampling probability."""
    if hasattr(dataset, 'sample_categories'):
        categories = np.asarray(dataset.sample_categories)
    else:
        sequence_object_indices = dataset.data['obj_code_idx']
        categories = []
        for object_index in sequence_object_indices:
            object_id = dataset.obj_code_name_list[int(object_index)]
            parts = object_id.split('-', 2)
            if len(parts) < 3:
                raise ValueError('Cannot infer category from object ID: {}'.format(object_id))
            categories.append(parts[1])
        if dataset.is_flat:
            categories = np.repeat(np.asarray(categories), dataset.num_frame)
    category_list = categories.tolist() if isinstance(categories, np.ndarray) else categories
    counts = collections.Counter(category_list)
    if online_sample_fraction is not None:
        if not hasattr(dataset, 'sample_sources'):
            raise ValueError('Online source balancing requires an augmented dataset')
        fraction = float(online_sample_fraction)
        if not 0.0 < fraction < 1.0:
            raise ValueError('online_sample_fraction must be in (0, 1)')
        sources = np.asarray(dataset.sample_sources)
        if len(sources) != len(category_list):
            raise ValueError('Sample source/category lengths differ')
        group_counts = collections.Counter(zip(sources.tolist(), category_list))
        source_targets = {0: 1.0 - fraction, 1: fraction}
        category_count = len(counts)
        weights = torch.as_tensor([
            source_targets[int(source)]
            / category_count
            / group_counts[(int(source), category)]
            for source, category in zip(sources, category_list)
        ], dtype=torch.double)
        print('Balanced sampler target online fraction: {:.3f}'.format(fraction))
    else:
        weights = torch.as_tensor(
            [1.0 / counts[category] for category in category_list], dtype=torch.double)
    generator = torch.Generator()
    generator.manual_seed(seed)
    print('Balanced sampler frame counts before weighting: {}'.format(dict(sorted(counts.items()))))
    return WeightedRandomSampler(
        weights, num_samples=len(weights), replacement=True, generator=generator)


def build_object_balanced_sampler(dataset, seed):
    """Give every training object equal expected probability per epoch."""
    sequence_object_indices = np.asarray(dataset.data['obj_code_idx'])
    if dataset.is_flat:
        sample_object_indices = np.repeat(
            sequence_object_indices, dataset.num_frame)
    else:
        sample_object_indices = sequence_object_indices
    if len(sample_object_indices) != len(dataset):
        raise ValueError(
            'Object/sample lengths differ: {} versus {}'.format(
                len(sample_object_indices), len(dataset)))
    counts = collections.Counter(sample_object_indices.tolist())
    weights = torch.as_tensor([
        1.0 / counts[int(object_index)]
        for object_index in sample_object_indices
    ], dtype=torch.double)
    generator = torch.Generator()
    generator.manual_seed(seed)
    readable_counts = {
        dataset.obj_code_name_list[int(object_index)]: count
        for object_index, count in sorted(counts.items())
    }
    print('Object-balanced sampler frame counts before weighting: {}'.format(
        readable_counts))
    print('Object-balanced sampler target probability: {:.6f} each'.format(
        1.0 / len(counts)))
    return WeightedRandomSampler(
        weights, num_samples=len(weights), replacement=True,
        generator=generator)


class OnlineAugmentedDataset(Dataset):
    """Append student-visited states labeled by the routed teacher.

    Offline samples retain the 70:30 teacher/demo target.  An online sample
    has no demonstration action, so both targets are set to the teacher action;
    its effective supervision is therefore 100% teacher without changing the
    loss implementation.
    """

    def __init__(self, offline_dataset, online_path):
        self.offline = offline_dataset
        online = np.load(online_path, allow_pickle=False)
        self.online_observations = online['observations'].astype(np.float32, copy=False)
        self.online_actions = online['teacher_actions'].astype(np.float32, copy=False)
        self.online_category_indices = online['category_indices'].astype(np.int64, copy=False)
        self.online_student_actions = online['student_actions'].astype(
            np.float32, copy=False)
        self.online_object_indices = online['object_indices'].astype(
            np.int64, copy=False)
        self.online_trajectory_indices = online['trajectory_indices'].astype(
            np.int64, copy=False)
        self.online_frame_indices = online['frame_indices'].astype(
            np.int64, copy=False)
        if len(self.online_observations) != len(self.online_actions):
            raise ValueError('Online observations/actions are not aligned')
        if self.online_observations.shape[1:] != self.offline.data['obs'].shape[1:]:
            raise ValueError(
                'Online observation shape {} does not match offline {}'.format(
                    self.online_observations.shape[1:],
                    self.offline.data['obs'].shape[1:]))
        category_names = np.asarray(['bottle', 'mug', 'bowl', 'camera'])
        if np.any(self.online_category_indices < 0) or np.any(
                self.online_category_indices >= len(category_names)):
            raise ValueError('Invalid online category index')
        sequence_categories = []
        for object_index in self.offline.data['obj_code_idx']:
            object_id = self.offline.obj_code_name_list[int(object_index)]
            sequence_categories.append(object_id.split('-', 2)[1])
        offline_categories = np.asarray(sequence_categories)
        if self.offline.is_flat:
            offline_categories = np.repeat(
                offline_categories, self.offline.num_frame)
        self.sample_categories = np.concatenate([
            offline_categories,
            category_names[self.online_category_indices],
        ])
        self.sample_sources = np.concatenate([
            np.zeros(len(self.offline), dtype=np.int8),
            np.ones(len(self.online_observations), dtype=np.int8),
        ])
        print('Online aggregation: offline={} online={} total={}'.format(
            len(self.offline), len(self.online_observations), len(self)))

    def __len__(self):
        return len(self.offline) + len(self.online_observations)

    def __getitem__(self, index):
        if index < len(self.offline):
            return self.offline[index]
        online_index = index - len(self.offline)
        observation = self.online_observations[online_index].copy()
        if self.offline.args.add_noise:
            add_observation_noise(
                observation, self.offline.args, self.offline.pro_dim)
        action = self.online_actions[online_index]
        return {
            'obs': observation,
            'actions': action,
            'teacher_actions': action,
            'obj_code_idx': np.int64(-1),
            'sample_index': np.int64(online_index),
            'task_index': np.int64(self.online_category_indices[online_index]),
            'task_onehot': np.eye(
                len(TASK_CATEGORIES), dtype=np.float32)[
                    self.online_category_indices[online_index]],
        }


class CategorySubset(Dataset):
    """Select one semantic category after all labels have been aligned."""

    def __init__(self, dataset, category):
        categories = np.asarray(dataset.sample_categories)
        self.indices = np.flatnonzero(categories == category)
        if len(self.indices) == 0:
            raise ValueError('No samples for category {}'.format(category))
        self.dataset = dataset
        self.sample_categories = categories[self.indices]
        sources = getattr(dataset, 'sample_sources', None)
        self.sample_sources = (
            None if sources is None else np.asarray(sources)[self.indices])
        print('Category subset {}: {} samples'.format(
            category, len(self.indices)))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.dataset[int(self.indices[index])]


class TemporalHistoryDataset(Dataset):
    """Add previous proprioception and executed actions to each BC sample."""

    def __init__(self, dataset, args):
        self.dataset = dataset
        self.history_frames, self.prop_dim, self.action_dim = (
            temporal_history_dimensions(args))
        self.history_steps = self.history_frames - 1
        self.history_lags = temporal_history_lags(args)
        self.action_chunk_horizon = (
            action_chunk_aux_parameters(args)[0]
            if action_chunk_aux_enabled(args) else 1)
        self.include_full_observation_history = (
            full_observation_gru_enabled(args)
            or full_observation_transformer_enabled(args))
        self.include_phase_feature = phase_conditioning_enabled(args)
        self.phase_max_frame_index = (
            phase_conditioning_parameters(args)["phase_max_frame_index"]
            if self.include_phase_feature else None)
        self.offline = (
            dataset.offline
            if isinstance(dataset, OnlineAugmentedDataset)
            else dataset)
        if not self.offline.is_flat:
            raise ValueError("Temporal history expects a frame-flat dataset")
        if self.prop_dim != self.offline.pro_dim:
            raise ValueError(
                "History prop dimension {} does not match dataset {}".format(
                    self.prop_dim, self.offline.pro_dim))
        self.offline_length = len(self.offline)
        self.sample_categories = getattr(
            dataset, "sample_categories", None)
        self.sample_sources = getattr(dataset, "sample_sources", None)
        self.online_lookup = {}
        if isinstance(dataset, OnlineAugmentedDataset):
            keys = zip(
                dataset.online_object_indices.tolist(),
                dataset.online_trajectory_indices.tolist(),
                dataset.online_frame_indices.tolist())
            for index, key in enumerate(keys):
                if key in self.online_lookup:
                    raise ValueError(
                        "Duplicate online temporal key: {}".format(key))
                self.online_lookup[key] = index

    def __len__(self):
        return len(self.dataset)

    def _maybe_noise_prop(self, prop):
        prop = prop.astype(np.float32, copy=True)
        if (self.offline.args.add_noise
                and self.offline.ds_name != "test"):
            prop += np.random.uniform(
                -self.offline.args.noise_val,
                self.offline.args.noise_val,
                size=self.prop_dim).astype(np.float32)
        return prop

    def _offline_history(self, index):
        frame = index % self.offline.num_frame
        sequence_start = index - frame
        props = []
        actions = []
        for lag in self.history_lags:
            previous_frame = frame - lag
            prop_frame = max(0, previous_frame)
            previous_index = sequence_start + prop_frame
            props.append(self._maybe_noise_prop(
                self.offline.data["obs"][
                    previous_index, :self.prop_dim]))
            if previous_frame < 0:
                actions.append(np.zeros(
                    self.action_dim, dtype=np.float32))
            else:
                actions.append(
                    self.offline.data["vis_unscale_actions"][
                        previous_index].astype(np.float32, copy=True))
        return np.concatenate(props + actions, axis=0)

    def _offline_full_observation_history(self, index):
        frame = index % self.offline.num_frame
        sequence_start = index - frame
        observations = []
        for lag in self.history_lags:
            previous_index = sequence_start + max(0, frame - lag)
            observation = self.offline.data["obs"][
                previous_index].astype(np.float32, copy=True)
            observation[:self.prop_dim] = self._maybe_noise_prop(
                observation[:self.prop_dim])
            observations.append(observation)
        return np.stack(observations)

    def _online_history(self, online_index):
        object_index = int(
            self.dataset.online_object_indices[online_index])
        trajectory_index = int(
            self.dataset.online_trajectory_indices[online_index])
        frame = int(self.dataset.online_frame_indices[online_index])
        current_prop = self.dataset.online_observations[
            online_index, :self.prop_dim]
        props = []
        actions = []
        for lag in self.history_lags:
            previous_frame = frame - lag
            key = (object_index, trajectory_index, previous_frame)
            previous_index = self.online_lookup.get(key)
            if previous_index is None:
                props.append(self._maybe_noise_prop(current_prop))
                actions.append(np.zeros(
                    self.action_dim, dtype=np.float32))
            else:
                props.append(self._maybe_noise_prop(
                    self.dataset.online_observations[
                        previous_index, :self.prop_dim]))
                actions.append(
                    self.dataset.online_student_actions[
                        previous_index].astype(np.float32, copy=True))
        return np.concatenate(props + actions, axis=0)

    def _online_full_observation_history(self, online_index):
        object_index = int(
            self.dataset.online_object_indices[online_index])
        trajectory_index = int(
            self.dataset.online_trajectory_indices[online_index])
        frame = int(self.dataset.online_frame_indices[online_index])
        current = self.dataset.online_observations[online_index]
        observations = []
        for lag in self.history_lags:
            # Inference initializes every history slot with the first frame.
            # Mirror that convention instead of crossing trajectories.
            key = (
                object_index, trajectory_index, max(0, frame - lag))
            previous_index = self.online_lookup.get(key)
            source = (
                current if previous_index is None
                else self.dataset.online_observations[previous_index])
            observation = source.astype(np.float32, copy=True)
            observation[:self.prop_dim] = self._maybe_noise_prop(
                observation[:self.prop_dim])
            observations.append(observation)
        return np.stack(observations)

    def __getitem__(self, index):
        item = self.dataset[index]
        if index < self.offline_length:
            history = self._offline_history(index)
            phase_frame = index % self.offline.num_frame
            if self.include_full_observation_history:
                full_history = (
                    self._offline_full_observation_history(index))
        else:
            online_index = index - self.offline_length
            history = self._online_history(online_index)
            phase_frame = int(
                self.dataset.online_frame_indices[online_index])
            if self.include_full_observation_history:
                full_history = (
                    self._online_full_observation_history(online_index))
        item["history_features"] = history
        if self.include_phase_feature:
            normalized_phase = (
                2.0 * min(phase_frame, self.phase_max_frame_index)
                / float(self.phase_max_frame_index)
                - 1.0)
            item["phase_feature"] = np.asarray(
                [normalized_phase], dtype=np.float32)
        if self.include_full_observation_history:
            item["full_history_observations"] = full_history
        if self.action_chunk_horizon > 1:
            if index < self.offline_length:
                teacher_chunk, demo_chunk, chunk_mask = (
                    self._offline_action_chunk(index))
            else:
                teacher_chunk, demo_chunk, chunk_mask = (
                    self._online_action_chunk(index - self.offline_length))
            item["teacher_action_chunk"] = teacher_chunk
            item["demo_action_chunk"] = demo_chunk
            item["action_chunk_mask"] = chunk_mask
        return item

    def _offline_action_chunk(self, index):
        frame = index % self.offline.num_frame
        sequence_start = index - frame
        teacher = []
        demo = []
        mask = []
        for offset in range(self.action_chunk_horizon):
            valid = frame + offset < self.offline.num_frame
            target_frame = min(
                frame + offset, self.offline.num_frame - 1)
            target_index = sequence_start + target_frame
            demo_action = self.offline.data[
                "vis_unscale_actions"][target_index].astype(
                    np.float32, copy=True)
            teacher_action = (
                demo_action
                if self.offline.teacher_actions is None
                else self.offline.teacher_actions[target_index].astype(
                    np.float32, copy=True))
            teacher.append(teacher_action)
            demo.append(demo_action)
            mask.append(valid)
        return (
            np.stack(teacher),
            np.stack(demo),
            np.asarray(mask, dtype=np.bool_),
        )

    def _online_action_chunk(self, online_index):
        object_index = int(
            self.dataset.online_object_indices[online_index])
        trajectory_index = int(
            self.dataset.online_trajectory_indices[online_index])
        frame = int(self.dataset.online_frame_indices[online_index])
        teacher = []
        mask = []
        last_action = self.dataset.online_actions[online_index]
        for offset in range(self.action_chunk_horizon):
            key = (object_index, trajectory_index, frame + offset)
            target_index = self.online_lookup.get(key)
            valid = target_index is not None
            if valid:
                last_action = self.dataset.online_actions[target_index]
            teacher.append(last_action.astype(np.float32, copy=True))
            mask.append(valid)
        teacher = np.stack(teacher)
        return teacher, teacher.copy(), np.asarray(mask, dtype=np.bool_)


class DistillationBCModel(LitBCModel):
    """Unified student trained against routed teachers and original demos."""

    def __init__(self, args, env_args):
        super().__init__(args, env_args)
        if task_conditioning_enabled(args):
            enable_task_conditioning(self, args, env_args)
        config = args.distillation
        self.teacher_weight = float(config.teacher_weight)
        self.demo_weight = float(config.demo_weight)
        self.action_chunk_auxiliary_weight = (
            action_chunk_aux_parameters(args)[1]
            if action_chunk_aux_enabled(args) else 0.0)
        if self.teacher_weight < 0 or self.demo_weight < 0:
            raise ValueError('Distillation weights must be non-negative')
        if abs(self.teacher_weight + self.demo_weight - 1.0) > 1e-6:
            raise ValueError('Distillation weights must sum to one')
        self.shadow_keypoint_kinematics = None

    def _keypoint_geometry_loss(self, prediction, target, mask):
        config = self.args.get("keypoint_geometry_loss", {})
        if not bool(config.get("enabled", False)):
            return prediction.sum() * 0.0
        if self.shadow_keypoint_kinematics is None:
            from custom_tools.shadow_keypoint_loss import (
                ShadowKeypointKinematics)
            self.shadow_keypoint_kinematics = ShadowKeypointKinematics(
                prediction.device)
        return self.shadow_keypoint_kinematics.mean_distance_loss(
            prediction, target, mask)

    def forward(self, batch):
        if phase_conditioning_enabled(self.args):
            return self.model(
                batch['obs'], batch['task_onehot'],
                batch['history_features'], batch['phase_feature'])
        if (full_observation_gru_enabled(self.args)
                or full_observation_transformer_enabled(self.args)):
            return self.model(
                batch['obs'], batch['task_onehot'],
                batch['history_features'],
                batch['full_history_observations'])
        if temporal_history_enabled(self.args):
            return self.model(
                batch['obs'], batch['task_onehot'],
                batch['history_features'])
        if task_conditioning_enabled(self.args):
            return self.model(batch['obs'], batch['task_onehot'])
        return super().forward(batch)

    def training_step(self, batch, batch_idx):
        if action_diffusion_enabled(self.args):
            return self._diffusion_chunk_step(batch, "train")
        if action_chunk_aux_enabled(self.args):
            return self._action_chunk_training_step(batch)
        prediction = self.forward(batch)
        teacher = self.cal_loss(prediction, batch['teacher_actions'])
        demo = self.cal_loss(prediction, batch['actions'])
        loss = (self.teacher_weight * teacher['loss']
                + self.demo_weight * demo['loss'])
        self.log_dict({
            'train_loss': loss,
            'teacher_loss': teacher['loss'],
            'demo_loss': demo['loss'],
            'teacher_wrist_loss': teacher['wrist_loss'],
            'teacher_ori_loss': teacher['ori_loss'],
            'teacher_finger_loss': teacher['finger_loss'],
        }, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if action_diffusion_enabled(self.args):
            return self._diffusion_chunk_step(batch, "val")
        return super().validation_step(batch, batch_idx)

    def _diffusion_chunk_step(self, batch, prefix):
        choose_demo = (
            torch.rand(
                batch["demo_action_chunk"].shape[0], 1, 1,
                device=batch["demo_action_chunk"].device)
            < self.demo_weight)
        target = torch.where(
            choose_demo, batch["demo_action_chunk"],
            batch["teacher_action_chunk"])
        with torch.no_grad():
            base = self.model.base_action_chunk(
                batch["obs"], batch["task_onehot"],
                batch["history_features"])
        residual_target = (target - base).clamp(-2.0, 2.0)
        timesteps = torch.randint(
            0, self.model.diffusion_steps,
            (target.shape[0],), device=target.device)
        alpha_bar = self.model.diffusion_alpha_bars[
            timesteps].reshape(-1, 1, 1)
        noise = torch.randn_like(residual_target)
        noisy = (
            torch.sqrt(alpha_bar) * residual_target
            + torch.sqrt(1.0 - alpha_bar) * noise)
        predicted = self.model.predict_diffusion_noise(
            noisy, batch["obs"], batch["task_onehot"],
            batch["history_features"], timesteps)
        mask = batch["action_chunk_mask"].unsqueeze(-1).to(predicted.dtype)
        loss = ((predicted - noise).square() * mask).sum() / (
            mask.sum() * predicted.shape[-1])
        residual_l1 = (residual_target.abs() * mask).sum() / (
            mask.sum() * residual_target.shape[-1])
        self.log_dict({
            "{}_diffusion_loss".format(prefix): loss,
            "{}_residual_l1".format(prefix): residual_l1,
        }, prog_bar=True, on_epoch=True)
        return loss

    def _masked_future_loss(self, prediction, target, mask):
        losses = []
        for step in range(1, prediction.shape[1]):
            valid = mask[:, step].to(dtype=torch.bool)
            if torch.any(valid):
                losses.append(self.cal_loss(
                    prediction[valid, step],
                    target[valid, step])["loss"])
        if not losses:
            return prediction.sum() * 0.0
        return torch.stack(losses).mean()

    def _action_chunk_training_step(self, batch):
        chunk_kwargs = {}
        if phase_conditioning_enabled(self.args):
            chunk_kwargs["phase_feature"] = batch["phase_feature"]
        if (full_observation_gru_enabled(self.args)
                or full_observation_transformer_enabled(self.args)):
            chunk_kwargs["full_history_observations"] = (
                batch["full_history_observations"])
        if multi_candidate_chunk_enabled(self.args):
            return self._multi_candidate_chunk_training_step(
                batch, chunk_kwargs)
        prediction = self.model.forward_action_chunk(
            batch["obs"], batch["task_onehot"],
            batch["history_features"], **chunk_kwargs)
        teacher_main = self.cal_loss(
            prediction[:, 0], batch["teacher_action_chunk"][:, 0])
        demo_main = self.cal_loss(
            prediction[:, 0], batch["demo_action_chunk"][:, 0])
        teacher_future = self._masked_future_loss(
            prediction, batch["teacher_action_chunk"],
            batch["action_chunk_mask"])
        demo_future = self._masked_future_loss(
            prediction, batch["demo_action_chunk"],
            batch["action_chunk_mask"])
        teacher_loss = (
            teacher_main["loss"]
            + self.action_chunk_auxiliary_weight * teacher_future)
        demo_loss = (
            demo_main["loss"]
            + self.action_chunk_auxiliary_weight * demo_future)
        loss = (
            self.teacher_weight * teacher_loss
            + self.demo_weight * demo_loss)
        geometry_loss = self._keypoint_geometry_loss(
            prediction, batch["demo_action_chunk"],
            batch["action_chunk_mask"])
        geometry_weight = float(self.args.get(
            "keypoint_geometry_loss", {}).get("weight", 0.0))
        loss = loss + geometry_weight * geometry_loss
        self.log_dict({
            "train_loss": loss,
            "keypoint_geometry_loss_m": geometry_loss,
            "teacher_loss": teacher_main["loss"],
            "demo_loss": demo_main["loss"],
            "teacher_future_loss": teacher_future,
            "demo_future_loss": demo_future,
            "teacher_wrist_loss": teacher_main["wrist_loss"],
            "teacher_ori_loss": teacher_main["ori_loss"],
            "teacher_finger_loss": teacher_main["finger_loss"],
        }, prog_bar=True, on_epoch=True)
        return loss

    def _multi_candidate_chunk_training_step(self, batch, chunk_kwargs):
        candidates, gate_logits = self.model.forward_action_candidates(
            batch["obs"], batch["task_onehot"],
            batch["history_features"], **chunk_kwargs)
        mask = batch["action_chunk_mask"][:, None, :, None].to(
            candidates.dtype)

        def per_candidate(target):
            expanded_target = target[:, None].expand_as(candidates)
            error = F.smooth_l1_loss(
                candidates, expanded_target, reduction="none")
            return (error * mask).sum(dim=(2, 3)) / (
                mask.sum(dim=(2, 3)) * candidates.shape[-1]).clamp_min(1.0)

        combined = (
            self.teacher_weight * per_candidate(batch["teacher_action_chunk"])
            + self.demo_weight * per_candidate(batch["demo_action_chunk"]))
        winner = combined.detach().argmin(dim=1)
        regression = combined.gather(1, winner[:, None]).mean()
        gate = F.cross_entropy(gate_logits, winner)
        gate_weight = float(
            self.args.multi_candidate_action_chunk.get(
                "gate_loss_weight", 0.1))
        loss = regression + gate_weight * gate
        self.log_dict({
            "train_loss": loss,
            "candidate_regression_loss": regression,
            "candidate_gate_loss": gate,
        }, prog_bar=True, on_epoch=True)
        return loss


class BCTrainer:
    def __init__(self, args,env_args,  train_loader=None,test_loader=None,
                 init_checkpoint=None):
        self.args = args
        self.env_args = env_args
        self.train_loader = train_loader
        self.test_loader = test_loader

        if args.get('distillation', {}).get('enabled', False):
            self.bc_model = DistillationBCModel(args, env_args.env)
        else:
            self.bc_model = LitBCModel(args, env_args.env)
        if init_checkpoint is not None:
            checkpoint = torch.load(init_checkpoint, map_location='cpu')
            state_dict = checkpoint.get('state_dict', checkpoint)
            expanded = False
            if task_conditioning_enabled(args):
                state_dict, expanded = expand_standard_state_dict_for_task_model(
                    self.bc_model, state_dict)
            self.bc_model.load_state_dict(state_dict, strict=True)
            print(
                'Initialized model weights from {} (epoch={}, optimizer state '
                'not loaded, task_input_expanded={})'
                .format(
                    init_checkpoint, checkpoint.get('epoch', 'unknown'),
                    expanded))
        if relative_action_chunk_enabled(args) and bool(
                args.relative_action_chunk.get("reset_correction_heads", True)):
            if init_checkpoint is None:
                raise ValueError(
                    "Resetting relative-action heads requires --init-checkpoint")
            self.bc_model.model.reset_correction_heads()
            print(
                "Reset current/future action heads; training corrections "
                "relative to the current normalized joint state")
        self.frozen_mode_modules = []
        if action_diffusion_enabled(args) and bool(
                args.action_diffusion.get("freeze_chunk_base", True)):
            self.frozen_mode_modules.extend(
                self.bc_model.model.freeze_chunk_base())
            print("Frozen deterministic Chunk8 base; training diffusion denoiser only")
        if multi_candidate_chunk_enabled(args) and bool(
                args.multi_candidate_action_chunk.get(
                    "freeze_chunk_base", True)):
            self.frozen_mode_modules.extend(
                self.bc_model.model.freeze_chunk_base())
            print(
                "Frozen deterministic Chunk8 base; training candidates and gate only")
        if wrist_residual_chunk_enabled(args) and bool(
                args.wrist_residual_chunk.get("freeze_chunk_base", True)):
            if init_checkpoint is None:
                raise ValueError(
                    "Freezing the Chunk8 base requires --init-checkpoint")
            self.frozen_mode_modules.extend(
                self.bc_model.model.freeze_chunk_base())
            print("Frozen Chunk8 base; training wrist residual only")
        if full_observation_gru_freeze_base(args):
            if init_checkpoint is None:
                raise ValueError(
                    "Freezing the Chunk8 base requires --init-checkpoint")
            self.frozen_mode_modules.extend(
                self.bc_model.model.freeze_chunk_base())
            print("Frozen Chunk8 base; training full-observation GRU only")
        if full_observation_transformer_freeze_base(args):
            if init_checkpoint is None:
                raise ValueError(
                    "Freezing the Chunk8 base requires --init-checkpoint")
            self.frozen_mode_modules.extend(
                self.bc_model.model.freeze_chunk_base())
            print("Frozen Chunk8 base; training full-observation Transformer only")
        if temporal_attention_freeze_base(args):
            if init_checkpoint is None:
                raise ValueError(
                    'Freezing the Temporal3 base requires --init-checkpoint')
            freeze_method = getattr(
                self.bc_model.model, 'freeze_temporal_base', None)
            if freeze_method is None:
                raise TypeError(
                    'Configured temporal attention model cannot freeze its base')
            self.frozen_mode_modules.extend(freeze_method())
            trainable = sum(
                p.numel() for p in self.bc_model.parameters()
                if p.requires_grad)
            frozen = sum(
                p.numel() for p in self.bc_model.parameters()
                if not p.requires_grad)
            if trainable == 0 or frozen == 0:
                raise RuntimeError(
                    'Invalid attention residual freeze: trainable={} '
                    'frozen={}'.format(trainable, frozen))
            print(
                'Temporal3 base frozen: trainable_attention_parameters={} '
                'frozen_base_parameters={}'.format(trainable, frozen))
        if args.get('freeze_feature_encoder', False):
            if temporal_attention_freeze_base(args):
                raise ValueError(
                    'freeze_feature_encoder is redundant when the entire '
                    'Temporal3 base is frozen')
            if init_checkpoint is None:
                raise ValueError(
                    'freeze_feature_encoder requires --init-checkpoint; freezing a '
                    'randomly initialized encoder is not meaningful')
            self.frozen_mode_modules.extend(
                freeze_feature_encoder(self.bc_model))

    def train(self, ckpt_path=None):

        callback = ModelCheckpoint(
            dirpath=self.args.exp_dir,
            filename='{epoch:03d}-{step}',
            save_top_k=-1,
            save_last=True,
            every_n_epochs=self.args.get('checkpoint_every_n_epochs', 1),
        )
        lr_monitor = LearningRateMonitor(logging_interval='step')
        callbacks = [callback, lr_monitor]
        if self.frozen_mode_modules:
            callbacks.append(FrozenEncoderModeCallback(
                self.frozen_mode_modules))
        trainer_kwargs = dict(
            accelerator='gpu', devices=1, precision=32, max_epochs=self.args.num_epochs,
            callbacks=callbacks, log_every_n_steps=5,
            check_val_every_n_epoch=self.args.get('check_val_every_n_epoch', 1),
            default_root_dir=os.path.join(self.args.exp_dir, "tensorboard_logs"))
        if self.args.get('limit_train_batches') is not None:
            trainer_kwargs['limit_train_batches'] = int(self.args.limit_train_batches)
        if self.args.get('limit_val_batches') is not None:
            trainer_kwargs['limit_val_batches'] = int(self.args.limit_val_batches)
        trainer = pl.Trainer(**trainer_kwargs)

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        trainer.fit(model=self.bc_model, train_dataloaders=self.train_loader,
                    ckpt_path=ckpt_path, val_dataloaders=self.test_loader)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        free_bytes, total_bytes = torch.cuda.mem_get_info(0)
        resource_summary = {
            'elapsed_seconds': float(elapsed),
            'global_step': int(trainer.global_step),
            'peak_allocated_mib': float(torch.cuda.max_memory_allocated() / (1024 ** 2)),
            'peak_reserved_mib': float(torch.cuda.max_memory_reserved() / (1024 ** 2)),
            'free_vram_after_mib': float(free_bytes / (1024 ** 2)),
            'total_vram_mib': float(total_bytes / (1024 ** 2)),
        }
        from omegaconf import OmegaConf
        OmegaConf.save(
            OmegaConf.create(resource_summary),
            os.path.join(self.args.exp_dir, 'resource_summary.yaml'))
        print(
            'Training resource summary: elapsed={:.2f}s, peak_allocated={:.0f} MiB, '
            'peak_reserved={:.0f} MiB'.format(
                resource_summary['elapsed_seconds'],
                resource_summary['peak_allocated_mib'],
                resource_summary['peak_reserved_mib']))


def main(args, env_args, resume_checkpoint=None, init_checkpoint=None,
         min_free_vram_mb=5000):

    require_free_vram(min_free_vram_mb)

    seed = int(args.get('seed', 0))
    pl.seed_everything(0 if seed < 0 else seed, workers=True)

    kstr = 'sim_action' if args.use_sim_action else 'vis_action'

    default_run_name = '1obj_seq2000_DexRep_pro100_start_uniform_{}_dsam_mod'.format(kstr)
    args.task_name = args.get('run_name', default_run_name)
    args.policy.actor_critic = 'ActorCriticDexRep'
    env_args.env.obs_dim.pop('pnG')

    # Keep historical defaults, but allow independent custom configs to turn
    # observation augmentation on for controlled experiments.
    if 'add_noise' not in args:
        args.add_noise = False
    if 'noise_val' not in args:
        args.noise_val = 0.02

    args.exp_dir = os.path.abspath(os.path.join(args.exp_dir, args.task_name))
    existing_checkpoints = list(pathlib.Path(args.exp_dir).glob('*.ckpt'))
    if existing_checkpoints and resume_checkpoint is None:
        raise FileExistsError(
            'Run directory already contains checkpoints: {}. '
            'Choose a new --run-name or pass --resume-checkpoint.'.format(args.exp_dir)
        )
    if resume_checkpoint is not None:
        resume_checkpoint = os.path.abspath(os.path.expanduser(resume_checkpoint))
        if not os.path.isfile(resume_checkpoint):
            raise FileNotFoundError('Resume checkpoint not found: {}'.format(resume_checkpoint))
    if init_checkpoint is not None and not os.path.isfile(init_checkpoint):
        raise FileNotFoundError('Initialization checkpoint not found: {}'.format(init_checkpoint))

    os.makedirs(args.exp_dir, exist_ok=True)

    from omegaconf import OmegaConf
    OmegaConf.save(args, os.path.join(args.exp_dir, 'resolved_config.yaml'))
    OmegaConf.save(env_args, os.path.join(args.exp_dir, 'resolved_env_config.yaml'))
    metadata = OmegaConf.create({
        'started_at': datetime.now().isoformat(timespec='seconds'),
        'hostname': socket.gethostname(),
        'python': sys.version.split()[0],
        'command': ' '.join(sys.argv),
        'resume_checkpoint': resume_checkpoint,
        'init_checkpoint': init_checkpoint,
        'init_checkpoint_sha256': (
            checkpoint_sha256(init_checkpoint) if init_checkpoint is not None else None),
    })
    OmegaConf.save(metadata, os.path.join(args.exp_dir, 'run_metadata.yaml'))

    ds_train = GraspM3DexRepDataset(args, ds_name='train')
    ds_test = GraspM3DexRepDataset(args, ds_name='test')

    online_action_file = args.get('distillation', {}).get('online_action_file')
    if online_action_file:
        online_path = pathlib.Path(str(online_action_file)).expanduser()
        if not online_path.is_absolute():
            online_path = pathlib.Path.cwd() / online_path
        online_path = online_path.resolve()
        if not online_path.is_file():
            raise FileNotFoundError('Online aggregation file: {}'.format(online_path))
        ds_train = OnlineAugmentedDataset(ds_train, str(online_path))
    if temporal_history_enabled(args):
        ds_train = TemporalHistoryDataset(ds_train, args)
        ds_test = TemporalHistoryDataset(ds_test, args)
    sample_category = args.get('sample_category')
    if sample_category:
        ds_train = CategorySubset(ds_train, str(sample_category))
        ds_test = CategorySubset(ds_test, str(sample_category))

    sampler = None
    category_balanced = args.get('category_balanced_sampling', False)
    object_balanced = args.get('object_balanced_sampling', False)
    if category_balanced and object_balanced:
        raise ValueError(
            'category_balanced_sampling and object_balanced_sampling are '
            'mutually exclusive')
    if category_balanced:
        sampler = build_category_balanced_sampler(
            ds_train, seed, args.get('online_sample_fraction'))
    elif object_balanced:
        sampler = build_object_balanced_sampler(ds_train, seed)
    train_loader = DataLoader(
        ds_train, batch_size=args.batch_size, shuffle=sampler is None,
        sampler=sampler, drop_last=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False, drop_last=True,num_workers=4, pin_memory=True)

    trainer = BCTrainer(
        args, env_args, train_loader, test_loader,
        init_checkpoint=init_checkpoint)
    trainer.train(ckpt_path=resume_checkpoint)


def parse_cli():
    parser = argparse.ArgumentParser(description='Train the DexRep behavior-cloning baseline.')
    parser.add_argument('--config', default=str(REPO_ROOT / 'ActionDiffusion/bc/config/lhm_bc.yaml'))
    parser.add_argument('--env-config', default=str(DEXGRASP_ROOT / 'cfg/shadow_hand_grasp_dexrep_ijrr.yaml'))
    parser.add_argument('--run-name', default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--num-epochs', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--learning-rate', type=float, default=None)
    parser.add_argument('--teacher-weight', type=float, default=None)
    parser.add_argument(
        '--teacher-action-file', default=None,
        help='Override the distillation label file for a custom run.')
    parser.add_argument('--online-sample-fraction', type=float, default=None)
    parser.add_argument('--noise-value', type=float, default=None)
    parser.add_argument('--seq-num', type=int, default=None)
    parser.add_argument('--val-seq-num', type=int, default=None)
    parser.add_argument('--action-chunk-horizon', type=int, default=None)
    parser.add_argument(
        '--train-category', choices=('bottle', 'mug', 'bowl', 'camera'),
        default=None,
        help='Train and validate only on one category from the frozen manifest.')
    parser.add_argument(
        '--sample-category', choices=('bottle', 'mug', 'bowl', 'camera'),
        default=None,
        help=('Filter the fully aligned offline/teacher/online dataset to one '
              'category; use this for category-specialist fine-tuning.'))
    parser.add_argument(
        '--category-train-size', type=int, choices=(4, 10, 20), default=None,
        help='Use a nested per-category object count from the category manifest.')
    parser.add_argument('--category-manifest', default=str(
        REPO_ROOT / 'custom_tools/configs/object_split_final.json'))
    parser.add_argument('--resume-checkpoint', default=None)
    parser.add_argument(
        '--init-checkpoint', default=None,
        help='Load model weights only and start a new run at epoch 0.')
    parser.add_argument(
        '--freeze-feature-encoder', action='store_true',
        help='Keep the loaded DexRep encoder and BatchNorm statistics fixed; train the actor head.')
    parser.add_argument(
        '--object-balanced-sampling', action='store_true',
        help='Sample each selected training object with equal expected probability.')
    parser.add_argument(
        '--min-free-vram-mb', type=int, default=5000,
        help='Abort before training if less VRAM is free (default: 5000 MiB).')
    parser.add_argument('--print-config', action='store_true')
    return parser.parse_args()

if __name__ == "__main__":
    from omegaconf import OmegaConf

    cli = parse_cli()
    if cli.resume_checkpoint is not None and cli.init_checkpoint is not None:
        raise ValueError('--resume-checkpoint and --init-checkpoint are mutually exclusive')
    resume_checkpoint = (
        str(Path(cli.resume_checkpoint).expanduser().resolve())
        if cli.resume_checkpoint is not None else None)
    init_checkpoint = (
        str(Path(cli.init_checkpoint).expanduser().resolve())
        if cli.init_checkpoint is not None else None)
    args = OmegaConf.load(str(Path(cli.config).expanduser().resolve()))
    env_args = OmegaConf.load(str(Path(cli.env_config).expanduser().resolve()))
    for cli_name, config_name in (
        ('run_name', 'run_name'),
        ('seed', 'seed'),
        ('num_epochs', 'num_epochs'),
        ('batch_size', 'batch_size'),
        ('learning_rate', 'lr'),
        ('online_sample_fraction', 'online_sample_fraction'),
        ('noise_value', 'noise_val'),
        ('seq_num', 'seq_num'),
        ('val_seq_num', 'val_seq_num'),
    ):
        value = getattr(cli, cli_name)
        if value is not None:
            OmegaConf.update(args, config_name, value)

    if cli.teacher_weight is not None:
        if not 0.0 <= cli.teacher_weight <= 1.0:
            raise ValueError('--teacher-weight must be in [0, 1]')
        OmegaConf.update(args, 'distillation.teacher_weight', cli.teacher_weight)
        OmegaConf.update(args, 'distillation.demo_weight', 1.0 - cli.teacher_weight)
    if cli.action_chunk_horizon is not None:
        if cli.action_chunk_horizon < 2:
            raise ValueError('--action-chunk-horizon must be at least two')
        OmegaConf.update(
            args, 'action_chunk_aux.horizon', cli.action_chunk_horizon)
    if cli.teacher_action_file is not None:
        teacher_action_file = str(
            Path(cli.teacher_action_file).expanduser().resolve())
        OmegaConf.update(
            args, 'distillation.teacher_action_file',
            teacher_action_file)

    if cli.freeze_feature_encoder:
        OmegaConf.update(args, 'freeze_feature_encoder', True)
    if cli.object_balanced_sampling:
        OmegaConf.update(args, 'object_balanced_sampling', True)
    if cli.sample_category is not None:
        OmegaConf.update(args, 'sample_category', cli.sample_category)

    if cli.train_category is not None:
        manifest_path = Path(cli.category_manifest).expanduser().resolve()
        with manifest_path.open('r', encoding='utf-8') as handle:
            manifest = json.load(handle)
        category_split = manifest['categories'][cli.train_category]
        if cli.category_train_size is not None:
            nested = category_split.get('train_nested', {})
            if str(cli.category_train_size) not in nested:
                raise ValueError(
                    'Manifest has no nested category size {}'.format(
                        cli.category_train_size))
            object_ids = nested[str(cli.category_train_size)]
        else:
            object_ids = category_split['train']
        OmegaConf.update(args, 'train_obj_code_list', object_ids)
        OmegaConf.update(args, 'val_obj_code_list', object_ids)
        OmegaConf.update(args, 'expert_category', cli.train_category)
        OmegaConf.update(args, 'expert_category_train_size', len(object_ids))
        OmegaConf.update(args, 'category_manifest', str(manifest_path))

    if cli.print_config:
        print(OmegaConf.to_yaml(args, resolve=False))
    else:
        original_cwd = Path.cwd()
        try:
            os.chdir(str(DEXGRASP_ROOT))
            main(
                args,
                env_args,
                resume_checkpoint=resume_checkpoint,
                init_checkpoint=init_checkpoint,
                min_free_vram_mb=cli.min_free_vram_mb,
            )
        finally:
            os.chdir(str(original_cwd))
