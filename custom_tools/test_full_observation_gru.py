"""CPU checks for the full-observation Temporal3 GRU extension."""

from pathlib import Path
from types import SimpleNamespace

import isaacgym  # Isaac Gym must precede torch.  # noqa: F401
import numpy as np
import torch
from omegaconf import OmegaConf

from ActionDiffusion.bc.model.policy.lhm_policy import LitBCModel
from custom_tools.task_conditioning import (
    enable_task_conditioning,
    expand_standard_state_dict_for_task_model,
)
from custom_tools.train_bc import TemporalHistoryDataset


ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = (
    ROOT / "custom_tools/configs/"
    / "unified_student_taskid_temporal3_v1.yaml")
GRU_CONFIG = (
    ROOT / "custom_tools/configs/"
    / "unified_student_taskid_temporal3_fullobs_gru_v1.yaml")
ENV_CONFIG = (
    ROOT / "dexgrasp/cfg/shadow_hand_grasp_dexrep_ijrr.yaml")
CHECKPOINT = (
    ROOT / "custom_tools/runs/bc/"
    / "unified_student_taskid_temporal3_seed2025_e4_v1/"
    / "epoch=003-step=5152.ckpt")


class Config(dict):
    __getattr__ = dict.__getitem__


class OfflineDataset:
    def __init__(self):
        self.args = SimpleNamespace(add_noise=False, noise_val=0.0)
        self.ds_name = "train"
        self.is_flat = True
        self.pro_dim = 100
        self.num_frame = 5
        observations = np.arange(
            10 * 2460, dtype=np.float32).reshape(10, 2460)
        actions = np.arange(
            10 * 28, dtype=np.float32).reshape(10, 28)
        self.teacher_actions = actions.copy()
        self.data = {
            "obs": observations,
            "vis_unscale_actions": actions,
        }
        self.sample_categories = np.asarray(["bottle"] * 10)
        self.sample_sources = np.zeros(10, dtype=np.int8)

    def __len__(self):
        return len(self.teacher_actions)

    def __getitem__(self, index):
        return {
            "obs": self.data["obs"][index].copy(),
            "actions": self.data["vis_unscale_actions"][index].copy(),
            "teacher_actions": self.teacher_actions[index].copy(),
            "task_onehot": np.asarray([1, 0, 0, 0], dtype=np.float32),
        }


def make_configs():
    policy = Config(
        actor_critic="ActorCriticDexRep",
        pi_hid_sizes=[32, 32],
        vf_hid_sizes=[32, 32],
        activation="elu",
        actions_shape=28)
    encoder = Config(emb_dim=8, n_obs_steps=1, bn_type="part")
    learn = Config(init_noise_std=0.8)
    task = Config(
        enabled=True,
        categories=["bottle", "mug", "bowl", "camera"])
    temporal = Config(
        enabled=True, history_frames=3, prop_dim=100, action_dim=28)
    source = Config(
        policy=policy, encoder=encoder, learn=learn,
        task_conditioning=task, temporal_history=temporal, lr=5e-5)
    target = Config(source)
    target["full_observation_gru"] = Config(
        enabled=True, observation_dim=2460, hidden_dim=16)
    env = Config(obs_dim=Config(
        prop=100, dexrep_sensor=1280, dexrep_pnl=1080))
    return source, target, env


def test_exact_temporal3_warm_start():
    torch.manual_seed(29)
    source_args, target_args, env = make_configs()
    source = LitBCModel(source_args, env)
    enable_task_conditioning(source, source_args, env)
    target = LitBCModel(target_args, env)
    enable_task_conditioning(target, target_args, env)
    expanded, changed = expand_standard_state_dict_for_task_model(
        target, source.state_dict())
    assert changed
    target.load_state_dict(expanded, strict=True)

    observations = torch.randn(3, 2460)
    task_ids = torch.eye(4)[:3]
    history = torch.randn(3, 2 * (100 + 28))
    full_history = torch.randn(3, 2, 2460)
    source.eval()
    target.eval()
    with torch.no_grad():
        source_action = source.model(
            observations, task_ids, history)
        target_action = target.model(
            observations, task_ids, history, full_history)
    torch.testing.assert_close(target_action, source_action)


def test_offline_full_history_boundaries():
    _, args, _ = make_configs()
    dataset = TemporalHistoryDataset(OfflineDataset(), args)
    first = dataset[0]["full_history_observations"]
    third = dataset[2]["full_history_observations"]
    next_sequence = dataset[5]["full_history_observations"]
    np.testing.assert_array_equal(
        first[0], dataset.offline.data["obs"][0])
    np.testing.assert_array_equal(
        first[1], dataset.offline.data["obs"][0])
    np.testing.assert_array_equal(
        third[0], dataset.offline.data["obs"][0])
    np.testing.assert_array_equal(
        third[1], dataset.offline.data["obs"][1])
    np.testing.assert_array_equal(
        next_sequence[0], dataset.offline.data["obs"][5])
    np.testing.assert_array_equal(
        next_sequence[1], dataset.offline.data["obs"][5])


def test_inference_history_and_partial_reset():
    _, target_args, env = make_configs()
    model = LitBCModel(target_args, env)
    enable_task_conditioning(model, target_args, env)
    model.eval()
    tasks = torch.eye(4)[:2]
    observations = torch.randn(2, 2460)
    with torch.no_grad():
        first = model.model.act_inference(observations, tasks)
        second = model.model.act_inference(observations + 1.0, tasks)
        reset = model.model.act_inference_for_reset(
            observations, torch.tensor([1]), tasks)
    assert tuple(first.shape) == (2, 28)
    assert tuple(second.shape) == (2, 28)
    assert tuple(reset.shape) == (1, 28)
    torch.testing.assert_close(
        model.model._inference_full_observation_history[1, -1],
        observations[1])


def test_real_checkpoint_and_config():
    required = (BASE_CONFIG, GRU_CONFIG, ENV_CONFIG, CHECKPOINT)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    source_args = OmegaConf.load(str(BASE_CONFIG))
    target_args = OmegaConf.load(str(GRU_CONFIG))
    env_args = OmegaConf.load(str(ENV_CONFIG))
    env_args.env.obs_dim.pop("pnG")
    checkpoint = torch.load(str(CHECKPOINT), map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    source = LitBCModel(source_args, env_args.env)
    enable_task_conditioning(source, source_args, env_args.env)
    source.load_state_dict(state_dict, strict=True)
    target = LitBCModel(target_args, env_args.env)
    enable_task_conditioning(target, target_args, env_args.env)
    expanded, changed = expand_standard_state_dict_for_task_model(
        target, state_dict)
    assert changed
    target.load_state_dict(expanded, strict=True)
    source.eval()
    target.eval()
    observations = torch.randn(2, 2460)
    tasks = torch.eye(4)[:2]
    history = torch.randn(2, 2 * (100 + 28))
    full_history = torch.randn(2, 2, 2460)
    with torch.no_grad():
        source_action = source.model(observations, tasks, history)
        target_action = target.model(
            observations, tasks, history, full_history)
    torch.testing.assert_close(target_action, source_action)


def main():
    test_exact_temporal3_warm_start()
    test_offline_full_history_boundaries()
    test_inference_history_and_partial_reset()
    test_real_checkpoint_and_config()
    print("FULL_OBSERVATION_GRU_TEST=PASS")


if __name__ == "__main__":
    main()
