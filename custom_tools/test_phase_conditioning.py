"""CPU checks for explicit rollout-progress conditioning."""

from types import SimpleNamespace

import isaacgym  # Isaac Gym must precede torch.  # noqa: F401
import numpy as np
import torch

from ActionDiffusion.bc.model.policy.lhm_policy import LitBCModel
from custom_tools.task_conditioning import (
    enable_task_conditioning,
    expand_standard_state_dict_for_task_model,
)
from custom_tools.train_bc import TemporalHistoryDataset


class Config(dict):
    __getattr__ = dict.__getitem__


class OfflineDataset:
    def __init__(self):
        self.args = SimpleNamespace(add_noise=False, noise_val=0.0)
        self.ds_name = "train"
        self.is_flat = True
        self.pro_dim = 100
        self.num_frame = 5
        self.teacher_actions = np.zeros((10, 28), dtype=np.float32)
        self.data = {
            "obs": np.zeros((10, 2460), dtype=np.float32),
            "vis_unscale_actions": np.zeros(
                (10, 28), dtype=np.float32),
        }
        self.sample_categories = np.asarray(["bottle"] * 10)
        self.sample_sources = np.zeros(10, dtype=np.int8)

    def __len__(self):
        return 10

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
    target["phase_conditioning"] = Config(
        enabled=True, max_frame_index=4)
    env = Config(obs_dim=Config(
        prop=100, dexrep_sensor=1280, dexrep_pnl=1080))
    return source, target, env


def test_exact_warm_start():
    torch.manual_seed(31)
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
    tasks = torch.eye(4)[:3]
    history = torch.randn(3, 256)
    phase = torch.tensor([[-1.0], [0.0], [1.0]])
    source.eval()
    target.eval()
    with torch.no_grad():
        source_action = source.model(observations, tasks, history)
        target_action = target.model(
            observations, tasks, history, phase)
    torch.testing.assert_close(target_action, source_action)


def test_dataset_phase_boundaries():
    _, args, _ = make_configs()
    dataset = TemporalHistoryDataset(OfflineDataset(), args)
    assert dataset[0]["phase_feature"].tolist() == [-1.0]
    assert dataset[2]["phase_feature"].tolist() == [0.0]
    assert dataset[4]["phase_feature"].tolist() == [1.0]
    assert dataset[5]["phase_feature"].tolist() == [-1.0]


def test_inference_counter_and_partial_reset():
    _, args, env = make_configs()
    model = LitBCModel(args, env)
    enable_task_conditioning(model, args, env)
    model.eval()
    observations = torch.randn(2, 2460)
    tasks = torch.eye(4)[:2]
    with torch.no_grad():
        model.model.act_inference(observations, tasks)
        model.model.act_inference(observations, tasks)
        assert model.model._inference_phase_steps.tolist() == [2, 2]
        reset = model.model.act_inference_for_reset(
            observations, torch.tensor([1]), tasks)
    assert tuple(reset.shape) == (1, 28)
    assert model.model._inference_phase_steps.tolist() == [2, 1]


def main():
    test_exact_warm_start()
    test_dataset_phase_boundaries()
    test_inference_counter_and_partial_reset()
    print("PHASE_CONDITIONING_TEST=PASS")


if __name__ == "__main__":
    main()
