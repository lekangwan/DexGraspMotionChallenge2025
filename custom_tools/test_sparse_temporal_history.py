"""CPU checks for sparse three-frame temporal history."""

from types import MethodType, SimpleNamespace

import isaacgym  # Isaac Gym must precede torch.  # noqa: F401
import numpy as np
import torch

from ActionDiffusion.bc.model.policy.lhm_policy import LitBCModel
from custom_tools.task_conditioning import (
    enable_task_conditioning,
    temporal_history_lags,
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
        self.num_frame = 10
        self.teacher_actions = None
        observations = np.zeros((10, 2460), dtype=np.float32)
        actions = np.zeros((10, 28), dtype=np.float32)
        for frame in range(10):
            observations[frame, :100] = frame
            actions[frame] = 100 + frame
        self.data = {
            "obs": observations,
            "vis_unscale_actions": actions,
        }

    def __len__(self):
        return 10

    def __getitem__(self, index):
        return {
            "obs": self.data["obs"][index].copy(),
            "actions": self.data["vis_unscale_actions"][index].copy(),
            "task_onehot": np.asarray([1, 0, 0, 0], dtype=np.float32),
        }


def make_configs():
    args = Config(
        policy=Config(
            actor_critic="ActorCriticDexRep",
            pi_hid_sizes=[32, 32],
            vf_hid_sizes=[32, 32],
            activation="elu",
            actions_shape=28),
        encoder=Config(emb_dim=8, n_obs_steps=1, bn_type="part"),
        learn=Config(init_noise_std=0.8),
        task_conditioning=Config(
            enabled=True,
            categories=["bottle", "mug", "bowl", "camera"]),
        temporal_history=Config(
            enabled=True,
            history_frames=3,
            history_lags=[6, 3],
            prop_dim=100,
            action_dim=28),
        lr=5e-5)
    env = Config(obs_dim=Config(
        prop=100, dexrep_sensor=1280, dexrep_pnl=1080))
    return args, env


def test_offline_sparse_history():
    args, _ = make_configs()
    assert temporal_history_lags(args) == (6, 3)
    dataset = TemporalHistoryDataset(OfflineDataset(), args)
    history = dataset[6]["history_features"]
    props = history[:200].reshape(2, 100)
    actions = history[200:].reshape(2, 28)
    np.testing.assert_array_equal(props[:, 0], [0, 3])
    np.testing.assert_array_equal(actions[:, 0], [100, 103])


def test_offline_multiscale_history():
    args, _ = make_configs()
    args.temporal_history["history_frames"] = 5
    args.temporal_history["history_lags"] = [6, 3, 2, 1]
    assert temporal_history_lags(args) == (6, 3, 2, 1)
    dataset = TemporalHistoryDataset(OfflineDataset(), args)
    history = dataset[6]["history_features"]
    props = history[:400].reshape(4, 100)
    actions = history[400:].reshape(4, 28)
    np.testing.assert_array_equal(props[:, 0], [0, 3, 4, 5])
    np.testing.assert_array_equal(actions[:, 0], [100, 103, 104, 105])


def test_inference_sparse_history():
    args, env = make_configs()
    model = LitBCModel(args, env)
    enable_task_conditioning(model, args, env)
    captured = []

    def capture_forward(
            self, observations, task_onehot=None, history_features=None):
        captured.append(history_features.detach().clone())
        return torch.zeros(
            observations.shape[0], 28, dtype=observations.dtype)

    model.model.forward = MethodType(capture_forward, model.model)
    task = torch.as_tensor([[1, 0, 0, 0]], dtype=torch.float32)
    for frame in range(7):
        observation = torch.zeros(1, 2460)
        observation[:, :100] = frame
        model.model.act_inference(observation, task)
    history = captured[-1][0]
    props = history[:200].reshape(2, 100)
    actions = history[200:].reshape(2, 28)
    torch.testing.assert_close(
        props[:, 0], torch.as_tensor([0.0, 3.0]))
    torch.testing.assert_close(actions, torch.zeros_like(actions))


def test_partial_inference_history_reset():
    args, env = make_configs()
    model = LitBCModel(args, env)
    enable_task_conditioning(model, args, env)
    model.eval()
    task = torch.eye(4)[:3]
    for frame in range(3):
        observation = torch.zeros(3, 2460)
        observation[:, :100] = torch.as_tensor(
            [frame, frame + 1, frame + 2])[:, None]
        model.model.act_inference(observation, task)
    old_props = model.model._inference_prop_history.clone()
    old_actions = model.model._inference_action_history.clone()

    reset_observation = observation.clone()
    reset_observation[1, :100] = 10.0
    reset_action = model.model.act_inference_for_reset(
        reset_observation, torch.as_tensor([1]), task)
    torch.testing.assert_close(
        model.model._inference_prop_history[[0, 2]],
        old_props[[0, 2]])
    torch.testing.assert_close(
        model.model._inference_action_history[[0, 2]],
        old_actions[[0, 2]])
    torch.testing.assert_close(
        model.model._inference_prop_history[1],
        torch.full_like(model.model._inference_prop_history[1], 10.0))
    torch.testing.assert_close(
        model.model._inference_action_history[1, :-1],
        torch.zeros_like(model.model._inference_action_history[1, :-1]))
    torch.testing.assert_close(
        model.model._inference_action_history[1, -1],
        reset_action[0])


def main():
    test_offline_sparse_history()
    test_offline_multiscale_history()
    test_inference_sparse_history()
    test_partial_inference_history_reset()
    print("SPARSE_TEMPORAL_HISTORY_TEST=PASS")


if __name__ == "__main__":
    main()
