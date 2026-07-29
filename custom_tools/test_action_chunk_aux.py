"""CPU checks for the Temporal3 future-action auxiliary objective."""

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
        self.teacher_actions = np.arange(
            10 * 28, dtype=np.float32).reshape(10, 28)
        self.data = {
            "obs": np.zeros((10, 2460), dtype=np.float32),
            "vis_unscale_actions": self.teacher_actions + 1000.0,
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
    source = Config(
        policy=policy, encoder=encoder, learn=learn,
        task_conditioning=task, lr=5e-5)
    target = Config(source)
    target["temporal_history"] = Config(
        enabled=True, history_frames=3, prop_dim=100, action_dim=28)
    target["action_chunk_aux"] = Config(
        enabled=True, horizon=4, auxiliary_weight=0.25)
    env = Config(obs_dim=Config(
        prop=100, dexrep_sensor=1280, dexrep_pnl=1080))
    return source, target, env


def test_model_warm_start():
    torch.manual_seed(23)
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
    history = torch.zeros(3, 2 * (100 + 28))
    source.eval()
    target.eval()
    with torch.no_grad():
        source_action = source.model(observations, task_ids)
        chunk = target.model.forward_action_chunk(
            observations, task_ids, history)
    assert tuple(chunk.shape) == (3, 4, 28)
    torch.testing.assert_close(chunk[:, 0], source_action)
    torch.testing.assert_close(
        chunk[:, 1:], source_action[:, None].expand(-1, 3, -1))


def test_offline_boundary_mask():
    _, args, _ = make_configs()
    dataset = TemporalHistoryDataset(OfflineDataset(), args)
    first = dataset[0]
    penultimate = dataset[3]
    last = dataset[4]
    next_sequence = dataset[5]
    assert first["action_chunk_mask"].tolist() == [True] * 4
    assert penultimate["action_chunk_mask"].tolist() == [
        True, True, False, False]
    assert last["action_chunk_mask"].tolist() == [
        True, False, False, False]
    assert next_sequence["action_chunk_mask"].tolist() == [True] * 4
    np.testing.assert_array_equal(
        last["teacher_action_chunk"][0],
        dataset.offline.teacher_actions[4])
    assert not np.array_equal(
        last["teacher_action_chunk"][1],
        dataset.offline.teacher_actions[5])


def main():
    test_model_warm_start()
    test_offline_boundary_mask()
    print("ACTION_CHUNK_AUX_TEST=PASS")


if __name__ == "__main__":
    main()
