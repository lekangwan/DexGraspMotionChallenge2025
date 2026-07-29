"""CPU checks for explicit Task-ID conditioning and Soup expansion."""

from types import SimpleNamespace

import torch

from ActionDiffusion.bc.model.policy.lhm_policy import LitBCModel
from custom_tools.task_conditioning import (
    ACTOR_INPUT_WEIGHT_KEY,
    TaskConditionedActorCriticDexRep,
    checkpoint_uses_task_conditioning,
    enable_task_conditioning,
    expand_standard_state_dict_for_task_model,
)


class Config(dict):
    __getattr__ = dict.__getitem__


def make_configs():
    args = Config(
        policy=Config(
            actor_critic="ActorCriticDexRep",
            pi_hid_sizes=[32, 32],
            vf_hid_sizes=[32, 32],
            activation="elu",
            actions_shape=28),
        encoder=Config(
            emb_dim=8,
            n_obs_steps=1,
            bn_type="part"),
        learn=Config(init_noise_std=0.8),
        task_conditioning=Config(
            enabled=True,
            categories=["bottle", "mug", "bowl", "camera"]),
        lr=5e-5)
    env = Config(obs_dim=Config(
        prop=100,
        dexrep_sensor=1280,
        dexrep_pnl=1080))
    return args, env


def main():
    torch.manual_seed(7)
    args, env = make_configs()
    standard = LitBCModel(args, env)
    standard_state = standard.state_dict()

    task_model = LitBCModel(args, env)
    enable_task_conditioning(task_model, args, env)
    assert isinstance(task_model.model, TaskConditionedActorCriticDexRep)
    assert task_model.model.actor[0].in_features == 28

    expanded, changed = expand_standard_state_dict_for_task_model(
        task_model, standard_state)
    assert changed
    assert expanded[ACTOR_INPUT_WEIGHT_KEY].shape[1] == 28
    assert torch.count_nonzero(
        expanded[ACTOR_INPUT_WEIGHT_KEY][:, -4:]).item() == 0
    task_model.load_state_dict(expanded, strict=True)

    standard.eval()
    task_model.eval()
    observations = torch.randn(3, 2460)
    task_ids = torch.eye(4)[:3]
    with torch.no_grad():
        standard_actions = standard.model.act_inference(observations)
        task_actions = task_model.model.act_inference(
            observations, task_ids)
    torch.testing.assert_close(standard_actions, task_actions)

    assert not checkpoint_uses_task_conditioning(
        args, env, standard_state)
    assert checkpoint_uses_task_conditioning(
        args, env, task_model.state_dict())
    print("TASK_CONDITIONING_TEST=PASS")


if __name__ == "__main__":
    main()
