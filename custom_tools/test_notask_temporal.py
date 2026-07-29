"""CPU test that Temporal3 can operate without receiving a category ID."""

import isaacgym  # noqa: F401  Must precede torch.
import torch

from ActionDiffusion.bc.model.policy.lhm_policy import LitBCModel
from custom_tools.task_conditioning import (
    ACTOR_INPUT_WEIGHT_KEY,
    enable_task_conditioning,
    expand_standard_state_dict_for_task_model,
)


class Config(dict):
    __getattr__ = dict.__getitem__


def main():
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
            enabled=True, input_enabled=False,
            categories=["bottle", "mug", "bowl", "camera"]),
        temporal_history=Config(
            enabled=True, history_frames=3,
            prop_dim=100, action_dim=28),
        lr=2e-5)
    env = Config(obs_dim=Config(
        prop=100, dexrep_sensor=1280, dexrep_pnl=1080))

    torch.manual_seed(11)
    standard = LitBCModel(args, env)
    temporal = LitBCModel(args, env)
    enable_task_conditioning(temporal, args, env)
    assert temporal.model.task_dim == 0
    assert temporal.model.actor[0].in_features == 24 + 2 * (100 + 28)

    expanded, changed = expand_standard_state_dict_for_task_model(
        temporal, standard.state_dict())
    assert changed
    temporal.load_state_dict(expanded, strict=True)
    standard.eval()
    temporal.eval()
    weight = temporal.state_dict()[ACTOR_INPUT_WEIGHT_KEY]
    assert torch.count_nonzero(weight[:, 24:]).item() == 0

    observations = torch.randn(3, 2460)
    history = torch.zeros(3, 256)
    task_a = torch.eye(4)[:3]
    task_b = torch.flip(task_a, dims=(1,))
    with torch.no_grad():
        standard_action = standard.model.act_inference(observations)
        action_a = temporal.model(observations, task_a, history)
        action_b = temporal.model(observations, task_b, history)
    torch.testing.assert_close(action_a, standard_action)
    torch.testing.assert_close(action_b, standard_action)
    print("NOTASK_TEMPORAL_TEST=PASS")


if __name__ == "__main__":
    main()
