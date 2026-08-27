from types import MethodType

import torch

from ActionDiffusion.bc.model.policy.lhm_policy import LitBCModel
from custom_tools.task_conditioning import (
    ActionChunkAuxTemporalActorCriticDexRep,
    enable_task_conditioning,
)


class Config(dict):
    __getattr__ = dict.__getitem__


def main():
    args = Config(
        policy=Config(actor_critic="ActorCriticDexRep", pi_hid_sizes=[32, 32],
                      vf_hid_sizes=[32, 32], activation="elu", actions_shape=28),
        encoder=Config(emb_dim=8, n_obs_steps=1, bn_type="part"),
        learn=Config(init_noise_std=0.8),
        task_conditioning=Config(enabled=True,
                                 categories=["bottle", "mug", "bowl", "camera"]),
        temporal_history=Config(enabled=True, history_frames=3,
                                prop_dim=100, action_dim=28),
        action_chunk_aux=Config(enabled=True, horizon=4, auxiliary_weight=1.0),
        action_chunk_execution=Config(enabled=True, temporal_ensemble_decay=0.2),
        lr=2e-5,
    )
    env = Config(obs_dim=Config(prop=100, dexrep_sensor=1280, dexrep_pnl=1080))
    model = LitBCModel(args, env)
    enable_task_conditioning(model, args, env)
    policy = model.model
    assert isinstance(policy, ActionChunkAuxTemporalActorCriticDexRep)
    call = {"step": 0}

    def fake_chunk(self, observations, task_onehot=None, history_features=None):
        start = call["step"]
        call["step"] += 1
        offsets = torch.arange(4, dtype=observations.dtype)[None, :, None]
        return (start + offsets).expand(observations.shape[0], 4, 28)

    policy.forward_action_chunk = MethodType(fake_chunk, policy)
    observations = torch.zeros(2, 2460)
    tasks = torch.eye(4)[:2]
    for step in range(6):
        action = policy.act_inference(observations, tasks)
        torch.testing.assert_close(action, torch.full_like(action, float(step)))
    policy.reset_inference_history()
    assert policy._inference_action_chunks == []
    print("ACTION_CHUNK_EXECUTION_TEST=PASS")


if __name__ == "__main__":
    main()
