"""CPU-only regression tests for the global action-blend wrapper."""

import torch

from custom_tools.action_blend import ActionBlendPolicy


class FakeOnline:
    def act_inference(self, observations):
        return observations[:, :2] + 1.0


class FakeTemporal:
    def __init__(self):
        self._inference_action_history = None

    def reset_inference_history(self):
        self._inference_action_history = None

    def act_inference(self, observations):
        batch = observations.shape[0]
        action = observations[:, :2] - 2.0
        if (
            self._inference_action_history is None
            or self._inference_action_history.shape[0] != batch
        ):
            self._inference_action_history = torch.zeros(batch, 2, 2)
        self._inference_action_history[:, -1] = action
        return action

    def act_inference_for_reset(self, observations, env_ids):
        action = observations[env_ids, :2] - 2.0
        self._inference_action_history[env_ids] = 0.0
        self._inference_action_history[env_ids, -1] = action
        return action


def make(weight):
    temporal = FakeTemporal()
    return ActionBlendPolicy(FakeOnline(), temporal, weight), temporal


def test_endpoints():
    observations = torch.tensor([
        [1.0, 2.0, 9.0],
        [3.0, 4.0, 8.0],
    ])
    online = FakeOnline().act_inference(observations)
    temporal_reference = FakeTemporal().act_inference(observations)
    zero, _ = make(0.0)
    one, _ = make(1.0)
    assert torch.equal(zero.act_inference(observations), online)
    assert torch.equal(one.act_inference(observations), temporal_reference)


def test_blend_and_history():
    observations = torch.tensor([
        [1.0, 2.0, 9.0],
        [3.0, 4.0, 8.0],
    ])
    policy, temporal = make(0.75)
    action = policy.act_inference(observations)
    expected = (
        0.25 * FakeOnline().act_inference(observations)
        + 0.75 * (observations[:, :2] - 2.0)
    )
    assert torch.allclose(action, expected)
    assert torch.equal(temporal._inference_action_history[:, -1], action)

    previous = temporal._inference_action_history.clone()
    reset_observations = observations + 10.0
    reset_action = policy.act_inference_for_reset(
        reset_observations, torch.tensor([1]))
    assert torch.equal(
        temporal._inference_action_history[1, -1], reset_action[0])
    assert torch.equal(
        temporal._inference_action_history[0], previous[0])


if __name__ == "__main__":
    test_endpoints()
    test_blend_and_history()
    print("ACTION_BLEND_TEST=PASS")
