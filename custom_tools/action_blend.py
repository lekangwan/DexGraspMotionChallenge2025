"""Global action-space interpolation between Online-R1 and Temporal3.

This is an evaluation-only policy wrapper.  It does not alter either
checkpoint and does not route by object category.
"""

from itertools import chain


class ActionBlendPolicy:
    """Blend two policy actions with one global, fixed coefficient."""

    def __init__(self, online_policy, temporal_policy, temporal_weight):
        self.online_policy = online_policy
        self.temporal_policy = temporal_policy
        self.temporal_weight = float(temporal_weight)
        if not 0.0 <= self.temporal_weight <= 1.0:
            raise ValueError("temporal_weight must be in [0, 1]")

    def reset_inference_history(self):
        reset = getattr(
            self.temporal_policy, "reset_inference_history", None)
        if reset is not None:
            reset()

    def _blend(self, online_action, temporal_action):
        # Explicit endpoint branches make the two controls bit-for-bit exact.
        if self.temporal_weight == 0.0:
            return online_action
        if self.temporal_weight == 1.0:
            return temporal_action
        return (
            (1.0 - self.temporal_weight) * online_action
            + self.temporal_weight * temporal_action
        )

    def _record_selected_action(self, action, env_ids=None):
        """Keep Temporal3 history consistent with the selected blend.

        The existing Temporal3 convention stores the policy output before the
        environment's final [-1, 1] safety clamp.  We preserve that convention.
        """
        history = getattr(
            self.temporal_policy, "_inference_action_history", None)
        if history is None:
            raise RuntimeError("Temporal action history was not initialized")
        if env_ids is None:
            history[:, -1] = action
        else:
            history[env_ids, -1] = action

    def act_inference(self, observations):
        online_action = self.online_policy.act_inference(observations)
        temporal_action = self.temporal_policy.act_inference(observations)
        action = self._blend(online_action, temporal_action)
        self._record_selected_action(action)
        return action

    def act_inference_for_reset(self, observations, env_ids):
        # Online-R1 is stateless, so evaluating the full batch and selecting
        # reset environments is equivalent to a dedicated partial-reset path.
        online_action = self.online_policy.act_inference(observations)[env_ids]
        temporal_action = self.temporal_policy.act_inference_for_reset(
            observations, env_ids)
        action = self._blend(online_action, temporal_action)
        self._record_selected_action(action, env_ids)
        return action


class BlendedBCModel:
    """Minimal LitBCModel-compatible container used by ResidualDexGraspEnv."""

    def __init__(self, online_model, temporal_model, temporal_weight):
        self.online_model = online_model
        self.temporal_model = temporal_model
        self.model = ActionBlendPolicy(
            online_model.model, temporal_model.model, temporal_weight)

    def eval(self):
        self.online_model.eval()
        self.temporal_model.eval()
        return self

    def parameters(self):
        return chain(
            self.online_model.parameters(),
            self.temporal_model.parameters(),
        )
