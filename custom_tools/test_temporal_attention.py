"""CPU checks for the zero-initialized Temporal3 attention residual."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
DEXGRASP_ROOT = ROOT / "dexgrasp"
for path in (str(ROOT), str(DEXGRASP_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import isaacgym  # Must precede torch.  # noqa: E402,F401
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from ActionDiffusion.bc.model.policy.lhm_policy import LitBCModel  # noqa: E402
from custom_tools.task_conditioning import (  # noqa: E402
    AttentionResidualTemporalActorCriticDexRep,
    enable_task_conditioning,
    expand_standard_state_dict_for_task_model,
)


BASE_CONFIG = (
    ROOT / "custom_tools/configs/unified_student_taskid_temporal3_v1.yaml"
)
ATTENTION_CONFIG = (
    ROOT / "custom_tools/configs/"
    / "unified_student_taskid_temporal3_attention_v1.yaml"
)
ENV_CONFIG = DEXGRASP_ROOT / "cfg/shadow_hand_grasp_dexrep_ijrr.yaml"
CHECKPOINT = (
    ROOT / "custom_tools/runs/bc/"
    / "unified_student_taskid_temporal3_seed2025_e4_v1/"
    / "epoch=003-step=5152.ckpt"
)


def make_model(args, env):
    model = LitBCModel(args, env)
    enable_task_conditioning(model, args, env)
    return model


def main():
    required = [BASE_CONFIG, ATTENTION_CONFIG, ENV_CONFIG, CHECKPOINT]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    base_args = OmegaConf.load(str(BASE_CONFIG))
    attention_args = OmegaConf.load(str(ATTENTION_CONFIG))
    env_args = OmegaConf.load(str(ENV_CONFIG))
    env_args.env.obs_dim.pop("pnG")
    env = env_args.env
    checkpoint = torch.load(str(CHECKPOINT), map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)

    torch.manual_seed(11)
    base = make_model(base_args, env)
    base.load_state_dict(state_dict, strict=True)
    torch.manual_seed(13)
    attention = make_model(attention_args, env)
    assert isinstance(
        attention.model, AttentionResidualTemporalActorCriticDexRep)
    expanded, changed = expand_standard_state_dict_for_task_model(
        attention, state_dict)
    assert changed
    attention.load_state_dict(expanded, strict=True)

    base.eval()
    attention.eval()
    observation_dim = sum(int(value) for value in env.obs_dim.values())
    observations = torch.randn(4, observation_dim)
    task_onehot = torch.eye(4)
    history = torch.randn(4, 256)
    with torch.no_grad():
        base_actions = base.model(
            observations, task_onehot, history)
        attention_actions = attention.model(
            observations, task_onehot, history)
    max_difference = float(
        torch.max(torch.abs(base_actions - attention_actions)).item())
    if max_difference > 1e-6:
        raise AssertionError(
            "Attention warm start changed actions: {}".format(
                max_difference))

    frozen_modules = attention.model.freeze_temporal_base()
    if not frozen_modules:
        raise AssertionError("No frozen base modules were returned")
    trainable_names = [
        name for name, parameter in attention.named_parameters()
        if parameter.requires_grad
    ]
    expected_prefixes = (
        "model.history_token_projection.",
        "model.history_position_embedding",
        "model.history_transformer.",
        "model.history_attention_norm.",
        "model.history_attention_action_head.",
    )
    unexpected = [
        name for name in trainable_names
        if not name.startswith(expected_prefixes)
    ]
    if unexpected:
        raise AssertionError(
            "Unexpected trainable base parameters: {}".format(unexpected))
    trainable = sum(
        parameter.numel() for parameter in attention.parameters()
        if parameter.requires_grad)
    frozen = sum(
        parameter.numel() for parameter in attention.parameters()
        if not parameter.requires_grad)
    if trainable <= 0 or frozen <= 0:
        raise AssertionError(
            "Invalid trainable/frozen counts: {}/{}".format(
                trainable, frozen))
    print(
        "TEMPORAL_ATTENTION_TEST=PASS max_initial_action_difference={:.3e} "
        "trainable={} frozen={}".format(
            max_difference, trainable, frozen))


if __name__ == "__main__":
    main()
