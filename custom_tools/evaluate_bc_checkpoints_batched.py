"""Evaluate many BC checkpoints in one multi-object Isaac Gym simulation."""

import argparse
import gc
from datetime import datetime
from pathlib import Path
import sys

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEXGRASP_ROOT = REPO_ROOT / "dexgrasp"
for root in (str(REPO_ROOT), str(DEXGRASP_ROOT)):
    if root not in sys.path:
        sys.path.insert(0, root)

from custom_tools import evaluate_bc as evaluation_support  # noqa: E402
from custom_tools.train_residual_ppo import build_task, create_residual_env  # noqa: E402
from custom_tools.grasp_quality_metrics import (  # noqa: E402
    instantaneous_official_success, stable_official_success)


def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--ensemble-checkpoint", default="")
    parser.add_argument("--ensemble-second-weight", type=float, default=0.5)
    parser.add_argument("--bc-config", required=True)
    parser.add_argument("--residual-config", required=True)
    parser.add_argument("--trajectory-root", required=True)
    parser.add_argument("--object-selection", required=True)
    parser.add_argument(
        "--object-id", default="",
        help="Evaluate only this object (used by the isolated-process driver).")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--min-free-vram-mb", type=int, default=4500)
    parser.add_argument("--sim-device", default="cuda:0")
    parser.add_argument("--rl-device", default="cuda:0")
    parser.add_argument("--env-config", default=str(
        DEXGRASP_ROOT / "cfg/shadow_hand_grasp_dexrep_ijrr.yaml"))
    parser.add_argument("--train-config", default=str(
        DEXGRASP_ROOT / "cfg/ppo1/config.yaml"))
    parser.add_argument("--show-viewer", action="store_true")
    parser.add_argument("--capture-dir", default="")
    parser.add_argument("--capture-env", type=int, default=0)
    parser.add_argument("--capture-width", type=int, default=960)
    parser.add_argument("--capture-height", type=int, default=720)
    parser.add_argument(
        "--meshdata-root", default="",
        help="Override the object URDF/mesh root used by the Isaac task.")
    parser.add_argument("--strict-lift-threshold", type=float, default=0.30)
    parser.add_argument("--strict-hold-steps", type=int, default=20)
    parser.add_argument("--strict-min-contact-count", type=int, default=2)
    parser.add_argument("--strict-max-terminal-drop", type=float, default=0.03)
    parser.add_argument("--policy-motion-steps", type=int, default=70)
    parser.add_argument("--temporal-ensemble-decay", type=float, default=None)
    parser.add_argument("--diffusion-residual-scale", type=float, default=None)
    parser.add_argument(
        "--full-observation-residual-scale", type=float, default=None)
    parser.add_argument("--dynamic-candidate-routing", action="store_true")
    parser.add_argument("--late-lift-z-boost", type=float, default=0.0)
    parser.add_argument("--late-lift-start-step", type=int, default=40)
    parser.add_argument("--late-lift-contact-gate", type=int, default=0)
    parser.add_argument("--hold-grip-scale", type=float, default=0.0)
    parser.add_argument("--hold-grip-reference-step", type=int, default=40)
    parser.add_argument("--max-trajectories-per-object", type=int, default=0)
    parser.add_argument("--use-expert-actions", action="store_true")
    return parser.parse_args()


def resolve(cli):
    for name in ("bc_config", "residual_config", "trajectory_root",
                 "object_selection", "output", "env_config", "train_config",
                 "capture_dir"):
        setattr(cli, name, str(Path(getattr(cli, name)).expanduser().resolve()))
    cli.checkpoint = [str(Path(path).expanduser().resolve())
                      for path in cli.checkpoint]
    if cli.ensemble_checkpoint:
        cli.ensemble_checkpoint = str(
            Path(cli.ensemble_checkpoint).expanduser().resolve())
    if cli.meshdata_root:
        cli.meshdata_root = str(Path(cli.meshdata_root).expanduser().resolve())
    missing = [path for path in cli.checkpoint if not Path(path).is_file()]
    if (cli.ensemble_checkpoint
            and not Path(cli.ensemble_checkpoint).is_file()):
        missing.append(cli.ensemble_checkpoint)
    if missing:
        raise FileNotFoundError("Missing checkpoints: {}".format(missing))


def evaluate_one(cli, config, task, object_ids, checkpoint_path, torch,
                 residual_env_class, loaded_model=None):
    if loaded_model is None:
        cli.bc_checkpoint = checkpoint_path
        loaded_model = evaluation_support.load_model(cli)
    bc_model, _, resolved_checkpoint, checkpoint = loaded_model
    set_inference_tasks(bc_model, object_ids, task.object_idxs)
    env, actor_obs, critic_obs = create_residual_env(
        task, bc_model, config, residual_env_class)
    del actor_obs, critic_obs
    object_index = torch.as_tensor(task.object_idxs, device=task.device)
    masks = {index: object_index == index for index in range(len(object_ids))}
    peak_counts = {index: 0 for index in masks}
    peak_masks = {index: torch.zeros(task.num_envs, dtype=torch.bool,
                                    device=task.device) for index in masks}
    ever_failure = torch.zeros(task.num_envs, dtype=torch.bool, device=task.device)
    maximum_height = torch.full(
        (task.num_envs,), -float("inf"), device=task.device)
    terminal_heights = []
    terminal_contacts = []
    terminal_official_conditions = []
    if not 1 <= int(cli.policy_motion_steps) <= int(config["horizon"]):
        raise ValueError("policy-motion-steps必须位于[1, horizon]")
    expert = torch.as_tensor(
        task.obj_trajs_info["grasp_seqs"],
        dtype=torch.float32, device=task.device)
    lower = task.shadow_hand_dof_lower_limits
    upper = task.shadow_hand_dof_upper_limits
    normalized_expert_actions = (
        2.0 * expert - upper[None, None] - lower[None, None]
    ) / (upper[None, None] - lower[None, None])
    expert_actions = normalized_expert_actions if cli.use_expert_actions else None
    zero = torch.zeros((task.num_envs, env.policy_action_dim), device=task.device)
    held_action = None
    grip_reference_action = None
    lift_contact_latched = torch.zeros(
        task.num_envs, dtype=torch.bool, device=task.device)
    minimum_palm_distance = torch.full(
        (task.num_envs,), float("inf"), device=task.device)
    minimum_finger_distance = minimum_palm_distance.clone()
    maximum_contact_count = torch.zeros(task.num_envs, device=task.device)
    first_contact_step = torch.full(
        (task.num_envs,), -1, dtype=torch.long, device=task.device)
    first_official_step = first_contact_step.clone()
    raw_action_squared_error = torch.zeros(task.num_envs, device=task.device)
    applied_action_squared_error = torch.zeros(task.num_envs, device=task.device)
    action_error_steps = 0
    for step in range(int(config["horizon"])):
        raw_action = env._bc_action.clone()
        if expert_actions is not None:
            frame = min(step, expert_actions.shape[1] - 1)
            env._bc_action = expert_actions[:, frame].clamp(-1.0, 1.0)
        elif step >= int(cli.policy_motion_steps):
            if held_action is None:
                raise RuntimeError("进入保持阶段前没有记录末帧动作")
            env._bc_action = held_action.clone()
            if cli.hold_grip_scale != 0.0:
                if grip_reference_action is None:
                    raise RuntimeError("保持阶段缺少手指闭合参考动作")
                env._bc_action[:, 6:] = (
                    held_action[:, 6:]
                    + float(cli.hold_grip_scale)
                    * (held_action[:, 6:] - grip_reference_action[:, 6:])
                ).clamp(-1.0, 1.0)
        elif cli.late_lift_z_boost != 0.0 and step >= cli.late_lift_start_step:
            if cli.late_lift_contact_gate > 0:
                _, _, current_contacts = env._distances_and_contacts()
                lift_contact_latched |= (
                    current_contacts >= int(cli.late_lift_contact_gate))
            ramp_steps = max(
                int(cli.policy_motion_steps) - 1 - cli.late_lift_start_step, 1)
            fraction = min(
                (step - cli.late_lift_start_step) / float(ramp_steps), 1.0)
            boost = float(cli.late_lift_z_boost) * fraction
            if cli.late_lift_contact_gate > 0:
                boost = boost * lift_contact_latched.to(
                    dtype=env._bc_action.dtype)
            env._bc_action[:, 2] = (
                env._bc_action[:, 2] + boost).clamp(-1.0, 1.0)
        if step < int(cli.policy_motion_steps):
            expert_frame = min(step, normalized_expert_actions.shape[1] - 1)
            target_action = normalized_expert_actions[:, expert_frame]
            raw_action_squared_error += (
                raw_action - target_action).square().mean(dim=-1)
            applied_action_squared_error += (
                env._bc_action - target_action).square().mean(dim=-1)
            action_error_steps += 1
        _, _, _, _, terms = env.step(zero, step + 1)
        task.capture_frame(step)
        if step == int(cli.hold_grip_reference_step):
            grip_reference_action = env._previous_final_action.clone()
        if step == int(cli.policy_motion_steps) - 1:
            held_action = env._previous_final_action.clone()
        success = task.successes > 0
        for index, mask in masks.items():
            count = int((success & mask).sum().item())
            if count > peak_counts[index]:
                peak_counts[index] = count
                peak_masks[index] = success & mask
        ever_failure |= terms["failure_penalty"] < 0
        maximum_height = torch.maximum(maximum_height, terms["height_delta"])
        terminal_heights.append(terms["height_delta"].clone())
        terminal_contacts.append(terms["contact_count"].clone())
        terminal_official_conditions.append(
            instantaneous_official_success(
                task.object_pos, task.goal_pos, tolerance_m=0.12))
        palm_distance, finger_distance, contact_count = (
            env._distances_and_contacts())
        minimum_palm_distance = torch.minimum(
            minimum_palm_distance, palm_distance)
        minimum_finger_distance = torch.minimum(
            minimum_finger_distance, finger_distance)
        maximum_contact_count = torch.maximum(
            maximum_contact_count, contact_count)
        new_contact = (first_contact_step < 0) & (contact_count >= 1)
        first_contact_step[new_contact] = step
        official_now = terminal_official_conditions[-1]
        new_official = (first_official_step < 0) & official_now
        first_official_step[new_official] = step
    hold_steps = int(cli.strict_hold_steps)
    if hold_steps <= 0 or hold_steps > len(terminal_heights):
        raise ValueError("strict-hold-steps必须位于[1, horizon]")
    final_heights = torch.stack(terminal_heights[-hold_steps:])
    final_contacts = torch.stack(terminal_contacts[-hold_steps:])
    final_official = torch.stack(
        terminal_official_conditions[-hold_steps:])
    height_thresholds = (0.18, 0.20, 0.25, 0.30)
    terminal_height_masks = {
        threshold: (final_heights >= threshold).all(dim=0)
        for threshold in height_thresholds
    }
    terminal_contact_mask = (
        final_contacts >= int(cli.strict_min_contact_count)).all(dim=0)
    terminal_drop_mask = (
        (maximum_height - final_heights[-1])
        <= float(cli.strict_max_terminal_drop))
    stable_official = stable_official_success(
        final_official, maximum_height, final_heights[-1],
        maximum_drop_m=cli.strict_max_terminal_drop)
    goal_center_30cm = terminal_height_masks[0.30] & terminal_drop_mask
    strict_success = stable_official
    final_goal_distance = torch.linalg.vector_norm(
        task.goal_pos - task.object_pos, dim=-1)
    peak_to_final_drop = maximum_height - final_heights[-1]
    raw_action_mse = raw_action_squared_error / max(action_error_steps, 1)
    applied_action_mse = applied_action_squared_error / max(
        action_error_steps, 1)
    objects = []
    for index, object_id in enumerate(object_ids):
        mask = masks[index]
        count = int(mask.sum().item())
        environment_indices = torch.where(mask)[0]
        trajectory_diagnostics = []
        for local_index, environment_index in enumerate(environment_indices):
            item = int(environment_index.item())
            trajectory_diagnostics.append({
                "local_trajectory_index": local_index,
                "official_peak_success": bool(peak_masks[index][item].item()),
                "stable_official_success": bool(stable_official[item].item()),
                "first_official_step": int(first_official_step[item].item()),
                "maximum_lift_m": float(maximum_height[item].item()),
                "final_lift_m": float(final_heights[-1, item].item()),
                "peak_to_final_drop_m": float(peak_to_final_drop[item].item()),
                "final_goal_distance_m": float(final_goal_distance[item].item()),
                "minimum_palm_object_distance_m": float(
                    minimum_palm_distance[item].item()),
                "minimum_mean_fingertip_object_distance_m": float(
                    minimum_finger_distance[item].item()),
                "maximum_fingertip_contact_count": int(
                    maximum_contact_count[item].item()),
                "first_fingertip_contact_step": int(
                    first_contact_step[item].item()),
                "raw_policy_to_expert_action_mse": float(
                    raw_action_mse[item].item()),
                "applied_action_to_expert_action_mse": float(
                    applied_action_mse[item].item()),
            })
        objects.append({
            "object_id": object_id,
            "category": object_id.split("-", 2)[1],
            "trajectory_count": count,
            "official_peak_success_count": peak_counts[index],
            "official_peak_success_rate": peak_counts[index] / count,
            "strict_terminal_success_count": int((strict_success & mask).sum().item()),
            "strict_terminal_success_rate": float(strict_success[mask].float().mean().item()),
            "stable_official_success_count": int(
                (stable_official & mask).sum().item()),
            "stable_official_success_rate": float(
                stable_official[mask].float().mean().item()),
            "goal_center_30cm_success_count": int(
                (goal_center_30cm & mask).sum().item()),
            "goal_center_30cm_success_rate": float(
                goal_center_30cm[mask].float().mean().item()),
            "terminal_height_hold_counts": {
                str(threshold): int((value & mask).sum().item())
                for threshold, value in terminal_height_masks.items()
            },
            "terminal_contact_hold_count": int(
                (terminal_contact_mask & mask).sum().item()),
            "terminal_low_drop_count": int(
                (terminal_drop_mask & mask).sum().item()),
            "terminal_stable_transport_count": int(
                (stable_official & mask).sum().item()),
            "mean_final_lift_m": float(final_heights[-1][mask].mean().item()),
            "mean_terminal_height_range_m": float(
                (final_heights.max(dim=0).values
                 - final_heights.min(dim=0).values)[mask].mean().item()),
            "official_peak_success_local_indices": torch.where(
                peak_masks[index][mask])[0].cpu().tolist(),
            "mean_maximum_lift_m": float(maximum_height[mask].mean().item()),
            "failure_rate": float(ever_failure[mask].float().mean().item()),
            "trajectory_diagnostics": trajectory_diagnostics,
        })
    result = {
        "checkpoint": str(resolved_checkpoint),
        "checkpoint_epoch": (
            int(checkpoint["epoch"]) if checkpoint.get("epoch") is not None
            else None),
        "total_success_count": sum(x["official_peak_success_count"] for x in objects),
        "total_trajectory_count": sum(x["trajectory_count"] for x in objects),
        "total_strict_terminal_success_count": sum(
            x["strict_terminal_success_count"] for x in objects),
        "total_stable_official_success_count": sum(
            x["stable_official_success_count"] for x in objects),
        "total_goal_center_30cm_success_count": sum(
            x["goal_center_30cm_success_count"] for x in objects),
        "macro_official_peak_success_rate": float(np.mean([
            x["official_peak_success_rate"] for x in objects])),
        "macro_mean_maximum_lift_m": float(np.mean([
            x["mean_maximum_lift_m"] for x in objects])),
        "macro_failure_rate": float(np.mean([x["failure_rate"] for x in objects])),
        "objects": objects,
    }
    result["overall_official_peak_success_rate"] = (
        result["total_success_count"] / result["total_trajectory_count"])
    result["overall_strict_terminal_success_rate"] = (
        result["total_strict_terminal_success_count"]
        / result["total_trajectory_count"])
    result["overall_stable_official_success_rate"] = (
        result["total_stable_official_success_count"]
        / result["total_trajectory_count"])
    result["overall_goal_center_30cm_success_rate"] = (
        result["total_goal_center_30cm_success_count"]
        / result["total_trajectory_count"])
    result["macro_strict_terminal_success_rate"] = float(np.mean([
        x["strict_terminal_success_rate"] for x in objects]))
    result["strict_terminal_definition"] = {
        "metric_name": "stable_official_success",
        "official_instantaneous_rule": (
            "goal_distance<=0.12m OR object_z>=goal_z, within workspace"),
        "terminal_hold_policy_steps": hold_steps,
        "policy_motion_steps": int(cli.policy_motion_steps),
        "minimum_contact_count": int(cli.strict_min_contact_count),
        "maximum_peak_to_terminal_drop_m": float(cli.strict_max_terminal_drop),
        "contact_is_diagnostic_only": True,
        "height_rule": "official instantaneous rule holds for every terminal step",
        "late_lift_z_boost_normalized": float(cli.late_lift_z_boost),
        "late_lift_start_step": int(cli.late_lift_start_step),
        "late_lift_contact_gate": int(cli.late_lift_contact_gate),
        "hold_grip_scale": float(cli.hold_grip_scale),
        "hold_grip_reference_step": int(cli.hold_grip_reference_step),
    }
    result["goal_center_30cm_diagnostic"] = {
        "lift_threshold_m": 0.30,
        "terminal_hold_policy_steps": hold_steps,
        "maximum_peak_to_terminal_drop_m": float(
            cli.strict_max_terminal_drop),
        "primary_metric": False,
    }
    result["action_source"] = (
        "graspm3_expert_replay" if cli.use_expert_actions
        else "checkpoint_policy")
    del env, bc_model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def load_ensemble(cli):
    """Load two same-architecture policies before simulator construction."""
    from custom_tools.task_conditioning import FixedPolicyEnsemble

    first = evaluation_support.load_model(cli)
    second_cli = argparse.Namespace(**vars(cli))
    second_cli.bc_checkpoint = cli.ensemble_checkpoint
    second = evaluation_support.load_model(second_cli)
    weight = float(cli.ensemble_second_weight)
    if not 0.0 <= weight <= 1.0:
        raise ValueError("ensemble-second-weight must be in [0, 1]")
    first_model, model_name, first_path, checkpoint = first
    second_model, second_name, second_path, _ = second
    if model_name != second_name:
        raise ValueError("Ensemble policy model names differ")
    first_model.model = FixedPolicyEnsemble(
        [first_model.model, second_model.model], [1.0 - weight, weight])
    label = "{}+{}@{:.2f}".format(first_path, second_path, weight)
    return first_model, model_name, label, checkpoint


def main():
    global set_inference_tasks
    cli = parse_cli()
    resolve(cli)
    output = Path(cli.output)
    if output.exists():
        raise FileExistsError(output)
    with Path(cli.object_selection).open(encoding="utf-8") as handle:
        selection = yaml.safe_load(handle)
    with Path(cli.residual_config).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if cli.meshdata_root:
        config["meshdata_root"] = cli.meshdata_root
    config["seed"] = int(cli.seed)
    object_ids = list(selection["object_ids"])
    if cli.object_id:
        if cli.object_id not in object_ids:
            raise ValueError("Object is absent from selection: {}".format(
                cli.object_id))
        object_ids = [cli.object_id]
    trajectory_data = []
    for object_id in object_ids:
        source = np.load(
            Path(cli.trajectory_root) / (object_id + ".npy"),
            allow_pickle=True).item()
        item = {"obj_code": object_id}
        count = source["grasp_seqs"].shape[0]
        if cli.max_trajectories_per_object > 0:
            count = min(count, cli.max_trajectories_per_object)
        for key in ("obj_scale", "obj_rotmat", "grasp_seqs"):
            item[key] = source[key][:count].copy()
        trajectory_data.append(item)
    cli.seed = int(cli.seed)
    cli.num_envs = sum(item["grasp_seqs"].shape[0] for item in trajectory_data)
    evaluation_support.initialize_cuda_runtime()
    evaluation_support.require_free_vram(cli.min_free_vram_mb)
    evaluation_support.initialize_runtime()
    import torch
    # Isaac Gym requires its modules to be imported before anything that
    # imports torch.  Task conditioning therefore has to remain a lazy import.
    from custom_tools.task_conditioning import set_inference_tasks
    from custom_tools.residual_env import ResidualDexGraspEnv
    original_cwd = Path.cwd()
    task = None
    first_loaded_model = None
    try:
        import os
        os.chdir(str(DEXGRASP_ROOT))
        official_args = evaluation_support.build_official_args(cli)
        base_cfg, cfg_train, _ = evaluation_support.load_cfg(official_args)
        evaluation_support.set_seed(
            cli.seed, cfg_train.get("torch_deterministic", False))
        # Preserve the verified evaluator's random-number consumption order:
        # load/initialize the BC network before creating the Isaac Gym task.
        cli.bc_checkpoint = cli.checkpoint[0]
        first_loaded_model = (
            load_ensemble(cli) if cli.ensemble_checkpoint
            else evaluation_support.load_model(cli))
        # Model constructors consume different amounts of torch randomness.
        # Reset here so simulator creation sees the same seed for every
        # architecture; otherwise a zero-effect wrapper can change results.
        evaluation_support.set_seed(
            cli.seed, cfg_train.get("torch_deterministic", False))
        task = build_task(
            cli, config, official_args, base_cfg, cfg_train, trajectory_data)
        results = []
        for index, checkpoint in enumerate(cli.checkpoint, 1):
            print("checkpoint {}/{}: {}".format(
                index, len(cli.checkpoint), checkpoint), flush=True)
            results.append(evaluate_one(
                cli, config, task, object_ids, checkpoint, torch,
                ResidualDexGraspEnv,
                loaded_model=first_loaded_model if index == 1 else None))
            if index == 1:
                first_loaded_model = None
        aggregate = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "evaluation_mode": "single_sim_multi_object_multi_checkpoint",
            "success_metric": "stable_official_success",
            "official_peak_metric_retained": True,
            "official_success_definition_changed": False,
            "seed": cli.seed,
            "trajectory_root": cli.trajectory_root,
            "object_ids": object_ids,
            "checkpoint_results": results,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(aggregate, handle, allow_unicode=True, sort_keys=False)
        print("BATCHED_BC_EVALUATION=COMPLETE")
    finally:
        if task is not None:
            task.clean_sim()
        import os
        os.chdir(str(original_cwd))


if __name__ == "__main__":
    main()
