"""Evaluate one global Online-R1/Temporal3 action blend in Isaac Gym."""

import argparse
from datetime import datetime
from pathlib import Path
import sys

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
DEXGRASP_ROOT = ROOT / "dexgrasp"
for import_root in (str(ROOT), str(DEXGRASP_ROOT)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from custom_tools import evaluate_bc as evaluation_support  # noqa: E402
from custom_tools.train_residual_ppo import (  # noqa: E402
    build_task,
    create_residual_env,
)


def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--online-checkpoint", required=True)
    parser.add_argument("--online-config", required=True)
    parser.add_argument("--temporal-checkpoint", required=True)
    parser.add_argument("--temporal-config", required=True)
    parser.add_argument("--temporal-weight", type=float, required=True)
    parser.add_argument("--residual-config", required=True)
    parser.add_argument("--trajectory-root", required=True)
    parser.add_argument("--object-selection", required=True)
    parser.add_argument("--object-id", default="")
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
    return parser.parse_args()


def resolve(cli):
    path_names = (
        "online_checkpoint", "online_config",
        "temporal_checkpoint", "temporal_config",
        "residual_config", "trajectory_root", "object_selection",
        "output", "env_config", "train_config",
    )
    for name in path_names:
        setattr(cli, name, str(
            Path(getattr(cli, name)).expanduser().resolve()))
    required_files = (
        cli.online_checkpoint, cli.online_config,
        cli.temporal_checkpoint, cli.temporal_config,
        cli.residual_config, cli.object_selection,
        cli.env_config, cli.train_config,
    )
    missing = [path for path in required_files if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError("Missing action-blend inputs: {}".format(
            missing))
    if not Path(cli.trajectory_root).is_dir():
        raise FileNotFoundError(cli.trajectory_root)
    if not 0.0 <= cli.temporal_weight <= 1.0:
        raise ValueError("--temporal-weight must be in [0, 1]")


def load_policy(cli, checkpoint, config):
    cli.bc_checkpoint = checkpoint
    cli.bc_config = config
    return evaluation_support.load_model(cli)


def main():
    cli = parse_cli()
    resolve(cli)
    output = Path(cli.output)
    if output.exists():
        raise FileExistsError(output)

    with Path(cli.object_selection).open(encoding="utf-8") as handle:
        selection = yaml.safe_load(handle)
    object_ids = list(selection["object_ids"])
    if cli.object_id:
        if cli.object_id not in object_ids:
            raise ValueError("Object is absent from selection: {}".format(
                cli.object_id))
        object_ids = [cli.object_id]

    trajectory_data = []
    for object_id in object_ids:
        path = Path(cli.trajectory_root) / (object_id + ".npy")
        source = np.load(path, allow_pickle=True).item()
        item = {"obj_code": object_id}
        for key in ("obj_scale", "obj_rotmat", "grasp_seqs"):
            item[key] = source[key].copy()
        trajectory_data.append(item)
    cli.num_envs = sum(
        item["grasp_seqs"].shape[0] for item in trajectory_data)

    with Path(cli.residual_config).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["seed"] = int(cli.seed)

    # Isaac Gym has to be imported before torch and all torch-based custom
    # policy modules.
    evaluation_support.initialize_cuda_runtime()
    evaluation_support.require_free_vram(cli.min_free_vram_mb)
    evaluation_support.initialize_runtime()
    import torch
    from custom_tools.action_blend import BlendedBCModel
    from custom_tools.residual_env import ResidualDexGraspEnv
    from custom_tools.task_conditioning import set_inference_tasks

    original_cwd = Path.cwd()
    task = None
    try:
        import os
        os.chdir(str(DEXGRASP_ROOT))
        official_args = evaluation_support.build_official_args(cli)
        base_cfg, cfg_train, _ = evaluation_support.load_cfg(official_args)
        evaluation_support.set_seed(
            cli.seed, cfg_train.get("torch_deterministic", False))

        # The load order is fixed and identical for every blend candidate.
        online_loaded = load_policy(
            cli, cli.online_checkpoint, cli.online_config)
        temporal_loaded = load_policy(
            cli, cli.temporal_checkpoint, cli.temporal_config)
        online_model, _, online_path, online_state = online_loaded
        temporal_model, _, temporal_path, temporal_state = temporal_loaded

        task = build_task(
            cli, config, official_args, base_cfg, cfg_train, trajectory_data)
        set_inference_tasks(online_model, object_ids, task.object_idxs)
        set_inference_tasks(temporal_model, object_ids, task.object_idxs)
        blended_model = BlendedBCModel(
            online_model, temporal_model, cli.temporal_weight)
        env, actor_obs, critic_obs = create_residual_env(
            task, blended_model, config, ResidualDexGraspEnv)
        del actor_obs, critic_obs

        object_index = torch.as_tensor(
            task.object_idxs, device=task.device)
        masks = {
            index: object_index == index
            for index in range(len(object_ids))
        }
        peak_counts = {index: 0 for index in masks}
        peak_masks = {
            index: torch.zeros(
                task.num_envs, dtype=torch.bool, device=task.device)
            for index in masks
        }
        ever_failure = torch.zeros(
            task.num_envs, dtype=torch.bool, device=task.device)
        maximum_height = torch.full(
            (task.num_envs,), -float("inf"), device=task.device)
        zero = torch.zeros(
            (task.num_envs, env.policy_action_dim), device=task.device)

        for step in range(int(config["horizon"])):
            _, _, _, _, terms = env.step(zero, step + 1)
            success = task.successes > 0
            for index, mask in masks.items():
                count = int((success & mask).sum().item())
                if count > peak_counts[index]:
                    peak_counts[index] = count
                    peak_masks[index] = success & mask
            ever_failure |= terms["failure_penalty"] < 0
            maximum_height = torch.maximum(
                maximum_height, terms["height_delta"])

        objects = []
        for index, object_id in enumerate(object_ids):
            mask = masks[index]
            count = int(mask.sum().item())
            objects.append({
                "object_id": object_id,
                "category": object_id.split("-", 2)[1],
                "trajectory_count": count,
                "official_peak_success_count": peak_counts[index],
                "official_peak_success_rate": (
                    peak_counts[index] / count),
                "official_peak_success_local_indices": torch.where(
                    peak_masks[index][mask])[0].cpu().tolist(),
                "mean_maximum_lift_m": float(
                    maximum_height[mask].mean().item()),
                "failure_rate": float(
                    ever_failure[mask].float().mean().item()),
            })

        total_success = sum(
            item["official_peak_success_count"] for item in objects)
        total_trajectories = sum(
            item["trajectory_count"] for item in objects)
        result = {
            "temporal_weight": float(cli.temporal_weight),
            "online_checkpoint": str(online_path),
            "online_checkpoint_epoch": (
                int(online_state["epoch"])
                if online_state.get("epoch") is not None else None),
            "temporal_checkpoint": str(temporal_path),
            "temporal_checkpoint_epoch": (
                int(temporal_state["epoch"])
                if temporal_state.get("epoch") is not None else None),
            "total_success_count": total_success,
            "total_trajectory_count": total_trajectories,
            "overall_official_peak_success_rate": (
                total_success / total_trajectories),
            "macro_official_peak_success_rate": float(np.mean([
                item["official_peak_success_rate"] for item in objects])),
            "macro_mean_maximum_lift_m": float(np.mean([
                item["mean_maximum_lift_m"] for item in objects])),
            "macro_failure_rate": float(np.mean([
                item["failure_rate"] for item in objects])),
            "objects": objects,
        }
        aggregate = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "evaluation_mode": "single_sim_action_blend",
            "success_metric": "official_peak_per_object",
            "official_success_definition_changed": False,
            "seed": int(cli.seed),
            "trajectory_root": cli.trajectory_root,
            "object_ids": object_ids,
            "blend_result": result,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(
                aggregate, handle, allow_unicode=True, sort_keys=False)
        print("ACTION_BLEND_EVALUATION=COMPLETE", flush=True)
    finally:
        if task is not None:
            task.clean_sim()
        import os
        os.chdir(str(original_cwd))


if __name__ == "__main__":
    main()
