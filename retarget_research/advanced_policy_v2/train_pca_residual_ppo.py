#!/usr/bin/env python3
"""在自主PCA轨迹上训练只修正手指的接触—运输Residual PPO。"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np


MODULE = Path(__file__).resolve().parent
RESEARCH = MODULE.parent
sys.path.insert(0, str(RESEARCH.parent))

from retarget_research.advanced_policy.train_residual_ppo_general import (  # noqa: E402
    Case, GeneralResidualEnv,
)
from retarget_research.advanced_policy.residual_ppo import (  # noqa: E402
    PPOConfig, ResidualActorCritic, RolloutStorage, ppo_update,
)
from retarget_research.advanced_policy_v2.runtime import GeometryPolicyRunner  # noqa: E402
import torch  # noqa: E402


BASE_MODEL = {"linker": "geometry_pca32", "xhand": "geometry_pca16", "wuji": "geometry_pca16"}


class EnvArgs:
    pass


def load_cases(args):
    """从训练split每类取一条，只用首态让PCA生成自主名义轨迹。"""
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    entries = {item["object_name"]: item for item in manifest["entries"]}
    mappings = json.loads((args.data_dir / "mappings.json").read_text(encoding="utf-8"))
    object_by_id = {int(value): key for key, value in mappings["object_to_id"].items()}
    category_by_id = {int(value): key for key, value in mappings["category_to_id"].items()}
    with np.load(args.data_dir / "train.npz", allow_pickle=False) as archive:
        data = {name: archive[name].copy() for name in archive.files}
    with np.load(args.data_dir / "geometry_train.npz", allow_pickle=False) as archive:
        geometry = {name: archive[name].copy() for name in archive.files}
    geometry_row = {int(value): row for row, value in enumerate(geometry["trajectory_id"])}
    runner = GeometryPolicyRunner(args.base_checkpoint, args.data_dir, args.device)
    cases, categories, target_metadata = [], set(), None
    for trajectory_id in np.unique(data["trajectory_id"]):
        indices = np.flatnonzero(data["trajectory_id"] == trajectory_id)
        category = category_by_id[int(data["category_id"][indices[0]])]
        if category in categories:
            continue
        object_name = object_by_id[int(data["object_id"][indices[0]])]
        source_index = int(data["source_trajectory_index"][indices[0]])
        entry = entries[object_name]
        row = geometry_row[int(trajectory_id)]
        initial = geometry["initial_command"][row].astype(np.float32)
        policy_open = initial.copy()
        policy_open[6:] = 0.0
        runner.object_points = geometry["object_points"][row].astype(np.float32)
        runner.reset(category, data["observations"][indices[0]], policy_open)
        base_frames = policy_open[None] + runner.generated_sequence.copy()
        source = np.load(entry["source_path"], allow_pickle=True).item()
        target = np.load(
            args.target_dir / f"{object_name}.npy", allow_pickle=True
        ).item()
        if target_metadata is None:
            target_metadata = target
        cases.append(Case(
            args.hand, category, object_name, source_index, base_frames,
            Path(entry["object_asset_path"]) / "coacd/decomposed.obj",
            float(np.asarray(source["obj_scale"])[source_index]),
            np.asarray(source["obj_rotmat"])[source_index],
            initial_command=policy_open,
        ))
        categories.add(category)
        if len(cases) >= args.num_envs:
            break
    if len(cases) != args.num_envs:
        raise ValueError(f"只找到{len(cases)}个训练类别，要求{args.num_envs}")
    return cases, target_metadata


def environment_args(args):
    values = {
        "device": args.device, "dt": 1.0 / 60.0, "substeps": 2,
        "finger_stiffness": 120.0, "finger_damping": 5.0,
        "mimic_stiffness": 120.0, "mimic_damping": 5.0,
        "clearance": 0.005, "object_friction": 1.0, "settle_steps": 30,
        "steps_per_frame": 1, "hold_steps": 0, "lift_threshold": 0.15,
        "reward_mode": "pca_contact_transport",
    }
    env_args = EnvArgs()
    for name, value in values.items():
        setattr(env_args, name, value)
    return env_args


def autonomous_success(metric):
    return bool(metric["success"] and metric.get("transport_stability_success", False))


def summarize(metrics):
    return {
        "success_count": int(sum(autonomous_success(item) for item in metrics)),
        "stable_lift_count": int(sum(item["success"] for item in metrics)),
        "mean_max_lift_m": float(np.mean([item["max_lift_m"] for item in metrics])),
        "mean_final_lift_m": float(np.mean([item["final_lift_m"] for item in metrics])),
        "mean_terminal_opposition": float(np.mean([
            item.get("terminal_thumb_opposition_ratio", 0.0) for item in metrics
        ])),
    }


def rollout(env, model, residual_scale, storage=None, deterministic=False):
    env.reset()
    previous = np.zeros((env.n, env.action_dim), dtype=np.float32)
    for step in range(env.horizon):
        observation = env.observation(step, previous)
        if model is None:
            action = torch.zeros((env.n, env.action_dim), device=env.device)
            log_prob = value = torch.zeros(env.n, device=env.device)
        else:
            action, log_prob, value = model.act(
                observation, observation, deterministic=deterministic
            )
        effective = float(residual_scale) * action.detach().cpu().numpy()
        reward = env.step(step, effective)
        done = torch.full(
            (env.n,), step == env.horizon - 1,
            dtype=torch.bool, device=env.device,
        )
        if storage is not None:
            storage.add(
                observation, observation, action, log_prob, value, reward, done
            )
        previous = effective
    metrics = env.metrics()
    if storage is not None:
        storage.rewards[-1] += torch.as_tensor([
            50.0 if autonomous_success(item) else 10.0 if item["success"] else 0.0
            for item in metrics
        ], dtype=torch.float32, device=env.device)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hand", choices=tuple(BASE_MODEL), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--num-envs", type=int, default=20)
    parser.add_argument("--residual-scale", type=float, default=0.12)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260902)
    args = parser.parse_args()
    np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    cases, target_metadata = load_cases(args)
    env = GeneralResidualEnv(cases, target_metadata, environment_args(args))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        baseline_metrics = rollout(env, None, args.residual_scale, deterministic=True)
        baseline = summarize(baseline_metrics)
        print("baseline", json.dumps(baseline, ensure_ascii=False), flush=True)
        model = ResidualActorCritic(
            env.obs_dim, env.obs_dim, action_dim=env.action_dim,
            hidden_dims=(256, 256), init_std=0.08,
        ).to(env.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        config = PPOConfig(
            learning_rate=1e-4, minibatches=4, update_epochs=4,
            target_kl=0.03, entropy_coef=0.001, gamma=0.995,
        )
        history, best_key = [], (-1, -1.0, -1.0)
        for iteration in range(1, args.iterations + 1):
            storage = RolloutStorage()
            rollout(env, model, args.residual_scale, storage, deterministic=False)
            batch = storage.finish(torch.zeros(env.n, device=env.device), config)
            update = ppo_update(model, optimizer, batch, config)
            metrics = rollout(
                env, model, args.residual_scale, deterministic=True
            ) if iteration == 1 or iteration % 2 == 0 else None
            if metrics is None:
                item = {"iteration": iteration, **update}
            else:
                result = summarize(metrics)
                item = {"iteration": iteration, **result, **update}
                key = (
                    result["success_count"], result["mean_final_lift_m"],
                    result["mean_terminal_opposition"],
                )
                if key > best_key:
                    best_key = key
                    torch.save({
                        "schema": "pca_autonomous_contact_residual_ppo_v1",
                        "model": model.state_dict(),
                        "obs_dim": env.obs_dim,
                        "action_dim": env.action_dim,
                        "residual_scale": args.residual_scale,
                        "base_checkpoint": str(args.base_checkpoint.resolve()),
                        "iteration": iteration,
                        "config": vars(args),
                    }, args.output_dir / "best.pt")
            history.append(item)
            print(json.dumps(item, ensure_ascii=False), flush=True)
        (args.output_dir / "training_log.json").write_text(
            json.dumps({
                "baseline": baseline,
                "categories": [case.category for case in cases],
                "iterations": history,
            }, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
