#!/usr/bin/env python3
"""Train residual PPO, then freeze its rollout as ordinary 70-frame trajectories."""

import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
ADVANCED = ROOT / "retarget_research" / "advanced_policy"
RUN = Path(__file__).resolve().parent
for path in (ADVANCED, RUN):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from train_residual_ppo_general import GeneralResidualEnv
import torch
from residual_ppo import PPOConfig, ResidualActorCritic, RolloutStorage, ppo_update
from physics_cem_refine import (
    PhysicsCase,
    make_env_args,
    phase_frames,
    physics_score,
    smoothstep,
    target_row,
)


def load_cases(hand, manifest_path, target_dir, limit):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cases, target_data, entries = [], {}, []
    for entry in manifest["entries"][:limit]:
        source_index = int(entry["trajectory_indices"][0])
        target = np.load(
            target_dir / f"{entry['object_name']}.npy", allow_pickle=True
        ).item()
        row = target_row(target, source_index)
        source = np.load(entry["source_path"], allow_pickle=True).item()
        cases.append(PhysicsCase(
            hand,
            entry["category"],
            entry["object_name"],
            source_index,
            np.asarray(target["grasp_seqs"][row], dtype=np.float32),
            Path(entry["object_asset_path"]) / "coacd" / "decomposed.obj",
            float(np.asarray(source["obj_scale"])[source_index]),
            np.asarray(source["obj_rotmat"])[source_index],
        ))
        target_data[entry["object_name"]] = target
        entries.append(entry)
    return cases, target_data, entries


def build_residual_gates(cases, horizon, steps_per_frame, mode):
    gates = np.ones((horizon, len(cases), 1), dtype=np.float32)
    if mode == "none":
        return gates
    for case_index, case in enumerate(cases):
        close, grasp = phase_frames(case.target_frames)
        frames = np.arange(horizon, dtype=np.float32) / float(steps_per_frame)
        if mode == "closure":
            values = smoothstep((frames - close) / float(max(1, grasp - close)))
        else:
            values = smoothstep((frames - grasp) / 10.0)
        gates[:, case_index, 0] = values
    return gates


def rollout(env, model, residual_scale, gates, storage=None,
            deterministic=False, collect_actions=False):
    env.reset()
    previous = np.zeros((env.n, env.action_dim), dtype=np.float32)
    applied = []
    for step in range(env.horizon):
        observation = env.observation(step, previous)
        if model is None:
            raw = torch.zeros((env.n, env.action_dim), device=env.device)
            log_prob = torch.zeros(env.n, device=env.device)
            value = torch.zeros(env.n, device=env.device)
        else:
            raw, log_prob, value = model.act(
                observation, observation, deterministic=deterministic)
        gate = torch.as_tensor(gates[step], device=env.device)
        effective = raw * float(residual_scale) * gate
        reward = env.step(step, effective.detach().cpu().numpy())
        done = torch.full(
            (env.n,), step == env.horizon - 1,
            dtype=torch.bool, device=env.device)
        if storage is not None:
            storage.add(
                observation, observation, raw, log_prob, value, reward, done)
        previous = effective.detach().cpu().numpy()
        if collect_actions:
            applied.append(previous.copy())
    metrics = env.metrics()
    if storage is not None:
        bonus = torch.as_tensor([
            30.0 if item["success"] and item["transport_stability_success"]
            else 10.0 if item["success"] else 0.0
            for item in metrics
        ], dtype=torch.float32, device=env.device)
        storage.rewards[-1] += bonus
    return metrics, None if not collect_actions else np.asarray(applied)


def metric_summary(metrics):
    return {
        "stable_count": int(sum(row["success"] for row in metrics)),
        "transport_count": int(sum(
            row["success"] and row["transport_stability_success"]
            for row in metrics)),
        "mean_score": float(np.mean([
            physics_score(row, 0.15, grasp_quality=True) for row in metrics
        ])),
        "mean_final_lift_m": float(np.mean([
            row["final_lift_m"] for row in metrics
        ])),
    }


def export_trajectories(cases, entries, target_data, actions, output_dir,
                        checkpoint_info):
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_actions = actions[:210].reshape(70, 3, len(cases), -1).mean(axis=1)
    for case_index, (case, entry) in enumerate(zip(cases, entries)):
        source_index = case.source_index
        original = target_data[case.object_name]
        row = target_row(original, source_index)
        result = dict(original)
        sequences = np.asarray(original["grasp_seqs"]).copy()
        sequences[row, :, 6:] += frame_actions[:, case_index]
        result["grasp_seqs"] = sequences.astype(np.float32)
        result["retarget_method"] = "trajectory_residual_ppo_v1"
        result["trajectory_residual_ppo"] = {
            **checkpoint_info,
            "source_index": source_index,
            "frame_residual_l2_mean": float(np.linalg.norm(
                frame_actions[:, case_index], axis=1).mean()),
            "frame_residual_l2_max": float(np.linalg.norm(
                frame_actions[:, case_index], axis=1).max()),
        }
        np.save(output_dir / f"{entry['object_name']}.npy", result, allow_pickle=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hand", choices=("linker", "xhand", "wuji"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--num-envs", type=int, default=50)
    parser.add_argument("--residual-scale", type=float, default=0.20)
    parser.add_argument("--gate-mode", choices=("none", "closure", "lift"),
                        default="none")
    parser.add_argument("--evaluation-interval", type=int, default=10)
    parser.add_argument("--selection-rollouts", type=int, default=1)
    parser.add_argument("--export-rollouts", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260829)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cases, target_data, entries = load_cases(
        args.hand, args.manifest, args.target_dir, args.num_envs)
    env = GeneralResidualEnv(
        cases, target_data[cases[0].object_name], make_env_args(args.device, 0.15))
    gates = build_residual_gates(
        cases, env.horizon, env.steps_per_frame, args.gate_mode)
    model = ResidualActorCritic(
        env.obs_dim, env.obs_dim, env.action_dim,
        hidden_dims=(128, 128), init_std=0.05).to(env.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    config = PPOConfig(
        learning_rate=3e-4, minibatches=4, update_epochs=4,
        target_kl=0.5)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    log, best_state, best_key, best_iteration = [], None, None, 0
    try:
        baseline, _ = rollout(env, None, args.residual_scale, gates)
        print("baseline", json.dumps(metric_summary(baseline), ensure_ascii=False))
        for iteration in range(1, args.iterations + 1):
            storage = RolloutStorage()
            metrics, _ = rollout(
                env, model, args.residual_scale, gates, storage=storage)
            batch = storage.finish(
                torch.zeros(env.n, device=env.device), config)
            update = ppo_update(model, optimizer, batch, config)
            item = {"iteration": iteration, **metric_summary(metrics), **update}
            if iteration == 1 or iteration % args.evaluation_interval == 0:
                repeats = [metric_summary(rollout(
                    env, model, args.residual_scale, gates,
                    deterministic=True)[0])
                    for _ in range(args.selection_rollouts)]
                deterministic_summary = {
                    key: float(np.mean([row[key] for row in repeats]))
                    for key in repeats[0]
                }
                item["deterministic_repeats"] = repeats
                item["deterministic"] = deterministic_summary
                key = (
                    deterministic_summary["transport_count"],
                    deterministic_summary["stable_count"],
                    deterministic_summary["mean_score"],
                )
                if best_key is None or key > best_key:
                    best_key = key
                    best_iteration = iteration
                    best_state = copy.deepcopy(model.state_dict())
            log.append(item)
            print(json.dumps(item, ensure_ascii=False), flush=True)
        model.load_state_dict(best_state)
        final_rollouts = [rollout(
            env, model, args.residual_scale, gates,
            deterministic=True, collect_actions=True)
            for _ in range(args.export_rollouts)]
        final_metrics = final_rollouts[-1][0]
        actions = np.mean([row[1] for row in final_rollouts], axis=0)
        checkpoint = args.run_dir / "best.pt"
        torch.save({
            "model": model.state_dict(),
            "hand": args.hand,
            "iteration": best_iteration,
            "obs_dim": env.obs_dim,
            "action_dim": env.action_dim,
            "residual_scale": args.residual_scale,
            "gate_mode": args.gate_mode,
            "selection_rollouts": args.selection_rollouts,
            "export_rollouts": args.export_rollouts,
            "metrics": metric_summary(final_metrics),
        }, checkpoint)
        (args.run_dir / "training_log.json").write_text(
            json.dumps(log, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
        export_trajectories(
            cases, entries, target_data, actions,
            args.run_dir / "independent_targets",
            {
                "checkpoint": str(checkpoint.resolve()),
                "best_iteration": best_iteration,
                "residual_scale": args.residual_scale,
                "gate_mode": args.gate_mode,
                "export_rollouts": args.export_rollouts,
                "training_metric": metric_summary(final_metrics),
            },
        )
        print("best", json.dumps({
            "iteration": best_iteration,
            **metric_summary(final_metrics),
        }, ensure_ascii=False))
    finally:
        env.close()


if __name__ == "__main__":
    main()
