"""Evaluate phase-aligned nearest-neighbor imitation blended with a BC policy."""

import argparse
from datetime import datetime
import os
from pathlib import Path
import sys

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
DEXGRASP_ROOT = ROOT / "dexgrasp"
for path in (str(ROOT), str(DEXGRASP_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from custom_tools import evaluate_bc as evaluation_support  # noqa: E402
from custom_tools.diagnose_bc_closed_loop import (  # noqa: E402
    process_observation, reset_task, success_update)
from custom_tools.train_residual_ppo import build_task  # noqa: E402


FRAMEWISE_CANDIDATES = (
    ("bc", 1, 0.0),
    ("retrieval_k1", 1, 1.0),
    ("retrieval_k3", 3, 1.0),
    ("hybrid_k3_a075", 3, 0.75),
    ("hybrid_k3_a050", 3, 0.50),
)

COHERENT_CANDIDATES = (
    ("bc", 1, 0.0, "framewise"),
    ("framewise_k1", 1, 1.0, "framewise"),
    ("coherent_k1", 1, 1.0, "coherent"),
    ("coherent_k1_a075", 1, 0.75, "coherent"),
    ("coherent_k1_a050", 1, 0.50, "coherent"),
)

PAIRED_CANDIDATES = (
    ("bc", 1, 0.0, "framewise"),
    ("framewise_k1", 1, 1.0, "framewise"),
)


def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--reference-root", required=True)
    parser.add_argument("--evaluation-root", required=True)
    parser.add_argument("--bc-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bc-config", default=str(
        ROOT / "custom_tools/configs/multicategory_bc_noise005.yaml"))
    parser.add_argument("--env-config", default=str(
        DEXGRASP_ROOT / "cfg/shadow_hand_grasp_dexrep_ijrr.yaml"))
    parser.add_argument("--train-config", default=str(
        DEXGRASP_ROOT / "cfg/ppo1/config.yaml"))
    parser.add_argument("--horizon", type=int, default=122)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--min-free-vram-mb", type=int, default=4500)
    parser.add_argument("--sim-device", default="cuda:0")
    parser.add_argument("--rl-device", default="cuda:0")
    parser.add_argument("--show-viewer", action="store_true")
    parser.add_argument("--num-envs", type=int, default=0)
    parser.add_argument(
        "--candidate-profile", choices=("framewise", "coherent", "paired"),
        default="framewise")
    parser.add_argument("--only-candidate", default="")
    parser.add_argument("--capture-dir", default="")
    parser.add_argument("--capture-env", type=int, default=0)
    parser.add_argument("--capture-width", type=int, default=640)
    parser.add_argument("--capture-height", type=int, default=480)
    parser.add_argument("--capture-stride", type=int, default=2)
    return parser.parse_args()


def absolute_paths(cli):
    for name in ("reference_root", "evaluation_root", "bc_checkpoint",
                 "output", "bc_config", "env_config", "train_config"):
        setattr(cli, name, str(Path(getattr(cli, name)).expanduser().resolve()))


def source(path, object_id):
    file_path = Path(path) / (object_id + ".npy")
    if not file_path.is_file():
        raise FileNotFoundError(file_path)
    return np.load(file_path, allow_pickle=True).item()


def task_data(item, object_id):
    data = {"obj_code": object_id}
    for key in ("obj_scale", "obj_rotmat", "grasp_seqs"):
        data[key] = item[key].copy()
    return [data]


def nearest(live_prop, reference_prop, k, torch):
    # Distances are computed only among demonstrations at the same phase/frame.
    distance = (live_prop[:, None, :] - reference_prop[None, :, :]).square().mean(-1)
    k = min(int(k), reference_prop.shape[0])
    values, indices = torch.topk(distance, k=k, dim=1, largest=False)
    if k == 1:
        weights = torch.ones_like(values)
        return indices, weights
    weights = (values + 1e-6).reciprocal()
    weights = weights / weights.sum(dim=1, keepdim=True)
    return indices, weights


def retrieval_action(reference_action, indices, weights):
    selected = reference_action[indices]
    return (selected * weights[..., None]).sum(dim=1)


def rollout(task, model, reference_obs, reference_actions, feature_scale,
            pro_dim, horizon, k, alpha, retrieval_mode, torch):
    raw_obs = reset_task(task, torch)
    peak_count = 0
    peak_mask = torch.zeros(task.num_envs, dtype=torch.bool, device=task.device)
    ever_success = torch.zeros_like(peak_mask)
    initial_height = task.object_pos[:, 2].clone()
    maximum_lift = torch.zeros(task.num_envs, device=task.device)
    fixed_indices = None
    fixed_weights = None
    for step in range(horizon):
        processed = process_observation(raw_obs, pro_dim, torch)
        with torch.no_grad():
            bc_action = model.model.act_inference(processed).clamp(-1.0, 1.0)
            frame = min(step, reference_actions.shape[1] - 1)
            live_prop = processed[:, :pro_dim] / feature_scale
            ref_prop = reference_obs[:, frame, :pro_dim] / feature_scale
            if retrieval_mode == "framewise" or fixed_indices is None:
                indices, weights = nearest(live_prop, ref_prop, k, torch)
                if retrieval_mode == "coherent":
                    fixed_indices, fixed_weights = indices, weights
            else:
                indices, weights = fixed_indices, fixed_weights
            retrieved = retrieval_action(
                reference_actions[:, frame], indices, weights)
            action = ((1.0 - alpha) * bc_action + alpha * retrieved).clamp(-1.0, 1.0)
        task.step(action, step + 1)
        task.capture_frame(step)
        peak_count, peak_mask, ever_success = success_update(
            task, peak_count, peak_mask, ever_success)
        maximum_lift = torch.maximum(
            maximum_lift, task.object_pos[:, 2] - initial_height)
        raw_obs = task.obs_buf.clone()
    return {
        "official_peak_success_count": peak_count,
        "official_peak_success_rate": peak_count / task.num_envs,
        "official_peak_success_local_indices": peak_mask.nonzero(
            as_tuple=False).flatten().cpu().tolist(),
        "diagnostic_ever_success_count": int(ever_success.sum().item()),
        "mean_maximum_lift_m": float(maximum_lift.mean().item()),
    }


def main():
    cli = parse_cli()
    absolute_paths(cli)
    output = Path(cli.output)
    if output.exists():
        raise FileExistsError(output)
    reference = source(cli.reference_root, cli.object_id)
    evaluation = source(cli.evaluation_root, cli.object_id)
    reference_obs_np = reference["obs"][:, :-1].copy()
    reference_actions_np = reference["vis_unscale_actions"][:, :-1].copy()

    evaluation_support.initialize_cuda_runtime()
    evaluation_support.require_free_vram(cli.min_free_vram_mb)
    evaluation_support.initialize_runtime()
    import torch

    original_cwd = Path.cwd()
    task = None
    try:
        os.chdir(str(DEXGRASP_ROOT))
        cli.num_envs = int(evaluation["grasp_seqs"].shape[0])
        official_args = evaluation_support.build_official_args(cli)
        base_cfg, cfg_train, _ = evaluation_support.load_cfg(official_args)
        evaluation_support.set_seed(
            cli.seed, cfg_train.get("torch_deterministic", False))
        model, model_name, checkpoint_path, _ = evaluation_support.load_model(cli)
        if model_name != "ActorCriticDexRep":
            raise ValueError("Retrieval evaluator requires ActorCriticDexRep")
        task = build_task(
            cli, {"seed": cli.seed}, official_args, base_cfg, cfg_train,
            task_data(evaluation, cli.object_id))
        pro_dim = int(task.cfg["env"]["obs_dim"]["prop"])
        reference_obs = process_observation(torch.as_tensor(
            reference_obs_np, device=cli.rl_device, dtype=torch.float32),
            pro_dim, torch)
        reference_actions = torch.as_tensor(
            reference_actions_np, device=cli.rl_device, dtype=torch.float32)
        # Robust floor avoids magnifying nearly constant state coordinates.
        feature_scale = reference_obs[..., :pro_dim].reshape(
            -1, pro_dim).std(dim=0).clamp_min(0.05)
        model.eval()
        results = []
        if cli.candidate_profile == "framewise":
            candidates = [(label, k, alpha, "framewise")
                          for label, k, alpha in FRAMEWISE_CANDIDATES]
        elif cli.candidate_profile == "coherent":
            candidates = COHERENT_CANDIDATES
        else:
            candidates = PAIRED_CANDIDATES
        if cli.only_candidate:
            candidates = [item for item in candidates
                          if item[0] == cli.only_candidate]
            if len(candidates) != 1:
                raise ValueError("Candidate not in profile: {}".format(
                    cli.only_candidate))
        for label, k, alpha, retrieval_mode in candidates:
            print("EVALUATE {} k={} alpha={} mode={}".format(
                label, k, alpha, retrieval_mode), flush=True)
            metrics = rollout(
                task, model, reference_obs, reference_actions, feature_scale,
                pro_dim, cli.horizon, k, alpha, retrieval_mode, torch)
            metrics.update({
                "label": label, "k": k, "retrieval_weight": alpha,
                "retrieval_mode": retrieval_mode})
            results.append(metrics)
        results.sort(key=lambda item: (
            -item["official_peak_success_rate"],
            -item["mean_maximum_lift_m"], item["label"]))
        report = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "method": "phase_aligned_state_retrieval_imitation",
            "formal_result": False,
            "official_success_definition_changed": False,
            "object_id": cli.object_id,
            "reference_root": cli.reference_root,
            "evaluation_root": cli.evaluation_root,
            "reference_trajectory_count": int(reference_obs.shape[0]),
            "evaluation_trajectory_count": int(task.num_envs),
            "bc_checkpoint": str(checkpoint_path),
            "feature": "standardized_prop100_at_same_demonstration_frame",
            "candidate_profile": cli.candidate_profile,
            "ranked_results": results,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(report, handle, allow_unicode=True, sort_keys=False)
        print("PHASE_RETRIEVAL_EVALUATION=COMPLETE output={}".format(output))
    finally:
        if task is not None:
            task.clean_sim()
        os.chdir(str(original_cwd))


if __name__ == "__main__":
    main()
