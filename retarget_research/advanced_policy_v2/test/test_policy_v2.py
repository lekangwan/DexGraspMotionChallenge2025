"""第二版策略的纯CPU形状、边界和几何回归测试。"""

import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parents[1]))

from dataset import GeometryPolicyDataset  # noqa: E402
from geometry import farthest_point_indices  # noqa: E402
from models import GeometryPolicy, build_model, build_pca_model, build_pca_mixture  # noqa: E402
from retarget_research.advanced_policy_v2.runtime import GeometryPolicyRunner  # noqa: E402


def make_data(root):
    observation_dim, action_dim = 10, 6
    normalization = {
        "observation_mean": np.zeros(observation_dim, dtype=np.float32),
        "observation_std": np.ones(observation_dim, dtype=np.float32),
        "action_mean": np.zeros(action_dim, dtype=np.float32),
        "action_std": np.ones(action_dim, dtype=np.float32),
        "action_delta_limit": np.ones(action_dim, dtype=np.float32),
        "initial_command_mean": np.zeros(action_dim, dtype=np.float32),
        "initial_command_std": np.ones(action_dim, dtype=np.float32),
        "point_mean": np.zeros(3, dtype=np.float32),
        "point_std": np.ones(3, dtype=np.float32),
        "initial_delta_mean": np.zeros(action_dim, dtype=np.float32),
        "initial_delta_std": np.ones(action_dim, dtype=np.float32),
    }
    np.savez(root / "geometry_normalization.npz", **normalization)
    for split in ("train", "valid"):
        trajectory_id = np.asarray([2, 2, 2, 7, 7], dtype=np.int64)
        arrays = {
            "observations": np.arange(50, dtype=np.float32).reshape(5, 10) / 50,
            "actions": np.arange(30, dtype=np.float32).reshape(5, 6) / 30,
            "trajectory_id": trajectory_id,
            "category_id": np.zeros(5, dtype=np.int64),
            "object_id": np.zeros(5, dtype=np.int64),
            "source_trajectory_index": np.zeros(5, dtype=np.int64),
            "source_frame_index": np.arange(5, dtype=np.int16),
            "is_hold": np.asarray([False, False, True, False, True]),
            "expert_replay_success": np.ones(5, dtype=bool),
        }
        np.savez(root / f"{split}.npz", **arrays)
        np.savez(
            root / f"geometry_{split}.npz",
            trajectory_id=np.asarray([2, 7], dtype=np.int64),
            initial_command=np.zeros((2, 6), dtype=np.float32),
            object_points=np.zeros((2, 8, 3), dtype=np.float32),
        )
    (root / "mappings.json").write_text(
        json.dumps({"category_to_id": {"x": 0}, "object_to_id": {}, "policy_action_order": [str(i) for i in range(6)]}),
        encoding="utf-8",
    )


def test_farthest_points_are_deterministic():
    points = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float32)
    first = farthest_point_indices(points, 6)
    second = farthest_point_indices(points, 6)
    assert np.array_equal(first, second)
    assert first.shape == (6,)


def test_dataset_windows_do_not_cross_trajectory(tmp_path):
    make_data(tmp_path)
    dataset = GeometryPolicyDataset(tmp_path, "train", "geometry_chunk", history=3, action_horizon=4)
    sample = dataset[2]
    assert sample["observation_history"].shape == (3, 10)
    assert sample["action_chunk"].shape == (4, 6)
    assert torch.equal(sample["action_chunk"][0], sample["action_chunk"][-1])
    next_sample = dataset[3]
    assert torch.equal(next_sample["observation_history"][0], next_sample["observation_history"][-1])


def test_all_candidate_output_shapes():
    batch, history, points = 4, 3, 16
    common = {
        "initial_observation": torch.zeros(batch, 10),
        "initial_command": torch.zeros(batch, 6),
        "object_points": torch.zeros(batch, points, 3),
        "observation_history": torch.zeros(batch, history, 10),
        "previous_delta_history": torch.zeros(batch, history, 6),
        "phase": torch.zeros(batch, 1),
    }
    for model_type, horizon in [
        ("geometry_phase", 1), ("geometry_chunk", 8),
        ("geometry_plan_chunk", 8), ("geometry_temporal_chunk", 8),
    ]:
        model = GeometryPolicy(
            10, 6, model_type=model_type, history=history,
            action_horizon=8, hidden_dim=64, point_feature_dim=32,
            transformer_layers=1,
        )
        assert model(**common).shape == (batch, horizon, 6)


def test_composite_runner_reuses_autonomous_models(tmp_path):
    """复合策略应保持动作维度，且不需要参考轨迹输入。"""
    make_data(tmp_path)
    observation_dim, action_dim, point_count = 10, 6, 8
    config = {
        "model_type": "geometry_phase", "history": 3, "action_horizon": 8,
        "hidden_dim": 16, "point_feature_dim": 8, "motion_steps": 20,
    }
    model = build_model(config, observation_dim, action_dim)
    base = tmp_path / "base.pt"
    torch.save({
        "schema": "geometry_action_chunk_policy_v1", "config": config,
        "dimensions": {"observation_dim": observation_dim, "action_dim": action_dim,
                       "point_count": point_count},
        "model_state": model.state_dict(),
    }, base)
    composite = tmp_path / "composite.pt"
    torch.save({
        "schema": "geometry_composite_policy_v1",
        "config": {"model_type": "phase_lead05", "composite_type": "phase_lead",
                   "finger_phase_lead": 0.05, "finger_scale": 1.0},
        "primary_checkpoint": str(base), "secondary_checkpoint": str(base),
    }, composite)
    runner = GeometryPolicyRunner(composite, tmp_path, "cpu")
    for child in (runner.primary, runner.secondary):
        child.object_points = np.zeros((point_count, 3), dtype=np.float32)
    runner.reset("unused", np.zeros(observation_dim, dtype=np.float32),
                 np.zeros(action_dim, dtype=np.float32))
    assert runner.act(np.zeros(observation_dim, dtype=np.float32)).shape == (action_dim,)


def test_geometry_pca_runner_generates_complete_sequence(tmp_path):
    """PCA策略应只靠初始条件生成固定长度动作序列。"""
    make_data(tmp_path)
    action_dim, rank, length = 6, 3, 5
    config = {
        "model_type": "geometry_pca3", "pca_rank": rank,
        "hidden_dim": 16, "point_feature_dim": 8,
    }
    model = build_pca_model(config, 10, action_dim)
    checkpoint = tmp_path / "pca.pt"
    torch.save({
        "schema": "geometry_trajectory_pca_policy_v1", "config": config,
        "dimensions": {"task_observation_dim": 10, "observation_dim": 10,
                       "action_dim": action_dim, "point_count": 8},
        "model_state": model.state_dict(),
        "pca_mean": np.zeros(length * action_dim, np.float32),
        "pca_components": np.zeros((rank, length * action_dim), np.float32),
        "coefficient_mean": np.zeros(rank, np.float32),
        "coefficient_std": np.ones(rank, np.float32),
        "sequence_shape": [length, action_dim],
    }, checkpoint)
    runner = GeometryPolicyRunner(checkpoint, tmp_path, "cpu")
    runner.object_points = np.zeros((8, 3), dtype=np.float32)
    runner.reset("unused", np.zeros(10, dtype=np.float32), np.zeros(action_dim, dtype=np.float32))
    for _ in range(length + 2):
        assert runner.act(np.zeros(10, dtype=np.float32)).shape == (action_dim,)


def test_pca_mixture_runner_selects_without_reference_trajectory(tmp_path):
    """多候选策略应由内部判别器选择，并生成完整动作序列。"""
    make_data(tmp_path)
    action_dim, rank, length = 6, 3, 5
    config = {"model_type": "geometry_mixture_critic", "pca_rank": rank,
              "mode_count": 4, "hidden_dim": 16, "point_feature_dim": 8,
              "selection": "critic", "critic_prior_weight": 0.15}
    model = build_pca_mixture(config, 10, action_dim)
    checkpoint = tmp_path / "mixture.pt"
    torch.save({
        "schema": "geometry_pca_mixture_policy_v1", "config": config,
        "dimensions": {"task_observation_dim": 10, "observation_dim": 10,
                       "action_dim": action_dim, "point_count": 8},
        "model_state": model.state_dict(),
        "pca_mean": np.zeros(length * action_dim, np.float32),
        "pca_components": np.zeros((rank, length * action_dim), np.float32),
        "coefficient_mean": np.zeros(rank, np.float32),
        "coefficient_std": np.ones(rank, np.float32),
        "sequence_shape": [length, action_dim],
    }, checkpoint)
    runner = GeometryPolicyRunner(checkpoint, tmp_path, "cpu")
    runner.object_points = np.zeros((8, 3), dtype=np.float32)
    runner.reset("unused", np.zeros(10, dtype=np.float32), np.zeros(action_dim, dtype=np.float32))
    assert runner.act(np.zeros(10, dtype=np.float32)).shape == (action_dim,)
