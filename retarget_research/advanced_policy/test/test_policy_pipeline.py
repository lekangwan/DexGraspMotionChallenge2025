"""对进阶策略的数据边界、模型维度和时序窗口做快速单元测试。"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch


POLICY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(POLICY_ROOT))

from dataset import TargetHandPolicyDataset
from models import ConditionalDiffusionPolicy, MLPBCPolicy, Temporal3BCPolicy, linear_beta_schedule, sample_diffusion
from observations import build_object_shape_descriptor, build_observation_batch
from runtime import PolicyRunner
from train import compute_loss, run_epoch
from evaluate_policy_manifest import build_tasks, stable_task_seed


class PolicyPipelineTest(unittest.TestCase):
    """验证无需Isaac Gym的策略核心纯逻辑。"""

    def test_observation_schema_dimension_and_relative_state(self):
        """17轴Linker加入14维形状后应为66维，初始相对位移必须为零。"""
        count, dofs = 3, 17
        position = np.asarray([[0, 0, 0.1], [0, 0, 0.15], [0, 0, 0.22]], dtype=np.float32)
        result = build_observation_batch(
            np.zeros((count, dofs)), np.zeros((count, dofs)), position,
            np.tile([0, 0, 0, 1], (count, 1)), np.zeros((count, 3)),
            np.zeros((count, 3)), position[0], np.asarray([0, 1, 3]),
            np.zeros(14, dtype=np.float32), 0.10,
        )
        self.assertEqual(result.shape, (3, 66))
        np.testing.assert_allclose(result[0, -5:-2], 0.0)
        self.assertAlmostEqual(float(result[1, -2]), 0.05, places=6)
        self.assertAlmostEqual(float(result[2, -2]), 0.0, places=6)

    def test_shape_descriptor_is_finite_and_scales_geometrically(self):
        """同一四面体放大2倍时尺寸应2倍、表面积4倍、体积8倍。"""
        with tempfile.TemporaryDirectory() as directory:
            mesh = Path(directory) / "tetra.obj"
            mesh.write_text(
                "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\n"
                "f 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n",
                encoding="utf-8",
            )
            first = build_object_shape_descriptor(mesh, 1.0)
            second = build_object_shape_descriptor(mesh, 2.0)
            self.assertEqual(first.shape, (14,))
            self.assertTrue(np.isfinite(first).all())
            np.testing.assert_allclose(second[:3], first[:3] * 2, rtol=1e-5)
            self.assertAlmostEqual(float(second[9] / first[9]), 4.0, places=5)
            self.assertAlmostEqual(float(second[10] / first[10]), 8.0, places=5)

    def test_temporal_history_never_crosses_trajectory(self):
        """第二条轨迹首步历史不能读到第一条轨迹动作或观测。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            observations = np.asarray([[1, 1], [2, 2], [9, 9], [10, 10]], dtype=np.float32)
            actions = np.asarray([[1], [2], [9], [10]], dtype=np.float32)
            np.savez_compressed(
                root / "data.npz", observations=observations, actions=actions,
                trajectory_id=np.asarray([0, 0, 1, 1]), category_id=np.zeros(4, dtype=np.int64),
            )
            np.savez_compressed(
                root / "normalization.npz", observation_mean=np.zeros(2),
                observation_std=np.ones(2), action_mean=np.zeros(1), action_std=np.ones(1),
            )
            dataset = TargetHandPolicyDataset(root / "data.npz", root / "normalization.npz", "temporal3", 3)
            sample = dataset[2]
            np.testing.assert_allclose(sample["observation_history"].numpy(), [[9, 9], [9, 9], [9, 9]])
            np.testing.assert_allclose(sample["previous_actions"].numpy(), [[0], [0]])

    def test_closed_loop_tasks_respect_requested_split(self):
        """valid选择只能读取valid轨迹，最终test也不能混入训练物体轨迹。"""
        with tempfile.TemporaryDirectory() as directory:
            target_dir = Path(directory)
            np.save(
                target_dir / "object_a.npy",
                {"source_trajectory_indices": np.asarray([3, 7])},
                allow_pickle=True,
            )
            source = target_dir / "source.npy"
            object_dir = target_dir / "object_a"
            manifest = {
                "entries": [{
                    "object_name": "object_a",
                    "category": "cup",
                    "source_path": str(source),
                    "object_asset_path": str(object_dir),
                }]
            }
            split = {
                "records": [
                    {"split": "valid", "object_name": "object_a", "category": "cup", "source_trajectory_index": 3},
                    {"split": "test", "object_name": "object_a", "category": "cup", "source_trajectory_index": 7},
                ]
            }
            valid = build_tasks(manifest, split, target_dir, "valid")
            test = build_tasks(manifest, split, target_dir, "test")
            self.assertEqual([(item["source_index"], item["target_index"]) for item in valid], [(3, 0)])
            self.assertEqual([(item["source_index"], item["target_index"]) for item in test], [(7, 1)])
            with self.assertRaisesRegex(ValueError, "未知策略划分"):
                build_tasks(manifest, split, target_dir, "unknown")

    def test_all_models_preserve_expected_batch_shapes(self):
        """三类策略分别输出一帧动作或固定长度动作片段。"""
        batch, observation_dim, action_dim, categories = 4, 20, 6, 5
        category = torch.arange(batch) % categories
        bc = MLPBCPolicy(observation_dim, action_dim, categories)
        self.assertEqual(tuple(bc(torch.randn(batch, observation_dim), category).shape), (batch, action_dim))
        temporal = Temporal3BCPolicy(observation_dim, action_dim, categories)
        self.assertEqual(
            tuple(temporal(torch.randn(batch, 3, observation_dim), torch.randn(batch, 2, action_dim), category).shape),
            (batch, action_dim),
        )
        diffusion = ConditionalDiffusionPolicy(
            observation_dim, action_dim, categories, action_horizon=4, observation_history=3
        )
        noisy = torch.randn(batch, 4, action_dim)
        predicted = diffusion(noisy, torch.randn(batch, 3, observation_dim), category, torch.arange(batch))
        self.assertEqual(tuple(predicted.shape), (batch, 4, action_dim))
        sampled = sample_diffusion(
            diffusion, torch.randn(batch, 3, observation_dim), category, linear_beta_schedule(3)
        )
        self.assertEqual(tuple(sampled.shape), (batch, 4, action_dim))
        self.assertTrue(torch.isfinite(sampled).all())

    def test_policy_runner_loads_checkpoint_and_denormalizes(self):
        """统一推理器应按checkpoint构模并把零标准化输出恢复到动作均值。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {
                "model_type": "bc",
                "category_embedding_dim": 2,
                "hidden_dims": [4],
                "dropout": 0.0,
            }
            dimensions = {"observation_dim": 3, "action_dim": 2, "category_count": 1}
            model = MLPBCPolicy(3, 2, 1, 2, (4,), 0.0)
            for parameter in model.parameters():
                parameter.data.zero_()
            torch.save(
                {"config": config, "dimensions": dimensions, "model_state": model.state_dict()},
                root / "model.pt",
            )
            np.savez_compressed(
                root / "normalization.npz",
                observation_mean=np.asarray([1, 2, 3], dtype=np.float32),
                observation_std=np.ones(3, dtype=np.float32),
                action_mean=np.asarray([0.25, -0.5], dtype=np.float32),
                action_std=np.asarray([2, 3], dtype=np.float32),
                action_delta_limit=np.asarray([0.1, 0.2], dtype=np.float32),
                action_delta_norm_limit=np.asarray(0.15, dtype=np.float32),
                action_delta_quantile=np.asarray(0.995, dtype=np.float32),
            )
            (root / "mappings.json").write_text(
                json.dumps({"category_to_id": {"cup": 0}, "object_to_id": {}, "policy_action_order": ["a", "b"]}), encoding="utf-8"
            )
            runner = PolicyRunner(root / "model.pt", root, "cpu")
            runner.reset("cup", np.asarray([1, 2, 3], dtype=np.float32))
            np.testing.assert_allclose(
                runner.act(np.asarray([1, 2, 3], dtype=np.float32)),
                [0.25, -0.5],
                atol=1e-6,
            )
            limited = PolicyRunner(
                root / "model.pt", root, "cpu", action_rate_limit_scale=1.0
            )
            with self.assertRaisesRegex(ValueError, "initial_action"):
                limited.reset("cup", np.asarray([1, 2, 3], dtype=np.float32))
            limited.reset(
                "cup",
                np.asarray([1, 2, 3], dtype=np.float32),
                initial_action=np.zeros(2, dtype=np.float32),
            )
            command = limited.act(np.asarray([1, 2, 3], dtype=np.float32))
            self.assertLessEqual(float(np.linalg.norm(command)), 0.150001)
            self.assertLessEqual(abs(float(command[0])), 0.100001)
            self.assertLessEqual(abs(float(command[1])), 0.200001)

    def test_diffusion_task_seed_is_stable_and_trajectory_specific(self):
        """相同轨迹键必须得到相同seed，不同源轨迹必须得到不同seed。"""
        first = {"object_name": "cup", "source_index": 3}
        second = {"object_name": "cup", "source_index": 4}
        self.assertEqual(stable_task_seed(7, first), stable_task_seed(7, first))
        self.assertNotEqual(stable_task_seed(7, first), stable_task_seed(7, second))

    def test_three_training_losses_are_finite_and_backwardable(self):
        """BC、Temporal3和Diffusion的监督目标都应产生可反传有限loss。"""
        device = torch.device("cpu")
        category = torch.tensor([0, 1], dtype=torch.long)
        bc = MLPBCPolicy(5, 3, 2, hidden_dims=(8,))
        loss, _ = compute_loss(
            bc,
            {"observations": torch.randn(2, 5), "actions": torch.randn(2, 3), "category_id": category},
            "bc",
            device,
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        temporal = Temporal3BCPolicy(5, 3, 2, hidden_dims=(8,))
        loss, _ = compute_loss(
            temporal,
            {
                "observation_history": torch.randn(2, 3, 5),
                "previous_actions": torch.randn(2, 2, 3),
                "actions": torch.randn(2, 3),
                "category_id": category,
            },
            "temporal3",
            device,
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        diffusion = ConditionalDiffusionPolicy(
            5,
            3,
            2,
            action_horizon=4,
            observation_history=3,
            hidden_dims=(8,),
        )
        loss, _ = compute_loss(
            diffusion,
            {
                "observation_history": torch.randn(2, 3, 5),
                "action_sequence": torch.randn(2, 4, 3),
                "category_id": category,
            },
            "diffusion",
            device,
            linear_beta_schedule(3),
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss))

    def test_smoke_epoch_respects_batch_limit_and_reports_work(self):
        """CPU冒烟必须只执行配置数量的batch并报告真实样本数。

        输入：5个各含2样本的人工BC batch，训练上限2。
        输出：仅处理2个batch和4个样本，loss有限；0上限被拒绝。
        内部逻辑：使用小MLP和SGD调用共享`run_epoch`，不写checkpoint。
        作用：防止所谓冒烟测试意外遍历完整正式数据集并长期占用CPU。
        """
        model = MLPBCPolicy(3, 2, 1, hidden_dims=(4,))
        batches = [
            {
                "observations": torch.randn(2, 3),
                "actions": torch.randn(2, 2),
                "category_id": torch.zeros(2, dtype=torch.long),
            }
            for _ in range(5)
        ]
        metrics = run_epoch(
            model,
            batches,
            "bc",
            torch.device("cpu"),
            torch.optim.SGD(model.parameters(), lr=1e-3),
            max_batches=2,
        )
        self.assertEqual(metrics["batch_count"], 2)
        self.assertEqual(metrics["sample_count"], 4)
        self.assertTrue(np.isfinite(metrics["loss"]))
        with self.assertRaises(ValueError):
            run_epoch(
                model,
                batches,
                "bc",
                torch.device("cpu"),
                max_batches=0,
            )


if __name__ == "__main__":
    unittest.main()
