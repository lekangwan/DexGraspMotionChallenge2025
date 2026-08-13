"""用合成trace验证完整策略数据物化和仅train归一化规则。"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest

import numpy as np


POLICY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(POLICY_ROOT / "prepare"))

from prepare_policy_dataset import prepare_dataset


def write_trace(path, object_name, source_index, position_offset):
    """写一条4步18轴、执行前对齐的最小XHand trace。"""
    length = 4
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "trace_alignment": "pre_action_state_to_command_v1",
        "hand": "xhand",
        "object_name": object_name,
        "source_trajectory_index": source_index,
        "policy_action_order": [f"joint_{index}" for index in range(18)],
    }
    np.savez_compressed(
        path,
        hand_dof_position=np.full((length, 18), position_offset, dtype=np.float32),
        hand_dof_velocity=np.zeros((length, 18), dtype=np.float32),
        policy_action=np.full((length, 18), source_index, dtype=np.float32),
        object_position=np.asarray([[0, 0, position_offset + i * 0.01] for i in range(length)], dtype=np.float32),
        object_quaternion_xyzw=np.tile([0, 0, 0, 1], (length, 1)).astype(np.float32),
        object_linear_velocity=np.zeros((length, 3), dtype=np.float32),
        object_angular_velocity=np.zeros((length, 3), dtype=np.float32),
        hand_object_contact_count=np.arange(length, dtype=np.int64),
        source_frame_index=np.arange(length, dtype=np.int16),
        is_hold=np.zeros(length, dtype=bool),
        metadata_json=np.asarray(json.dumps(metadata)),
    )


class PrepareDatasetTest(unittest.TestCase):
    """验证trace到train/valid/test的完整最小路径。"""

    def test_prepare_uses_train_only_statistics_and_keeps_test(self):
        """valid/test极端状态不能改变均值，test成功标签为假也必须保留。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.npy"
            np.save(
                source_path,
                {
                    "obj_scale": np.ones(4, dtype=np.float32),
                    "obj_rotmat": np.tile(np.eye(3), (4, 1, 1)),
                    "grasp_seqs": np.zeros((4, 70, 28), dtype=np.float32),
                },
                allow_pickle=True,
            )
            asset = root / "asset" / "coacd"
            asset.mkdir(parents=True)
            (asset / "decomposed.obj").write_text(
                "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\n"
                "f 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n",
                encoding="utf-8",
            )
            manifest_path = root / "manifest.json"
            manifest = {
                "entries": [
                    {
                        "object_name": "train_object", "category": "cup",
                        "source_path": str(source_path), "object_asset_path": str(asset.parent),
                    },
                    {
                        "object_name": "test_object", "category": "cup",
                        "source_path": str(source_path), "object_asset_path": str(asset.parent),
                    },
                ]
            }
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            split_path = root / "split.json"
            split = {
                "records": [
                    {"split": "train", "object_name": "train_object", "category": "cup", "source_trajectory_index": 1},
                    {"split": "valid", "object_name": "train_object", "category": "cup", "source_trajectory_index": 2},
                    {"split": "test", "object_name": "test_object", "category": "cup", "source_trajectory_index": 3},
                ]
            }
            split_path.write_text(json.dumps(split), encoding="utf-8")
            trace_dir = root / "traces"
            write_trace(trace_dir / "train_object" / "source_1_trace.npz", "train_object", 1, 1.0)
            write_trace(trace_dir / "train_object" / "source_2_trace.npz", "train_object", 2, 100.0)
            write_trace(trace_dir / "test_object" / "source_3_trace.npz", "test_object", 3, 200.0)
            evaluation_path = root / "evaluation.json"
            evaluation = {
                "manifest": str(manifest_path.resolve()),
                "hand": "xhand",
                "results": [
                    {"object_name": "train_object", "source_trajectory_index": 1, "success": True},
                    {"object_name": "train_object", "source_trajectory_index": 2, "success": True},
                    {"object_name": "test_object", "source_trajectory_index": 3, "success": False},
                ],
            }
            evaluation_path.write_text(json.dumps(evaluation), encoding="utf-8")
            specs_path = root / "specs.json"
            specs_path.write_text(
                json.dumps({"hands": {"xhand": {"policy_action_dimension": 18, "physics_dof_dimension": 18}}}),
                encoding="utf-8",
            )
            output = root / "output"
            summary = prepare_dataset(
                SimpleNamespace(
                    manifest=manifest_path, policy_split=split_path, evaluation_summary=evaluation_path,
                    hand_specs=specs_path, hand="xhand", trace_dir=trace_dir,
                    output_dir=output, lift_goal=0.10,
                )
            )
            self.assertEqual(summary["split_summaries"]["test"]["trajectory_count"], 1)
            self.assertEqual(summary["observation_dimension"], 68)
            with np.load(output / "normalization.npz") as normalization, np.load(output / "train.npz") as train:
                np.testing.assert_allclose(
                    normalization["observation_mean"], train["observations"].mean(axis=0), atol=1e-6
                )
            with np.load(output / "test.npz") as test:
                self.assertFalse(test["expert_replay_success"].any())


if __name__ == "__main__":
    unittest.main()
