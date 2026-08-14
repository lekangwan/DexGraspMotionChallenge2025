"""Tests for frozen-manifest Wuji running and cross-hand evaluation guards."""

from argparse import Namespace
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


RETARGET_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RETARGET_ROOT / "run"))
sys.path.insert(0, str(RETARGET_ROOT / "evaluate"))

from evaluate_hand_manifest import (  # noqa: E402
    load_completed_evaluation,
    summarize_results,
    verify_target,
)
from retarget_xhand_reference import select_source_trajectories  # noqa: E402
from run_xhand_manifest import existing_output_matches as xhand_matches  # noqa: E402
from run_wuji_manifest import existing_output_matches  # noqa: E402
from wuji_candidate_utils import (  # noqa: E402
    physics_selection_score,
    trajectory_mapping_metadata,
)


class ManifestBatchingTest(unittest.TestCase):
    """Check metadata matching, target dimensions, and aggregate arithmetic."""

    def setUp(self):
        """Create one minimal two-trajectory manifest entry."""
        self.entry = {
            "object_name": "object",
            "trajectory_indices": [2, 7],
        }

    def test_wuji_resume_requires_exact_mapping_and_method(self):
        """A v2 output matches, while another mapping configuration does not."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mapping = root / "v2.json"
            mapping.write_text(json.dumps({"pairs": []}), encoding="utf-8")
            output = root / "candidate.npy"
            np.save(
                output,
                {
                    "grasp_seqs": np.zeros((2, 70, 26), dtype=np.float32),
                    "source_trajectory_indices": np.asarray([2, 7]),
                    "mapping_config": str(mapping.resolve()),
                    "maxeval": 100,
                    "source_z_offset": 0.4,
                    "joint_temporal_weight": 0.0,
                    "translation_temporal_weight": 0.0,
                    "rotation_temporal_weight": 0.0,
                },
                allow_pickle=True,
            )
            args = Namespace(
                mapping_config=mapping,
                maxeval=100,
                source_z_offset=0.4,
                joint_temporal_weight=0.0,
                translation_temporal_weight=0.0,
                rotation_temporal_weight=0.0,
            )

            self.assertTrue(existing_output_matches(output, self.entry, args))
            args.maxeval = 50
            self.assertFalse(existing_output_matches(output, self.entry, args))

    def test_cross_hand_verifier_checks_dimension_and_indices(self):
        """The same manifest accepts 26-D Wuji but rejects it as 18-D XHand."""
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "candidate.npy"
            np.save(
                output,
                {
                    "grasp_seqs": np.zeros((2, 70, 26), dtype=np.float32),
                    "source_trajectory_indices": np.asarray([2, 7]),
                },
                allow_pickle=True,
            )

            self.assertEqual(verify_target(self.entry, output, "wuji"), 2)
            with self.assertRaises(ValueError):
                verify_target(self.entry, output, "xhand")

    def test_xhand_subset_keeps_per_trajectory_fields_aligned(self):
        """Selecting source indices also selects matching rotation and scale rows."""
        source = {
            "grasp_seqs": np.arange(4 * 70 * 28).reshape(4, 70, 28),
            "obj_rotmat": np.arange(4 * 9).reshape(4, 3, 3),
            "obj_scale": np.asarray([0.1, 0.2, 0.3, 0.4]),
        }

        selected = select_source_trajectories(source, [2, 0])

        self.assertEqual(selected["grasp_seqs"].shape, (2, 70, 28))
        self.assertEqual(selected["obj_scale"].tolist(), [0.3, 0.1])
        np.testing.assert_array_equal(selected["obj_rotmat"][0], source["obj_rotmat"][2])

    def test_xhand_resume_rejects_old_reference_output_without_indices(self):
        """Only a normalized 18-D output with exact method metadata can resume."""
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "candidate.npy"
            payload = {
                "grasp_seqs": np.zeros((2, 70, 18), dtype=np.float32),
                "source_trajectory_indices": np.asarray([2, 7]),
                "iter_num": 100,
                "sample_frame_num": 5,
                "trans_lr": 5e-3,
                "ang_lr": 1e-2,
                "trans_bound": 2.0,
                "enlarge_scale": 1.0,
            }
            np.save(output, payload, allow_pickle=True)
            args = Namespace(**{key: value for key, value in payload.items() if key not in {"grasp_seqs", "source_trajectory_indices"}})

            self.assertTrue(xhand_matches(output, self.entry, args))
            del payload["source_trajectory_indices"]
            np.save(output, payload, allow_pickle=True)
            self.assertFalse(xhand_matches(output, self.entry, args))

    def test_summary_counts_success_by_trajectory(self):
        """Two objects and three trajectories produce the expected 2/3 rate."""
        results = [
            {
                "object_name": "a",
                "category": "cat1",
                "evaluation_split": "calibration",
                "success": True,
                "keypoint_mean_distance_m": 0.01,
                "max_lift_m": 0.2,
                "final_lift_m": 0.2,
                "source_trajectory_index": 0,
            },
            {
                "object_name": "a",
                "category": "cat1",
                "evaluation_split": "heldout",
                "success": False,
                "keypoint_mean_distance_m": 0.02,
                "max_lift_m": 0.0,
                "final_lift_m": 0.0,
                "source_trajectory_index": 1,
            },
            {
                "object_name": "b",
                "category": "cat2",
                "evaluation_split": "heldout",
                "success": True,
                "keypoint_mean_distance_m": 0.03,
                "max_lift_m": 0.3,
                "final_lift_m": 0.3,
                "source_trajectory_index": 2,
            },
        ]

        summary = summarize_results(results)

        self.assertEqual(summary["success_count"], 2)
        self.assertAlmostEqual(summary["success_rate"], 2 / 3)
        self.assertEqual(summary["per_object"]["a"]["success_count"], 1)
        self.assertAlmostEqual(summary["object_macro_success_rate"], 0.75)
        self.assertAlmostEqual(summary["category_macro_success_rate"], 0.75)
        self.assertEqual(summary["per_split"]["heldout"]["trajectory_count"], 2)

    def test_resume_requires_matching_reports_and_complete_trace(self):
        """续跑只接受路径/索引匹配且长度与执行速度一致的完整产物。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.npy"
            target = root / "target.npy"
            source.touch()
            target.touch()
            entry = {
                "object_name": "object",
                "category": "category",
                "source_path": str(source),
                "trajectory_indices": [2],
                "heldout_indices": [2],
            }
            item = root / "reports" / "object"
            item.mkdir(parents=True)
            common = {
                "source": str(source.resolve()),
                "target": str(target.resolve()),
                "source_trajectory_index": 2,
                "target_trajectory_index": 0,
            }
            geometry = {
                **common,
                "keypoint_mean_distance_m": 0.01,
                "keypoint_max_distance_m": 0.02,
                "max_joint_step_l2_rad": 0.03,
            }
            physics = {
                **common,
                "object_name": "object",
                "target_dimensions": 18,
                "max_lift_m": 0.2,
                "final_lift_m": 0.15,
                "hand_object_contact_steps": 40,
                "longest_sustained_lift_time_s": 0.5,
                "success": True,
            }
            (item / "source_2_geometry.json").write_text(json.dumps(geometry))
            (item / "source_2_physics.json").write_text(json.dumps(physics))
            trace = root / "traces" / "object" / "source_2_trace.npz"
            trace.parent.mkdir(parents=True)
            metadata = {
                "trace_alignment": "pre_action_state_to_command_v1",
                "hand": "xhand",
                "object_name": "object",
                "source": str(source.resolve()),
                "target": str(target.resolve()),
                "source_trajectory_index": 2,
                "target_trajectory_index": 0,
            }
            np.savez_compressed(
                trace,
                policy_action=np.zeros((240, 18), dtype=np.float32),
                source_frame_index=np.zeros(240, dtype=np.int64),
                metadata_json=np.asarray(json.dumps(metadata)),
            )

            result = load_completed_evaluation(
                "xhand", entry, target, 0, root / "reports", policy_trace_dir=root / "traces"
            )
            self.assertTrue(result["resumed_existing"])
            self.assertEqual(result["evaluation_split"], "heldout")

            physics.update({"steps_per_frame": 4, "hold_steps": 30})
            (item / "source_2_physics.json").write_text(json.dumps(physics))
            np.savez_compressed(
                trace,
                policy_action=np.zeros((310, 18), dtype=np.float32),
                source_frame_index=np.zeros(310, dtype=np.int64),
                metadata_json=np.asarray(json.dumps(metadata)),
            )
            slower_result = load_completed_evaluation(
                "xhand",
                entry,
                target,
                0,
                root / "reports",
                physics_extra_args=(
                    "--steps-per-frame",
                    "4",
                    "--hold-steps",
                    "30",
                ),
                policy_trace_dir=root / "traces",
            )
            self.assertTrue(slower_result["resumed_existing"])

            metadata["target_trajectory_index"] = 1
            np.savez_compressed(
                trace,
                policy_action=np.zeros((310, 18), dtype=np.float32),
                source_frame_index=np.zeros(310, dtype=np.int64),
                metadata_json=np.asarray(json.dumps(metadata)),
            )
            self.assertIsNone(
                load_completed_evaluation(
                    "xhand",
                    entry,
                    target,
                    0,
                    root / "reports",
                    physics_extra_args=(
                        "--steps-per-frame",
                        "4",
                        "--hold-steps",
                        "30",
                    ),
                    policy_trace_dir=root / "traces",
                )
            )

    def test_physics_selection_prefers_sustained_success(self):
        """A stable success outranks a higher instantaneous but failed lift."""
        stable = {
            "success": True,
            "longest_sustained_lift_time_s": 0.5,
            "final_lift_m": 0.11,
            "max_lift_m": 0.12,
            "hand_object_contact_steps": 40,
        }
        transient = {
            "success": False,
            "longest_sustained_lift_time_s": 0.1,
            "final_lift_m": 0.0,
            "max_lift_m": 0.4,
            "hand_object_contact_steps": 100,
        }

        self.assertGreater(
            physics_selection_score(stable), physics_selection_score(transient)
        )

    def test_mixed_wuji_metadata_is_selected_per_trajectory(self):
        """A merged candidate returns its own mapping and semantics per row."""
        data = {
            "mapping_config_per_trajectory": ["v1.json", "v2.json"],
            "mapping_semantics_per_trajectory": [["palm"], ["thumb_tip"]],
        }

        mapping, semantics = trajectory_mapping_metadata(data, 1, "default.json")

        self.assertEqual(mapping, Path("v2.json"))
        self.assertEqual(semantics, ["thumb_tip"])


if __name__ == "__main__":
    unittest.main()
