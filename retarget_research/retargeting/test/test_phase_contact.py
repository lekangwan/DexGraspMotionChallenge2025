"""Tests for phase inference and virtual carried-object transforms."""

import sys
from pathlib import Path
import unittest

import numpy as np


RUN_DIR = Path(__file__).resolve().parents[1] / "run"
sys.path.insert(0, str(RUN_DIR))

from phase_contact import (  # noqa: E402
    TIP_SEMANTICS,
    infer_motion_phases,
    friction_wrench_residual,
    load_pad_config,
    move_with_wrist,
    select_opposing_contact_regions,
    select_reachable_opposing_contact_regions,
)


class PhaseContactTest(unittest.TestCase):
    """Check that data-derived phases and rigid motion have clear semantics."""

    def test_infers_close_and_lift_from_distances_and_wrist_z(self):
        """Two tips touch at frame2 and wrist rises 3cm at frame4."""
        frames = np.zeros((6, 28), dtype=np.float64)
        frames[:, 2] = [0.10, 0.08, 0.07, 0.07, 0.10, 0.12]
        object_vertices = np.asarray([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]])
        far = np.tile([0.1, 0.0, 0.0], (6, 1))
        source_tips = {name: far.copy() for name in TIP_SEMANTICS}
        source_tips["index"][2:] = [0.005, 0.0, 0.0]
        source_tips["thumb"][2:] = [0.006, 0.0, 0.0]

        result = infer_motion_phases(
            frames, source_tips, object_vertices, 0.01, 2, 0.03
        )

        self.assertEqual(result["close_start_frame"], 2)
        self.assertEqual(result["lift_start_frame"], 4)
        self.assertEqual(result["grasp_frame"], 4)
        self.assertEqual(result["close_detection"], "threshold")
        self.assertFalse(result["contact_fallback_used"])

    def test_nearest_fallback_uses_best_multi_tip_frame(self):
        """双指均未入阈值时，选择第二近指尖距离最小的帧并显式标记。"""
        frames = np.zeros((6, 28), dtype=np.float64)
        frames[:, 2] = [0.10, 0.08, 0.07, 0.07, 0.10, 0.12]
        object_vertices = np.asarray([[0.0, 0.0, 0.0]])
        far = np.tile([0.10, 0.0, 0.0], (6, 1))
        source_tips = {name: far.copy() for name in TIP_SEMANTICS}
        source_tips["index"][2] = [0.015, 0.0, 0.0]
        source_tips["thumb"][2] = [0.030, 0.0, 0.0]
        source_tips["index"][3] = [0.016, 0.0, 0.0]
        source_tips["thumb"][3] = [0.025, 0.0, 0.0]

        result = infer_motion_phases(
            frames,
            source_tips,
            object_vertices,
            0.01,
            2,
            0.03,
            contact_fallback="nearest",
        )

        self.assertEqual(result["close_start_frame"], 3)
        self.assertEqual(result["lift_start_frame"], 4)
        self.assertEqual(result["close_detection"], "nearest_min_contact_tips")
        self.assertTrue(result["contact_fallback_used"])
        self.assertAlmostEqual(result["close_contact_order_distance_m"], 0.025)

    def test_missing_multi_tip_contact_still_errors_without_opt_in(self):
        """默认行为保持严格，避免其他方法在不知情时改变。"""
        frames = np.zeros((4, 28), dtype=np.float64)
        object_vertices = np.asarray([[0.0, 0.0, 0.0]])
        source_tips = {
            name: np.tile([0.1, 0.0, 0.0], (4, 1)) for name in TIP_SEMANTICS
        }
        with self.assertRaisesRegex(ValueError, "没有足够指尖"):
            infer_motion_phases(
                frames, source_tips, object_vertices, 0.01, 2, 0.03
            )

    def test_moves_points_with_wrist_translation(self):
        """A 5cm wrist translation moves points equally and leaves normals fixed."""
        base = np.zeros(28)
        current = np.zeros(28)
        current[0] = 0.05
        points, normals = move_with_wrist(
            np.asarray([[0.1, 0.2, 0.3]]),
            np.asarray([[0.0, 0.0, 1.0]]),
            base,
            current,
        )
        np.testing.assert_allclose(points, [[0.15, 0.2, 0.3]], atol=1e-8)
        np.testing.assert_allclose(normals, [[0.0, 0.0, 1.0]], atol=1e-8)

    def test_opposing_side_contacts_can_balance_gravity_with_friction(self):
        """Symmetric side contacts should support gravity better than top contacts."""
        side_residual, _ = friction_wrench_residual(
            [[-1, 0, 0], [1, 0, 0], [0, 1, 0]],
            [[-1, 0, 0], [1, 0, 0], [0, 1, 0]],
            [0, 0, 0],
            friction_coefficient=1.0,
            cone_edges=4,
            torque_scale=2.0,
        )
        top_residual, _ = friction_wrench_residual(
            [[-1, 0, 1], [1, 0, 1], [0, 1, 1]],
            [[0, 0, 1], [0, 0, 1], [0, 0, 1]],
            [0, 0, 0],
            friction_coefficient=0.2,
            cone_edges=4,
            torque_scale=2.0,
        )
        self.assertLess(side_residual, top_residual)

    def test_generated_pad_configs_are_complete(self):
        """Load both calibrated configs and verify their hand identities."""
        config_dir = Path(__file__).resolve().parents[1] / "configs"
        self.assertEqual(
            load_pad_config(config_dir / "xhand_contact_pads_v1.json", "xhand")["hand"],
            "xhand",
        )
        self.assertEqual(
            load_pad_config(config_dir / "linker_contact_pads_v1.json", "linker")["hand"],
            "linker",
        )
        self.assertEqual(
            load_pad_config(config_dir / "wuji_contact_pads_v1.json", "wuji")["hand"],
            "wuji",
        )

    def test_selects_thumb_and_fingers_on_opposing_surfaces(self):
        """Opposing selector keeps semantic proximity and yields 180-degree normals."""
        vertices = np.asarray(
            [
                [-0.01, -0.02, 0.00],
                [-0.01, -0.01, 0.00],
                [-0.01, 0.00, 0.00],
                [0.01, -0.02, 0.00],
                [0.01, -0.01, 0.00],
                [0.01, 0.00, 0.00],
            ],
            dtype=np.float64,
        )
        normals = np.asarray(
            [[-1.0, 0.0, 0.0]] * 3 + [[1.0, 0.0, 0.0]] * 3,
            dtype=np.float64,
        )
        source_tips = {
            "thumb": np.asarray([-0.011, -0.01, 0.00]),
            "index": np.asarray([0.011, -0.02, 0.00]),
            "middle": np.asarray([0.011, -0.01, 0.00]),
            "ring": np.asarray([0.011, 0.00, 0.00]),
            "little": np.asarray([0.011, 0.00, 0.00]),
        }

        regions, diagnostics = select_opposing_contact_regions(
            source_tips,
            vertices,
            normals,
            candidate_neighbors=6,
            region_neighbors=1,
            distance_scale=0.03,
            opposition_weight=1.0,
        )

        self.assertEqual(set(regions), set(TIP_SEMANTICS))
        for semantic in TIP_SEMANTICS[:-1]:
            self.assertAlmostEqual(
                diagnostics[semantic]["thumb_normal_angle_deg"], 180.0
            )

    def test_reachable_selector_prefers_target_hand_pad_neighborhoods(self):
        """Target-hand pads, rather than remote Shadow tips, define reachable contacts."""
        vertices = np.asarray(
            [[-0.02, 0, 0], [-0.01, 0, 0], [0.01, 0, 0], [0.02, 0, 0]],
            dtype=np.float64,
        )
        normals = np.asarray(
            [[-1, 0, 0], [-1, 0, 0], [1, 0, 0], [1, 0, 0]],
            dtype=np.float64,
        )
        pads = {}
        for semantic in TIP_SEMANTICS:
            if semantic == "thumb":
                pads[semantic] = (
                    np.asarray([[-0.011, 0, 0]]),
                    np.asarray([[1.0, 0, 0]]),
                )
            else:
                pads[semantic] = (
                    np.asarray([[0.011, 0, 0]]),
                    np.asarray([[-1.0, 0, 0]]),
                )
        regions, diagnostics = select_reachable_opposing_contact_regions(
            pads,
            vertices,
            normals,
            candidate_neighbors=2,
            region_neighbors=1,
            distance_scale=0.02,
            opposition_weight=2.0,
            pad_alignment_weight=1.0,
            min_opposing_fingers=4,
        )
        self.assertAlmostEqual(float(regions["thumb"][0][0, 0]), -0.01)
        self.assertAlmostEqual(float(regions["index"][0][0, 0]), 0.01)
        self.assertEqual(diagnostics["selector"], "linker_reachable_pads")

    def test_reachable_selector_can_ignore_an_unreachable_finger(self):
        """A three-point grasp keeps the two closest opposing fingers."""
        vertices = np.asarray(
            [[-0.01, 0, 0], [0.01, 0, 0], [0.20, 0, 0]], dtype=np.float64
        )
        normals = np.asarray(
            [[-1, 0, 0], [1, 0, 0], [1, 0, 0]], dtype=np.float64
        )
        pads = {
            "thumb": (np.asarray([[-0.011, 0, 0]]), np.asarray([[1, 0, 0]])),
            "index": (np.asarray([[0.011, 0, 0]]), np.asarray([[-1, 0, 0]])),
            "middle": (np.asarray([[0.012, 0, 0]]), np.asarray([[-1, 0, 0]])),
            "ring": (np.asarray([[0.50, 0, 0]]), np.asarray([[-1, 0, 0]])),
            "little": (np.asarray([[0.60, 0, 0]]), np.asarray([[-1, 0, 0]])),
        }
        regions, diagnostics = select_reachable_opposing_contact_regions(
            pads, vertices, normals, 3, 1, 0.02, 2.0, 1.0, 2
        )
        self.assertEqual(set(regions), {"thumb", "index", "middle"})
        self.assertEqual(
            diagnostics["selected_opposing_fingers"], ["index", "middle"]
        )


if __name__ == "__main__":
    unittest.main()
