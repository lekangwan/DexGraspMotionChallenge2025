"""Tests for XHand contact-refinement data layout and shared contact terms."""

from argparse import Namespace
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from retarget_research.retargeting.run.phase_contact import pad_contact_terms
from retarget_research.retargeting.run.retarget_xhand_contact import (
    internal_to_saved,
    saved_to_internal,
)
from retarget_research.retargeting.run.run_xhand_contact_manifest import (
    existing_output_matches,
)


class XHandContactRefinementTest(unittest.TestCase):
    """Lock down layout conversion and the sign of contact/penetration losses."""

    def test_saved_internal_round_trip(self):
        """Reordering 18 distinguishable values twice returns the original frame."""
        frames = np.arange(36, dtype=np.float32).reshape(2, 18)
        internal = saved_to_internal(frames)
        np.testing.assert_array_equal(internal[:, :12], frames[:, 6:18])
        np.testing.assert_array_equal(internal_to_saved(internal), frames)

    def test_opposite_normals_have_zero_normal_loss(self):
        """One coincident pad/target pair with opposite normals is ideal."""
        pads = {
            "thumb": (
                torch.tensor([[0.0, 0.0, 0.0]], requires_grad=True),
                torch.tensor([[1.0, 0.0, 0.0]]),
            )
        }
        targets = {
            "thumb": (
                np.asarray([[0.0, 0.0, 0.0]]),
                np.asarray([[-1.0, 0.0, 0.0]]),
            )
        }
        terms = pad_contact_terms(
            pads,
            targets,
            object_surface=None,
            contact_offset=0.0,
            min_signed_distance=-0.003,
        )
        self.assertAlmostEqual(float(terms["contact"]), 0.0)
        self.assertAlmostEqual(float(terms["normal"]), 0.0)

    def test_nearest_resume_reuses_successful_strict_output_only_forward(self):
        """严格模式成功产物可复用给nearest，反向不允许。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "baseline.npy"
            pad_config = root / "pads.json"
            output = root / "output.npy"
            baseline.touch()
            pad_config.touch()
            common = {
                "method": "xhand_official_baseline_phase_contact_refinement_v1",
                "grasp_seqs": np.zeros((2, 70, 18), dtype=np.float32),
                "source_trajectory_indices": np.asarray([2, 7]),
                "initial_target": str(baseline.resolve()),
                "contact_pad_config": str(pad_config.resolve()),
                "maxeval": 20,
                "contact_weight": 5.0,
                "normal_weight": 0.05,
                "penetration_weight": 2.0,
                "joint_prior_weight": 0.0,
                "contact_offset": -0.003,
                "min_signed_distance": -0.006,
                "contact_threshold": 0.02,
                "min_contact_tips": 2,
                "lift_delta": 0.03,
                "region_neighbors": 32,
                "contact_fallback": "error",
            }
            np.save(output, common, allow_pickle=True)
            args = Namespace(
                contact_pad_config=pad_config,
                maxeval=20,
                contact_weight=5.0,
                normal_weight=0.05,
                penetration_weight=2.0,
                joint_prior_weight=0.0,
                contact_offset=-0.003,
                min_signed_distance=-0.006,
                contact_threshold=0.02,
                min_contact_tips=2,
                lift_delta=0.03,
                region_neighbors=32,
                contact_fallback="nearest",
            )
            entry = {"trajectory_indices": [2, 7]}
            self.assertTrue(existing_output_matches(output, entry, baseline, args))

            common["contact_fallback"] = "nearest"
            np.save(output, common, allow_pickle=True)
            args.contact_fallback = "error"
            self.assertFalse(existing_output_matches(output, entry, baseline, args))


if __name__ == "__main__":
    unittest.main()
