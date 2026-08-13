"""Tests for XHand contact-refinement data layout and shared contact terms."""

import unittest

import numpy as np
import torch

from retarget_research.retargeting.run.phase_contact import pad_contact_terms
from retarget_research.retargeting.run.retarget_xhand_contact import (
    internal_to_saved,
    saved_to_internal,
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


if __name__ == "__main__":
    unittest.main()
