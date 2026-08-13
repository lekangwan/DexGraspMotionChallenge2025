"""Tests for extracting hand-local contact points from Isaac-style records."""

import sys
from pathlib import Path
import unittest

import numpy as np


EVALUATE_DIR = Path(__file__).resolve().parents[1] / "evaluate"
sys.path.insert(0, str(EVALUATE_DIR))

from contact_sample_utils import collect_hand_object_local_contacts  # noqa: E402


class ContactSampleCollectionTest(unittest.TestCase):
    """Verify body-side handling and local-position selection."""

    def test_extracts_local_position_from_both_contact_sides(self):
        """Create two opposite-order contacts and preserve the hand-side point."""
        vector = np.dtype([("x", "f4"), ("y", "f4"), ("z", "f4")])
        dtype = np.dtype(
            [
                ("body0", "i4"),
                ("body1", "i4"),
                ("local_pos0", vector),
                ("local_pos1", vector),
                ("normal", vector),
            ]
        )
        contacts = np.zeros(2, dtype=dtype)
        contacts[0]["body0"], contacts[0]["body1"] = 3, 9
        contacts[0]["local_pos0"] = (0.1, 0.2, 0.3)
        contacts[0]["normal"] = (1.0, 0.0, 0.0)
        contacts[1]["body0"], contacts[1]["body1"] = 9, 4
        contacts[1]["local_pos1"] = (-0.1, -0.2, -0.3)
        contacts[1]["normal"] = (0.0, 1.0, 0.0)

        result = collect_hand_object_local_contacts(
            contacts,
            {3: "index_tip", 4: "thumb_tip"},
            [9],
            physics_step=12,
            trajectory_frame=4,
        )

        self.assertEqual([item["hand_body"] for item in result], ["index_tip", "thumb_tip"])
        self.assertEqual([item["hand_body_side"] for item in result], [0, 1])
        np.testing.assert_allclose(result[0]["hand_local_position_m"], [0.1, 0.2, 0.3])
        np.testing.assert_allclose(result[1]["hand_local_position_m"], [-0.1, -0.2, -0.3])
        self.assertEqual(result[0]["physics_step"], 12)
        self.assertEqual(result[0]["trajectory_frame"], 4)

    def test_accepts_camel_case_binding_fields(self):
        """Use the field spelling returned by Isaac Gym's NumPy binding."""
        vector = np.dtype([("x", "f4"), ("y", "f4"), ("z", "f4")])
        dtype = np.dtype(
            [
                ("body0", "i4"),
                ("body1", "i4"),
                ("localPos0", vector),
                ("localPos1", vector),
                ("normal", vector),
            ]
        )
        contacts = np.zeros(1, dtype=dtype)
        contacts[0]["body0"], contacts[0]["body1"] = 3, 9
        contacts[0]["localPos0"] = (0.01, 0.02, 0.03)
        result = collect_hand_object_local_contacts(contacts, {3: "tip"}, [9], 1, 2)
        np.testing.assert_allclose(result[0]["hand_local_position_m"], [0.01, 0.02, 0.03])


if __name__ == "__main__":
    unittest.main()
