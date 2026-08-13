"""Tests for compact commanded-versus-actual DOF diagnostics."""

import sys
from pathlib import Path
import unittest

import numpy as np


EVALUATE_DIR = Path(__file__).resolve().parents[1] / "evaluate"
sys.path.insert(0, str(EVALUATE_DIR))

from tracking_metrics import summarize_dof_tracking  # noqa: E402


class TrackingSummaryTest(unittest.TestCase):
    """Verify that error magnitudes, timing, ranges, and endpoints agree."""

    def test_reports_per_dof_details(self):
        """Input two short traces and check the most important output fields."""
        commanded = np.asarray([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]])
        actual = np.asarray([[0.0, 1.2], [0.5, 2.0], [1.5, 2.0]])

        report = summarize_dof_tracking(actual, commanded, ["a", "b"])

        self.assertEqual(report["worst_tracking_dof"], "b")
        self.assertEqual(report["max_tracking_error_step_by_dof"]["b"], 2)
        self.assertEqual(report["commanded_position_range_by_dof"]["a"], [0.0, 2.0])
        self.assertEqual(report["actual_position_range_by_dof"]["b"], [1.2, 2.0])
        self.assertEqual(report["final_commanded_position_by_dof"]["b"], 3.0)
        self.assertEqual(report["final_actual_position_by_dof"]["b"], 2.0)


if __name__ == "__main__":
    unittest.main()
