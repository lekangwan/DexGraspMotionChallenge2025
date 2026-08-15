"""测试Wuji解剖学边界与关节协调配置。"""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


RETARGET_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RETARGET_ROOT / "run"))

from retarget_wuji_keypoints import apply_anatomy_profile  # noqa: E402


class WujiAnatomyTest(unittest.TestCase):
    """保证手型配置只能收紧真实关节范围且能正确解析协调关系。"""

    def setUp(self):
        """建立两个最小关节及其URDF边界，供各测试复用。"""
        self.names = ["finger2_joint3", "finger2_joint4"]
        self.lower = np.asarray([-0.4932, -0.4932])
        self.upper = np.asarray([1.6272, 1.6272])

    def test_profile_tightens_bounds_and_resolves_coupling_indices(self):
        """合法配置应把-28.3度收紧为-5度，并建立PIP到DIP索引。"""
        profile = {
            "lower_bound_overrides_rad": {
                "finger2_joint3": -0.08726646,
                "finger2_joint4": -0.08726646,
            },
            "soft_flexion_couplings": [
                {
                    "proximal_joint": "finger2_joint3",
                    "distal_joint": "finger2_joint4",
                    "ratio": 2.0 / 3.0,
                    "weight": 0.02,
                }
            ],
        }
        lower, upper, couplings = apply_anatomy_profile(
            self.names, self.lower, self.upper, profile
        )
        np.testing.assert_allclose(lower, [-0.08726646, -0.08726646])
        np.testing.assert_allclose(upper, self.upper)
        self.assertEqual(couplings[0]["proximal_index"], 0)
        self.assertEqual(couplings[0]["distal_index"], 1)
        self.assertAlmostEqual(couplings[0]["ratio"], 2.0 / 3.0)

    def test_profile_cannot_widen_urdf_range(self):
        """比URDF更低的下界不是解剖约束，必须明确拒绝。"""
        with self.assertRaises(ValueError):
            apply_anatomy_profile(
                self.names,
                self.lower,
                self.upper,
                {"lower_bound_overrides_rad": {"finger2_joint4": -0.6}},
            )

    def test_profile_rejects_unknown_joint(self):
        """拼错关节名时不能静默忽略，否则实验看似运行但约束未生效。"""
        with self.assertRaises(ValueError):
            apply_anatomy_profile(
                self.names,
                self.lower,
                self.upper,
                {"lower_bound_overrides_rad": {"finger9_joint4": 0.0}},
            )


if __name__ == "__main__":
    unittest.main()
