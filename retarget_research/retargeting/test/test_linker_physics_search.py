"""测试Linker物理残差上界搜索的纯数组规则。"""

import unittest

import numpy as np

from retarget_research.retargeting.evaluate.search_linker_residual_upper_bound import (
    apply_residual,
    generate_residuals,
    generate_wrist_probes,
)


class LinkerPhysicsSearchTest(unittest.TestCase):
    """覆盖候选可复现性、阶段渐进和关节限位。"""

    def test_residual_generation_is_deterministic_and_starts_at_zero(self):
        """相同种子应生成相同候选，第一项必须复现原轨迹。"""
        first = generate_residuals(20, 0.1, 7)
        second = generate_residuals(20, 0.1, 7)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(first[0], 0)

    def test_residual_ramps_only_after_close_and_clips_limits(self):
        """闭合前不变、抬升后完整叠加，过大命令按O6限位裁剪。"""
        frames = np.zeros((6, 12), dtype=np.float32)
        frames[:, 6:] = 0.5
        actual = apply_residual(
            frames, 2, 4, np.asarray([2, 2, 2, 2, 2, 2], dtype=np.float32)
        )
        np.testing.assert_allclose(actual[:3], frames[:3])
        np.testing.assert_allclose(actual[-1, 6:], [1.36, 0.58, 1.6, 1.6, 1.6, 1.6])

    def test_wrist_probes_cover_all_coordinate_directions(self):
        """13个探针应包含零点及6维手腕的正负方向。"""
        probes = generate_wrist_probes(0.003, 0.03)
        self.assertEqual(probes.shape, (13, 6))
        np.testing.assert_array_equal(probes[0], 0)
        self.assertAlmostEqual(float(probes[1, 0]), -0.003, places=6)
        self.assertAlmostEqual(float(probes[-1, 5]), 0.03, places=6)


if __name__ == "__main__":
    unittest.main()
