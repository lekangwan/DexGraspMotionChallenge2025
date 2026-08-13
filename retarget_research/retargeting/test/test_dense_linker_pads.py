"""测试Linker稠密接触表面的确定性纯数组部分。"""

import unittest

import numpy as np

from retarget_research.retargeting.prepare.build_dense_linker_pads import (
    farthest_point_indices,
)


class DenseLinkerPadsTest(unittest.TestCase):
    """确保稠密采样可复现、无重复并覆盖空间两端。"""

    def test_farthest_sampling_is_deterministic_and_covers_extent(self):
        """一维排列点应稳定选择中心附近点和两端点。"""
        points = np.stack(
            [np.arange(7, dtype=np.float64), np.zeros(7), np.zeros(7)], axis=1
        )
        first = farthest_point_indices(points, 3)
        second = farthest_point_indices(points, 3)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(set(first.tolist())), 3)
        self.assertIn(0, first)
        self.assertIn(6, first)


if __name__ == "__main__":
    unittest.main()
