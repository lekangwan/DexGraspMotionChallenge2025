"""测试目标手接触分布和表面法向对抗指标。

输入：两个相对表面点、五个合成指尖和0.1 m阈值。
输出：接触数量、跨度、180度法向夹角及对向点对计数。
内部逻辑：直接调用不依赖手模型的接触统计函数。
作用：防止后续法向损失建立在角度方向或阈值判断错误的指标上。
"""

import unittest

import numpy as np

from retarget_research.retargeting.prepare.analyze_target_contact_distribution import (
    contact_distribution_metrics,
)


class ContactDistributionTest(unittest.TestCase):
    """用最小相对表面构造检查力闭合代理指标。"""

    def test_opposite_surface_normals_are_counted(self):
        """检查物体两侧的两个近指尖形成一个180度对向点对。

        输入：x轴正负表面点、相反法向、两个近指尖和三个远指尖。
        输出：接触2指、跨度2 m、最大夹角180度和1个对向点对。
        内部逻辑：KD-tree最近点与两两法向统计使用同一帧。
        作用：锁定成功/失败接触分布比较中最核心的量化含义。
        """
        names = [f"tip_{index}" for index in range(5)]
        tips = np.asarray(
            [[[-1.01, 0.0, 0.0], [1.01, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 3.0, 0.0], [0.0, 4.0, 0.0]]]
        )
        vertices = np.asarray([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        normals = np.asarray([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        report, _ = contact_distribution_metrics(
            names, tips, vertices, normals, threshold=0.1
        )
        best = report["best_frame_metrics"]
        self.assertEqual(best["contact_tip_count"], 2)
        self.assertAlmostEqual(best["max_contact_point_separation_m"], 2.0)
        self.assertAlmostEqual(best["max_contact_normal_angle_deg"], 180.0)
        self.assertEqual(best["opposing_normal_pair_count_ge_120deg"], 1)


if __name__ == "__main__":
    unittest.main()
