"""测试Wuji二阶段精修的对向法向接触对选择。

输入：两个相反表面点、拇指/中指近点和其他远指尖。
输出：只选择拇指与中指，并生成外偏锚点的测试结果。
内部逻辑：直接调用纯KD-tree与法向筛选函数。
作用：防止新方法误选同侧手指或把锚点偏移到物体内部。
"""

import unittest

import numpy as np

from retarget_research.retargeting.run.refine_wuji_opposing_contacts import (
    select_opposing_contact_anchors,
)


class WujiContactRefinementTest(unittest.TestCase):
    """覆盖拇指—非拇指最大法向夹角选择规则。"""

    def test_selects_opposing_thumb_middle_pair(self):
        """检查只有满足距离和120度法向阈值的接触对被选择。

        输入：拇指靠近正x面、中指靠近负x面，其余指尖远离。
        输出：掩码仅thumb/middle为True，锚点沿外法向偏移5 cm。
        内部逻辑：两个最近表面法向夹角180度，满足筛选条件。
        作用：锁定二阶段方法的选择和外偏方向。
        """
        names = ["index_tip", "middle_tip", "ring_tip", "little_tip", "thumb_tip"]
        tips = np.asarray(
            [[[0.0, 3.0, 0.0], [-1.01, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 5.0, 0.0], [1.01, 0.0, 0.0]]]
        )
        vertices = np.asarray([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        normals = np.asarray([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        mask, anchors, records = select_opposing_contact_anchors(
            names,
            tips,
            vertices,
            normals,
            maximum_distance=0.02,
            minimum_normal_angle_deg=120.0,
            target_surface_offset=0.05,
        )
        np.testing.assert_array_equal(mask[0], [False, True, False, False, True])
        np.testing.assert_allclose(anchors[0, 1], [-1.05, 0.0, 0.0])
        np.testing.assert_allclose(anchors[0, 4], [1.05, 0.0, 0.0])
        self.assertEqual(records[0]["thumb_partner"], "middle_tip")


if __name__ == "__main__":
    unittest.main()
