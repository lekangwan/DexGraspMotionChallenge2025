"""测试Wuji 26维候选到物理DOF的名称映射。

输入：每维可辨识的人工帧、20个手指名和打乱后的26个物理名。
输出：所有物理位置都取到名称对应值的测试结果。
内部逻辑：直接调用不依赖Isaac的纯映射函数。
作用：在启动昂贵物理仿真前发现手腕或五指顺序错误。
"""

import unittest

import numpy as np

from retarget_research.retargeting.evaluate.wuji_replay_utils import (
    WRIST_NAMES,
    reorder_wuji_frame,
)


class WujiReplayMappingTest(unittest.TestCase):
    """覆盖手腕和20个手指关节的乱序重排。"""

    def test_reorder_uses_names_instead_of_positions(self):
        """检查打乱物理名称后各值仍来自正确保存维度。

        输入：值为0到25的帧、标准手指名和反序物理名。
        输出：结果等于名称字典按反序查询的数组。
        内部逻辑：构造期望名称到值映射并与函数输出逐项比较。
        作用：锁定Wuji物理适配器最关键的关节顺序约束。
        """
        joints = [
            f"finger{finger}_joint{joint}"
            for finger in range(1, 6)
            for joint in range(1, 5)
        ]
        frame = np.arange(26, dtype=np.float32)
        physics_names = list(reversed(WRIST_NAMES + joints))
        actual = reorder_wuji_frame(frame, joints, physics_names)
        expected_by_name = {
            name: frame[index]
            for index, name in enumerate(WRIST_NAMES + joints)
        }
        expected = np.asarray(
            [expected_by_name[name] for name in physics_names], dtype=np.float32
        )
        np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
