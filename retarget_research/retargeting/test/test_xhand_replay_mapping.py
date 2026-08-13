"""测试XHand优化器输出到Isaac物理DOF的名称重排。

输入：每一维均可辨识的人工18维帧和打乱后的DOF名称。
输出：unittest通过，或指出错误的手腕/手指关节对应。
内部逻辑：直接调用纯NumPy映射函数并核对指定名称对应的原始维度。
作用：防止Isaac内部顺序变化导致XHand轨迹控制错手指。
"""

import unittest

import numpy as np

from retarget_research.retargeting.evaluate.xhand_replay_utils import (
    reorder_xhand_frame,
)


class XHandReplayMappingTest(unittest.TestCase):
    """覆盖XHand手腕和不同手指的按名称重排。"""

    def test_reorder_uses_semantic_joint_names(self):
        """检查打乱后的名称仍取到正确保存维度。

        输入：值为0到17的候选帧和包含手腕、食指、小指、拇指的名称列表。
        输出：按给定名称顺序排列的原始数值。
        逻辑：根据固定优化器关节表人工推导期望索引。
        作用：锁定XHand 18维动作在物理执行前的关键适配规则。
        """
        frame = np.arange(18, dtype=np.float32)
        names = [
            "right_hand_index_joint1",
            "virtual_z",
            "right_hand_pinky_joint2",
            "right_hand_thumb_bend_joint",
        ]
        actual = reorder_xhand_frame(frame, names)
        np.testing.assert_array_equal(actual, [10.0, 2.0, 17.0, 6.0])


if __name__ == "__main__":
    unittest.main()
