"""测试Wuji真实指腹细化的轨迹维度转换。

输入：人工构造的两帧26维轨迹。
输出：保存/内部顺序互换测试结果。
内部逻辑：调用两个纯列重排函数并检查可逆性。
作用：防止20个手指关节与6维手腕错位后仍被物理入口静默执行。
"""

import unittest

import numpy as np

from retarget_research.retargeting.run.refine_wuji_pad_contacts import (
    internal_to_saved,
    saved_to_internal,
)


class WujiPadRefinementTest(unittest.TestCase):
    """覆盖Wuji 6+20维布局的双向变换。"""

    def test_saved_internal_round_trip(self):
        """检查保存顺序转内部顺序后能无损恢复。

        输入：数值互不相同的`(2,26)`数组。
        输出：往返结果与输入逐元素相等。
        内部逻辑：先移动前6维到末尾，再执行逆变换。
        作用：锁定Wuji细化器和Isaac重放器之间的数据协议。
        """
        saved = np.arange(52, dtype=np.float32).reshape(2, 26)
        internal = saved_to_internal(saved)
        np.testing.assert_array_equal(internal[:, :20], saved[:, 6:])
        np.testing.assert_array_equal(internal_to_saved(internal), saved)


if __name__ == "__main__":
    unittest.main()
