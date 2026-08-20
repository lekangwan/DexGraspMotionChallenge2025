"""测试Wuji拇指只在闭合阶段由展开姿态过渡到抓取姿态。

输入：人工构造的短轨迹和close/lift帧。
输出：unittest通过，或指出展开、插值、抬升保持中的具体错误。
内部逻辑：不加载URDF，只验证纯NumPy阶段调度函数。
作用：防止后续修改再次让拇指在第1帧突然折叠，或破坏抬升期抓取解。
"""

import unittest

import numpy as np

from retarget_research.retargeting.run.refine_wuji_thumb_nullspace import (
    phase_aware_thumb_schedule,
    thumb_transition_frames,
)


class WujiThumbPhaseScheduleTest(unittest.TestCase):
    """覆盖接近保持、闭合平滑过渡和抬升原样保留。"""

    def test_opens_then_smoothly_blends_and_preserves_lift(self):
        """检查close前为首帧、lift后为原解且混合系数单调。

        输入：8帧轨迹，close=3、lift=6，拇指每帧线性增大。
        输出：阶段边界、结果轨迹和alpha符合定义。
        内部逻辑：比较前三帧、闭合中间帧及最后两帧。
        作用：锁定用户要求的“初始平放，抓取时才弯曲”。
        """
        optimized = np.zeros((8, 26), dtype=np.float32)
        optimized[:, :4] = np.arange(8, dtype=np.float32)[:, None]
        optimized[:, 4:] = np.arange(22, dtype=np.float32)[None, :]
        result, alpha = phase_aware_thumb_schedule(optimized, 3, 6)
        np.testing.assert_allclose(result[:4, :4], 0.0)
        self.assertEqual(float(alpha[3]), 0.0)
        self.assertEqual(float(alpha[6]), 1.0)
        self.assertTrue(np.all(np.diff(alpha) >= 0.0))
        np.testing.assert_allclose(result[6:], optimized[6:])
        np.testing.assert_allclose(result[:, 4:], optimized[:, 4:])
        self.assertGreater(float(result[5, 0]), float(result[4, 0]))

    def test_rejects_invalid_phase_order(self):
        """检查close不早于lift时明确报错，避免静默生成错误轨迹。"""
        with self.assertRaises(ValueError):
            phase_aware_thumb_schedule(np.zeros((8, 26)), 6, 3)

    def test_closes_before_contact_and_settles_before_lift(self):
        """检查接触前6帧启动，并在抬升前3帧完成闭合。"""
        start, end = thumb_transition_frames(28, 37, 70, 6, 3)
        self.assertEqual((start, end), (22, 34))


if __name__ == "__main__":
    unittest.main()
