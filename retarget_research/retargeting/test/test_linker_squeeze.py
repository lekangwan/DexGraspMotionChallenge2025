"""测试Linker O6渐进夹紧后处理的纯数组规则。

输入：人工12维轨迹、阶段帧和统一夹紧增量。
输出：阶段、动态保留与关节限位测试结果。
内部逻辑：不加载模型或Isaac，只核对方法定义中最容易出现的索引错误。
作用：保证夹紧只改变6个主动关节，且不会再次把抬升段冻结成单帧抓形。
"""

import unittest

import numpy as np

from retarget_research.retargeting.run.refine_linker_squeeze import (
    apply_squeeze,
    lift_squeeze_progress,
    shape_lift_translation,
    squeeze_progress,
)


class LinkerSqueezeTest(unittest.TestCase):
    """覆盖夹紧进度、动作布局和物理限位。"""

    def test_progress_ramps_then_holds(self):
        """检查接近为0、闭合线性增大、抬升保持1。"""
        actual = squeeze_progress(7, close_start=2, lift_start=4)
        np.testing.assert_allclose(actual, [0, 0, 0, 0.5, 1, 1, 1])

    def test_progress_allows_immediate_lift(self):
        """首次接触与抬升同帧时，应从该帧直接保持完整夹紧。"""
        actual = squeeze_progress(6, close_start=3, lift_start=3)
        np.testing.assert_allclose(actual, [0, 0, 0, 1, 1, 1])

    def test_lift_progress_starts_after_lift_and_ramps(self):
        """检查二次夹紧只在抬升阶段逐渐启用。"""
        actual = lift_squeeze_progress(8, lift_start=3, tighten_frames=2)
        np.testing.assert_allclose(actual, [0, 0, 0, 0, 0.5, 1, 1, 1])

    def test_squeeze_preserves_wrist_and_dynamic_joint_changes(self):
        """检查抬升期是在原轨迹上加常量，而不是复制闭合末帧。"""
        frames = np.zeros((7, 12), dtype=np.float32)
        frames[:, :6] = np.arange(7, dtype=np.float32)[:, None]
        frames[:, 6:] = np.arange(7, dtype=np.float32)[:, None] * 0.05
        actual, clipped = apply_squeeze(
            frames,
            close_start=2,
            lift_start=4,
            thumb_pitch_delta=0.1,
            finger_delta=0.2,
        )
        np.testing.assert_allclose(actual[:, :6], frames[:, :6])
        self.assertAlmostEqual(float(actual[5, 8] - actual[4, 8]), 0.05, places=6)
        np.testing.assert_array_equal(clipped, 0)

    def test_lift_translation_scale_keeps_anchor_and_slows_motion(self):
        """检查减速抬升只缩放锚点之后的XYZ相对位移。"""
        frames = np.zeros((6, 12), dtype=np.float32)
        frames[:, 0] = np.arange(6, dtype=np.float32)
        frames[:, 2] = np.arange(6, dtype=np.float32) * 2
        frames[:, 3] = np.arange(6, dtype=np.float32) * 0.1
        frames[:, 6:] = 0.3
        actual = shape_lift_translation(frames, lift_start=2, scale=0.5)
        np.testing.assert_allclose(actual[:3], frames[:3])
        self.assertAlmostEqual(float(actual[4, 0]), 3.0, places=6)
        self.assertAlmostEqual(float(actual[4, 2]), 6.0, places=6)
        np.testing.assert_allclose(actual[:, 3:], frames[:, 3:])

    def test_lift_translation_cap_creates_hold_phase(self):
        """检查手腕达到统一抬升距离后保持位置。"""
        frames = np.zeros((6, 12), dtype=np.float32)
        frames[:, 2] = np.arange(6, dtype=np.float32) * 0.05
        actual = shape_lift_translation(
            frames, lift_start=1, scale=1.0, distance_cap=0.10
        )
        np.testing.assert_allclose(actual[:, 2], [0, 0.05, 0.10, 0.15, 0.15, 0.15])

    def test_lift_squeeze_adds_to_close_squeeze(self):
        """检查抬升二次增量与闭合增量相加，而不覆盖原动作。"""
        frames = np.zeros((7, 12), dtype=np.float32)
        actual, _ = apply_squeeze(
            frames,
            close_start=1,
            lift_start=3,
            finger_delta=0.2,
            lift_finger_delta=0.1,
            lift_tighten_frames=2,
        )
        self.assertAlmostEqual(float(actual[3, 8]), 0.2, places=6)
        self.assertAlmostEqual(float(actual[4, 8]), 0.25, places=6)
        self.assertAlmostEqual(float(actual[5, 8]), 0.3, places=6)

    def test_finger_floor_only_closes_under_flexed_fingers(self):
        """检查闭合下限只补偿不够弯曲的手指，不继续挤压已经闭合的手指。"""
        frames = np.zeros((7, 12), dtype=np.float32)
        frames[:, 8:] = [0.8, 1.3, 1.0, 1.4]
        actual, _ = apply_squeeze(
            frames,
            close_start=1,
            lift_start=3,
            lift_tighten_frames=2,
            lift_finger_floor=1.2,
        )
        np.testing.assert_allclose(actual[3, 8:], [0.8, 1.3, 1.0, 1.4])
        np.testing.assert_allclose(actual[5, 8:], [1.2, 1.3, 1.2, 1.4])

    def test_joint_limits_are_enforced(self):
        """检查过强夹紧不会超过O6真实主动关节上界。"""
        frames = np.zeros((5, 12), dtype=np.float32)
        frames[:, 6:] = [1.3, 0.55, 1.5, 1.5, 1.5, 1.5]
        actual, clipped = apply_squeeze(
            frames,
            close_start=1,
            lift_start=3,
            thumb_yaw_delta=0.2,
            thumb_pitch_delta=0.2,
            finger_delta=0.3,
        )
        np.testing.assert_allclose(actual[-1, 6:], [1.36, 0.58, 1.6, 1.6, 1.6, 1.6])
        self.assertTrue(np.all(clipped > 0))

    def test_signed_joint_residuals_redistribute_finger_closure(self):
        """检查独立残差可以同时放松部分手指并收紧另一根手指。"""
        frames = np.zeros((5, 12), dtype=np.float32)
        frames[:, 6:] = 0.5
        actual, _ = apply_squeeze(
            frames,
            close_start=1,
            lift_start=3,
            joint_residuals=[-0.1, 0.0, -0.1, -0.05, 0.1, 0.0],
        )
        np.testing.assert_allclose(
            actual[-1, 6:], [0.4, 0.5, 0.4, 0.45, 0.6, 0.5]
        )


if __name__ == "__main__":
    unittest.main()
