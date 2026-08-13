"""测试Linker接触阶段的语义关键点加权规则。

输入：掌心、普通指尖、拇指中段和拇指尖四类人工点对。
输出：阶段切换前后权重测试结果。
内部逻辑：直接调用纯权重函数，核对接触前全1和接触后分组赋权。
作用：防止调参时因语义匹配错误而意外削弱真正需要强调的指尖。
"""

import unittest

import numpy as np

from retarget_research.retargeting.run.retarget_linker_keypoints import (
    compose_frozen_lift_values,
    compose_lift_joint_residual,
    clip_start_to_bounds,
    grip_tightening_vector,
    intersect_joint_trust_region,
    semantic_weights,
    source_contact_point_weights,
    source_tip_contact_mask,
)


class LinkerObjectiveWeightsTest(unittest.TestCase):
    """覆盖阶段切换和三类语义权重分配。"""

    def setUp(self):
        """准备覆盖结构点、普通指尖和拇指点的最小pair列表。

        输入：无外部输入。
        输出：写入当前测试实例的四个语义pair。
        逻辑：只保留权重函数实际读取的semantic字段。
        作用：让两个测试共享同一份清晰、最小的输入。
        """
        self.pairs = [
            {"semantic": "palm"},
            {"semantic": "index_tip"},
            {"semantic": "thumb_middle"},
            {"semantic": "thumb_tip"},
        ]

    def test_before_contact_all_weights_are_one(self):
        """检查接触阶段开始前仍使用等权几何基线。

        输入：当前帧34、接触起始帧35和任意后期权重。
        输出：四个权重全部等于1。
        逻辑：调用阶段权重函数并与全1数组比较。
        作用：确保改进只作用于闭合/抬升段，不破坏前期接近轨迹。
        """
        actual = semantic_weights(self.pairs, 34, 35, 2.0, 3.0, 0.5)
        np.testing.assert_array_equal(actual, np.ones(4, dtype=np.float32))

    def test_after_contact_weights_follow_semantics(self):
        """检查接触阶段按结构、普通指尖和拇指正确赋权。

        输入：当前帧35及结构0.5、普通指尖2、拇指3的配置。
        输出：`[0.5,2,3,3]`。
        逻辑：普通tip只取tip权重，所有thumb语义优先取thumb权重。
        作用：锁定我们“后期优先保持指尖接触”的核心实验定义。
        """
        actual = semantic_weights(self.pairs, 35, 35, 2.0, 3.0, 0.5)
        np.testing.assert_allclose(actual, [0.5, 2.0, 3.0, 3.0])

    def test_start_values_are_clipped_inside_bounds(self):
        """检查略微越过正负π的初值会被夹回边界内侧。

        输入：包含`π+1e-7`和`-π-1e-7`的两个初值。
        输出：结果严格小于上界且严格大于下界。
        逻辑：以1e-6安全距离调用边界裁剪纯函数。
        作用：回归保护批处理中实际出现的NLopt欧拉角边界错误。
        """
        lower = np.asarray([-np.pi, -np.pi])
        upper = np.asarray([np.pi, np.pi])
        actual = clip_start_to_bounds(
            [np.pi + 1e-7, -np.pi - 1e-7], lower, upper
        )
        self.assertTrue(np.all(actual < upper))
        self.assertTrue(np.all(actual > lower))

    def test_joint_trust_region_only_limits_finger_variables(self):
        """检查硬残差范围只约束前J维手指关节，不改变手腕边界。"""
        lower = np.full(12, -2.0)
        upper = np.full(12, 2.0)
        baseline = np.arange(12, dtype=np.float64) * 0.05
        actual_lower, actual_upper = intersect_joint_trust_region(
            lower, upper, baseline, joint_count=6, delta=0.1
        )
        np.testing.assert_allclose(actual_lower[:6], baseline[:6] - 0.1)
        np.testing.assert_allclose(actual_upper[:6], baseline[:6] + 0.1)
        np.testing.assert_allclose(actual_lower[6:], lower[6:])
        np.testing.assert_allclose(actual_upper[6:], upper[6:])

    def test_frozen_lift_keeps_grasp_and_optional_wrist_residual(self):
        """检查抬升合成同时保留抓形和闭合末帧手腕修正。

        输入：当前基线、闭合优化结果和闭合末帧基线三组人工12维姿态。
        输出：前6维来自抓形，后6维等于当前基线加固定残差。
        逻辑：直接调用纯合成函数，并比较预期数组。
        作用：防止抬升首帧把优化形成的接触因手腕跳回基线而破坏。
        """
        baseline = np.zeros(12, dtype=np.float64)
        baseline[:6] = np.arange(6)
        baseline[6:9] = [1.0, 2.0, 3.0]
        baseline_before = np.zeros(12, dtype=np.float64)
        baseline_before[6:9] = [10.0, 11.0, 12.0]
        grasp = baseline_before.copy()
        grasp[:6] = np.arange(6) + 20.0
        grasp[6:9] += 0.1

        actual = compose_frozen_lift_values(
            baseline, grasp, baseline_before, carry_wrist_residual=True
        )

        np.testing.assert_allclose(actual[:6], grasp[:6])
        np.testing.assert_allclose(actual[6:9], baseline[6:9] + 0.1)
        np.testing.assert_allclose(actual[9:12], 0.0)

    def test_lift_joint_residual_preserves_dynamic_baseline(self):
        """检查固定残差叠加后，原逐帧关节变化和手腕仍被保留。"""
        baseline = np.arange(12, dtype=np.float64)
        baseline_grasp = np.arange(12, dtype=np.float64) + 10
        optimized_grasp = baseline_grasp.copy()
        optimized_grasp[:6] += 0.02
        actual = compose_lift_joint_residual(
            baseline, optimized_grasp, baseline_grasp, joint_count=6
        )
        np.testing.assert_allclose(actual[:6], baseline[:6] + 0.02)
        np.testing.assert_allclose(actual[6:], baseline[6:])

    def test_independent_joint_layout_helpers(self):
        """检查11轴模式的抓形冻结和闭合增量均使用动态切片。

        输入：17维内部姿态及拇指/四指增量。
        输出：前11维抓形被完整保留，随后6维手腕不被误当成关节。
        逻辑：调用两个纯函数并核对11轴的原机构顺序。
        作用：防止从6轴扩展时仍残留固定`[:6]`切片而破坏手腕轨迹。
        """
        baseline = np.zeros(17, dtype=np.float64)
        grasp = np.arange(17, dtype=np.float64)
        actual = compose_frozen_lift_values(
            baseline,
            grasp,
            baseline.copy(),
            carry_wrist_residual=False,
            joint_count=11,
        )
        np.testing.assert_allclose(actual[:11], grasp[:11])
        np.testing.assert_allclose(actual[11:], baseline[11:])

        tightening = grip_tightening_vector(11, 0.1, 0.2)
        np.testing.assert_allclose(
            tightening,
            [0.0, 0.1, 0.186, 0.2, 0.178, 0.2, 0.178, 0.2, 0.178, 0.2, 0.178],
        )

    def test_source_contact_only_weights_nearby_fingertips(self):
        """检查动态规则只强调专家中真正靠近物体的指尖。

        输入：两帧掌心/食指尖/拇指中段/拇指尖和原点附近物体顶点。
        输出：仅第一帧食指尖、第二帧拇指尖取得权重10。
        逻辑：构造小于或大于1厘米阈值的确定性坐标并调用KD-tree权重函数。
        作用：防止非指尖点被误加权，也锁定逐帧而非固定阶段的接触定义。
        """
        target_points = np.asarray(
            [
                [[0.0, 0.0, 0.0], [0.005, 0.0, 0.0], [0.0, 0.0, 0.0], [0.02, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.02, 0.0, 0.0], [0.0, 0.0, 0.0], [0.005, 0.0, 0.0]],
            ],
            dtype=np.float32,
        )
        actual = source_contact_point_weights(
            target_points,
            self.pairs,
            np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
            distance_threshold=0.01,
            contact_point_weight=10.0,
        )
        np.testing.assert_allclose(actual, [[1.0, 10.0, 1.0, 1.0], [1.0, 1.0, 1.0, 10.0]])

        mask = source_tip_contact_mask(
            target_points,
            self.pairs,
            np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
            distance_threshold=0.01,
        )
        np.testing.assert_array_equal(mask, np.asarray(actual) > 1.0)


if __name__ == "__main__":
    unittest.main()
