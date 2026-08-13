"""测试Linker 6轴耦合与11轴解耦候选到Isaac DOF的映射规则。

输入：人工构造的一帧手腕6维和主动关节6维。
输出：unittest通过，或指出具体mimic倍率/名称重排错误。
内部逻辑：用与Isaac不同的名称顺序调用纯映射函数并逐项核对。
作用：防止物理重放看似运行，实际却把某根手指命令发给另一DOF。
"""

import unittest

import numpy as np

from retarget_research.retargeting.evaluate.linker_replay_utils import (
    expand_active_frame,
    expand_independent_frame,
    expand_linker_frame,
    linker_dof_gains,
    longest_true_run,
)
from retarget_research.retargeting.evaluate.evaluate_linker_geometry import (
    coupling_residuals,
)


class LinkerReplayMappingTest(unittest.TestCase):
    """覆盖主动关节展开和持续成功计数两个关键纯函数。"""

    def test_expand_uses_names_and_mimic_multipliers(self):
        """检查输出遵循名称顺序且正确应用1.86/0.89倍率。

        输入：自定义5个DOF顺序和可辨识的12维候选帧。
        输出：五个值分别匹配手腕、主动关节或mimic计算结果。
        逻辑：故意打乱名称，避免测试只验证固定数字索引。
        作用：锁定Isaac DOF顺序变化时仍然可靠的名称映射行为。
        """
        frame = np.asarray(
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0, 0.2, 0.3, 0.4, 0.5, 0.6],
            dtype=np.float32,
        )
        names = [
            "rh_index_dip",
            "virtual_joint_z",
            "rh_thumb_ip",
            "rh_pinky_mcp_pitch",
            "rh_ring_dip",
        ]
        actual = expand_active_frame(frame, names)
        expected = np.asarray([0.3 * 0.89, 0.3, 0.2 * 1.86, 0.6, 0.5 * 0.89])
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)

    def test_independent_mode_preserves_each_joint(self):
        """检查11轴模式不再把DIP覆盖为MCP固定倍率。

        输入：17维可辨识候选和故意打乱的四个物理DOF名称。
        输出：拇指IP、食指MCP/DIP和手腕Y分别取得各自原值。
        逻辑：让MCP和DIP值明显不满足0.89倍率，并核对名称映射。
        作用：锁定“提高自由度”在回放端真实生效，而非只改优化器输出维度。
        """
        frame = np.arange(17, dtype=np.float32) / 10.0
        names = [
            "rh_index_dip",
            "rh_thumb_ip",
            "virtual_joint_y",
            "rh_index_mcp_pitch",
        ]
        actual = expand_independent_frame(frame, names)
        np.testing.assert_allclose(actual, [1.0, 0.8, 0.1, 0.9])
        np.testing.assert_allclose(expand_linker_frame(frame, names), actual)

    def test_original_mimic_pose_has_zero_coupling_residual(self):
        """检查耦合偏差指标以原O6倍率为零点。

        输入：由6个主动量严格按1.86/0.89展开出的11轴姿态。
        输出：五个残差全部为零。
        逻辑：直接调用几何报告使用的纯计算函数。
        作用：确保后续报告的非零值确实表示新增自由度被使用。
        """
        joints = np.asarray(
            [[0.1, 0.2, 0.372, 0.3, 0.267, 0.4, 0.356, 0.5, 0.445, 0.6, 0.534]],
            dtype=np.float32,
        )
        np.testing.assert_allclose(coupling_residuals(joints), 0.0, atol=1e-7)

    def test_soft_mimic_only_changes_follower_gains(self):
        """检查柔顺参数只施加到5个从动IP/DIP轴。

        输入：平移手腕、旋转手腕、主动MCP和从动DIP四类名称。
        输出：手腕保持固定高增益，主动指保持120/5，从动指取得10/1。
        内部逻辑：故意打乱名称后调用纯增益分组函数。
        作用：防止柔顺消融意外软化手腕或6个真正的主动控制轴。
        """
        names = [
            "rh_index_dip",
            "virtual_joint_y",
            "rh_index_mcp_pitch",
            "virtual_joint_roll",
        ]
        stiffness, damping = linker_dof_gains(
            names, mimic_stiffness=10.0, mimic_damping=1.0
        )
        np.testing.assert_allclose(stiffness, [10.0, 20000.0, 120.0, 2000.0])
        np.testing.assert_allclose(damping, [1.0, 500.0, 5.0, 80.0])

    def test_longest_true_run(self):
        """检查连续成功长度不会把分散的抬升步相加。

        输入：包含两段True的布尔序列。
        输出：较长第二段的长度3。
        逻辑：调用物理评估器使用的单次扫描函数。
        作用：保证瞬时或间断抬升不会被错误累计成持续成功。
        """
        self.assertEqual(longest_true_run([True, True, False, True, True, True]), 3)


if __name__ == "__main__":
    unittest.main()
