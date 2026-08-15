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
    apply_anatomy_dof_limits,
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

    def test_anatomy_limits_are_applied_only_after_hash_verification(self):
        """候选配置应同步收紧物理下界，文件变化后必须拒绝。"""
        import hashlib
        import json
        from pathlib import Path
        import tempfile

        dtype = np.dtype(
            [("lower", "f4"), ("upper", "f4"), ("hasLimits", "?")]
        )
        properties = np.zeros(2, dtype=dtype)
        properties["lower"] = [-0.5, -0.5]
        properties["upper"] = [1.6, 1.6]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "anatomy.json"
            path.write_text(
                json.dumps(
                    {
                        "lower_bound_overrides_rad": {"finger2_joint4": -0.087},
                        "upper_bound_overrides_rad": {},
                    }
                ),
                encoding="utf-8",
            )
            target = {
                "anatomy_config": str(path),
                "anatomy_config_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            metadata = apply_anatomy_dof_limits(
                properties, ["finger2_joint3", "finger2_joint4"], target
            )
            self.assertTrue(metadata["anatomy_limits_enforced"])
            self.assertAlmostEqual(float(properties["lower"][1]), -0.087, places=5)
            self.assertTrue(bool(properties["hasLimits"][1]))
            path.write_text("{}", encoding="utf-8")
            with self.assertRaises(ValueError):
                apply_anatomy_dof_limits(
                    properties, ["finger2_joint3", "finger2_joint4"], target
                )


if __name__ == "__main__":
    unittest.main()
