"""检查语义关键点配置与参考资产的实际展开顺序一致。

输入：Shadow/XHand参考关键点JSON和我们的XHand语义映射JSON。
输出：unittest通过，或指出点数、索引、link名称、唯一性错误。
内部逻辑：复现`use_joint21`展开规则，再逐项比对15对配置。
作用：防止底层JSON顺序变化或人工改配置后产生静默错配。
"""

from __future__ import annotations

import json
from pathlib import Path
import unittest


RETARGET_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = RETARGET_ROOT.parent
ASSETS = (
    PROJECT_ROOT
    / "reference"
    / "HandRetargetTask2026"
    / "scripts"
    / "assets"
)


def _triples(value):
    """把两种JSON局部关键点格式统一成三元组列表。

    输入：空列表、单点`[x,y,z]`或多点`[[x,y,z], ...]`。
    输出：由浮点三元组组成的列表。
    逻辑：识别嵌套层级，扁平数组则每3个数切成一个点。
    作用：兼容Shadow关键点JSON中单点和多点混用的格式。
    """
    if not value:
        return []
    if isinstance(value[0], list):
        return [tuple(map(float, point)) for point in value]
    if len(value) % 3:
        raise ValueError(f"关键点数组长度不是3的倍数: {value}")
    return [tuple(map(float, value[i : i + 3])) for i in range(0, len(value), 3)]


def _enumerate_points(data):
    """按JSON插入顺序为所有局部关键点编号。

    输入：`link名称 -> 一个或多个局部点`字典。
    输出：按模型展开顺序排列的`(link名称, 三维点)`列表。
    逻辑：逐link调用`_triples`，忽略没有关键点的link。
    作用：在不导入重型手模型的情况下复现全局关键点索引。
    """
    points = []
    for link, local_points in data.items():
        for point in _triples(local_points):
            points.append((link, point))
    return points


class XHandKeypointConfigTest(unittest.TestCase):
    """XHand关键点语义配置的轻量回归测试集合。"""

    def test_reference_indices_and_link_names(self):
        """确认21→30关键点中的15对索引和link语义一致。

        输入：无测试参数；从固定项目路径读取三个JSON。
        输出：断言通过，或报告点数、link名称及重复索引错误。
        逻辑：先复现Shadow特殊展开，再按配置逐对查询双方link。
        作用：把人工完成的一次语义核对固化成持续自动检查。
        """
        config = json.loads(
            (RETARGET_ROOT / "configs" / "xhand_keypoint_map.json").read_text()
        )

        shadow = json.loads(
            (ASSETS / "mjcf_free" / "penetration_points.json").read_text()
        )
        # 与参考 `HandModel(..., use_joint21=True)` 完全相同的展开规则。
        shadow["robot0:palm"] = [0.0, 0.0, 0.01]
        old_thumb_proximal = shadow["robot0:thproximal_child"]
        shadow["robot0:thproximal_child"] = [
            [0.0, 0.0, 0.005],
            old_thumb_proximal,
        ]
        shadow_points = _enumerate_points(shadow)

        xhand = json.loads(
            (
                ASSETS
                / "xhand_right"
                / "urdf"
                / "penetration_xhand.json"
            ).read_text()
        )
        xhand_points = _enumerate_points(xhand)

        self.assertEqual(len(shadow_points), config["source_keypoint_count"])
        self.assertEqual(len(xhand_points), config["target_keypoint_count"])
        self.assertEqual(len(config["pairs"]), 15)

        source_indices = []
        target_indices = []
        for pair in config["pairs"]:
            source_index = pair["shadow_index"]
            target_index = pair["xhand_index"]
            source_indices.append(source_index)
            target_indices.append(target_index)
            self.assertEqual(shadow_points[source_index][0], pair["shadow_link"])
            self.assertEqual(xhand_points[target_index][0], pair["xhand_link"])

        self.assertEqual(len(source_indices), len(set(source_indices)))
        self.assertEqual(len(target_indices), len(set(target_indices)))


if __name__ == "__main__":
    unittest.main()
