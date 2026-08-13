"""检查Wuji 20自由度模型和Shadow语义关键点映射。

输入：Wuji映射JSON、右手URDF和26点局部关键点JSON。
输出：关节数、五指顺序、索引/link一致性测试结果。
内部逻辑：解析静态配置并按URDF/JSON顺序重建关键点索引。
作用：防止把finger1误当食指，或在优化能运行时静默匹配到错误手指。
"""

import json
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET


RETARGET_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = RETARGET_ROOT.parent
ASSET_ROOT = (
    PROJECT_ROOT
    / "reference"
    / "HandRetargetTask2026"
    / "scripts"
    / "assets"
    / "wujihand_urdf"
    / "urdf"
)


class WujiConfigTest(unittest.TestCase):
    """锁定Wuji自由度、点数量和五指语义顺序。"""

    @classmethod
    def setUpClass(cls):
        """读取测试共享的映射、URDF和局部点。

        输入：仓库内三个静态文件。
        输出：类属性config、URDF根节点和按文件顺序的点link。
        逻辑：JSON保持插入顺序，URDF用标准XML解析。
        作用：避免每个测试重复读取相同资产。
        """
        cls.config = json.loads(
            (RETARGET_ROOT / "configs" / "wuji_keypoint_map.json").read_text()
        )
        cls.urdf = ET.parse(ASSET_ROOT / "right.urdf").getroot()
        points = json.loads(
            (ASSET_ROOT / "penetration_wuji_right.json").read_text()
        )
        cls.point_links = list(points)

    def test_model_has_twenty_independent_joints_and_twenty_six_points(self):
        """检查Wuji结构规模符合本项目定义。

        输入：右手URDF全部关节和局部点键。
        输出：20个非固定关节、26个关键点link。
        逻辑：排除5个tip固定关节后计数，并核对配置声明。
        作用：保证最终动作维度应为手腕6加手指20，即26维。
        """
        active = [
            joint
            for joint in self.urdf.findall("joint")
            if joint.attrib["type"] != "fixed"
        ]
        self.assertEqual(len(active), 20)
        self.assertEqual(len(self.point_links), 26)
        self.assertEqual(self.config["target_active_dof"], 20)

    def test_each_mapping_index_names_the_same_link(self):
        """检查每个目标索引和声明link完全一致。

        输入：15个映射pair和按JSON顺序展开的26个link。
        输出：全部`point_links[wuji_index] == wuji_link`。
        逻辑：逐对查询，另外检查双方索引无重复且均不越界。
        作用：锁定优化器选择的数组位置与人类可读语义相同。
        """
        pairs = self.config["pairs"]
        self.assertEqual(len(pairs), 15)
        self.assertEqual(len({pair["semantic"] for pair in pairs}), 15)
        self.assertEqual(len({pair["shadow_index"] for pair in pairs}), 15)
        self.assertEqual(len({pair["wuji_index"] for pair in pairs}), 15)
        for pair in pairs:
            self.assertEqual(
                self.point_links[pair["wuji_index"]], pair["wuji_link"]
            )

    def test_finger_numbers_follow_thumb_to_little_order(self):
        """检查Wuji数字手指和语义手指的对应关系。

        输入：五个tip映射。
        输出：thumb/index/middle/ring/little依次对应finger1/2/3/4/5。
        逻辑：从pair中按semantic查找link前缀并与固定字典比较。
        作用：回归保护本项目最容易发生的五指编号误解。
        """
        pairs = {pair["semantic"]: pair for pair in self.config["pairs"]}
        expected = {
            "thumb_tip": "finger1_tip_link",
            "index_tip": "finger2_tip_link",
            "middle_tip": "finger3_tip_link",
            "ring_tip": "finger4_tip_link",
            "little_tip": "finger5_tip_link",
        }
        for semantic, link in expected.items():
            self.assertEqual(pairs[semantic]["wuji_link"], link)

    def test_v2_mapping_indices_name_the_same_links(self):
        """检查可视化校准后的v2仍保持合法索引和五指语义。

        输入：`wuji_keypoint_map_v2.json`和26点link顺序。
        输出：15个索引均指向配置声明的link。
        内部逻辑：读取第二份配置并逐对查询`point_links[wuji_index]`。
        作用：允许v1/v2做可复现实验，同时防止校准时手工改错索引。
        """
        config = json.loads(
            (RETARGET_ROOT / "configs" / "wuji_keypoint_map_v2.json").read_text()
        )
        self.assertEqual(len(config["pairs"]), 15)
        for pair in config["pairs"]:
            self.assertEqual(
                self.point_links[pair["wuji_index"]], pair["wuji_link"]
            )


if __name__ == "__main__":
    unittest.main()
