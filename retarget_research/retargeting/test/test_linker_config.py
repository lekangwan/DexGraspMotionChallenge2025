"""检查Linker O6语义点是否对应有效link并落在物理mesh范围内。

输入：Linker映射JSON、右手URDF和局部STL mesh。
输出：unittest通过，或报告link、关节中心、mesh范围或语义配置错误。
内部逻辑：解析URDF关系，并用trimesh检查配置点的局部坐标范围。
作用：防止校准点虽然数组合法，却落到错误link或手指实体之外。
"""

from __future__ import annotations

import json
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET

import numpy as np
import trimesh


RETARGET_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = RETARGET_ROOT.parent
ASSET_ROOT = (
    PROJECT_ROOT
    / "reference"
    / "HandRetargetTask2026"
    / "scripts"
    / "assets"
    / "linkerhand"
    / "o6"
    / "right"
)


def load_config():
    """读取Linker O6语义关键点配置。

    输入：无显式参数；使用固定项目路径。
    输出：解析后的配置字典。
    逻辑：读取UTF-8 JSON。
    作用：为多个测试共享同一份配置来源。
    """
    return json.loads(
        (RETARGET_ROOT / "configs" / "linker_o6_keypoint_map.json").read_text()
    )


class LinkerConfigTest(unittest.TestCase):
    """Linker物理语义点配置的结构与几何测试。"""

    def test_semantics_and_reference_indices_are_unique(self):
        """确认语义唯一且参考基线仍保持原来的10对点。

        输入：Linker配置JSON。
        输出：断言通过，或报告重复语义/索引与基线点数错误。
        逻辑：分别收集全部pair及启用参考基线的pair进行集合检查。
        作用：允许新增拇指中段点，同时保证参考对比口径明确。
        """
        pairs = load_config()["pairs"]
        semantics = [pair["semantic"] for pair in pairs]
        self.assertEqual(len(semantics), len(set(semantics)))
        baseline = [pair for pair in pairs if pair["use_in_reference_baseline"]]
        indices = [pair["reference_target_index"] for pair in baseline]
        self.assertEqual(10, len(baseline))
        self.assertEqual(len(indices), len(set(indices)))

    def test_dense_set_has_same_fifteen_source_semantics_as_other_hands(self):
        """确认Linker稠密消融由官方10点、四指中段和拇指中段组成。

        输入：Linker映射JSON的基线标记、稠密标记和拇指语义。
        输出：断言通过，或报告点数、源索引或普通指中段缺失。
        内部逻辑：合并参考点、`use_in_dense15`点和`thumb_middle`，检查15个唯一源索引。
        作用：防止所谓15点实验因重复或漏点而与XHand/Wuji口径不同。
        """
        pairs = load_config()["pairs"]
        dense = [
            pair
            for pair in pairs
            if pair["use_in_reference_baseline"]
            or pair.get("use_in_dense15", False)
            or pair["semantic"] == "thumb_middle"
        ]
        self.assertEqual(15, len(dense))
        self.assertEqual(15, len({pair["shadow_index"] for pair in dense}))
        expected_middle = {
            "index_middle_end",
            "middle_middle_end",
            "ring_middle_end",
            "little_middle_end",
        }
        self.assertTrue(expected_middle.issubset({pair["semantic"] for pair in dense}))

        by_name = {pair["semantic"]: pair for pair in pairs}
        tip = np.asarray(by_name["index_tip"]["linker_local_xyz"], dtype=float)
        for semantic in expected_middle:
            point = np.asarray(by_name[semantic]["linker_local_xyz"], dtype=float)
            np.testing.assert_allclose(point, 0.37 * tip, atol=1e-8)

    def test_proximal_points_equal_next_joint_origins(self):
        """确认四指近端标志使用URDF中的下一关节中心。

        输入：Linker配置和URDF joint父子关系。
        输出：断言通过，或报告近端点偏离DIP关节中心。
        逻辑：查找以对应proximal link为parent的关节并比较origin xyz。
        作用：证明近端点来自机械结构，而不是凭视觉填写。
        """
        root = ET.parse(ASSET_ROOT / "linkerhand_o6_right.urdf").getroot()
        child_joint_origin = {}
        for joint in root.findall("joint"):
            parent = joint.find("parent").get("link")
            origin = [float(value) for value in joint.find("origin").get("xyz").split()]
            child_joint_origin[parent] = origin
        for pair in load_config()["pairs"]:
            if pair["semantic"].endswith("proximal_end"):
                expected = child_joint_origin[pair["linker_link"]]
                np.testing.assert_allclose(pair["linker_local_xyz"], expected, atol=1e-9)

    def test_all_points_lie_inside_link_mesh_bounds(self):
        """确认每个局部标志点都位于对应link的mesh包围盒内。

        输入：Linker配置和各link的STL mesh。
        输出：断言通过，或报告落在实体范围之外的语义点。
        逻辑：按link名称找到同名mesh，检查xyz是否处于AABB并留1 mm容差。
        作用：排除把源手位置硬投影成目标局部点后落到手指实体外的情况。
        """
        mesh_dir = ASSET_ROOT / "meshes"
        tolerance = 1e-3
        for pair in load_config()["pairs"]:
            stem = pair["linker_link"]
            if stem.startswith("rh_"):
                stem = stem[3:]
            if stem.endswith("_link"):
                stem = stem[:-5]
            if stem == "hand_base":
                stem = "hand_base_link"
            mesh = trimesh.load(mesh_dir / f"{stem}.STL", process=False, force="mesh")
            point = np.asarray(pair["linker_local_xyz"], dtype=float)
            self.assertTrue(
                np.all(point >= mesh.bounds[0] - tolerance)
                and np.all(point <= mesh.bounds[1] + tolerance),
                f"{pair['semantic']} outside {stem}: {point} vs {mesh.bounds}",
            )


if __name__ == "__main__":
    unittest.main()
