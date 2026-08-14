"""测试类别均衡A/B抽样、候选复用切片和XHand动态残差合成。

输入：人工正式manifest与可辨认的NumPy候选数组。
输出：确定性、实例互斥、字段对齐和抬升动态的单元测试结果。
内部逻辑：直接调用纯函数，不启动运动学优化或Isaac Gym。
作用：确保节省时间的轨迹复用不会引入样本泄漏、错行或关节冻结。
"""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from retarget_research.retargeting.prepare.build_method_selection_ab import (
    build_ab_manifests,
)
from retarget_research.retargeting.prepare.slice_manifest_candidates import (
    slice_candidate,
)
from retarget_research.retargeting.run.refine_xhand_dynamic_residual import (
    blend_dynamic_residual,
    residual_factor,
)


class MethodSelectionReuseTest(unittest.TestCase):
    """锁定两阶段小样本协议和轻量后处理的关键不变量。"""

    @staticmethod
    def artificial_formal_manifest() -> dict:
        """构造50类、每类2物体、每物体2条calibration的最小清单。

        输入：无外部数据。
        输出：满足A/B构造器结构要求的人工manifest字典。
        内部逻辑：使用可辨认的类别、物体和索引编号。
        作用：避免单元测试依赖本机不提交Git的正式绝对路径清单。
        """
        categories = [f"category_{index:02d}" for index in range(50)]
        entries = []
        for category_index, category in enumerate(categories):
            for object_index in range(2):
                entries.append(
                    {
                        "object_name": f"{category}_object_{object_index}",
                        "category": category,
                        "source_path": f"/{category}_{object_index}.npy",
                        "source_sha256": str(object_index) * 64,
                        "object_asset_path": f"/assets/{category}_{object_index}",
                        "available_trajectory_count": 20,
                        "frame_count": 70,
                        "action_dimension": 28,
                        "trajectory_indices": [0, 1, 2, 3],
                        "calibration_indices": [category_index % 2, 2 + object_index],
                        "heldout_indices": [3],
                    }
                )
        return {"categories": categories, "entries": entries}

    def test_ab_is_deterministic_balanced_and_object_disjoint(self):
        """相同种子应生成各50条、同类不同实例且只来自calibration。

        输入：人工50类正式manifest和固定种子。
        输出：重复结果相同，A/B各一条每类且物体集合无交集。
        内部逻辑：保存临时源JSON以提供真实哈希路径，再调用构造器两次。
        作用：证明小集不是根据成功率人工挑选，也不会漏掉困难类别。
        """
        source = self.artificial_formal_manifest()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "formal.json"
            path.write_text("{}", encoding="utf-8")
            first_a, first_b = build_ab_manifests(source, 20260814, path)
            second_a, second_b = build_ab_manifests(source, 20260814, path)
        self.assertEqual(first_a, second_a)
        self.assertEqual(first_b, second_b)
        self.assertEqual(first_a["trajectory_count"], 50)
        self.assertEqual(first_b["trajectory_count"], 50)
        self.assertEqual(
            {item["category"] for item in first_a["entries"]},
            {item["category"] for item in first_b["entries"]},
        )
        self.assertFalse(
            {item["object_name"] for item in first_a["entries"]}
            & {item["object_name"] for item in first_b["entries"]}
        )
        source_by_object = {item["object_name"]: item for item in source["entries"]}
        for entry in first_a["entries"] + first_b["entries"]:
            self.assertIn(
                entry["trajectory_indices"][0],
                source_by_object[entry["object_name"]]["calibration_indices"],
            )

    def test_candidate_slice_preserves_requested_order_and_metadata_alignment(self):
        """按源编号反序请求时，动作、尺度和阶段列表必须同步反序。

        输入：三条带可辨认字段的候选和请求索引`[9,2]`。
        输出：所有轨迹级字段顺序均对应原候选第2、0行。
        内部逻辑：同时覆盖ndarray、list和全局标量三种字段类型。
        作用：防止复用完整候选时把某物体姿态或lift帧配给错误动作。
        """
        data = {
            "grasp_seqs": np.stack(
                [np.full((70, 18), value, dtype=np.float32) for value in (2, 5, 9)]
            ),
            "source_trajectory_indices": np.asarray([2, 5, 9]),
            "obj_scale": np.asarray([0.2, 0.5, 0.9]),
            "phase_metadata": [{"id": 2}, {"id": 5}, {"id": 9}],
            "method": "global_config",
        }
        output = slice_candidate(data, [9, 2], 18)
        self.assertEqual(output["source_trajectory_indices"].tolist(), [9, 2])
        self.assertEqual(output["obj_scale"].tolist(), [0.9, 0.2])
        self.assertEqual([item["id"] for item in output["phase_metadata"]], [9, 2])
        self.assertEqual(float(output["grasp_seqs"][0, 0, 0]), 9.0)
        self.assertEqual(output["method"], "global_config")

    def test_dynamic_residual_restores_official_lift_changes(self):
        """lift段应保留官方逐帧变化，并在三帧内把残差从1衰减到0。

        输入：线性变化的官方关节、闭合末帧0.2 rad残差和旧式冻结接触轨迹。
        输出：闭合段原样保留；lift系数为1、0.5、0，之后完全跟随官方动态。
        内部逻辑：使用宽关节限位排除裁剪影响并核对代表帧。
        作用：直接锁定本轮修复目标，防止代码又退回“冻结抓形”。
        """
        official = np.zeros((70, 18), dtype=np.float32)
        official[:, 6:] = np.arange(70, dtype=np.float32)[:, None] * 0.01
        contact = official.copy()
        lift_start = 40
        contact[35:lift_start, 6:] += 0.2
        contact[lift_start:, 6:] = contact[lift_start - 1, 6:]
        lower = np.full(12, -10.0, dtype=np.float32)
        upper = np.full(12, 10.0, dtype=np.float32)

        output, audit = blend_dynamic_residual(
            official, contact, lift_start, 0.0, 3, lower, upper
        )

        np.testing.assert_array_equal(output[:lift_start], contact[:lift_start])
        np.testing.assert_allclose(output[lift_start, 6:], official[lift_start, 6:] + 0.2)
        np.testing.assert_allclose(
            output[lift_start + 1, 6:], official[lift_start + 1, 6:] + 0.1
        )
        np.testing.assert_allclose(output[lift_start + 2, 6:], official[lift_start + 2, 6:])
        np.testing.assert_allclose(output[-1, 6:], official[-1, 6:])
        self.assertEqual(audit["joint_limit_clipped_value_count"], 0)
        self.assertEqual(residual_factor(lift_start, lift_start, 0.0, 3), 1.0)
        self.assertEqual(residual_factor(lift_start + 2, lift_start, 0.0, 3), 0.0)


if __name__ == "__main__":
    unittest.main()
