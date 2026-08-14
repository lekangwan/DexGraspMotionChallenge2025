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
from retarget_research.retargeting.prepare.build_independent_confirmation_c import (
    build_confirmation_manifest,
)
from retarget_research.retargeting.run.refine_xhand_dynamic_residual import (
    blend_dynamic_residual,
    residual_factor,
)
from retarget_research.retargeting.run.refine_phase_retiming import (
    add_pre_lift_settle,
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

    def test_confirmation_c_prefers_new_objects_and_never_reuses_formal_key(self):
        """C组应优先第三实例；无第三实例时只能使用正式物体未选轨迹。

        输入：50类人工正式清单，其中49类有第三物体、最后1类没有。
        输出：49个新实例和1条已知物体新轨迹，且与正式轨迹键交集为空。
        内部逻辑：临时源文件为新物体提供真实哈希输入，再调用C组构造函数。
        作用：锁定正式结果之后的新方法不能继续复用已经看过的轨迹。
        """
        categories = [f"category_{index:02d}" for index in range(50)]
        formal_entries = []
        for category in categories:
            for object_index in range(2):
                formal_entries.append(
                    {
                        "object_name": f"{category}_formal_{object_index}",
                        "category": category,
                        "source_path": f"/{category}_{object_index}.npy",
                        "source_sha256": str(object_index) * 64,
                        "object_asset_path": f"/assets/{category}_{object_index}",
                        "available_trajectory_count": 12,
                        "frame_count": 70,
                        "action_dimension": 28,
                        "trajectory_indices": list(range(10)),
                        "calibration_indices": [0, 1],
                        "heldout_indices": list(range(2, 10)),
                    }
                )
        formal = {"categories": categories, "entries": formal_entries}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.npy"
            source.write_bytes(b"confirmation-source")
            formal_path = root / "formal.json"
            formal_path.write_text("{}", encoding="utf-8")
            inventory = [
                {
                    "object_name": f"{category}_new",
                    "category": category,
                    "source_path": str(source),
                    "object_asset_path": str(root),
                    "available_trajectory_count": 20,
                }
                for category in categories[:-1]
            ]
            confirmation = build_confirmation_manifest(
                formal, inventory, 20260814, formal_path
            )
        self.assertEqual(confirmation["new_object_instance_count"], 49)
        self.assertEqual(confirmation["known_object_new_trajectory_count"], 1)
        formal_keys = {
            (entry["object_name"], index)
            for entry in formal_entries
            for index in entry["trajectory_indices"]
        }
        confirmation_keys = {
            (entry["object_name"], entry["trajectory_indices"][0])
            for entry in confirmation["entries"]
        }
        self.assertFalse(formal_keys & confirmation_keys)

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

    def test_phase_retiming_keeps_total_length_and_full_lift_unchanged(self):
        """四帧稳定段应提前闭合，同时逐元素保留原lift段和70帧总长度。

        输入：每帧都有唯一数值的70×12轨迹，close=28、lift=37、settle=4。
        输出：原close段前移4帧，33–36帧重复第36帧，37帧以后完全不变。
        内部逻辑：通过可辨认帧编号检查拼接边界，不依赖物理仿真。
        作用：锁定“从接近阶段借时间，而不是截断或减慢抬升”的核心方法定义。
        """
        frames = np.repeat(
            np.arange(70, dtype=np.float32)[:, None], 12, axis=1
        )
        output, audit = add_pre_lift_settle(frames, 28, 37, 4)
        self.assertEqual(output.shape, (70, 12))
        np.testing.assert_array_equal(output[24:33], frames[28:37])
        np.testing.assert_array_equal(
            output[33:37], np.repeat(frames[36:37], 4, axis=0)
        )
        np.testing.assert_array_equal(output[37:], frames[37:])
        self.assertTrue(audit["lift_segment_unchanged"])
        self.assertEqual(audit["retimed_close_start_frame"], 24)

    def test_phase_retiming_supports_equal_close_and_lift_frame(self):
        """阶段回退导致close等于lift时，应提前保持完整闭合首帧而不报错。

        输入：70×12轨迹、close=lift=20及2帧稳定时间。
        输出：18–19帧重复原20帧，20帧以后仍等于原轨迹。
        内部逻辑：覆盖没有显式闭合区间的边界分支。
        作用：保证正式数据中的少数阶段回退轨迹也能使用同一全局方法。
        """
        frames = np.repeat(
            np.arange(70, dtype=np.float32)[:, None], 12, axis=1
        )
        output, audit = add_pre_lift_settle(frames, 20, 20, 2)
        np.testing.assert_array_equal(
            output[18:20], np.repeat(frames[20:21], 2, axis=0)
        )
        np.testing.assert_array_equal(output[20:], frames[20:])
        self.assertEqual(audit["settle_pose_source_frame"], 20)


if __name__ == "__main__":
    unittest.main()
