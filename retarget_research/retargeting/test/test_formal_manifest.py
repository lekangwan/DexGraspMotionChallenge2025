"""测试正式50类抽样工具的数据协议和确定性。

输入：人工类别/物体inventory行及临时标准npy。
输出：数量、字段、划分、复现性和文件形状检查结果。
内部逻辑：直接调用纯抽样/检查函数，不依赖真实完整数据集。
作用：防止正式数据到位后才发现manifest与三手批处理入口不兼容。
"""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from retarget_research.scripts.build_manifest import (
    inspect_trajectory_file,
    sample_manifest,
)


class FormalManifestTest(unittest.TestCase):
    """覆盖正式manifest最关键的结构与输入校验。"""

    def test_sample_manifest_matches_batch_entry_protocol(self):
        """检查三级抽样数量、入口字段和calibration互斥划分。

        输入：3类、每类3物体、每物体12轨迹的人工inventory。
        输出：2类×2物体×5轨迹，且每物体2条calibration、3条heldout。
        内部逻辑：使用固定seed重复调用，并核对结果完全相同。
        作用：锁定正式manifest可被现有`run_*_manifest.py`直接读取。
        """
        rows = []
        for category in ("a", "b", "c"):
            for object_index in range(3):
                rows.append(
                    {
                        "object_id": f"{category}{object_index}",
                        "category": category,
                        "source_path": f"/{category}{object_index}.npy",
                        "source_sha256": str(object_index) * 64,
                        "object_asset_path": f"/assets/{category}{object_index}",
                        "available_trajectory_count": 12,
                        "frame_count": 70,
                        "action_dimension": 28,
                    }
                )

        first = sample_manifest(rows, 7, 2, 2, 5, 2)
        second = sample_manifest(rows, 7, 2, 2, 5, 2)

        self.assertEqual(first, second)
        self.assertEqual(first["object_count"], 4)
        self.assertEqual(first["trajectory_count"], 20)
        self.assertEqual(len(first["entries"]), 4)
        for entry in first["entries"]:
            self.assertIn("source_path", entry)
            self.assertIn("source_sha256", entry)
            self.assertEqual(len(entry["trajectory_indices"]), 5)
            self.assertEqual(len(entry["calibration_indices"]), 2)
            self.assertEqual(len(entry["heldout_indices"]), 3)
            self.assertFalse(
                set(entry["calibration_indices"]) & set(entry["heldout_indices"])
            )

    def test_inspect_trajectory_file_checks_declared_count(self):
        """检查标准npy可通过且错误声明数量会被拒绝。

        输入：临时`(3,70,28)`轨迹及同长度物体尺度/旋转。
        输出：返回`(3,70,28)`；声明为4时抛出ValueError。
        内部逻辑：把最小合法字典保存为npy后调用真实检查函数。
        作用：防止inventory轨迹数与文件内容不一致造成索引越界。
        """
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "object.npy"
            np.save(
                path,
                {
                    "grasp_seqs": np.zeros((3, 70, 28), dtype=np.float32),
                    "obj_scale": np.ones(3, dtype=np.float32),
                    "obj_rotmat": np.tile(np.eye(3), (3, 1, 1)),
                },
                allow_pickle=True,
            )
            self.assertEqual(inspect_trajectory_file(path, 3), (3, 70, 28))
            with self.assertRaises(ValueError):
                inspect_trajectory_file(path, 4)


if __name__ == "__main__":
    unittest.main()
