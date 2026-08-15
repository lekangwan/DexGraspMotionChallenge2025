"""测试Wuji手型初筛清单只从train确定性抽样。"""

from pathlib import Path
import sys
import unittest


RETARGET_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RETARGET_ROOT / "prepare"))

from build_wuji_anatomy_screen import build_screen  # noqa: E402


class WujiAnatomyScreenTest(unittest.TestCase):
    """锁定不读取物理结果、每类一条和train-only三项性质。"""

    def test_screen_is_train_only_and_deterministic(self):
        """相同种子必须得到相同两类选择，并排除valid轨迹。"""
        entries = []
        records = []
        for category in ("a", "b", "c"):
            for object_index in range(2):
                name = f"{category}{object_index}"
                entries.append(
                    {
                        "object_name": name,
                        "category": category,
                        "source_path": f"/{name}.npy",
                        "source_sha256": "0" * 64,
                        "object_asset_path": f"/{name}",
                        "available_trajectory_count": 3,
                        "frame_count": 70,
                        "action_dimension": 28,
                        "trajectory_indices": [0, 1, 2],
                    }
                )
                records.extend(
                    [
                        {"split": "train", "category": category, "object_name": name, "source_trajectory_index": 0},
                        {"split": "valid", "category": category, "object_name": name, "source_trajectory_index": 1},
                    ]
                )
        source = {"purpose": "source", "entries": entries}
        split = {"purpose": "split", "records": records}
        first = build_screen(source, split, 2, 7)
        second = build_screen(source, split, 2, 7)
        self.assertEqual(first, second)
        self.assertEqual(len(first["entries"]), 2)
        self.assertTrue(
            all(item["trajectory_indices"] == [0] for item in first["entries"])
        )
        self.assertEqual(len({item["category"] for item in first["entries"]}), 2)


if __name__ == "__main__":
    unittest.main()
